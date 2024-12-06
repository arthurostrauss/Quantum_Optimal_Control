import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Callable

from qiskit import QuantumCircuit, QiskitError, QuantumRegister, ClassicalRegister
from qiskit.primitives.backend_estimator import _pauli_expval_with_variance
from qiskit.primitives.backend_estimator_v2 import Options, _PreprocessedData
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.quantum_info import PauliList, Pauli

import numpy as np
from qiskit.primitives import (
    PubResult,
    BaseEstimatorV2,
    PrimitiveResult,
    PrimitiveJob,
    DataBin,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.result import Counts
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.qasm3 import dumps as qasm3_dumps
from qibo.backends import set_backend

from qibo import Circuit as QiboCircuit

from .utils import execute_action, resolve_gate_rule
from .qibo_environment import from_custom_to_baseline_circuit


@dataclass
class QiboOptions(Options):
    """Options for the Qibo backend."""

    qubits: Tuple[int] = (0, 1)
    gate_rule: str | Tuple[str, Callable] = "rx"


def qibo_execute(
    circuits: List[QuantumCircuit],
    parameter_values: List[np.ndarray],
    platform,
    **run_options,
) -> list[dict]:
    """Execute the circuits on a platform.
    Args:
        circuits: The circuits
        parameter_values: The parameter values
        platform: The platform
        **run_options: run_options (notably the shots)
    Returns:
        The result
    """
    qasm_circuits = [qasm3_dumps(circuit) for circuit in circuits]
    qibo_circuits = [
        QiboCircuit.from_qasm(qasm_circuit) for qasm_circuit in qasm_circuits
    ]

    set_backend("qibolab", platform=platform)
    hardware_qubit_pair = run_options.get(
        "qubits", (0, 1)
    )  # TODO: Figure out how to retrieve it

    # for parameter_value in parameter_values:
    param_shape = parameter_values[0].shape
    assert len(param_shape) == 1, "Only 1D parameter values are supported"

    rounded_params = np.round(parameter_values, decimals=6)
    unique_params = np.unique(rounded_params, axis=0)
    circuit_batches = []
    for param_set in unique_params:
        batch = []
        for qiskit_circ, qibo_circ in zip(circuits, qibo_circuits):
            if np.allclose(
                qiskit_circ.metadata["parameter_values"], param_set, atol=1e-6
            ):
                batch.append(qibo_circ)
        circuit_batches.append(batch)

    results = []
    for param_set, circuit_batch in zip(unique_params, circuit_batches):
        results.extend(
            execute_action(
                platform,
                circuit_batch,
                hardware_qubit_pair,
                param_set,
                run_options["shots"],
                resolve_gate_rule(run_options.get("gate_rule", "rx")),
            )
        )

    return results


def _run_circuits(
    # circuits: QuantumCircuit | list[QuantumCircuit], #TODO: Fix type
    circuits,
    platform,  # Qibo Backend
    **run_options,
) -> tuple[list[dict], list[dict]]:  # Counts and metadata:
    """Remove metadata of circuits and run the circuits on a platform.
    Args:
        circuits: The circuits
        platform: The platform
        monitor: Enable job minotor if True
        **run_options: run_options
    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        # circ.metadata = {}
    counts = qibo_execute(
        circuits,
        [metadata_["parameter_values"] for metadata_ in metadata],
        platform,
        **run_options,
    )
    return counts, metadata


class QiboEstimatorV2(BaseEstimatorV2):
    """
    Estimator for a Qibo platform.
    """

    def __init__(self, platform, options=None):

        self._platform = platform
        self._options = QiboOptions(**options) if options else QiboOptions()
        opt1q = Optimize1qGatesDecomposition(
            basis=["h", "x", "y", "z", "t", "s", "sdg", "tdg"]
        )
        self._passmanager = PassManager([opt1q])

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        for i, pub in enumerate(pubs):
            if pub.precision <= 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than or equal to 0 ({pub.precision}). ",
                    "But precision should be larger than 0.",
                )

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            shots = int(math.ceil(1.0 / pub.precision**2))
            pub_dict[shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs([pubs[i] for i in lst], shots)
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results, metadata={"version": 2})

    def _bind_and_add_measurements(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray,
        param_obs_map: dict[tuple[int, ...], set[str]],
    ) -> list[QuantumCircuit]:
        """Bind the given circuit against each parameter value set, and add necessary measurements
        to each.

        Args:
            circuit: The (possibly parametric) circuit of interest.
            parameter_values: An array of parameter value sets that can be applied to the circuit.
            param_obs_map: A mapping from locations in ``parameter_values`` to a sets of
                Pauli terms whose expectation values are required in those locations.

        Returns:
            A flat list of circuits sufficient to measure all Pauli terms in the ``param_obs_map``
            values at the corresponding ``parameter_values`` location, where requisite
            book-keeping is stored as circuit metadata.
        """
        circuits = []
        gate_map = get_standard_gate_name_mapping()
        for param_index, pauli_strings in param_obs_map.items():
            bound_circuit = parameter_values.bind(circuit, param_index)

            # sort pauli_strings so that the order is deterministic
            meas_paulis = PauliList(sorted(pauli_strings))
            new_circuits = self._create_measurement_circuits(
                bound_circuit, meas_paulis, param_index
            )
            new_circuits2 = []
            for circ in new_circuits:
                circ.metadata["parameter_values"] = parameter_values[
                    param_index
                ].as_array()
                new_circ = from_custom_to_baseline_circuit(circ)
                new_circuits2.append(new_circ)
            circuits.extend(new_circuits2)
        return circuits

    def _run_pubs(self, pubs: list[EstimatorPub], shots: int) -> list[PubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        preprocessed_data = []
        flat_circuits = []
        for pub in pubs:
            data = self._preprocess_pub(pub)
            preprocessed_data.append(data)
            flat_circuits.extend(data.circuits)

        counts, metadata = _run_circuits(
            flat_circuits,
            self._platform,
            qubits=self.options.qubits,
            shots=shots,
            seed_simulator=self._options.seed_simulator,
            gate_rule=self.options.gate_rule,
        )

        results = []
        start = 0
        for pub, data, count in zip(pubs, preprocessed_data, counts):
            end = start + len(data.circuits)
            expval_map = self._calc_expval_map(counts[start:end], metadata[start:end])
            start = end
            results.append(self._postprocess_pub(pub, expval_map, data, shots, count))
        return results

    def _preprocess_pub(self, pub: EstimatorPub) -> _PreprocessedData:
        """Converts a pub into a list of bound circuits necessary to estimate all its observables.

        The circuits contain metadata explaining which bindings array index they are with respect to,
        and which measurement basis they are measuring.

        Args:
            pub: The pub to preprocess.

        Returns:
            The values ``(circuits, bc_param_ind, bc_obs)`` where ``circuits`` are the circuits to
            execute on the backend, ``bc_param_ind`` are indices of the pub's bindings array and
            ``bc_obs`` is the observables array, both broadcast to the shape of the pub.
        """
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values

        # calculate broadcasting of parameters and observables
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(
            param_shape
        )
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        param_obs_map = defaultdict(set)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            param_obs_map[param_index].update(bc_obs[index])

        bound_circuits = self._bind_and_add_measurements(
            circuit, parameter_values, param_obs_map
        )
        return _PreprocessedData(bound_circuits, bc_param_ind, bc_obs)

    def _postprocess_pub(
        self,
        pub: EstimatorPub,
        expval_map: dict,
        data: _PreprocessedData,
        shots: int,
        counts: list[Counts],
    ) -> PubResult:
        """Computes expectation values (evs) and standard errors (stds).

        The values are stored in arrays broadcast to the shape of the pub.

        Args:
            pub: The pub to postprocess.
            expval_map: The map
            data: The result data of the preprocessing.
            shots: The number of shots.

        Returns:
            The pub result.
        """
        bc_param_ind = data.parameter_indices
        bc_obs = data.observables
        evs = np.zeros_like(bc_param_ind, dtype=float)
        variances = np.zeros_like(bc_param_ind, dtype=float)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            for pauli, coeff in bc_obs[index].items():
                expval, variance = expval_map[param_index, pauli]
                evs[index] += expval * coeff
                variances[index] += np.abs(coeff) * variance**0.5
        stds = variances / np.sqrt(shots)
        data_bin = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data_bin,
            metadata={
                "target_precision": pub.precision,
                "shots": shots,
                "circuit_metadata": pub.circuit.metadata,
                "counts": counts,
            },
        )

    def _calc_expval_map(
        self,
        counts: list[Counts],
        metadata: dict,
    ) -> dict[tuple[tuple[int, ...], str], tuple[float, float]]:
        """Computes the map of expectation values.

        Args:
            counts: The counts data.
            metadata: The metadata.

        Returns:
            The map of expectation values takes a pair of an index of the bindings array and
            a pauli string as a key and returns the expectation value of the pauli string
            with the the pub's circuit bound against the parameter value set in the index of
            the bindings array.
        """
        expval_map: dict[tuple[tuple[int, ...], str], tuple[float, float]] = {}
        for count, meta in zip(counts, metadata):
            orig_paulis = meta["orig_paulis"]
            meas_paulis = meta["meas_paulis"]
            param_index = meta["param_index"]
            expvals, variances = _pauli_expval_with_variance(count, meas_paulis)
            for pauli, expval, variance in zip(orig_paulis, expvals, variances):
                expval_map[param_index, pauli.to_label()] = (expval, variance)
        return expval_map

    def _create_measurement_circuits(
        self,
        circuit: QuantumCircuit,
        observable: PauliList,
        param_index: tuple[int, ...],
    ) -> list[QuantumCircuit]:
        """Generate a list of circuits sufficient to estimate each of the given Paulis.

        Paulis are divided into qubitwise-commuting subsets to reduce the total circuit count.
        Metadata is attached to circuits in order to remember what each one measures, and
        where it belongs in the output.

        Args:
            circuit: The circuit of interest.
            observable: Which Pauli terms we would like to observe.
            param_index: Where to put the data we estimate (only passed to metadata).

        Returns:
            A list of circuits sufficient to estimate each of the given Paulis.
        """
        meas_circuits: list[QuantumCircuit] = []
        if self._options.abelian_grouping:
            for obs in observable.group_commuting(qubit_wise=True):
                basis = Pauli(
                    (np.logical_or.reduce(obs.z), np.logical_or.reduce(obs.x))
                )
                meas_circuit, indices = _measurement_circuit(circuit.num_qubits, basis)
                paulis = PauliList.from_symplectic(
                    obs.z[:, indices],
                    obs.x[:, indices],
                    obs.phase,
                )
                meas_circuit.metadata = {
                    "orig_paulis": obs,
                    "meas_paulis": paulis,
                    "param_index": param_index,
                }
                meas_circuits.append(meas_circuit)
        else:
            for basis in observable:
                meas_circuit, indices = _measurement_circuit(circuit.num_qubits, basis)
                obs = PauliList(basis)
                paulis = PauliList.from_symplectic(
                    obs.z[:, indices],
                    obs.x[:, indices],
                    obs.phase,
                )
                meas_circuit.metadata = {
                    "orig_paulis": obs,
                    "meas_paulis": paulis,
                    "param_index": param_index,
                }
                meas_circuits.append(meas_circuit)

        # unroll basis gates
        meas_circuits = self._passmanager.run(meas_circuits)

        # combine measurement circuits
        preprocessed_circuits = []
        for meas_circuit in meas_circuits:
            circuit_copy = circuit.copy()
            # meas_circuit is supposed to have a classical register whose name is different from
            # those of the transpiled_circuit
            clbits = meas_circuit.cregs[0]
            for creg in circuit_copy.cregs:
                if clbits.name == creg.name:
                    raise QiskitError(
                        "Classical register for measurements conflict with those of the input "
                        f"circuit: {clbits}. "
                        "Recommended to avoid register names starting with '__'."
                    )
            circuit_copy.add_register(clbits)
            circuit_copy.compose(meas_circuit, clbits=clbits, inplace=True)
            circuit_copy.metadata = meas_circuit.metadata
            preprocessed_circuits.append(circuit_copy)
        return preprocessed_circuits

    @property
    def options(self):
        return self._options


def _measurement_circuit(num_qubits: int, pauli: Pauli):
    # Note: if pauli is I for all qubits, this function generates a circuit to measure only
    # the first qubit.
    # Although such an operator can be optimized out by interpreting it as a constant (1),
    # this optimization requires changes in various methods. So it is left as future work.
    qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
    if not np.any(qubit_indices):
        qubit_indices = [0]
    meas_circuit = QuantumCircuit(
        QuantumRegister(num_qubits, "q"),
        ClassicalRegister(len(qubit_indices), f"__c_{pauli}"),
    )
    for clbit, i in enumerate(qubit_indices):
        if pauli.x[i]:
            if pauli.z[i]:
                meas_circuit.sdg(i)
            meas_circuit.h(i)
        meas_circuit.measure(i, clbit)
    return meas_circuit, qubit_indices
