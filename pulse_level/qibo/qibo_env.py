from typing import List

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.primitives.backend_estimator_v2 import Options
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import PauliList

import numpy as np
from qiskit.primitives import BackendEstimatorV2, PubResult
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.qasm3 import dumps as qasm3_dumps


def qibo_execute(
    circuits: List[QuantumCircuit],
    parameter_values: List[np.ndarray],
    backend,
    **run_options,
) -> list[dict]:
    """Execute the circuits on a backend.
    Args:
        circuits: The circuits
        parameter_values: The parameter values
        backend: The backend
        **run_options: run_options (notably the shots)
    Returns:
        The result
    """
    qasm_circuits = [qasm3_dumps(circuit) for circuit in circuits]

    pass


def _run_circuits(
    circuits: QuantumCircuit | list[QuantumCircuit],
    backend,  # Qibo Backend
    **run_options,
) -> tuple[list[dict], list[dict]]:  # Counts and metadata:
    """Remove metadata of circuits and run the circuits on a backend.
    Args:
        circuits: The circuits
        backend: The backend
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
        circ.metadata = {}

    counts = qibo_execute(
        circuits,
        [metadata_["parameter_values"] for metadata_ in metadata],
        backend,
        **run_options,
    )
    return counts, metadata


class QiboEstimatorV2(BackendEstimatorV2):
    """
    Estimator for a Qibo backend.
    """

    def __init__(self, backend, options=None):

        self._backend = backend
        self._options = Options(**options) if options else Options()
        opt1q = Optimize1qGatesDecomposition(
            basis=["h", "x", "y", "z", "t", "s", "sdg", "tdg"]
        )
        self._passmanager = PassManager([opt1q])

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
                dag = circuit_to_dag(circ)
                op_nodes = dag.op_nodes()
                for node in op_nodes:
                    if node.name not in gate_map:
                        gate_name = node.name.split("_")[0]
                        qc = QuantumCircuit(list(node.qargs))
                        qc.append(gate_map[gate_name], node.qargs)
                        dag.substitute_node_with_dag(node, circuit_to_dag(qc))
                new_circuits2.append(dag_to_circuit(dag))
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
            self._backend,
            shots=shots,
            seed_simulator=self._options.seed_simulator,
        )

        results = []
        start = 0
        for pub, data in zip(pubs, preprocessed_data):
            end = start + len(data.circuits)
            expval_map = self._calc_expval_map(counts[start:end], metadata[start:end])
            start = end
            results.append(self._postprocess_pub(pub, expval_map, data, shots))
        return results
