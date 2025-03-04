from collections import defaultdict
from typing import List, Tuple, Optional
from .base_reward import Reward
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, pauli_basis
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit_experiments.library.tomography.basis import Pauli6PreparationBasis
import numpy as np

from ..helpers import shots_to_precision

Indices = Tuple[int]
from ..environment.backend_info import BackendInfo
from ..environment.target import GateTarget
from ..environment.configuration.execution_config import ExecutionConfig
from ..helpers.circuit_utils import (
    handle_n_reps,
    extend_observables,
    extend_input_state_prep,
    observables_to_indices,
    pauli_input_to_indices,
    precision_to_shots,
)


@dataclass
class ChannelReward(Reward):
    """
    Configuration for computing the reward based on channel fidelity estimation
    """

    print_debug = True
    num_eigenstates_per_pauli: int = 1
    fiducials_seed: int = 2000
    input_states_seed: int = 2001

    def __post_init__(self):
        super().__post_init__()
        self._observables: Optional[SparsePauliOp] = None
        self._pauli_shots: Optional[List[int]] = None
        self._fiducials: List[Tuple[Pauli, SparsePauliOp]] = []
        self._full_fiducials: List[Tuple[List[QuantumCircuit], SparsePauliOp]] = []
        self._fiducials_indices: List[Tuple[List[Indices], List[Indices]]] = []
        self.fiducials_rng = np.random.default_rng(self.fiducials_seed)
        self.input_states_rng = np.random.default_rng(self.input_states_seed)
        self.id_coeff = 0.0
        self.id_count = 0
        self.total_counts = 0

    def set_reward_seed(self, seed: int):
        """
        Set the seed for the random number generators used in the reward computation
        """
        self.fiducials_seed = seed + 30
        self.input_states_seed = seed + 31
        self.fiducials_rng = np.random.default_rng(self.fiducials_seed)
        self.input_states_rng = np.random.default_rng(self.input_states_seed)

    @property
    def reward_args(self):
        return {"num_eigenstates_per_pauli": self.num_eigenstates_per_pauli}

    @property
    def reward_method(self):
        return "channel"

    @property
    def observables(self) -> SparsePauliOp:
        """
        Pauli observables to sample
        """
        return self._observables

    @property
    def pauli_shots(self) -> List[int]:
        """
        Number of shots per Pauli for the fidelity estimation
        """
        return self._pauli_shots

    @property
    def fiducials(self) -> List[Tuple[Pauli, SparsePauliOp]]:
        """
        Fiducial states and observables to sample
        """
        return self._fiducials

    @property
    def full_fiducials(self) -> List[Tuple[List[QuantumCircuit], SparsePauliOp]]:
        """
        Fiducial states and observables to sample
        """
        return self._full_fiducials

    @property
    def fiducials_indices(self) -> List[Tuple[List[Indices], List[Indices]]]:
        """
        Indices of the input states and observables to sample
        """
        return self._fiducials_indices

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ) -> List[EstimatorPub]:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            dfe_precision: Tuple (Ɛ, δ) from DFE paper

        Returns:
            List of pubs related to the reward method
        """
        if not isinstance(target, GateTarget):
            raise ValueError("Channel reward can only be computed for a target gate")

        if target.causal_cone_size > 3:
            raise ValueError(
                "Channel reward can only be computed for a target gate with causal cone size <= 3"
            )

        n_qubits = target.causal_cone_size
        dim = 2**n_qubits
        nb_states = self.num_eigenstates_per_pauli
        if nb_states >= dim:
            raise ValueError(
                f"Number of eigenstates per Pauli should be less than or equal to {dim}"
            )

        n_reps = execution_config.current_n_reps
        n_shots = execution_config.n_shots
        control_flow = execution_config.control_flow_enabled
        c_factor = execution_config.c_factor

        Chi = target.Chi(
            n_reps
        )  # Characteristic function for cycle circuit repeated n_reps times

        # Reset storage variables
        self._fiducials_indices = []
        self._full_fiducials = []
        self.id_coeff = 0.0
        self.id_count = 0

        # Build repeated circuit
        repeated_circuit = handle_n_reps(qc, n_reps, backend_info.backend, control_flow)
        prep_basis = Pauli6PreparationBasis()

        probabilities = Chi**2 / (dim**2)
        cutoff = 1e-8
        non_zero_indices = np.nonzero(probabilities > cutoff)[0]
        non_zero_probabilities = probabilities[non_zero_indices]
        non_zero_probabilities /= np.sum(non_zero_probabilities)

        basis = pauli_basis(
            num_qubits=n_qubits
        )  # all n_fold tensor-product single qubit Paulis

        if (
            dfe_precision is not None
        ):  # Choose if we want to sample Paulis based on DFE precision guarantee
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            pauli_sampling = execution_config.sampling_paulis

        # Sample a list of input/observable pairs
        # If one pair is sampled repeatedly, it increases number of shots used to estimate its expectation value

        self.total_counts = pauli_sampling

        samples, counts = np.unique(
            self.fiducials_rng.choice(
                non_zero_indices, size=pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )

        # Convert samples to a pair of indices in the Pauli basis
        pauli_indices = np.array(
            [np.unravel_index(sample, (dim**2, dim**2)) for sample in samples],
            dtype=int,
        )

        # Filter out the case where the identity observable is sampled (trivial case)
        identity_terms = np.where(pauli_indices[:, 1] == 0)[0]
        self.id_coeff = c_factor * np.sum(
            counts[identity_terms] / (dim * Chi[samples[identity_terms]])
        )
        self.id_count = len(identity_terms)

        pauli_indices = np.delete(pauli_indices, identity_terms, axis=0)
        samples = np.delete(samples, identity_terms)
        counts = np.delete(counts, identity_terms)

        reward_factor = (
            c_factor * counts / (dim * Chi[samples])
        )  # Based on DFE estimator
        fiducials_list = [
            (basis[p[0]], SparsePauliOp(basis[p[1]], r))
            for p, r in zip(pauli_indices, reward_factor)
        ]

        self._observables = SparsePauliOp("I" * n_qubits, self.id_coeff) + sum(
            [obs for _, obs in fiducials_list]
        )

        obs_dict = {}
        for i, (prep, obs) in enumerate(fiducials_list):
            label = prep.to_label()
            if label not in obs_dict:
                obs_dict[label] = (prep, obs, counts[i])
            else:
                _, ref_obs, ref_count = obs_dict[label]
                obs_dict[label] = (
                    prep,
                    (ref_obs + obs).simplify(),
                    max(ref_count, counts[i]),
                )

        filtered_fiducials_list = [(prep, obs) for prep, obs, _ in obs_dict.values()]
        filtered_pauli_shots = [count for _, _, count in obs_dict.values()]

        self._fiducials = filtered_fiducials_list
        self._pauli_shots = filtered_pauli_shots

        pubs = []
        total_shots = 0
        default_factory = lambda: {
            "input_circuit": None,
            "observables": None,
            "shots": None,
            "pub": None,
        }
        used_prep_indices = defaultdict(default_factory)
        fiducial_indices = {}
        full_fiducials = {}
        for (prep, obs_list), shots in zip(self._fiducials, self._pauli_shots):
            # Each prep is a Pauli input state, that we need to decompose in its pure eigenbasis
            # Below, we select at random a subset of pure input states to prepare for each prep
            # If nb_states = 1, we prepare all pure input states for each Pauli prep (no random selection)

            # self._fiducials_indices.append(([], observables_to_indices(obs_list)))
            # self._full_fiducials.append(([], obs_list))
            max_input_states = dim // nb_states
            selected_input_states: List[int] = self.input_states_rng.choice(
                dim, size=max_input_states, replace=False
            )
            prep_label = prep.to_label()
            dedicated_shots = (
                shots * n_shots // max_input_states
            )  # Number of shots per Pauli eigenstate (divided equally)
            if dedicated_shots == 0:
                continue

            for pure_eig_state in selected_input_states:
                # Convert input state to Pauli6 basis:
                # preparing pure eigenstates of Pauli_prep

                inputs = np.unravel_index(pure_eig_state, (2,) * n_qubits)
                parity = np.prod(
                    [
                        (-1) ** inputs[q_idx]
                        for q_idx, term in enumerate(reversed(prep_label))
                        if term != "I"
                    ]
                )

                prep_indices = pauli_input_to_indices(prep, inputs)

                # self._fiducials_indices[-1][0].append(tuple(prep_indices))

                # Create input state preparation circuit
                input_circuit = qc.copy_empty_like().compose(
                    prep_basis.circuit(prep_indices),
                    target.causal_cone_qubits,
                    inplace=False,
                )
                input_circuit, prep_indices = extend_input_state_prep(
                    input_circuit, qc, target, prep_indices
                )  # Add random input state on other qubits
                prep_indices = tuple(prep_indices)
                input_circuit.metadata["indices"] = prep_indices
                # Prepend input state to custom circuit with front composition
                prep_circuit = repeated_circuit.compose(
                    input_circuit,
                    front=True,
                    inplace=False,
                )
                # Transpile circuit to decompose input state preparation
                prep_circuit = backend_info.custom_transpile(
                    prep_circuit,
                    initial_layout=target.layout,
                    scheduling=False,
                    optimization_level=0,
                )

                obs_ = parity * obs_list
                pub_obs = extend_observables(obs_, prep_circuit, target).apply_layout(
                    prep_circuit.layout
                )
                if prep_indices not in used_prep_indices:  # Add new PUB
                    # Add PUB
                    pub = (
                        prep_circuit,
                        pub_obs,
                        params,
                        shots_to_precision(dedicated_shots),
                    )
                    used_prep_indices[prep_indices]["input_circuit"] = input_circuit
                    used_prep_indices[prep_indices]["observables"] = obs_
                    used_prep_indices[prep_indices]["shots"] = dedicated_shots
                    used_prep_indices[prep_indices]["pub"] = pub
                else:  # Update PUB (regroup observables for same input circuit, redundant for I/Z terms)
                    used_prep_indices[prep_indices]["observables"] += pub_obs
                    ref_prep, ref_obs, _, ref_shots = used_prep_indices[prep_indices][
                        "pub"
                    ]
                    used_prep_indices[prep_indices]["pub"] = (
                        ref_prep,
                        (ref_obs + pub_obs).simplify(),
                        params,
                        ref_shots,
                    )

        for prep_indices_, items in used_prep_indices.items():
            pub = items["pub"]
            obs_list = items["observables"]
            obs_indices = observables_to_indices(obs_list)
            input_circ = items["input_circuit"]
            pubs.append(pub)
            total_shots += (
                pub[3] * params.shape[0] * len(pub[1].group_commuting(qubit_wise=True))
            )
            fiducial_indices[prep_indices_] = obs_indices
            full_fiducials[prep_indices_] = (
                input_circ,
                obs_list.group_commuting(qubit_wise=True),
            )

        self._fiducials_indices = [
            (prep_indices, obs_indices)
            for prep_indices, obs_indices in fiducial_indices.items()
        ]
        self._full_fiducials = [
            (input_circ, obs_list) for input_circ, obs_list in full_fiducials.values()
        ]
        fiducials = [
            (prep, obs.group_commuting(qubit_wise=True))
            for prep, obs in self._fiducials
        ]
        self._fiducials = fiducials
        self._total_shots = total_shots
        return [EstimatorPub.coerce(pub) for pub in pubs]

    def get_shot_budget(self, pubs: List[EstimatorPub]) -> int:
        """
        Retrieve number of shots associated to the input pub list
        """
        total_shots = 0
        for pub in pubs:
            observables = []
            for obs_dict in pub.observables.ravel().tolist():
                for pauli_string, coeff in obs_dict.items():
                    observables.append((pauli_string, coeff))
            observables = SparsePauliOp.from_list(observables).simplify()

            total_shots += (
                pub.parameter_values.shape[0]
                * len(observables.group_commuting(qubit_wise=True))
                * precision_to_shots(pub.precision)
            )
        return total_shots

    def compute_expectation_values(
        self,
        qc: QuantumCircuit,
        target: GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ) -> List[float]:
        """
        Compute the expectation values of the desired observables on the given circuit

        Args:
            qc: Quantum circuit to be executed on quantum system
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            dfe_precision: Tuple (Ɛ, δ) from DFE paper

        Returns:
            List of expectation values of the desired observables
        """
        from qiskit.quantum_info import Operator, Statevector

        if not isinstance(target, GateTarget):
            raise ValueError("Channel reward can only be computed for a target gate")

        if target.causal_cone_size > 3:
            raise ValueError(
                "Channel reward can only be computed for a target gate with causal cone size <= 3"
            )

        n_qubits = target.causal_cone_size
        n_reps = execution_config.current_n_reps
        control_flow = execution_config.control_flow_enabled
        c_factor = execution_config.c_factor
        dim = 2**n_qubits
        Chi = target.Chi(
            n_reps
        )  # Characteristic function for cycle circuit repeated n_reps times

        # Reset storage variables
        self._fiducials_indices = []
        self._full_fiducials = []
        self.id_coeff = 0.0
        self.id_count = 0

        # Build repeated circuit
        repeated_circuit = handle_n_reps(qc, n_reps, backend_info.backend, control_flow)

        nb_states = (
            self.num_eigenstates_per_pauli
        )  # Number of eigenstates per Pauli input to sample
        if nb_states >= dim:
            raise ValueError(
                f"Number of eigenstates per Pauli should be less than or equal to {dim}"
            )

        probabilities = Chi**2 / (dim**2)
        non_zero_indices = np.nonzero(probabilities)[0]  # Filter out zero probabilities
        non_zero_probabilities = probabilities[non_zero_indices]

        basis = pauli_basis(
            num_qubits=n_qubits
        )  # all n_fold tensor-product single qubit Paulis

        if (
            dfe_precision is not None
        ):  # Choose if we want to sample Paulis based on DFE precision guarantee
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            pauli_sampling = execution_config.sampling_paulis

        # Sample a list of input/observable pairs
        # If one pair is sampled repeatedly, it increases number of shots used to estimate its expectation value

        self.total_counts = pauli_sampling

        samples, counts = np.unique(
            self.fiducials_rng.choice(
                non_zero_indices, size=pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )

        # Convert samples to a pair of indices in the Pauli basis
        pauli_indices = np.array(
            [np.unravel_index(sample, (dim**2, dim**2)) for sample in samples],
            dtype=int,
        )

        # Filter the case where the identity observable is sampled (trivial case as it serves as offset coefficient)
        filtered_pauli_indices, filtered_samples, filtered_counts = [], [], []
        for i, (indices, sample) in enumerate(zip(pauli_indices, samples)):
            if indices[1] == 0:  # 0 is the index corresponding to 'I'*n_qubits
                self.id_coeff += c_factor * counts[i] / (dim * Chi[sample])
                self.id_count += 1
            else:
                filtered_pauli_indices.append(indices)
                filtered_samples.append(sample)
                filtered_counts.append(counts[i])

        pauli_indices = filtered_pauli_indices
        samples = filtered_samples
        counts = filtered_counts

        reward_factor = [
            c_factor * count / (dim * Chi[p]) for count, p in zip(counts, samples)
        ]  # Based on DFE estimator
        fiducials_list = [
            (basis[p[0]], extend_observables(SparsePauliOp(basis[p[1]], r), qc, target))
            for p, r in zip(pauli_indices, reward_factor)
        ]

        self._observables = SparsePauliOp("I" * n_qubits, self.id_coeff) + sum(
            [obs for _, obs in fiducials_list]
        )
        filtered_fiducials_list, filtered_pauli_shots, used_prep_paulis = [], [], []
        for i, (prep, obs) in enumerate(fiducials_list):
            # Regroup observables for same input Pauli
            label = prep.to_label()
            should_add_pauli = label not in used_prep_paulis
            if should_add_pauli:
                used_prep_paulis.append(label)
                filtered_fiducials_list.append((prep, obs))
                filtered_pauli_shots.append(counts[i])

            else:
                index = used_prep_paulis.index(label)
                _, ref_obs = filtered_fiducials_list[index]
                filtered_fiducials_list[index] = (
                    prep,
                    (ref_obs + obs).simplify(),
                )
                filtered_pauli_shots[index] = max(
                    filtered_pauli_shots[index], counts[i]
                )

        for i, (prep, obs) in enumerate(filtered_fiducials_list):
            filtered_fiducials_list[i] = (prep, obs.group_commuting(qubit_wise=True))

        self._fiducials = filtered_fiducials_list
        self._pauli_shots = filtered_pauli_shots

        noisy_expectation_values = []
        expectation_values = []
        repeated_unitary = Operator(repeated_circuit)

        for (prep, obs_list), shots in zip(self._fiducials, self._pauli_shots):
            prep: Pauli
            obs_list: List[SparsePauliOp]
            self._fiducials_indices.append(([], observables_to_indices(obs_list)))
            self._full_fiducials.append(([], obs_list))
            max_input_states = dim // nb_states
            selected_input_states: List[int] = self.input_states_rng.choice(
                dim, size=max_input_states, replace=False
            )

            for pure_eig_state in selected_input_states:
                # Convert input state to Pauli6 basis: preparing pure eigenstates of Pauli_prep
                prep_label = prep.to_label()
                inputs = np.unravel_index(pure_eig_state, (2,) * n_qubits)
                parity = 1
                for q_idx, symbol in enumerate(reversed(prep_label)):
                    if symbol != "I":
                        parity *= (-1) ** inputs[q_idx]

                # Create input state preparation circuit
                input_circuit = qc.copy_empty_like()
                input_circuit.compose(
                    Pauli6PreparationBasis().circuit(inputs),
                    target.causal_cone_qubits,
                    inplace=True,
                )  # Apply input state on causal cone qubits
                input_circuit, prep_indices = extend_input_state_prep(
                    input_circuit, qc, target, inputs
                )  # Add random input state on other qubits

                prep_circuit = repeated_circuit.compose(
                    input_circuit,
                    front=True,
                    inplace=False,
                )
                prep_circuit.save_expectation_value(parity * sum(obs_list), qc.qubits)
                backend = backend_info.backend
                prep_circuit = backend_info.custom_transpile(
                    prep_circuit,
                    initial_layout=target.layout,
                    scheduling=False,
                    optimization_level=0,
                )
                prep_circuit.metadata["indices"] = prep_indices
                self._full_fiducials[-1][0].append(prep_circuit)
                job = backend.run(prep_circuit, shots=shots)
                result = job.result()
                noisy_exp_value = result.data(0).get("expectation_value")
                noisy_expectation_values.append(noisy_exp_value)

                # Prepare the input state vector
                input_unitary = Operator(input_circuit)
                output_state = Statevector.from_int(0, (2,) * n_qubits)
                output_state = output_state.evolve(input_unitary).evolve(
                    repeated_unitary
                )

                ideal_exp_value = output_state.expectation_value(parity * sum(obs_list))
                expectation_values.append(ideal_exp_value)

        return noisy_expectation_values, expectation_values
