from typing import List, Tuple, Optional
from .base_reward import Reward
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, pauli_basis
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit_experiments.library.tomography.basis import Pauli6PreparationBasis
import numpy as np

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

        samples, self._pauli_shots = np.unique(
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
        filtered_pauli_indices, filtered_samples = [], []
        for i, (indices, sample) in enumerate(zip(pauli_indices, samples)):
            if indices[1] == 0:  # 0 is the index corresponding to 'I'*n_qubits
                self.id_coeff += c_factor / (dim * Chi[sample])
                self.id_count += 1
            else:
                filtered_pauli_indices.append(indices)
                filtered_samples.append(sample)

        pauli_indices = filtered_pauli_indices
        samples = filtered_samples

        reward_factor = [
            c_factor / (dim * Chi[p]) for p in samples
        ]  # Based on DFE estimator
        fiducials_list = [
            (basis[p[0]], extend_observables(SparsePauliOp(basis[p[1]], r), qc, target))
            for p, r in zip(pauli_indices, reward_factor)
        ]

        self._observables = sum([obs for _, obs in fiducials_list])
        filtered_fiducials_list, filtered_pauli_shots, used_prep_paulis = [], [], []
        for i, (prep, obs) in enumerate(fiducials_list):
            # Regroup observables for same input Pauli
            label = prep.to_label()
            should_add_pauli = label not in used_prep_paulis
            if should_add_pauli:
                used_prep_paulis.append(label)
                filtered_fiducials_list.append((prep, obs))
                filtered_pauli_shots.append(self._pauli_shots[i])

            else:
                index = used_prep_paulis.index(label)
                _, ref_obs = filtered_fiducials_list[index]
                filtered_fiducials_list[index] = (
                    prep,
                    (ref_obs + obs).simplify(),
                )
                filtered_pauli_shots[index] += self._pauli_shots[i]

        for i, (prep, obs) in enumerate(filtered_fiducials_list):
            filtered_fiducials_list[i] = (prep, obs.group_commuting(qubit_wise=True))

        self._fiducials = filtered_fiducials_list
        self._pauli_shots = filtered_pauli_shots

        pubs = []
        total_shots = 0
        used_prep_indices = []  # Track used input states to reduce number of PUBs
        for (prep, obs_list), shots in zip(self._fiducials, self._pauli_shots):
            # Each prep is a Pauli input state, that we need to decompose in its pure eigenbasis
            # Below, we select at random a subset of pure input states to prepare for each prep
            # If nb_states = 1, we prepare all pure input states for each Pauli prep (no random selection)
            prep: Pauli
            obs_list: List[SparsePauliOp]
            self._fiducials_indices.append(([], observables_to_indices(obs_list)))
            self._full_fiducials.append(([], obs_list))
            max_input_states = dim // nb_states
            selected_input_states: List[int] = self.input_states_rng.choice(
                dim, size=max_input_states, replace=False
            )

            for pure_eig_state in selected_input_states:
                dedicated_shots = (
                    shots * execution_config.n_shots // max_input_states
                )  # Number of shots per Pauli eigenstate (divided equally)
                if dedicated_shots == 0:
                    continue
                # Convert input state to Pauli6 basis: preparing pure eigenstates of Pauli_prep
                prep_label = prep.to_label()
                inputs = np.unravel_index(pure_eig_state, (2,) * n_qubits)
                parity = 1
                for q_idx, symbol in enumerate(reversed(prep_label)):
                    if symbol != "I":
                        parity *= (-1) ** inputs[q_idx]

                batch_size = params.shape[0]

                prep_indices = pauli_input_to_indices(prep, inputs)
                used_prep_indices.append(tuple(prep_indices))

                self._fiducials_indices[-1][0].append(tuple(prep_indices))

                # Create input state preparation circuit
                input_circuit = qc.copy_empty_like()
                input_circuit.compose(
                    Pauli6PreparationBasis().circuit(prep_indices),
                    target.causal_cone_qubits,
                    inplace=True,
                )  # Apply input state on causal cone qubits
                input_circuit, prep_indices = extend_input_state_prep(
                    input_circuit, qc, target, prep_indices
                )  # Add random input state on other qubits
                input_circuit.metadata["indices"] = prep_indices
                self._full_fiducials[-1][0].append(input_circuit)
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
                # Add PUB to list
                layout = prep_circuit.layout
                pub_obs: SparsePauliOp = (parity * sum(obs_list)).apply_layout(layout)
                pub = (prep_circuit, pub_obs, params, 1 / np.sqrt(dedicated_shots))
                pubs.append(pub)

                total_shots += dedicated_shots * batch_size * len(obs_list)
                # else:  # If input state already used, reuse PUB and just update observable and precision
                #     pub_ref_index: int = used_prep_indices.index(tuple(prep_indices))
                #
                #     prep_circuit, ref_obs, ref_params, ref_precision = pubs[
                #         pub_ref_index
                #     ]
                #     ref_shots = precision_to_shots(ref_precision)
                #     new_precision = min(
                #         ref_precision, shots_to_precision(dedicated_shots)
                #     )
                #     new_shots = precision_to_shots(new_precision)
                #     new_pub = (
                #         prep_circuit,
                #         ref_obs + parity * obs.apply_layout(prep_circuit.layout),
                #         ref_params,
                #         new_precision,
                #     )
                #     pubs[pub_ref_index] = new_pub
                #     total_shots -= (
                #             ref_shots
                #             * batch_size
                #             * len(ref_obs.group_commuting(qubit_wise=True))
                #     )
                #     total_shots += (
                #             new_shots
                #             * batch_size
                #             * len(new_pub[1].group_commuting(qubit_wise=True))
                #     )

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
