from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp, pauli_basis
import numpy as np
from ..base_reward import Reward
from .state_reward_data import StateRewardData, StateRewardDataList
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import StateTarget, GateTarget, InputState
from ...helpers.circuit_utils import (
    extend_input_state_prep,
    extend_observables,
    observables_to_indices,
    shots_to_precision,
    precision_to_shots,
    get_input_states_cardinality_per_qubit,
    handle_n_reps,
)

Indices = Tuple[int]


@dataclass
class StateReward(Reward):
    """
    Configuration for computing the reward based on state fidelity estimation
    """

    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"
    input_state_seed: int = 2000
    observables_seed: int = 2001
    input_states_rng: np.random.Generator = field(init=False)
    observables_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.input_states_rng = np.random.default_rng(self.input_state_seed)
        self.observables_rng = np.random.default_rng(self.observables_seed)

    @property
    def reward_method(self):
        return "state"

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}

    def set_reward_seed(self, seed: int = None):
        """
        Set seed for input states and observables sampling
        """
        if seed is not None:
            self.input_state_seed = seed + 30
            self.observables_seed = seed + 50
            self.input_states_rng = np.random.default_rng(self.input_state_seed)
            self.observables_rng = np.random.default_rng(self.observables_seed)

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: StateTarget | GateTarget,
        env_config: QEnvConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ) -> StateRewardDataList:
        """
        Compute pubs related to the reward method.
        This is used when real-time action sampling is not enabled on the backend.

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
            dfe_precision: Tuple (Ɛ, δ) from DFE paper

        Returns:
            List of pubs related to the reward method
        """
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info
        input_circuit = qc.copy_empty_like()

        prep_circuit = qc
        target_instance = target
        target_state = (
            target_instance if isinstance(target_instance, StateTarget) else None
        )
        n_reps = execution_config.current_n_reps

        if isinstance(target_instance, GateTarget):
            num_qubits = target_instance.causal_cone_size
        else:
            num_qubits = target_instance.n_qubits
        input_state_indices = (0,) * num_qubits
        if isinstance(target_instance, GateTarget):
            # State reward: sample a random input state for target gate
            input_state_index = self.input_states_rng.choice(
                len(target_instance.input_states)
            )

            input_choice = target_instance.input_states_choice
            input_state_indices = np.unravel_index(
                input_state_index,
                (get_input_states_cardinality_per_qubit(input_choice),) * num_qubits,
            )

            input_state: InputState = target_instance.input_states[input_state_index]

            # Modify target state to match input state and target gate
            target_state = input_state.target_state(n_reps)  # (Gate |input>=|target>)

            # Prepend input state to custom circuit with front composition

            prep_circuit = handle_n_reps(
                qc, n_reps, backend_info.backend, execution_config.control_flow_enabled
            )
            input_circuit = qc.copy_empty_like().compose(
                input_state.circuit, qubits=target_instance.causal_cone_qubits
            )
            input_circuit, input_state_indices = extend_input_state_prep(
                input_circuit, qc, target_instance, list(input_state_indices)
            )
            input_circuit.metadata["input_indices"] = input_state_indices
            prep_circuit.compose(input_circuit, front=True, inplace=True)

        # DFE: Retrieve observables for fidelity estimation
        Chi = target_state.Chi
        probabilities = Chi**2
        dim = target_state.dm.dim
        cutoff = 1e-8
        non_zero_indices = np.nonzero(probabilities > cutoff)[0]
        non_zero_probabilities = probabilities[non_zero_indices]
        non_zero_probabilities /= np.sum(non_zero_probabilities)

        basis = pauli_basis(num_qubits)

        if dfe_precision is not None:
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            pauli_sampling = execution_config.sampling_paulis

        pauli_indices, counts = np.unique(
            self.observables_rng.choice(
                non_zero_indices, pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )
        identity_term = np.where(pauli_indices == 0)[0]
        c_factor = execution_config.c_factor
        if len(identity_term) > 0:
            id_coeff = c_factor * np.sum(
                counts[identity_term]
                / (np.sqrt(dim) * Chi[pauli_indices[identity_term]])
            )
        else:
            id_coeff = 0.0

        pauli_indices = np.delete(pauli_indices, identity_term)
        counts = np.delete(counts, identity_term)
        reward_factor = (
            execution_config.c_factor * counts / (np.sqrt(dim) * Chi[pauli_indices])
        )

        if dfe_precision is not None:
            eps, delta = dfe_precision
            pauli_shots = np.ceil(
                2
                * np.log(2 / delta)
                / (eps**2)
                * dim
                * Chi[pauli_indices] ** 2
                * pauli_sampling
            )
        else:
            pauli_shots = execution_config.n_shots * counts

        observables = SparsePauliOp(basis[pauli_indices], reward_factor)
        observable_indices = observables_to_indices(observables)

        shots_per_basis = []
        # Group observables by qubit-wise commuting groups to reduce the number of PUBs
        for i, commuting_group in enumerate(
            observables.paulis.group_qubit_wise_commuting()
        ):
            max_pauli_shots = 0
            for pauli in commuting_group:
                pauli_index = list(basis).index(pauli)
                ref_index = list(pauli_indices).index(pauli_index)
                max_pauli_shots = max(max_pauli_shots, pauli_shots[ref_index])
            shots_per_basis.append(max_pauli_shots)
        pauli_shots = shots_per_basis

        prep_circuit = backend_info.custom_transpile(
            prep_circuit, initial_layout=target_instance.layout, scheduling=False
        )
        if isinstance(target_instance, GateTarget):
            observables = extend_observables(observables, prep_circuit, target_instance)
        else:
            observables = observables.apply_layout(prep_circuit.layout)

        pub = (
            prep_circuit,
            observables,
            params,
            shots_to_precision(max(pauli_shots)),
        )

        reward_data = StateRewardData(
            pub=pub,
            id_coeff=id_coeff,
            pauli_sampling=pauli_sampling,
            input_circuit=input_circuit,
            observables=observables,
            input_indices=input_state_indices,
            observables_indices=observable_indices,
            shots=max(pauli_shots),
            n_reps=n_reps,
        )

        return StateRewardDataList([reward_data])

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

    def get_reward_with_primitive(
        self,
        reward_data: StateRewardDataList,
        estimator: BaseEstimatorV2,
    ) -> np.array:
        """
        Retrieve the reward from the PUBs and the primitive
        """
        job = estimator.run(reward_data.pubs)
        pub_results = job.result()
        reward = np.sum([pub_result.data.evs for pub_result in pub_results], axis=0)
        reward += reward_data.id_coeff
        reward /= reward_data.pauli_sampling

        return reward

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
