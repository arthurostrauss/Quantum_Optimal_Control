from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from .base_reward import Reward
from ..environment.backend_info import BackendInfo
from ..environment.configuration.execution_config import ExecutionConfig
from ..environment.target import StateTarget, GateTarget, InputState
from ..helpers.circuit_utils import (
    extend_input_state_prep,
    extend_observables,
    observables_to_indices,
    retrieve_observables,
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

    def __post_init__(self):
        super().__post_init__()
        self._observables: Optional[SparsePauliOp] = None
        self._pauli_shots: Optional[List[int]] = None
        self.input_states_rng = np.random.default_rng(self.input_state_seed)
        self.observables_rng = np.random.default_rng(self.observables_seed)
        self._fiducials_indices: List[Tuple[List[Indices], List[Indices]]] = []
        self._fiducials = []

    @property
    def reward_method(self):
        return "state"

    @property
    def fiducials(self) -> List[Tuple[QuantumCircuit, List[SparsePauliOp]]]:
        """
        Fiducials to sample
        """
        return self._fiducials

    @property
    def fiducials_indices(self) -> List[Tuple[List[Indices], List[Indices]]]:
        """
        Indices of the input states and observables to sample
        """
        return self._fiducials_indices

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

    def get_reward_real_time_inputs(
        self,
        target: StateTarget | GateTarget,
        execution_config: ExecutionConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ):
        """
        Get input states and observables for reward computation
        """
        self._fiducials_indices = []
        target_instance = target
        target_state = (
            target_instance if isinstance(target_instance, StateTarget) else None
        )
        n_reps = execution_config.current_n_reps
        input_state_indices = []
        if isinstance(target_instance, GateTarget):
            num_qubits = target_instance.causal_cone_size
        else:
            num_qubits = target_instance.n_qubits

        if isinstance(target_instance, GateTarget):
            # State reward: sample a random input state for target gate
            input_state_index = self.input_states_rng.choice(
                len(target_instance.input_states)
            )
            input_choice = target_instance.input_states_choice
            indices = np.unravel_index(
                input_state_index,
                (get_input_states_cardinality_per_qubit(input_choice),) * num_qubits,
            )
            input_state_indices.append((int(i) for i in indices))
            input_state: InputState = target_instance.input_states[input_state_index]

            # Modify target state to match input state and target gate
            target_state = input_state.target_state(n_reps)  # (Gate |input>=|target>)

        self._observables, self._pauli_shots = retrieve_observables(
            target_state,
            dfe_tuple=dfe_precision,
            c_factor=execution_config.c_factor,
            sampling_paulis=execution_config.sampling_paulis,
            observables_rng=self.observables_rng,
        )

        observable_indices = observables_to_indices(self._observables)
        self._fiducials_indices.append((input_state_indices, observable_indices))

        return input_state_indices, observable_indices, self._pauli_shots

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: StateTarget | GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ) -> List[EstimatorPub]:
        """
        Compute pubs related to the reward method.
        This is used when real-time action sampling is not enabled on the backend.

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
        self._fiducials_indices = [(0, [])]
        input_circuit = qc.copy_empty_like()
        self._fiducials = [(input_circuit, [])]

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
            input_state: InputState = self.input_states_rng.choice(
                target_instance.input_states
            )
            # Modify target state to match input state and target gate
            target_state = input_state.target_state(n_reps)  # (Gate |input>=|target>)

            # Prepend input state to custom circuit with front composition

            prep_circuit = handle_n_reps(
                qc, n_reps, backend_info.backend, execution_config.control_flow_enabled
            )
            input_circuit, input_state_indices = extend_input_state_prep(
                input_state.circuit, qc, target_instance, input_state_indices
            )
            prep_circuit.compose(input_circuit, front=True, inplace=True)

        self._observables, self._pauli_shots = retrieve_observables(
            target_state,
            dfe_tuple=dfe_precision,
            c_factor=execution_config.c_factor,
            sampling_paulis=execution_config.sampling_paulis,
            observables_rng=self.observables_rng,
        )
        if isinstance(target_instance, GateTarget):
            self._observables = extend_observables(
                self._observables, prep_circuit, target_instance
            )
        observable_indices = observables_to_indices(self._observables)
        self._fiducials_indices = [(input_state_indices, observable_indices)]
        self._fiducials = [(input_circuit, self._observables)]

        prep_circuit = backend_info.custom_transpile(
            prep_circuit, initial_layout=target_instance.layout, scheduling=False
        )

        pubs = [
            (
                prep_circuit,
                obs.apply_layout(prep_circuit.layout),
                params,
                shots_to_precision(execution_config.n_shots * pauli_shots),
            )
            for obs, pauli_shots in zip(
                self._observables.group_commuting(qubit_wise=True),
                self._pauli_shots,
            )
        ]
        self._total_shots = params.shape[0] * sum(
            self._pauli_shots * execution_config.n_shots
        )

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
