from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional, Union, TYPE_CHECKING

from qiskit import ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical.types import Uint
from qiskit.primitives import BaseEstimatorV2, BitArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp, pauli_basis
import numpy as np
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis

from ..base_reward import Reward
from .state_reward_data import StateRewardData, StateRewardDataList
from ..real_time_utils import handle_real_time_n_reps
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import StateTarget, GateTarget, InputState
from ...helpers import validate_circuit_and_target
from ...helpers.circuit_utils import (
    extend_input_state_prep,
    extend_observables,
    get_2design_input_states,
    observables_to_indices,
    shots_to_precision,
    precision_to_shots,
    get_input_states_cardinality_per_qubit,
    handle_n_reps,
    get_single_qubit_input_states,
)

if TYPE_CHECKING:
    from qiskit_qm_provider.parameter_table import ParameterTable, Parameter as QuaParameter
    from ...qua.circuit_params import CircuitParams
    from qm import Program

Indices = Tuple[int]
Target = Union[StateTarget, GateTarget]


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
        self.input_states_rng = np.random.default_rng(self.input_state_seed)
        self.observables_rng = np.random.default_rng(self.observables_seed)

    @property
    def reward_method(self):
        return "state"

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice,
        "input_state_seed": self.input_state_seed,
        "observables_seed": self.observables_seed}

    def set_reward_seed(self, seed: int):
        """
        Set seed for input states and observables sampling
        """
        self.input_state_seed = seed + 30
        self.observables_seed = seed + 50
        self.input_states_rng = np.random.default_rng(self.input_state_seed)
        self.observables_rng = np.random.default_rng(self.observables_seed)

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.ndarray,
        env_config: QEnvConfig,
        *args,
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
        backend_info = env_config.backend_config
        dfe_precision = execution_config.dfe_precision
        target = env_config.target
        input_circuit = qc.copy_empty_like()

        prep_circuit = qc
        target_instance = target
        target_state = target_instance if isinstance(target_instance, StateTarget) else None
        n_reps = execution_config.current_n_reps

        if isinstance(target_instance, GateTarget):
            num_qubits = target_instance.causal_cone_size
        else:
            num_qubits = target_instance.n_qubits
        input_state_indices = (0,) * num_qubits
        if isinstance(target_instance, GateTarget):
            # State reward: sample a random input state for target gate
            input_choice = self.input_states_choice
            input_states = target_instance.input_states(input_choice)
            input_state_index = self.input_states_rng.choice(len(input_states))
            input_state_indices = np.unravel_index(
                input_state_index,
                (get_input_states_cardinality_per_qubit(input_choice),) * num_qubits,
            )

            input_state: InputState = input_states[input_state_index]

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
        cutoff = 1e-5
        non_zero_indices = np.nonzero(probabilities > cutoff)[0][1:]
        non_zero_probabilities = probabilities[non_zero_indices]
        non_zero_probabilities /= np.sum(non_zero_probabilities)

        basis = pauli_basis(num_qubits)

        if dfe_precision is not None:
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            pauli_sampling = execution_config.sampling_paulis

        pauli_indices, counts = np.unique(
            self.observables_rng.choice(non_zero_indices, pauli_sampling, p=non_zero_probabilities),
            return_counts=True,
        )
        c_factor = execution_config.c_factor
        id_coeff = c_factor / dim
        reward_factor = c_factor * counts * (dim - 1) / (dim * np.sqrt(dim) * Chi[pauli_indices])

        if dfe_precision is not None:
            eps, delta = dfe_precision
            pauli_shots = np.ceil(
                2 * np.log(2 / delta) / (eps**2) * dim * Chi[pauli_indices] ** 2 * pauli_sampling
            )
        else:
            pauli_shots = execution_config.n_shots * counts

        observables = SparsePauliOp(basis[pauli_indices], reward_factor)
        observable_indices = observables_to_indices(observables)

        shots_per_basis = []
        # Group observables by qubit-wise commuting groups to reduce the number of PUBs
        for i, commuting_group in enumerate(observables.paulis.group_qubit_wise_commuting()):
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
            pub_obs = extend_observables(
                observables, prep_circuit, target_instance.causal_cone_qubits_indices
            )
        else:
            pub_obs = observables.apply_layout(prep_circuit.layout)

        pub = (
            prep_circuit,
            pub_obs,
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
            n_reps=n_reps,
        )

        return StateRewardDataList([reward_data])

    def get_reward_data_multi_target(
        self,
        qc: QuantumCircuit,
        params: np.ndarray,
        env_config: "MultiTargetQEnvConfig",
        *args,
    ) -> StateRewardDataList:
        """
        Compute reward data for MultiTarget with joint PUB building.
        Combines all input state preparations (acting on disjoint qubits) into a single circuit.
        Structures observables to enable per-target reward extraction in post-processing.
        
        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            env_config: MultiTargetQEnvConfig containing the backend information and execution configuration
            *args: Additional arguments
            
        Returns:
            StateRewardDataList containing reward data with structured observables for per-target extraction
        """
        from ...environment.configuration.multi_target_qconfig import MultiTargetQEnvConfig
        from ...environment.target import MultiTarget
        
        execution_config = env_config.execution_config
        backend_info = env_config.backend_config
        dfe_precision = execution_config.dfe_precision
        multi_target: MultiTarget = env_config.target
        n_reps = execution_config.current_n_reps
        
        # Prepare the base circuit with n_reps
        prep_circuit = handle_n_reps(
            qc, n_reps, backend_info.backend, execution_config.control_flow_enabled
        )
        
        # Collect input state preparations and observables for all targets
        combined_input_circuit = qc.copy_empty_like()
        all_observables_list = []  # List of (target_index, observable, qubit_indices, id_coeff, pauli_sampling)
        all_input_indices = []  # List of (target_index, input_indices)
        all_observable_indices = []  # List of (target_index, observable_indices)
        target_metadata = []  # Store metadata per target
        
        # Process each target to collect input states and observables
        for target_idx, gate_target in enumerate(multi_target.gate_targets):
            if not isinstance(gate_target, GateTarget):
                continue
                
            num_qubits = gate_target.causal_cone_size
            input_choice = self.input_states_choice
            input_states = gate_target.input_states(input_choice)
            input_state_index = self.input_states_rng.choice(len(input_states))
            input_state_indices = np.unravel_index(
                input_state_index,
                (get_input_states_cardinality_per_qubit(input_choice),) * num_qubits,
            )
            
            input_state: InputState = input_states[input_state_index]
            target_state = input_state.target_state(n_reps)
            
            # Create input state preparation circuit for this target's qubits
            target_input_circuit = qc.copy_empty_like().compose(
                input_state.circuit, qubits=gate_target.causal_cone_qubits
            )
            target_input_circuit, extended_input_indices = extend_input_state_prep(
                target_input_circuit, qc, gate_target, list(input_state_indices)
            )
            
            # Compose this target's input preparation into the combined circuit
            # Since qubits are disjoint, this can be done in parallel
            combined_input_circuit.compose(target_input_circuit, inplace=True)
            all_input_indices.append((target_idx, tuple(extended_input_indices)))
            
            # DFE: Retrieve observables for this target
            Chi = target_state.Chi
            probabilities = Chi**2
            dim = target_state.dm.dim
            cutoff = 1e-5
            non_zero_indices = np.nonzero(probabilities > cutoff)[0][1:]
            non_zero_probabilities = probabilities[non_zero_indices]
            non_zero_probabilities /= np.sum(non_zero_probabilities)
            
            basis = pauli_basis(num_qubits)
            
            if dfe_precision is not None:
                eps, delta = dfe_precision
                pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
            else:
                pauli_sampling = execution_config.sampling_paulis
            
            pauli_indices, counts = np.unique(
                self.observables_rng.choice(non_zero_indices, pauli_sampling, p=non_zero_probabilities),
                return_counts=True,
            )
            c_factor = execution_config.c_factor
            id_coeff = c_factor / dim
            reward_factor = c_factor * counts * (dim - 1) / (dim * np.sqrt(dim) * Chi[pauli_indices])
            
            if dfe_precision is not None:
                eps, delta = dfe_precision
                pauli_shots = np.ceil(
                    2 * np.log(2 / delta) / (eps**2) * dim * Chi[pauli_indices] ** 2 * pauli_sampling
                )
            else:
                pauli_shots = execution_config.n_shots * counts
            
            # Create observables for this target's subsystem
            target_observables = SparsePauliOp(basis[pauli_indices], reward_factor)
            observable_indices = observables_to_indices(target_observables)
            all_observable_indices.append((target_idx, observable_indices))
            
            # Store metadata for this target
            target_metadata.append({
                'target_idx': target_idx,
                'gate_target': gate_target,
                'id_coeff': id_coeff,
                'pauli_sampling': pauli_sampling,
                'pauli_shots': pauli_shots,
                'observables': target_observables,
            })
            
            # Extend observables to full circuit qubits and store with target index
            extended_obs = extend_observables(
                target_observables, prep_circuit, gate_target.causal_cone_qubits_indices
            )
            # Store the extended observables in metadata for later use
            target_metadata[target_idx]['extended_observables'] = extended_obs
            all_observables_list.append((target_idx, extended_obs, gate_target.causal_cone_qubits_indices, id_coeff, pauli_sampling))
        
        # Combine all observables into a single SparsePauliOp
        # We'll structure this so post-processing can extract per-target rewards
        combined_observables = None
        max_shots = 0
        total_id_coeff = 0.0
        total_pauli_sampling = 0
        
        for target_idx, obs, qubit_indices, id_coeff, pauli_sampling in all_observables_list:
            if combined_observables is None:
                combined_observables = obs
            else:
                combined_observables = combined_observables + obs
            max_shots = max(max_shots, max(target_metadata[target_idx]['pauli_shots']) if isinstance(target_metadata[target_idx]['pauli_shots'], (list, np.ndarray)) else target_metadata[target_idx]['pauli_shots'])
            total_id_coeff += id_coeff
            total_pauli_sampling += pauli_sampling
        
        if combined_observables is None:
            raise ValueError("No observables generated for any target")
        
        combined_observables = combined_observables.simplify()
        
        # Create the combined preparation circuit
        combined_prep_circuit = prep_circuit.compose(combined_input_circuit, front=True, inplace=False)
        
        # Use the layout from the first target (or combine layouts if needed)
        first_target = multi_target.gate_targets[0]
        if isinstance(first_target, GateTarget):
            layout = first_target.layout
        else:
            layout = None
        
        combined_prep_circuit = backend_info.custom_transpile(
            combined_prep_circuit, initial_layout=layout, scheduling=False
        )
        
        # Store target mapping in metadata for post-processing
        combined_prep_circuit.metadata["multi_target_mapping"] = {
            'target_indices': [t['target_idx'] for t in target_metadata],
            'target_qubits': [t['gate_target'].causal_cone_qubits_indices if isinstance(t['gate_target'], GateTarget) else [] for t in target_metadata],
            'target_id_coeffs': [t['id_coeff'] for t in target_metadata],
            'target_pauli_samplings': [t['pauli_sampling'] for t in target_metadata],
            'input_indices': all_input_indices,
            'observable_indices': all_observable_indices,
        }
        
        # Create PUB with combined circuit and observables
        pub = (
            combined_prep_circuit,
            combined_observables,
            params,
            shots_to_precision(max_shots),
        )
        
        # Create reward data with combined structure
        reward_data = StateRewardData(
            pub=pub,
            id_coeff=total_id_coeff,
            pauli_sampling=total_pauli_sampling,
            input_circuit=combined_input_circuit,
            observables=combined_observables,
            input_indices=tuple([idx for _, idx in all_input_indices]),  # Flatten for compatibility
            observables_indices=[idx for _, idx in all_observable_indices],  # List of lists
            n_reps=n_reps,
        )
        
        # Store target mapping in reward data for post-processing
        reward_data.target_mapping = combined_prep_circuit.metadata["multi_target_mapping"]
        
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
    ) -> np.ndarray:
        """
        Retrieve the reward from the PUBs and the primitive
        """
        num_qubits = reward_data[0].observables.num_qubits
        dim = 2**num_qubits
        job = estimator.run(reward_data.pubs)
        pub_results = job.result()
        reward = np.sum([pub_result.data.evs for pub_result in pub_results], axis=0)
        reward /= reward_data.pauli_sampling
        reward += reward_data.id_coeff

        return reward

    def get_reward_with_primitive_multi_target(
        self,
        reward_data: "MultiTargetStateRewardDataList",
        estimator: BaseEstimatorV2,
    ) -> np.ndarray:
        """
        Retrieve per-target rewards from the PUBs and primitive.
        Performs centralized execution (single PUB run) and decentralized post-processing
        to extract individual fidelity estimators for each target.
        
        Args:
            reward_data: MultiTargetStateRewardDataList containing reward data with separate observables per target
            estimator: BaseEstimatorV2 estimator to run the PUBs
            
        Returns:
            Reward array of shape (num_targets, batch_size) - one reward per target
        """
        from .multi_target_state_reward_data import MultiTargetStateRewardDataList
        
        # Centralized execution: run all PUBs once
        job = estimator.run(reward_data.pubs)
        pub_results = job.result()
        
        # Get batch size from first PUB result
        if not pub_results:
            return np.zeros((reward_data.num_targets, 1))
        batch_size = pub_results[0].data.evs.shape[0]
        
        num_targets = reward_data.num_targets
        target_indices = reward_data.target_indices
        
        # Initialize reward array: shape (num_targets, batch_size)
        target_rewards = np.zeros((num_targets, batch_size))
        
        # For state reward, we typically have one PUB with combined observables
        # We need to extract per-target contributions
        if len(pub_results) == 1 and len(reward_data.reward_data) == 1:
            # Single PUB case - extract per-target contributions
            target_data = reward_data.reward_data[0]
            combined_evs = pub_results[0].data.evs  # Shape: [batch_size]
            
            for target_idx in target_indices:
                idx = target_data.target_indices.index(target_idx)
                id_coeff = target_data.target_id_coeffs[idx]
                pauli_sampling = target_data.target_pauli_samplings[idx]
                target_obs = target_data.target_observables[idx]
                
                # Extract expectation values for this target's observables
                # Since observables are combined in the PUB, we need to estimate the contribution
                target_obs_norm = np.sum(np.abs(target_obs.coeffs))
                combined_obs_norm = np.sum(np.abs(target_data.combined_observables.coeffs))
                
                if combined_obs_norm > 0:
                    target_contribution = target_obs_norm / combined_obs_norm
                    # Remove total ID coeff and scale by target contribution
                    target_ev = (combined_evs - target_data.total_id_coeff) * target_contribution
                else:
                    target_ev = np.zeros_like(combined_evs)
                
                # Compute target reward
                target_reward = target_ev / pauli_sampling + id_coeff
                
                # Store in output array
                target_array_idx = target_indices.index(target_idx)
                target_rewards[target_array_idx] = target_reward
        else:
            # Multiple PUBs case - group by target
            # This would require tracking which PUB belongs to which target
            # For now, distribute evenly or use metadata
            raise NotImplementedError("Multiple PUBs per target not yet fully supported for state reward")
        
        return target_rewards  # Shape: (num_targets, batch_size)

    def get_real_time_circuit(
        self,
        circuits: QuantumCircuit | List[QuantumCircuit],
        target: Target | List[Target],
        env_config: QEnvConfig,
        skip_transpilation: bool = False,
        *args,
    ) -> QuantumCircuit:

        n_reps = env_config.current_n_reps
        all_n_reps = env_config.n_reps

        prep_circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits

        is_gate_target = isinstance(target, GateTarget)

        qubits = [qc.qubits for qc in prep_circuits]
        if not all(qc.qubits == qubits[0] for qc in prep_circuits):
            raise ValueError("All circuits must be defined on the same qubits")

        qc = prep_circuits[0].copy_empty_like("real_time_state_qc")
        qc.reset(qc.qubits)
        n_reps_var = qc.add_input("n_reps", Uint(32)) if len(all_n_reps) > 1 else n_reps
        if is_gate_target:
            causal_cone_qubits = target.causal_cone_qubits
            causal_cone_size = target.causal_cone_size
        else:
            causal_cone_qubits = qc.qubits
            causal_cone_size = target.n_qubits

        if not qc.clbits:
            meas = ClassicalRegister(causal_cone_size, name="meas")
            qc.add_register(meas)
        else:
            meas = qc.cregs[0]
            if meas.size != causal_cone_size:
                raise ValueError("Classical register size must match the target causal cone size")

        observables_vars = [
            qc.add_input(f"observable_{i}", Uint(32)) for i in range(causal_cone_size)
        ]

        input_choice = self.input_states_choice
        if input_choice in ["pauli4", "pauli6"]:
            input_state_vars = [qc.add_input(f"input_state_{i}", Uint(32)) for i in range(causal_cone_size)]
            input_circuits = get_single_qubit_input_states(input_choice)
            for q, qubit in enumerate(causal_cone_qubits):
                # Input state prep (over all qubits of the circuit context)
                with qc.switch(input_state_vars[q]) as case_input_state:
                    for i, input_circuit in enumerate(input_circuits):
                        with case_input_state(i):
                            if input_circuit.data:
                                qc.compose(input_circuit, qubit, inplace=True)
                            else:
                                qc.delay(16, qubit)
        else:  # 2-design
            input_states = get_2design_input_states(2**causal_cone_size)
            input_circuits = [QuantumCircuit(causal_cone_size) for _ in input_states]
            for input_circuit, input_state in zip(input_circuits, input_states):
                input_circuit.prepare_state(input_state)
            input_state_var = qc.add_input("input_state", Uint(32))
            with qc.switch(input_state_var) as case_input_state:
                for i, input_circuit in enumerate(input_circuits):
                    with case_input_state(i):
                        if input_circuit.data:
                            qc.compose(input_circuit, causal_cone_qubits, inplace=True)
                        else:
                            qc.delay(16, causal_cone_qubits)

        if len(prep_circuits) > 1:  # Switch over possible contexts
            circuit_choice = qc.add_input("circuit_choice", Uint(32))
            with qc.switch(circuit_choice) as case_circuit:
                for i, prep_circuit in enumerate(prep_circuits):
                    with case_circuit(i):
                        handle_real_time_n_reps(all_n_reps, n_reps_var, prep_circuit, qc)
        else:
            handle_real_time_n_reps(all_n_reps, n_reps_var, prep_circuits[0], qc)

        # Local Basis rotation handling
        meas_basis = PauliMeasurementBasis()
        for q, qubit in enumerate(causal_cone_qubits):
            with qc.switch(observables_vars[q]) as case_observable:
                for i in range(3):
                    basis_rot_circuit = (
                        meas_basis.circuit([i]).decompose().remove_final_measurements(False)
                    )
                    with case_observable(i):
                        if basis_rot_circuit.data:
                            # Apply the measurement basis circuit to the qubit
                            qc.compose(basis_rot_circuit, qubit, inplace=True)
                        else:
                            qc.delay(16, qubit)

        # Measurement
        qc.measure(causal_cone_qubits, meas)

        if skip_transpilation:
            return qc

        return env_config.backend_config.custom_transpile(
            qc,
            optimization_level=1,
            initial_layout=target.layout,
            scheduling=False,
            remove_final_measurements=False,
        )

    def qm_step(
        self,
        reward_data: StateRewardDataList,
        fetching_index: int,
        fetching_size: int,
        circuit_params: CircuitParams,
        reward: QuaParameter,
        config: QEnvConfig,
        **push_args,
    ):
        """
        This function is used to compute the reward for the state reward.
        It is used in the QMEnvironment.step() function.
        Args:
            reward_data: Reward data to be used to compute the reward (can be used to send inputs to the QUA program and also to post-process measurement outcomes/counts coming out of the QUA program)
            fetching_index: Index of the first measurement outcome to be fetched in stream processing / DGX Quantum stream
            fetching_size: Number of measurement outcomes to be fetched
            circuit_params: Parameters defining the quantum program to be executed, those are entrypoints towards streaming values to control-flow that define the program adaptively (e.g. input state, number of repetitions, observable, etc.)
            reward: Reward parameter to be used to fetch measurement outcomes from the QUA program and compute the reward
            config: Environment configuration
            **push_args: Additional arguments to pass necessary entrypoints to communicate with the OPX (e.g. job, qm, verbosity, etc.)

        Returns:
            Reward array of shape (batch_size,)
        """
        from ...qua.qm_config import QMConfig

        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")

        reward_array = np.zeros(shape=(config.batch_size,))
        num_qubits = (
            config.target.causal_cone_size
            if isinstance(config.target, GateTarget)
            else config.target.n_qubits
        )
        dim = 2**num_qubits
        binary = lambda n, l: bin(n)[2:].zfill(l)
        if isinstance(config.target, GateTarget):
            from ..real_time_utils import push_circuit_context

            push_circuit_context(circuit_params, config.target, **push_args)

        if circuit_params.n_reps_var is not None:
            circuit_params.n_reps_var.push_to_opx(config.current_n_reps, **push_args)
        if len(reward_data.input_indices) > 1:
            raise ValueError("StateRewardDataList with multiple input indices is not supported")

        input_state_dict = {
            input_var: input_state_val
            for input_var, input_state_val in zip(
                circuit_params.input_state_vars.parameters, reward_data.input_indices[0]
            )
        }
        circuit_params.input_state_vars.push_to_opx(input_state_dict, **push_args)
        circuit_params.max_observables.push_to_opx(
            len(reward_data.observables_indices[0]), **push_args
        )
        circuit_params.n_shots.push_to_opx(reward_data.shots[0], **push_args)
        for i, observable in enumerate(reward_data.observables_indices[0]):
            observable_dict = {
                observable_var: observable_val
                for observable_var, observable_val in zip(
                    circuit_params.observable_vars.parameters, observable
                )
            }
            circuit_params.observable_vars.push_to_opx(observable_dict, **push_args)
        collected_counts = reward.fetch_from_opx(
            push_args["job"],
            fetching_index=fetching_index,
            fetching_size=fetching_size,
            verbosity=config.backend_config.verbosity,
            time_out=config.backend_config.timeout,
        )
        # sampled_actions = circuit_params.real_time_circuit_parameters.fetch_from_opx(
        #     push_args["job"],
        #     fetching_index=fetching_index,
        #     fetching_size=fetching_size,
        #     verbosity=config.backend_config.verbosity,
        #     time_out=config.backend_config.timeout,
        # )
        # print("sampled_actions", sampled_actions)
        counts = []
        formatted_counts = []
        count_idx = 0
        num_obs = len(reward_data.observables_indices[0])
        for o_idx in range(num_obs):
            formatted_counts.append([])
            counts_array = np.array(collected_counts[count_idx], dtype=int)
            formatted_counts[o_idx] = counts_array
            count_idx += 1
        for batch_idx in range(config.batch_size):
            counts.append([])
            for o_idx in range(num_obs):
                counts[batch_idx].append(formatted_counts[o_idx][batch_idx])

        for batch_idx in range(config.batch_size):
            exp_value = 0.0
            obs_group = reward_data[0].observables.group_commuting(True)
            for o_idx, obs in enumerate(obs_group):
                counts_dict = {
                    binary(i, num_qubits): int(counts[batch_idx][o_idx][i]) for i in range(dim)
                }
                bit_array = BitArray.from_counts(counts_dict, num_bits=num_qubits)
                diag_obs = SparsePauliOp("I" * num_qubits, 0.0)
                for obs_, coeff in zip(obs.paulis, obs.coeffs):
                    diag_obs_label = ""
                    for char in obs_.to_label():
                        diag_obs_label += char if char == "I" else "Z"
                    diag_obs += SparsePauliOp(diag_obs_label, coeff)
                exp_value += bit_array.expectation_values(diag_obs.simplify())
            exp_value /= reward_data.pauli_sampling
            exp_value += reward_data.id_coeff
            reward_array[batch_idx] = exp_value
        return reward_array

    def rl_qoc_training_qua_prog(
        self,
        qc: QuantumCircuit,
        policy: ParameterTable,
        reward: QuaParameter,
        circuit_params: CircuitParams,
        config: QEnvConfig,
        num_updates: int = 1000,
        test: bool = False,
    ) -> Program:
        from qm.qua import program, declare, Random, for_, stream_processing, assign, fixed
        from qiskit_qm_provider import QMBackend
        from ...qua.qua_utils import rand_gauss_moller_box, rescale_and_clip_wrapper
        from qiskit_qm_provider.backend import get_measurement_outcomes
        from ...qua.qm_config import QMConfig
        from ..real_time_utils import load_circuit_context, benchmark_cycle_macro

        if not isinstance(config.backend, QMBackend):
            raise ValueError("Backend must be a QMBackend")
        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")
        if circuit_params.input_state_vars is None:
            raise ValueError("input_state_vars should be set for State reward")
        if circuit_params.n_shots is None:
            raise ValueError("n_shots should be set for State reward")
        if circuit_params.max_observables is None:
            raise ValueError("max_observables should be set for State reward")
        if circuit_params.observable_vars is None:
            raise ValueError("observable_vars should be set for State reward")

        policy.reset()
        reward.reset()
        circuit_params.reset()
        num_qubits = (
            config.target.causal_cone_size
            if isinstance(config.target, GateTarget)
            else config.target.n_qubits
        )
        dim = int(2**num_qubits)
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) >= 2:
                raise ValueError("Overlapping classical registers are not supported")

        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)
            circuit_params.declare_variables()
            policy.declare_variables()
            reward.declare_variable()
            reward.declare_stream()

            n_u = declare(int)
            shots = declare(int)
            o_idx = declare(int)
            b = declare(int)
            j = declare(int)
            tmp1 = declare(fixed, size=config.n_actions)
            tmp2 = declare(fixed, size=config.n_actions)

            mu = policy.get_variable("mu")
            sigma = policy.get_variable("sigma")
            counts = reward.var
            batch_r = Random(config.seed)

            if test:
                circuit_params.declare_streams()
                policy.declare_streams()

            if config.backend.init_macro is not None:
                config.backend.init_macro()

            with for_(n_u, 0, n_u < num_updates, n_u + 1):
                policy.load_input_values()
                # Benchmark cycle 
                benchmark_cycle_macro(circuit_params.benchmark_cycle_var, policy, config)
                # Load context
                load_circuit_context(circuit_params)

                n_reps_var = circuit_params.n_reps_var
                if n_reps_var is not None and n_reps_var.input_type is not None:
                    n_reps_var.load_input_value()

                circuit_params.input_state_vars.load_input_values()
                circuit_params.max_observables.load_input_value()
                circuit_params.n_shots.load_input_value()
                with for_(o_idx, 0, o_idx < circuit_params.max_observables.var, o_idx + 1):
                    circuit_params.observable_vars.load_input_values()
                    batch_r.set_seed(config.seed + n_u)
                    with for_(b, 0, b < config.batch_size, b + 2):
                        # Sample from a multivariate Gaussian distribution (Muller-Box method)
                        tmp1, tmp2 = rand_gauss_moller_box(
                            mu,
                            sigma,
                            batch_r,
                            tmp1,
                            tmp2,
                        )
                        if (
                            config.backend_config.wrapper_data.get("rescale_and_clip", None)
                            is not None
                        ):
                            new_box = config.backend_config.wrapper_data["rescale_and_clip"]
                            tmp1, tmp2 = rescale_and_clip_wrapper(
                                [tmp1, tmp2],
                                config.action_space,
                                new_box,
                            )

                        # Assign 1 or 2 depending on batch_size being even or odd (only relevant at last iteration)
                        with for_(j, 0, j < 2, j + 1):
                            # Assign the sampled actions to the action batch
                            for i, parameter in enumerate(
                                circuit_params.real_time_circuit_parameters.parameters
                            ):
                                parameter.assign(tmp1[i], condition=(j == 0), value_cond=tmp2[i])

                            with for_(shots, 0, shots < circuit_params.n_shots.var, shots + 1):
                                result = config.backend.quantum_circuit_to_qua(
                                    qc, circuit_params.circuit_variables
                                )
                                state_int = get_measurement_outcomes(qc, result)[qc.cregs[0].name][
                                    "state_int"
                                ]
                                assign(counts[state_int], counts[state_int] + 1)

                            reward.stream_back(reset=True)

            with stream_processing():
                buffer = (config.batch_size, dim)
                reward.stream_processing(buffer=buffer)

        return rl_qoc_training_prog
