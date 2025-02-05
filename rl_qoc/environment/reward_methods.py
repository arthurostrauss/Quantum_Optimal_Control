from __future__ import annotations

from qiskit import ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical.expr import Var
from qiskit.circuit.classical.types import Uint
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.quantum_info import (
    SparsePauliOp,
    pauli_basis,
    PauliList,
    random_clifford,
    Operator,
)
from qiskit_aer import AerSimulator
from qiskit_experiments.library.tomography.basis import Pauli6PreparationBasis, PauliPreparationBasis, \
    PauliMeasurementBasis
from .backend_info import BackendInfo
from typing import Optional, Tuple, List, Literal, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .calibration_pubs import (
    CalibrationEstimatorPub,
    CalibrationEstimatorPubLike,
    CalibrationSamplerPub,
    CalibrationSamplerPubLike,
)
from .configuration.execution_config import ExecutionConfig
from .target import GateTarget, StateTarget, InputState, get_2design_input_states
from ..helpers import (
    shots_to_precision,
    precision_to_shots,
    handle_n_reps,
    causal_cone_circuit,
)

PubLike = Union[
    EstimatorPubLike,
    SamplerPubLike,
    CalibrationEstimatorPubLike,
    CalibrationSamplerPubLike,
]
Pub = Union[EstimatorPub, SamplerPub, CalibrationEstimatorPub, CalibrationSamplerPub]


@dataclass
class RewardConfig(ABC):
    """
    Configuration for how to compute the reward in the RL workflow
    """

    print_debug = False

    def __post_init__(self):
        if self.reward_method == "channel" or self.reward_method == "state":
            self.dfe = True
        else:
            self.dfe = False
        self._total_shots = 0

    @property
    def reward_args(self):
        return {}

    @property
    @abstractmethod
    def reward_method(self) -> str:
        """
        String identifier for the reward method
        """
        raise NotImplementedError

    @property
    def total_shots(self) -> int:
        """
        Total number of shots involved for the last call of the reward computation
        """
        return self._total_shots

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: StateTarget | GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
    ) -> List[Pub]:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            n_reps: Number of repetitions of cycle circuit

        Returns:
            List of pubs related to the reward method
        """
        pass

    def get_shot_budget(self, pubs: List[Pub]):
        """
        Compute the total number of shots to be used for the reward computation
        """
        pass


@dataclass
class FidelityConfig(RewardConfig):
    """
    Configuration for computing the reward based on fidelity estimation
    """

    @property
    def reward_method(self):
        return "fidelity"


def get_single_qubit_input_states(input_state_choice):
    """
    Get single qubit input states for a given choice of input states
    (pauli4, pauli6, 2-design)
    """
    if input_state_choice == "pauli4":
        input_states = [PauliPreparationBasis().circuit([i]) for i in range(4)]
    elif input_state_choice == "pauli6":
        input_states = [Pauli6PreparationBasis().circuit([i]) for i in range(6)]
    elif input_state_choice == "2-design":
        states = get_2design_input_states(2)
        input_circuits = [QuantumCircuit(1) for _ in states]
        for input_circ, state in zip(input_circuits, states):
            input_circ.prepare_state(state)
        input_states = input_circuits
    else:
        raise ValueError("Invalid input state choice")
    return input_states


def get_real_time_reward_pub(circuits: QuantumCircuit | List[QuantumCircuit],
                             params:np.array,
                             target: List[GateTarget]|GateTarget|StateTarget,
                             backend_info: BackendInfo,
                             execution_config: ExecutionConfig,
                             reward_method: Literal["channel", "state", "cafe"] = "state",
                             dfe_precision: Optional[Tuple[float, float]] = None) -> SamplerPub:
    """
    Compute pubs related to the reward method for real-time execution (relevant for backend enabling real-time
    control flow)
    
    Args:
        circuits: Quantum circuit to be executed on quantum system
        params: Parameters to feed the parametrized circuit
        target: List of target gates
        backend_info: Backend information
        execution_config: Execution configuration
        reward_method: Method to compute the reward (channel, state or cafe)
        dfe_precision: Tuple (Ɛ, δ) from DFE paper
    """
    
    prep_circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
    target_instances = [target] if isinstance(target, (GateTarget, StateTarget)) else target
    if len(prep_circuits) != len(target_instances):
        raise ValueError("Number of circuits and targets must be the same")
    if not all(isinstance(target_instance, (GateTarget, StateTarget)) for target_instance in target_instances):
        raise ValueError("All targets must be gate targets")
    
    
    target_instance = target
    is_gate_target = isinstance(target_instance, GateTarget)
    target_state = target_instance if isinstance(target_instance, StateTarget) else None
    
    # Compare qubits of each circuit between each other and ensure they are the same
    qubits = [qc.qubits for qc in prep_circuits]
    if len(qubits) > 1 and not all(qubits[0] == qubits[i] for i in range(1, len(qubits))):
        raise ValueError("All circuits must have the same qubits")
    
    qc = prep_circuits[0].copy_empty_like(name="real_time_qc")
    num_qubits = qc.num_qubits # all qubits (even those outside of causal cone)
    n_reps = execution_config.current_n_reps
    
    if len(execution_config.n_reps)>1: # Switch over possible number of repetitions
        n_reps_var = qc.add_input("n_reps", Uint(8))
    else:
        n_reps_var = n_reps
    
    if is_gate_target:
        causal_cone_size = target_instance.causal_cone_size
        causal_cone_qubits = target_instance.causal_cone_qubits
    else:
        causal_cone_size = num_qubits
        causal_cone_qubits = qc.qubits
    
    # Add classical register for measurements
    if not qc.clbits:
        meas = ClassicalRegister(causal_cone_size, name="meas")
        qc.add_register(meas)
    else:
        meas = qc.clbits
    
    if is_gate_target: # Declare input states variables
        input_state_vars = [qc.add_input(f"input_state_{i}", 
                                Uint(4)) for i in range(num_qubits)] 
    else:
        input_state_vars = None

    observables_vars = [qc.add_input(f"observable_{i}", Uint(4)) for i in range(causal_cone_size)]
    
    if is_gate_target:
        input_circuits = get_single_qubit_input_states(target_instance.input_states_choice)
     
        for q_idx, qubit in enumerate(qc.qubits): # Input state preparation
            with qc.switch(input_state_vars[q_idx]) as case_input_state:
                for i, input_circuit in enumerate(input_circuits):
                    with case_input_state(i):
                        qc.compose(input_circuit, [qubit], inplace=True)

    if len(prep_circuits) > 1: # Switch over possible circuit contexts
        circuit_choice = qc.add_input("circuit_choice", Uint(8))
        with qc.switch(circuit_choice) as circuit_case:
            for i, prep_circuit in enumerate(prep_circuits):
                with circuit_case(i):
                    handle_real_time_n_reps(execution_config.n_reps, n_reps_var, prep_circuit, qc)     
    else:
        prep_circuit = prep_circuits[0]
        handle_real_time_n_reps(execution_config.n_reps, n_reps_var, prep_circuit, qc)
    
    if reward_method in ["state", "channel"]:
        for q_idx, qubit in enumerate(causal_cone_qubits):
            with qc.switch(observables_vars[q_idx]) as case_observable:
                for i in range(3):
                    with case_observable(i):
                        qc.compose(PauliMeasurementBasis().circuit([i]).remove_final_measurements(False).decompose(),
                                   [qubit], inplace=True)
        for qubit, clbit in zip(causal_cone_qubits, meas):
            qc.measure(qubit, clbit)
            
    elif reward_method == "cafe":
        for circ in prep_circuits:
            if circ.metadata.get("baseline_circuit") is None:
                raise ValueError("Baseline circuit not found in metadata")
        ref_circuits: List[QuantumCircuit] = [circ.metadata["baseline_circuit"].copy() for circ in prep_circuits]
        layout = target_instance.layout
        cycle_circuit_inverses = [[] for _ in range(len(ref_circuits))]
        for i, ref_circ in enumerate(ref_circuits):
            for n in execution_config.n_reps:
                cycle_circuit = ref_circ.repeat(n)
                cycle_circuit = causal_cone_circuit(cycle_circuit, causal_cone_qubits)[0]
                cycle_circuit.save_unitary()
                sim_unitary = AerSimulator(method='unitary').run(cycle_circuit).result().get_unitary()
                inverse_circuit = ref_circ.copy_empty_like()
                inverse_circuit.unitary(sim_unitary.adjoint(), causal_cone_qubits, label="U_inv")
                inverse_circuit.measure(causal_cone_qubits, meas)
                cycle_circuit_inverses[i].append(inverse_circuit)
                
            
            
            with qc.switch(circuit_choice) as case_circuit:
                with case_circuit(i):
                    qc.compose(cycle_circuit, inplace=True)
        
            
    
    qc = backend_info.custom_transpile(qc, initial_layout=target_instance.layout, 
                                       scheduling=False, remove_final_measurements=False)
    
    return SamplerPub.coerce((qc, params, execution_config.n_shots))


def handle_real_time_n_reps(n_reps: List[int], n_reps_var: int|Var, prep_circuit: QuantumCircuit,
                            qc: QuantumCircuit):
    """
    Handle the number of repetitions of the circuit in the real-time reward computation
    
    Args:
        n_reps: List of possible number of repetitions
        n_reps_var: Variable for the number of repetitions
        prep_circuit: Circuit to be executed
        qc: Quantum circuit to add the repetitions to
    """
    if isinstance(n_reps_var, int):
        if n_reps_var > 1:
            with qc.for_loop(range(n_reps_var)):
                qc.compose(prep_circuit, inplace=True)
        else:
            qc.compose(prep_circuit, inplace=True)
    elif isinstance(n_reps_var, Var):
        with qc.switch(n_reps_var) as case_reps:
            for n in n_reps:
                with case_reps(n):
                    if n > 1:
                        with qc.for_loop(range(n)):
                            qc.compose(prep_circuit, inplace=True)
                    else:
                        qc.compose(prep_circuit, inplace=True)


@dataclass
class StateRewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on state fidelity estimation
    """

    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"

    def __post_init__(self):
        super().__post_init__()
        self._observables: Optional[SparsePauliOp] = None
        self._pauli_shots: Optional[List[int]] = None

    @property
    def reward_method(self):
        return "state"

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: StateTarget | GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ) -> List[Pub]:
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
        prep_circuit = qc
        target_instance = target
        target_state = (
            target_instance if isinstance(target_instance, StateTarget) else None
        )
        n_reps = execution_config.current_n_reps

        if isinstance(target_instance, GateTarget):
            # State reward: sample a random input state for target gate
            input_state: InputState = np.random.choice(target_instance.input_states)
            # Modify target state to match input state and target gate
            target_state = input_state.target_state(n_reps)  # (Gate |input>=|target>)

            # Prepend input state to custom circuit with front composition

            prep_circuit = handle_n_reps(
                qc, n_reps, backend_info.backend
            )
            input_circuit = extend_input_state_prep(
                input_state.circuit, qc, target_instance
            )
            prep_circuit.compose(input_circuit, front=True, inplace=True)

        self._observables, self._pauli_shots = retrieve_observables(
            target_state,
            dfe_tuple=dfe_precision,
            c_factor=execution_config.c_factor,
            sampling_paulis=execution_config.sampling_paulis,
        )
        if isinstance(target_instance, GateTarget):
            self._observables = extend_observables(
                self._observables, prep_circuit, target_instance
            )

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


@dataclass
class ChannelRewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on channel fidelity estimation
    """

    print_debug = True
    num_eigenstates_per_pauli: int = 1

    def __post_init__(self):
        super().__post_init__()
        self._observables: Optional[SparsePauliOp] = None
        self._pauli_shots: Optional[List[int]] = None
        self._input_states: List[QuantumCircuit] = []
        self._fiducials: List[Tuple[QuantumCircuit, SparsePauliOp]] = []

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
    def input_states(self) -> List[QuantumCircuit]:
        """
        Input states to sample
        """
        return self._input_states

    @property
    def fiducials(self) -> List[Tuple[QuantumCircuit, SparsePauliOp]]:
        """
        Fiducial states and observables to sample
        """
        return self._fiducials

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
        dim = 2**n_qubits
        Chi = target.Chi(n_reps)
        self._input_states = []

        nb_states = self.num_eigenstates_per_pauli
        if nb_states >= dim:
            raise ValueError(
                f"Number of eigenstates per Pauli should be less than or equal to {dim}"
            )

        probabilities = Chi**2 / (dim**2)
        non_zero_indices = np.nonzero(probabilities)[0]  # Filter out zero probabilities
        non_zero_probabilities = probabilities[non_zero_indices]

        basis = pauli_basis(num_qubits=n_qubits)

        if dfe_precision is not None:
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            eps, delta = None, None
            pauli_sampling = execution_config.sampling_paulis

        samples, self._pauli_shots = np.unique(
            np.random.choice(
                non_zero_indices, size=pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )
        pauli_indices = np.array(
            [np.unravel_index(sample, (dim**2, dim**2)) for sample in samples],
            dtype=int,
        )

        pauli_prep, pauli_meas = zip(
            *[(basis[p[1]], basis[p[0]]) for p in pauli_indices]
        )
        pauli_prep, pauli_meas = PauliList(pauli_prep), PauliList(pauli_meas)
        reward_factor = [
            execution_config.c_factor / (dim * Chi[p]) for p in samples
        ]

        observables = SparsePauliOp(pauli_meas, reward_factor, ignore_pauli_phase=True)

        self._observables = extend_observables(observables, qc, target)

        if dfe_precision is not None:
            self._pauli_shots = np.ceil(
                2
                * np.log(2 / delta)
                / (dim * pauli_sampling * eps**2 * Chi[samples] ** 2)
            )

        pubs, total_shots = [], 0
        used_prep_indices = []  # Track used input states to reduce number of PUBs

        for prep, obs, shots in zip(pauli_prep, self._observables, self._pauli_shots):

            # Each prep is a Pauli input state, that we need to decompose in its pure eigenbasis
            # Below, we select at random a subset of pure input states to prepare for each prep
            # If nb_states = 1, we prepare all pure input states for each prep (no random selection)
            max_input_states = dim // nb_states
            selected_input_states = np.random.choice(
                dim, size=max_input_states, replace=False
            )
            for input_state in selected_input_states:
                prep_indices = []
                dedicated_shots = (
                    shots * execution_config.n_shots // max_input_states
                )  # Number of shots per Pauli eigenstate (divided equally)
                if dedicated_shots == 0:
                    continue
                # Convert input state to Pauli6 basis: preparing pure eigenstates of Pauli_prep
                inputs = np.unravel_index(input_state, (2,) * n_qubits)
                parity = (-1) ** np.sum(
                    inputs
                )  # Parity of input state (weighting factor in Pauli prep decomposition)
                for i, pauli_op in enumerate(reversed(prep.to_label())):
                    # Build input state in Pauli6 basis from Pauli prep: look at each qubit individually
                    if pauli_op == "I" or pauli_op == "Z":
                        prep_indices.append(inputs[i])
                    elif pauli_op == "X":
                        prep_indices.append(2 + inputs[i])
                    elif pauli_op == "Y":
                        prep_indices.append(4 + inputs[i])

                batch_size = params.shape[0]

                if (
                    tuple(prep_indices) not in used_prep_indices
                ):  # If input state not already used, add a new PUB
                    used_prep_indices.append(tuple(prep_indices))
                    prep_circuit = handle_n_reps(
                        qc, execution_config.current_n_reps, backend_info.backend
                    )

                    # Create input state preparation circuit
                    input_circuit = qc.copy_empty_like()
                    input_circuit.compose(
                        Pauli6PreparationBasis().circuit(prep_indices),
                        target.causal_cone_qubits,
                        inplace=True,
                    )  # Apply input state on causal cone qubits
                    input_circuit = extend_input_state_prep(
                        input_circuit, qc, target
                    )  # Add random input state on other qubits
                    self._input_states.append(input_circuit)
                    # Prepend input state to custom circuit with front composition
                    prep_circuit.compose(
                        input_circuit,
                        front=True,
                        inplace=True,
                    )

                    # Transpile circuit to decompose input state preparation
                    prep_circuit = backend_info.custom_transpile(
                        prep_circuit,
                        initial_layout=target.layout,
                        scheduling=False,
                        optimization_level=0,
                    )
                    # Add PUB to list
                    pubs.append(
                        (
                            prep_circuit,
                            parity * obs.apply_layout(prep_circuit.layout),
                            params,
                            1 / np.sqrt(dedicated_shots),
                        )
                    )

                    total_shots += (
                        dedicated_shots
                        * batch_size
                        * len(obs.group_commuting(qubit_wise=True))
                    )
                else:  # If input state already used, reuse PUB and just update observable and precision
                    pub_ref_index: int = used_prep_indices.index(tuple(prep_indices))

                    prep_circuit, ref_obs, ref_params, ref_precision = pubs[
                        pub_ref_index
                    ]
                    ref_shots = precision_to_shots(ref_precision)
                    new_precision = min(
                        ref_precision, shots_to_precision(dedicated_shots)
                    )
                    new_shots = precision_to_shots(new_precision)
                    new_pub = (
                        prep_circuit,
                        ref_obs + parity * obs.apply_layout(prep_circuit.layout),
                        ref_params,
                        new_precision,
                    )
                    pubs[pub_ref_index] = new_pub
                    total_shots -= (
                        ref_shots
                        * batch_size
                        * len(ref_obs.group_commuting(qubit_wise=True))
                    )
                    total_shots += (
                        new_shots
                        * batch_size
                        * len(new_pub[1].group_commuting(qubit_wise=True))
                    )

        if len(pubs) == 0:  # If nothing was sampled, retry
            pubs = self.get_reward_pubs(
                qc, params, target, backend_info, execution_config, dfe_precision
            )

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


@dataclass
class XEBRewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    @property
    def reward_method(self):
        return "xeb"


@dataclass
class CAFERewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on Context-Aware Fidelity Estimation (CAFE)
    """

    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"

    def __post_init__(self):
        super().__post_init__()
        self._ideal_pubs: Optional[List[SamplerPub]] = None

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}

    @property
    def reward_method(self):
        return "cafe"

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> List[SamplerPub]:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            baseline_circuit: Ideal circuit that qc should implement
        """
        if not isinstance(target, GateTarget):
            raise ValueError("CAFE reward can only be computed for a target gate")

        pubs, ideal_pubs, total_shots = [], [], 0
        if baseline_circuit is not None:
            circuit_ref = baseline_circuit.copy()
        else:
            circuit_ref = qc.metadata["baseline_circuit"].copy()
        layout = target.layout
        batch_size = params.shape[0]

        # samples, shots = np.unique(
        #     np.random.choice(len(input_circuits), self.sampling_Pauli_space),
        #     return_counts=True,
        # )
        # for sample, shot in zip(samples, shots):
        for input_state in target.input_states:
            run_qc = QuantumCircuit.copy_empty_like(
                qc, name="cafe_circ"
            )  # Circuit with custom target gate
            ref_qc = QuantumCircuit.copy_empty_like(
                circuit_ref, name="cafe_ref_circ"
            )  # Circuit with reference gate

            for circuit, context, control_flow in zip(
                [run_qc, ref_qc], [qc, circuit_ref], [True, False]
            ):
                # Bind input states to the circuits
                circuit.compose(input_state.circuit, inplace=True)
                circuit.barrier()
                cycle_circuit = handle_n_reps(
                    context,
                    execution_config.current_n_reps,
                    backend_info.backend,
                    control_flow=control_flow,
                )
                circuit.compose(cycle_circuit, inplace=True)

            # Compute inverse unitary for reference circuit
            sim_qc = causal_cone_circuit(
                ref_qc.decompose(), target.causal_cone_qubits
            )[0]
            sim_qc.save_unitary()
            sim_unitary = (
                AerSimulator(method="unitary").run(sim_qc).result().get_unitary()
            )
            reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
            reverse_unitary_qc.unitary(
                sim_unitary.adjoint(),  # Inverse unitary
                target.causal_cone_qubits,
                label="U_inv",
            )
            reverse_unitary_qc.measure_all()

            reverse_unitary_qc = backend_info.custom_transpile(
                reverse_unitary_qc,
                initial_layout=layout,
                scheduling=False,
                optimization_level=3,  # Find smallest circuit implementing inverse unitary
                remove_final_measurements=False,
            )

            # Bind inverse unitary + measurement to run circuit
            for circ, pubs_ in zip([run_qc, ref_qc], [pubs, ideal_pubs]):
                transpiled_circuit = backend_info.custom_transpile(
                    circ, initial_layout=layout, scheduling=False
                )
                transpiled_circuit.barrier()
                # Add the inverse unitary + measurement to the circuit
                transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                pubs_.append((transpiled_circuit, params, execution_config.n_shots))
            total_shots += batch_size * execution_config.n_shots

        self._total_shots = total_shots
        self._ideal_pubs = ideal_pubs

        return [SamplerPub.coerce(pub) for pub in pubs]

    def get_shot_budget(self, pubs: List[SamplerPub]) -> int:
        """
        Retrieve number of shots associated to the input pub list
        """
        return sum([pub.shots * pub.parameter_values.shape[0] for pub in pubs])


@dataclass
class ORBITRewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on ORBIT
    """

    use_interleaved: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._ideal_pubs: [List[SamplerPub]] = []

    @property
    def reward_method(self):
        return "orbit"

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> List[SamplerPub]:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            baseline_circuit: Ideal circuit that qc should implement

        Returns:
            List of pubs related to the reward method
        """
        if not isinstance(target, GateTarget):
            raise ValueError("ORBIT reward can only be computed for a target gate")

        layout = target.layout
        if baseline_circuit is not None:
            circuit_ref = baseline_circuit
        else:
            circuit_ref = qc.metadata["baseline_circuit"]
        pubs, ideal_pubs, total_shots = [], [], 0
        batch_size = params.shape[0]

        if self.use_interleaved:
            # try:
            #     Clifford(circuit_ref)
            # except QiskitError as e:
            #     raise ValueError(
            #         "Circuit should be a Clifford circuit for using interleaved RB directly"
            #     ) from e
            # ref_element = circuit_ref.to_gate(label="ref_circ")
            # custom_element = qc.to_gate(label="custom_circ")
            # exp = InterleavedRB(
            #     ref_element,
            #     target.causal_cone_qubits,
            #     [execution_config.current_n_reps],
            #     backend_info.backend,
            #     execution_config.sampling_paulis,
            #     execution_config.seed,
            #     circuit_order="RRRIII",
            # )
            # # ref_circuits = exp.circuits()[0: self.n_reps]
            # interleaved_circuits = exp.circuits()[execution_config.current_n_reps:]
            # run_circuits = [
            #     substitute_target_gate(circ, ref_element, custom_element)
            #     for circ in interleaved_circuits
            # ]
            # run_circuits = backend_info.custom_transpile(
            #     run_circuits,
            #     initial_layout=layout,
            #     scheduling=False,
            #     remove_final_measurements=False,
            # )
            # pubs = [(circ, params, execution_config.n_shots) for circ in run_circuits]
            # total_shots += batch_size * execution_config.n_shots * len(pubs)
            # self._ideal_pubs = [
            #     (circ, params, execution_config.n_shots) for circ in interleaved_circuits
            # ]
            pass
        else:
            for seq in range(execution_config.sampling_paulis):
                ref_qc = QuantumCircuit.copy_empty_like(
                    circuit_ref,
                    name="orbit_ref_circ",
                )
                run_qc = QuantumCircuit.copy_empty_like(qc, name="orbit_run_circ")
                for l in range(execution_config.current_n_reps):
                    r_cliff = random_clifford(qc.num_qubits)
                    for circ, context in zip([run_qc, ref_qc], [qc, circuit_ref]):
                        circ.compose(r_cliff.to_circuit(), inplace=True)
                        circ.barrier()
                        circ.compose(context, inplace=True)
                        circ.barrier()

                reverse_unitary = Operator(ref_qc).adjoint()
                reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
                reverse_unitary_qc.unitary(
                    reverse_unitary, reverse_unitary_qc.qubits, label="U_inv"
                )
                reverse_unitary_qc.measure_all()

                reverse_unitary_qc = backend_info.custom_transpile(
                    reverse_unitary_qc,
                    initial_layout=layout,
                    scheduling=False,
                    optimization_level=3,
                    remove_final_measurements=False,
                )  # Try to get the smallest possible circuit for the reverse unitary

                for circ, pubs_ in zip([run_qc, ref_qc], [pubs, self._ideal_pubs]):
                    transpiled_circuit = backend_info.custom_transpile(
                        circ, initial_layout=layout, scheduling=False
                    )
                    transpiled_circuit.barrier()
                    # Add the inverse unitary + measurement to the circuit
                    transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                    pubs_.append((transpiled_circuit, params, execution_config.n_shots))

                total_shots += batch_size * execution_config.n_shots
        self._total_shots = total_shots

        return [SamplerPub.coerce(pub) for pub in pubs]

    def get_shot_budget(self, pubs: List[SamplerPub]) -> int:
        """
        Retrieve number of shots associated to the input pub list
        """
        return sum([pub.shots * pub.parameter_values.shape[0] for pub in pubs])


def default_reward_config():
    return StateRewardConfig()


reward_dict = {
    "fidelity": FidelityConfig,
    "channel": ChannelRewardConfig,
    "state": StateRewardConfig,
    "xeb": XEBRewardConfig,
    "cafe": CAFERewardConfig,
    "orbit": ORBITRewardConfig,
}


def extend_observables(
    observables: SparsePauliOp, qc: QuantumCircuit, target: GateTarget
) -> SparsePauliOp:
    """
    Extend the observables to all qubits in the quantum circuit if necessary

    Args:
        observables: Pauli observables to sample
        qc: Quantum circuit to be executed on quantum system
        target: Target gate to prepare (possibly within a wider circuit context)

    Returns:
        Extended Pauli observables
    """

    if qc.num_qubits > target.causal_cone_size:
        other_qubits_indices = set(range(qc.num_qubits)) - set(
            target.causal_cone_qubits_indices
        )
        observables = observables.apply_layout(None, qc.num_qubits).apply_layout(
            target.causal_cone_qubits_indices + list(other_qubits_indices)
        )

    return observables


def extend_input_state_prep(
    input_circuit: QuantumCircuit, qc: QuantumCircuit, target: GateTarget
) -> QuantumCircuit:
    """
    Extend the input state preparation to all qubits in the quantum circuit if necessary

    Args:
        input_circuit: Input state preparation circuit
        qc: Quantum circuit to be executed on quantum system
        target: Target gate to prepare (possibly within a wider circuit context)
    """
    if (
        qc.num_qubits > target.causal_cone_size
    ):  # Add random input state on all qubits (not part of reward calculation)
        other_qubits_indices = set(range(qc.num_qubits)) - set(
            target.causal_cone_qubits_indices
        )
        other_qubits = [qc.qubits[i] for i in other_qubits_indices]
        random_input_context = Pauli6PreparationBasis().circuit(
            np.random.randint(0, 6, len(other_qubits)).tolist()
        )
        return input_circuit.compose(random_input_context, other_qubits, front=True)
    return input_circuit


def retrieve_observables(
    target_state: StateTarget,
    dfe_tuple: Optional[Tuple[float, float]] = None,
    sampling_paulis: int = 100,
    c_factor: float = 1.0,
) -> Tuple[SparsePauliOp, List[int]]:
    """
    Retrieve observables to sample for the DFE protocol (PhysRevLett.106.230501) for given target state

    :param target_state: Target state to prepare
    :param dfe_tuple: Optional Tuple (Ɛ, δ) from DFE paper
    :param sampling_paulis: Number of Pauli observables to sample
    :param c_factor: Constant factor for reward calculation
    :return: Observables to sample, number of shots for each observable
    """
    # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
    probabilities = target_state.Chi**2
    full_basis = pauli_basis(target_state.n_qubits)
    if not np.isclose(np.sum(probabilities), 1, atol=1e-5):
        print("probabilities sum um to", np.sum(probabilities))
        print("probabilities normalized")
        probabilities = probabilities / np.sum(probabilities)

    sample_size = (
        sampling_paulis
        if dfe_tuple is None
        else int(np.ceil(1 / (dfe_tuple[0] ** 2 * dfe_tuple[1])))
    )
    k_samples = np.random.choice(len(probabilities), size=sample_size, p=probabilities)

    pauli_indices, pauli_shots = np.unique(k_samples, return_counts=True)
    reward_factor = c_factor / (
        np.sqrt(target_state.dm.dim) * target_state.Chi[pauli_indices]
    )

    if dfe_tuple is not None:
        pauli_shots = np.ceil(
            2
            * np.log(2 / dfe_tuple[1])
            / (
                target_state.dm.dim
                * sample_size
                * dfe_tuple[0] ** 2
                * target_state.Chi[pauli_indices] ** 2
            )
        )
    # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
    observables = SparsePauliOp(full_basis[pauli_indices], reward_factor, copy=False)

    shots_per_basis = []
    # Group observables by qubit-wise commuting groups to reduce the number of PUBs
    for i, commuting_group in enumerate(
        observables.paulis.group_qubit_wise_commuting()
    ):
        max_pauli_shots = 0
        for pauli in commuting_group:
            pauli_index = list(full_basis).index(pauli)
            ref_index = list(pauli_indices).index(pauli_index)
            max_pauli_shots = max(max_pauli_shots, pauli_shots[ref_index])
        shots_per_basis.append(max_pauli_shots)

    return observables, shots_per_basis
