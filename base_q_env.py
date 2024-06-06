"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 16/02/2024
"""

from __future__ import annotations
from abc import ABC, abstractmethod

# For compatibility for options formatting between Estimators.
import json
import signal
from dataclasses import asdict, dataclass
from itertools import product, chain
from typing import Optional, List, Callable, Any, SupportsFloat, Tuple, Dict

from gymnasium import Env
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box
from qiskit import schedule, transpile, QiskitError

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    Gate,
    CircuitInstruction,
    ParameterVector,
)

# Qiskit Estimator Primitives: for computing Pauli expectation value sampling easily
from qiskit.primitives import (
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSamplerV2,
    BaseSamplerV1,
)
from qiskit.quantum_info import Clifford, random_clifford

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.transpiler import (
    Layout,
    InstructionProperties,
    CouplingMap,
    InstructionDurations,
)
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.providers import BackendV2

# Qiskit dynamics for pulse simulation (& benchmarking)
from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.library import InterleavedRB

# Qiskit Experiments for generating reliable baseline for complex gate calibrations / state preparations
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
    Pauli6PreparationBasis,
)
from qiskit_ibm_runtime import (
    EstimatorV1 as RuntimeEstimatorV1,
    EstimatorV2 as RuntimeEstimatorV2,
    SamplerV1 as RuntimeSamplerV1,
    IBMBackend,
)

from helper_functions import (
    retrieve_primitives,
    handle_session,
    retrieve_backend_info,
    simulate_pulse_schedule,
    retrieve_neighbor_qubits,
    substitute_target_gate,
    get_hardware_runtime_single_circuit,
)
from qconfig import (
    QiskitConfig,
    QEnvConfig,
    QuaConfig,
    CAFEConfig,
    ChannelConfig,
    ORBITConfig,
)


def _calculate_chi_target(target: DensityMatrix | Operator | QuantumCircuit | Gate):
    """
    Calculate characteristic function for the given target. Based on DFE scheme
    :param target: Density matrix of the target state or Operator of the target gate
    :return: Characteristic function of the target state
    """

    if not isinstance(target, (DensityMatrix, Operator)):
        try:  # Try to convert to Operator (in case Gate or QuantumCircuit is provided)
            target = Operator(target)
        except Exception as e:
            raise ValueError(
                "Target should be a DensityMatrix or an Operator (Gate or QuantumCircuit) object"
            ) from e
    d = 2**target.num_qubits
    basis = pauli_basis(num_qubits=target.num_qubits)
    if isinstance(target, DensityMatrix):
        chi = np.real(
            [target.expectation_value(basis[k]) for k in range(d**2)]
        ) / np.sqrt(d)
    else:
        dms = [DensityMatrix(pauli).evolve(target) for pauli in basis]
        chi = (
            np.real(
                [
                    dms[k_].expectation_value(basis[k])
                    for k, k_ in product(range(d**2), repeat=2)
                ]
            )
            / d
        )
    # Real part is taken to convert it in good format,
    # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
    return chi


@dataclass
class BaseTarget(ABC):
    """
    Base class for the target of the quantum environment
    """

    def __init__(
        self,
        physical_qubits: int | List[int],
        target_type: str,
        tgt_register: Optional[QuantumRegister] = None,
        layout: Optional[List[Layout]] = None,
    ):
        self.physical_qubits = (
            list(range(physical_qubits))
            if isinstance(physical_qubits, int)
            else physical_qubits
        )
        self.target_type = target_type
        self._tgt_register = (
            QuantumRegister(len(self.physical_qubits), "tgt")
            if tgt_register is None
            else tgt_register
        )
        self._layout: List[Layout] = (
            [
                Layout(
                    {
                        self._tgt_register[i]: self.physical_qubits[i]
                        for i in range(len(self.physical_qubits))
                    }
                )
            ]
            if layout is None
            else layout
        )
        self._n_qubits = len(self.physical_qubits)

    @property
    def tgt_register(self):
        return self._tgt_register

    @property
    def layout(self) -> List[Layout]:
        return self._layout

    @layout.setter
    def layout(self, layout: List[Layout] | Layout):
        if isinstance(layout, Layout):
            layout = [layout]
        self._layout = layout

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits: int):
        self._n_qubits = n_qubits


class StateTarget(BaseTarget):
    """
    Class to represent the state target for the quantum environment
    """

    def __init__(
        self,
        dm: Optional[DensityMatrix] = None,
        circuit: Optional[QuantumCircuit] = None,
        physical_qubits: Optional[List[int]] = None,
    ):

        if dm is None and circuit is None:
            raise ValueError("No target provided")
        if dm is not None and circuit is not None:
            assert (
                DensityMatrix(circuit) == dm
            ), "Provided circuit does not generate the provided density matrix"
        self.dm = DensityMatrix(circuit) if dm is None else dm
        self.circuit = circuit if isinstance(circuit, QuantumCircuit) else None
        self.Chi = _calculate_chi_target(self.dm)
        super().__init__(
            self.dm.num_qubits if physical_qubits is None else physical_qubits, "state"
        )


class InputState(StateTarget):
    """
    Class to represent the input state for the quantum environment
    """

    def __init__(
        self,
        input_circuit: QuantumCircuit,
        target_op: Gate | QuantumCircuit,
        tgt_register: QuantumRegister,
        n_reps: int = 1,
    ):
        """
        Initialize the input state for the quantum environment
        :param circuit: Quantum circuit representing the input state
        :param target_op: Gate to be calibrated
        :param tgt_register: Quantum register for the target gate
        :param n_reps: Number of repetitions of the target gate
        """

        if isinstance(target_op, Gate):
            circ = QuantumCircuit(tgt_register)
        else:
            circ = target_op.copy_empty_like()

        circ.compose(input_circuit, inplace=True)
        super().__init__(circuit=input_circuit)
        for _ in range(n_reps):
            if isinstance(target_op, Gate):
                circ.append(target_op, tgt_register)
            else:
                for instr in target_op.data:
                    circ.append(instr)
                # circ.compose(target_op, inplace=True)
        self.target_state = StateTarget(circuit=circ)

    @property
    def layout(self):
        raise AttributeError("Input state does not have a layout")

    @layout.setter
    def layout(self, layout: List[Layout] | Layout):
        raise AttributeError("Input state does not have a layout")

    @property
    def tgt_register(self):
        raise AttributeError("Input state does not have a target register")


class GateTarget(BaseTarget):
    """
    Class to represent the gate target for the quantum environment
    """

    def __init__(
        self,
        gate: Gate,
        physical_qubits: Optional[List[int]] = None,
        n_reps: int = 1,
        target_op: Optional[Gate | QuantumCircuit | List[QuantumCircuit]] = None,
        tgt_register: Optional[QuantumRegister] = None,
        layout: Optional[List[Layout]] = None,
    ):
        """
        Initialize the gate target for the quantum environment
        :param gate: Gate to be calibrated
        :param physical_qubits: Physical qubits forming the target gate
        :param n_reps: Number of repetitions of the target gate / circuit
        :param target_op: Element to be repeated in the calibration (default is the gate to be calibrated)
        :param tgt_register: Specify target QuantumRegister if already declared
        :param layout: Specify layout or list of layouts if already declared
        """
        self.gate = gate
        if target_op is not None:
            if not isinstance(target_op, List):
                target_op = [target_op]
            if not all([isinstance(op, (QuantumCircuit, Gate)) for op in target_op]):
                raise ValueError("target_op should be either Gate or QuantumCircuit")

        else:
            target_op = [gate]
        self.target_op = target_op
        self.n_reps = n_reps
        super().__init__(
            gate.num_qubits if physical_qubits is None else physical_qubits,
            "gate",
            tgt_register,
            layout,
        )
        input_circuits = [
            [
                PauliPreparationBasis().circuit(s)
                for s in product(range(4), repeat=op.num_qubits)
            ]
            for op in self.target_op
        ]
        input_states = [
            [
                InputState(
                    input_circuit=circ,
                    target_op=op,
                    tgt_register=self.tgt_register,
                    n_reps=n_reps,
                )
                for circ in input_circuits_list
            ]
            for op, input_circuits_list in zip(self.target_op, input_circuits)
        ]
        self.input_states = input_states
        self.Chi = _calculate_chi_target(Operator(gate))


class QiskitBackendInfo:
    """
    Class to store information on Qiskit backend (can also generate some dummy information for the case of no backend)
    """

    def __init__(
        self,
        backend: Optional[BackendV2] = None,
        custom_instruction_durations: Optional[InstructionDurations] = None,
    ):
        self.backend = backend
        self._instruction_durations = custom_instruction_durations

    def custom_transpile(
        self,
        qc: QuantumCircuit | List[QuantumCircuit],
        initial_layout: Optional[Layout] = None,
        scheduling: bool = True,
        optimization_level: int = 0,
        remove_final_measurements: bool = True,
    ):
        """
        Custom transpile function to transpile the quantum circuit
        """
        if self.backend is None and self.instruction_durations is None and scheduling:
            raise QiskitError(
                "Backend or instruction durations should be provided for scheduling"
            )
        if remove_final_measurements:
            if isinstance(qc, QuantumCircuit):
                circuit = qc.remove_final_measurements(inplace=False)
            else:
                circuit = [circ.remove_final_measurements(inplace=False) for circ in qc]
        else:
            circuit = qc
        return transpile(
            circuit,
            backend=self.backend,
            scheduling_method=(
                "asap"
                if self.instruction_durations is not None and scheduling
                else None
            ),
            basis_gates=self.basis_gates,
            coupling_map=(self.coupling_map if self.coupling_map.size() != 0 else None),
            instruction_durations=self.instruction_durations,
            optimization_level=optimization_level,
            initial_layout=initial_layout,
            dt=self.dt,
        )

    @property
    def coupling_map(self):
        """
        Retrieve the coupling map of the backend (default is fully connected if backend is None)
        """
        return (
            self.backend.coupling_map
            if self.backend is not None
            else CouplingMap.from_full(5)
        )

    @property
    def basis_gates(self):
        """
        Retrieve the basis gates of the backend (default is ['x', 'sx', 'cx', 'rz', 'measure', 'reset'])
        """
        return (
            self.backend.operation_names
            if self.backend is not None
            else ["x", "sx", "cx", "rz", "measure", "reset"]
        )

    @property
    def dt(self):
        """
        Retrieve the time unit of the backend (default is 1e-9)
        """
        return self.backend.dt if self.backend is not None else 1e-9

    @property
    def instruction_durations(self):
        """
        Retrieve the instruction durations of the backend (default is None)
        """
        return (
            self.backend.instruction_durations
            if self.backend is not None
            else self._instruction_durations
        )


class BaseQuantumEnvironment(ABC, Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, training_config: QEnvConfig):
        """
        Initialize the quantum environment
        Args:
            training_config: QEnvConfig object containing the training configuration
        """
        self._training_config = training_config
        self.n_shots = training_config.n_shots
        self.n_reps = training_config.n_reps
        self.sampling_Pauli_space = training_config.sampling_paulis
        self.c_factor = training_config.c_factor
        self.batch_size = training_config.batch_size

        self.training_with_cal = training_config.training_with_cal

        self.action_space: Box = training_config.action_space

        self.parametrized_circuit_func: Callable = training_config.parametrized_circuit
        self._func_args = training_config.parametrized_circuit_kwargs
        self.backend = training_config.backend
        self.backend_info = QiskitBackendInfo(
            self.backend, training_config.backend_config.instruction_durations_dict
        )
        self._physical_target_qubits = training_config.target.get(
            "physical_qubits", None
        )

        # gate = training_config.target.get("gate", None)
        # gate_name_base = gate.name if gate is not None else "G"
        # gate_name_base += "_cal"
        # self.custom_gates = []
        # for param_vec in self.parameters:
        #     if not isinstance(param_vec, ParameterVector):
        #         self.custom_gates.append(
        #             Gate(
        #                 gate_name_base + str(0),
        #                 len(self.physical_target_qubits),
        #                 params=self.parameters.params,
        #             )
        #         )
        #         break
        #     else:
        #         self.custom_gates.append(
        #             Gate(
        #                 gate_name_base + param_vec.name[-1],
        #                 len(self.physical_target_qubits),
        #                 params=param_vec.params,
        #             )
        #         )
        # if self.backend is not None:
        #     for custom_gate in self.custom_gates:
        #         if custom_gate.name not in self.backend.operation_names:
        #             self.backend.target.add_instruction(
        #                 custom_gate,
        #                 properties={
        #                     tuple(self.physical_target_qubits): InstructionProperties()
        #                 },
        #             )

        self._physical_neighbor_qubits = retrieve_neighbor_qubits(
            self.backend_info.coupling_map, self.physical_target_qubits
        )
        self._physical_next_neighbor_qubits = retrieve_neighbor_qubits(
            self.backend_info.coupling_map,
            self.physical_target_qubits + self.physical_neighbor_qubits,
        )
        if isinstance(training_config.backend_config, QiskitConfig):
            estimator_options = training_config.backend_config.estimator_options
        else:
            estimator_options = None

        self.target, self.circuits, self.baseline_circuits = (
            self.define_target_and_circuits()
        )
        self.abstraction_level = "pulse" if self.circuits[0].calibrations else "circuit"
        self._estimator, self._sampler = retrieve_primitives(
            self.backend,
            self.config.backend_config,
            estimator_options,
            self.circuits[0],
        )
        if not isinstance(self.sampler, BaseSamplerV1):
            if hasattr(self.estimator, "session"):
                self.fidelity_checker = ComputeUncompute(
                    RuntimeSamplerV1(session=self.estimator.session)
                )
            else:
                # TODO: Account for BackendSampler vs AerSampler
                self.fidelity_checker = None
        else:
            self.fidelity_checker = ComputeUncompute(self._sampler)

        self._mean_action = np.zeros(self.action_space.shape[-1])
        self._std_action = np.ones(self.action_space.shape[-1])

        # Data storage
        self._optimal_action = np.zeros(self.action_space.shape[-1])
        self._seed = training_config.seed
        self._session_counts = 0
        self._step_tracker = 0
        self._inside_trunc_tracker = 0
        self._total_shots = []
        self._hardware_runtime = []
        self._max_return = 0
        self._episode_ended = False
        self._episode_tracker = 0
        self._benchmark_cycle = training_config.benchmark_cycle
        self.action_history = []
        self.density_matrix_history = []
        self.reward_history = []
        self.qc_history = []
        self._pubs = []
        self._observables, self._pauli_shots = None, None
        self._index_input_state = 0
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        if isinstance(self.target, GateTarget):
            self._input_state = self.target.input_states[self.trunc_index][
                self._index_input_state
            ]
            self.process_fidelity_history = []
            self.avg_fidelity_history = []

        else:
            self.state_fidelity_history = []

    @abstractmethod
    def define_target_and_circuits(
        self,
    ) -> tuple[BaseTarget, List[QuantumCircuit], List[QuantumCircuit]]:
        raise NotImplementedError("Define target method not implemented")

    @abstractmethod
    def episode_length(self, global_step: int) -> int:
        """
        Args:
            global_step: Step in the training loop

        Returns: Episode length

        """
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def compute_benchmarks(self, params: np.array) -> np.array:
        """
        Benchmark through tomography the fidelity of the quantum system
        Args:
            params:

        Returns:

        """

    def perform_action(self, actions: np.array):
        """
        Send the action batch to the quantum system and retrieve the reward
        :param actions: action vectors to execute on quantum system
        :return: Reward table (reward for each action in the batch)
        """

        trunc_index = self._inside_trunc_tracker
        qc = self.circuits[trunc_index].copy()
        input_state_circ = None
        params, batch_size = np.array(actions), actions.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} != {self.batch_size} ")

        if self.do_benchmark():  # Benchmarking or fidelity access
            fids = self.compute_benchmarks(params)
        
        reward_methods = {
            "channel": self.channel_reward_pubs,
            "state": self.state_reward_pubs,
            "cafe": self.cafe_reward_pubs,
            "xeb": self.xeb_reward_pubs,
            "orbit": self.orbit_reward_pubs
        }
        # Get the reward method from the configuration
        reward_method = self.config.reward_method
        # Check if the reward method exists in the dictionary
        if reward_method in reward_methods:
            if reward_method == "fidelity":
                return fids
            else:
                self._pubs, total_shots = reward_methods[reward_method](qc, params)
        else:
            raise NotImplementedError(f"Reward method not implemented. Only ({list(reward_methods.keys())}) are supported.")

        self._total_shots.append(total_shots)
        self._hardware_runtime.append(
            get_hardware_runtime_single_circuit(qc, self.config.backend_config.instruction_durations_dict.duration_by_name_qubits) 
            * self.total_shots[-1]
        )
        print("Hardware runtime taken:", sum(self.hardware_runtime))

        counts = (
            self._session_counts
            if isinstance(self.estimator, (RuntimeEstimatorV1, RuntimeEstimatorV2))
            else trunc_index
        )
        self.estimator = handle_session(
            self.estimator, self.backend, counts, qc, input_state_circ
        )
        print(
            f"Sending {'Estimator' if isinstance(self.primitive, BaseEstimatorV2) else 'Sampler'} job..."
        )
        if isinstance(self.estimator, BaseEstimatorV1):
            # TODO: Remove V1 support (once pulse support for V2 is added)
            reward_table = self.run_v1_primitive(qc, params)
        else:  # EstimatorV2
            job = self.primitive.run(pubs=self._pubs)
            pub_results = job.result()

            if self.config.dfe:
                reward_table = np.sum(
                    [pub_result.data.evs for pub_result in pub_results], axis=0
                ) / len(self._observables)
            else:
                # TODO: Implement XEB and CAFE/ORBIT reward computation using Sampler
                pub_counts = [
                    [pub_result.data.meas.get_counts(i) for i in range(self.batch_size)]
                    for pub_result in pub_results
                ]
                if self.config.reward_method == "xeb":
                    reward_table = np.zeros(self.batch_size)
                else:
                    reward_table = np.mean(
                        [
                            [
                                count.get("0" * qc.num_qubits, 0) / self.n_shots
                                for count in counts
                            ]
                            for counts in pub_counts
                        ],
                        axis=0,
                    )

        print("Finished Estimator job")

        return reward_table  # Shape [batchsize]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment to its initial state
        :param seed: Seed for random number generator
        :param options: Options for the reset
        :return: Initial observation
        """
        super().reset(seed=seed)
        self._episode_tracker += 1
        self._episode_ended = False

        if isinstance(self.estimator, RuntimeEstimatorV2):
            self.estimator.options.update(job_tags=[f"rl_qoc_step{self._step_tracker}"])

        target_state = None
        if isinstance(self.target, GateTarget):
            if self.config.reward_method == "state":
                input_states = self.target.input_states[self.trunc_index]
                self._index_input_state = np.random.randint(len(input_states))
                self._input_state = input_states[self._index_input_state]
                target_state = self._input_state.target_state  # (Gate |input>=|target>)

        else:  # State preparation task
            target_state = self.target

        if target_state is not None:
            self._observables, self._pauli_shots = self.retrieve_observables(
                target_state, self.circuits[self.trunc_index]
            )

        return self._get_obs(), self._get_info()

    def retrieve_observables(
        self,
        target_state: StateTarget,
        qc: QuantumCircuit,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ):
        """
        Retrieve observables to sample for the DFE protocol (PhysRevLett.106.230501) for given target state

        :param target_state: Target state to prepare
        :param qc: Quantum circuit to be executed on quantum system
        :param dfe_precision: Optional Tuple (Ɛ, δ) from DFE paper
        :return: Observables to sample, number of shots for each observable
        """
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        probabilities = target_state.Chi**2
        full_basis = pauli_basis(qc.num_qubits)
        if not np.isclose(np.sum(probabilities), 1, atol=1e-5):
            print("probabilities sum um to", np.sum(probabilities))
            print("probabilities renormalized")
            probabilities = probabilities / np.sum(probabilities)

        if dfe_precision is not None:
            eps, delta = dfe_precision
            l = int(np.ceil(1 / (eps**2 * delta)))
        else:
            l = self.sampling_Pauli_space
        k_samples = np.random.choice(len(probabilities), size=l, p=probabilities)

        pauli_indices, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = self.c_factor / (
            np.sqrt(target_state.dm.dim) * target_state.Chi[pauli_indices]
        )

        if dfe_precision is not None:
            pauli_shots = np.ceil(
                2
                * np.log(2 / delta)
                / (
                    target_state.dm.dim
                    * l
                    * eps**2
                    * target_state.Chi[pauli_indices] ** 2
                )
            )
        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp(
            full_basis[pauli_indices], reward_factor, copy=False
        )
        shots_per_basis = []

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

    def state_reward_pubs(self, qc, params):
        """
        Retrieve observables and input state to sample for the DFE protocol for a target state
        """
        if isinstance(self.target, GateTarget):
            for _ in range(self.n_reps - 1):  # Repeat circuit for noise amplification
                qc.compose(self.circuits[self._inside_trunc_tracker], inplace=True)
            # Append input state prep circuit to the custom circuit with front composition
            qc.compose(self._input_state.circuit, inplace=True, front=True)

        qc = self.backend_info.custom_transpile(
            qc, initial_layout=self.layout[self._inside_trunc_tracker], scheduling=False
        )

        pubs = [
            (
                qc,
                obs.apply_layout(qc.layout),
                params,
                1 / np.sqrt(self.n_shots * pauli_shots),
            )
            for obs, pauli_shots in zip(
                self._observables.group_commuting(qubit_wise=True),
                self._pauli_shots,
            )
        ]
        total_shots = self.batch_size * np.sum(self._pauli_shots * self.n_shots)

        return pubs, total_shots

    def channel_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        dfe_precision: Optional[Tuple[float, float]] = None,
    ):
        """
        Retrieve observables and input state to sample for the DFE protocol for a target gate

        :param qc: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        :param dfe_precision: Optional Tuple (Ɛ, δ) from DFE paper
        :return: Observables to sample, input state to prepare
        """
        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, ChannelConfig
        ), "ChannelConfig object required for channel reward method"
        nb_states = self.config.reward_config.num_eigenstates_per_pauli
        assert (
            nb_states <= 2**qc.num_qubits
        ), f"Number of eigenstates per Pauli should be less than {2 ** qc.num_qubits}"
        d = 2**qc.num_qubits
        probabilities = self.target.Chi**2 / (d**2)
        basis = pauli_basis(num_qubits=qc.num_qubits)

        if dfe_precision is not None:
            eps, delta = dfe_precision
            l = int(np.ceil(1 / (eps**2 * delta)))
        else:
            l = self.sampling_Pauli_space

        samples, self._pauli_shots = np.unique(
            np.random.choice(len(probabilities), size=l, p=probabilities),
            return_counts=True,
        )

        pauli_indices = [np.unravel_index(sample, (d**2, d**2)) for sample in samples]
        pauli_prep, pauli_meas = zip(
            *[(basis[p[0]], basis[p[1]]) for p in pauli_indices]
        )
        reward_factor = [self.c_factor / (d * self.target.Chi[p]) for p in samples]
        self._observables = [
            SparsePauliOp(pauli_meas[i], reward_factor[i]) for i in range(len(samples))
        ]
        if dfe_precision is not None:
            self._pauli_shots = np.ceil(
                2
                * np.log(2 / delta)
                / (d * l * eps**2 * self.target.Chi[pauli_indices] ** 2)
            )

        pubs = []
        total_shots = 0
        for prep, obs, shot in zip(pauli_prep, self._observables, self._pauli_shots):
            max_input_states = 2**qc.num_qubits // nb_states
            selected_input_states = np.random.choice(
                2**qc.num_qubits, size=max_input_states, replace=False
            )
            for input_state in selected_input_states:
                prep_indices = []
                dedicated_shots = shot // (
                    2**qc.num_qubits
                )  # Number of shots per Pauli eigenstate (divided equally)
                if dedicated_shots == 0:
                    continue
                input = np.unravel_index(input_state, (2,) * qc.num_qubits)
                parity = (-1) ** np.sum(input)
                for i, pauli_op in enumerate(reversed(prep.to_label())):
                    if pauli_op == "I" or pauli_op == "Z":
                        prep_indices.append(input[i])
                    elif pauli_op == "X":
                        prep_indices.append(2 + input[i])
                    else:
                        prep_indices.append(4 + input[i])

                prep_circuit = qc.compose(
                    Pauli6PreparationBasis().circuit(prep_indices),
                    front=True,
                    inplace=False,
                )
                prep_circuit = self.backend_info.custom_transpile(
                    prep_circuit,
                    initial_layout=self.layout[self._inside_trunc_tracker],
                    scheduling=False,
                )

                pubs.append(
                    (
                        prep_circuit,
                        parity * obs.apply_layout(prep_circuit.layout),
                        params,
                        1 / np.sqrt(dedicated_shots * self.n_shots),
                    )
                )
                total_shots += dedicated_shots * self.n_shots * self.batch_size

        return pubs, total_shots

    def cafe_reward_pubs(self, circuit: QuantumCircuit, params):
        """
        Retrieve PUBs for Context-Aware Fidelity Estimation (CAFE) protocol

        :param qc: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        """

        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, CAFEConfig
        ), "CAFEConfig object required for CAFE reward method"

        pubs = []
        total_shots = 0
        circuit_ref = self.baseline_circuits[self._inside_trunc_tracker]

        input_circuits = [
            Pauli6PreparationBasis().circuit(s)
            for s in product(range(4), repeat=circuit.num_qubits)
        ]

        for input_circ in input_circuits:
            run_qc = QuantumCircuit.copy_empty_like(
                circuit, name="cafe_circ"
            )  # Circuit with the custom target gate
            ref_qc = QuantumCircuit.copy_empty_like(
                circuit_ref, name="cafe_ref_circ"
            )  # Circuit with the reference gate

            # Bind input states to the circuits
            for qc in [run_qc, ref_qc]:
                qc.compose(input_circ, inplace=True)
                qc.barrier()

            # Add the custom target gate to the run circuit n_reps times
            for qc, context in zip([run_qc, ref_qc], [circuit, circuit_ref]):
                for _ in range(self.n_reps):
                    qc.compose(context, inplace=True)
                    qc.barrier()

            reverse_unitary = Operator(ref_qc).adjoint()
            reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
            reverse_unitary_qc.unitary(
                reverse_unitary,
                reverse_unitary_qc.qubits,
                label="U_inv",
            )
            reverse_unitary_qc.measure_all()
            layout = self.layout[self._inside_trunc_tracker]
            reverse_unitary_qc = self.backend_info.custom_transpile(
                reverse_unitary_qc,
                initial_layout=layout,
                scheduling=False,
                optimization_level=3,
                remove_final_measurements=False,
            )  # Try to get the smallest possible circuit for the reverse unitary

            qc = self.backend_info.custom_transpile(
                run_qc, initial_layout=layout, scheduling=False
            )
            qc.barrier()
            # Add the inverse unitary + measurement to the circuit
            qc.compose(reverse_unitary_qc, inplace=True)

            pubs.append((qc, params, self.n_shots))
            total_shots += self.batch_size * self.n_shots

        return pubs, total_shots

    def xeb_reward_pubs(self, circuit: QuantumCircuit, params):
        pass

    def orbit_reward_pubs(self, circuit: QuantumCircuit, params):
        """
        Retrieve PUBs for ORBIT protocol

        :param circuit: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        """
        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, ORBITConfig
        ), "ORBITConfig object required for ORBIT reward method"
        layout = self.layout[self._inside_trunc_tracker]
        circuit_ref = self.baseline_circuits[self._inside_trunc_tracker]
        pubs = []
        total_shots = 0
        if self.config.reward_config.use_interleaved:
            try:
                clifford = Clifford(circuit_ref)
            except QiskitError as e:
                raise ValueError(
                    "Circuit should be a Clifford circuit for using interleaved RB directly"
                ) from e
            physical_qubits = list(
                self.layout[self._inside_trunc_tracker].get_physical_bits().keys()
            )
            ref_element = self.baseline_circuits[self._inside_trunc_tracker].to_gate(
                label="ref_circ"
            )
            custom_element = circuit.to_gate(label="custom_circ")
            exp = InterleavedRB(
                ref_element,
                physical_qubits,
                [self.n_reps],
                self.backend,
                self.sampling_Pauli_space,
                self.seed,
                circuit_order="RRRIII",
            )
            ref_circuits = exp.circuits()[0 : self.n_reps]
            interleaved_circuits = exp.circuits()[self.n_reps :]
            run_circuits = [
                substitute_target_gate(circ, ref_element, custom_element)
                for circ in interleaved_circuits
            ]
            run_circuits = self.backend_info.custom_transpile(
                run_circuits,
                initial_layout=layout,
                scheduling=False,
                remove_final_measurements=False,
            )
            pubs = [(qc, params, self.n_shots) for qc in run_circuits]
            total_shots += self.batch_size * self.n_shots * len(pubs)
        else:
            for seq in range(self.sampling_Pauli_space):

                ref_qc = QuantumCircuit.copy_empty_like(
                    self.baseline_circuits[self._inside_trunc_tracker],
                    name="orbit_ref_circ",
                )
                run_qc = QuantumCircuit.copy_empty_like(circuit, name="orbit_run_circ")
                for l in range(self.n_reps):
                    r_cliff = random_clifford(circuit.num_qubits, self.seed)
                    for qc, context in zip([run_qc, ref_qc], [circuit, circuit_ref]):
                        qc.compose(r_cliff.to_circuit(), inplace=True)
                        qc.barrier()
                        qc.compose(context, inplace=True)
                        qc.barrier()

                reverse_unitary = Operator(ref_qc).adjoint()
                reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
                reverse_unitary_qc.unitary(
                    reverse_unitary, reverse_unitary_qc.qubits, label="U_inv"
                )
                reverse_unitary_qc.measure_all()

                reverse_unitary_qc = self.backend_info.custom_transpile(
                    reverse_unitary_qc,
                    initial_layout=layout,
                    scheduling=False,
                    optimization_level=3,
                    remove_final_measurements=False,
                )  # Try to get the smallest possible circuit for the reverse unitary

                qc = self.backend_info.custom_transpile(
                    run_qc, initial_layout=layout, scheduling=False
                )
                qc.barrier()
                # Add the inverse unitary + measurement to the circuit
                qc.compose(reverse_unitary_qc, inplace=True)

                pubs.append((qc, params, self.n_shots))
                total_shots += self.batch_size * self.n_shots

        return pubs, total_shots

    def _observable_to_observation(self):
        """
        Convert the observable to an observation to be given to the agent
        """
        if self.config.reward_method == "state":
            n_qubits = self.observables.num_qubits
            d = 2**n_qubits
            pauli_to_index = {pauli: i for i, pauli in enumerate(pauli_basis(n_qubits))}
            array_obs = np.zeros(d**2)
            for pauli in self.observables:
                array_obs[pauli_to_index[pauli.paulis[0]]] = pauli.coeffs[0]

            array_obs = []
            return array_obs
        else:
            raise NotImplementedError("Channel estimator not yet implemented")

    @property
    def config(self):
        return self._training_config

    @property
    def estimator(self) -> BaseEstimatorV1 | BaseEstimatorV2:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimatorV1 | BaseEstimatorV2):
        self._estimator = estimator

    @property
    def sampler(self) -> BaseSamplerV1 | BaseSamplerV2:
        return self._sampler

    @property
    def primitive(self) -> BaseEstimatorV2 | BaseSamplerV2:
        if (
            self.config.reward_method == "state"
            or self.config.reward_method == "channel"
        ):
            return self.estimator
        else:
            return self.sampler

    @property
    def physical_target_qubits(self):
        return self._physical_target_qubits

    @property
    def physical_neighbor_qubits(self):
        return self._physical_neighbor_qubits

    @property
    def physical_next_neighbor_qubits(self):
        return self._physical_next_neighbor_qubits

    @property
    @abstractmethod
    def tgt_instruction_counts(self) -> int:
        """
        Number of occurrences of the target instruction in the circuit
        """
        raise NotImplementedError("Target instruction counts not implemented")

    @property
    def fidelity_history(self):
        return (
            self.avg_fidelity_history
            if self.target.target_type == "gate"
            else self.state_fidelity_history
        )

    @property
    def step_tracker(self):
        return self._step_tracker

    @step_tracker.setter
    def step_tracker(self, step: int):
        assert step >= 0, "step must be positive integer"
        self._step_tracker = step

    def signal_handler(self, signum, frame):
        """Signal handler for SIGTERM and SIGINT signals."""
        print(f"Received signal {signum}, closing environment...")
        self.close()

    def close(self) -> None:
        if isinstance(self.backend, IBMBackend):
            self.estimator.session.close()

    def clear_history(self):
        """
        Clear all stored data to start new training.
        """
        self._step_tracker = 0
        self._episode_tracker = 0
        self.qc_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        if isinstance(self.target, GateTarget):
            self.avg_fidelity_history.clear()
            self.process_fidelity_history.clear()

        else:
            self.state_fidelity_history.clear()
            self.density_matrix_history.clear()

    def update_gate_calibration(self):
        """
        Update backend target with the optimal action found during training

        :return: Pulse calibration for the target gate
        """
        if not isinstance(self.target, GateTarget):
            raise ValueError("Target type should be a gate for gate calibration task.")

        if self.abstraction_level == "pulse":
            sched = schedule(self.circuits[0], self.backend).assign_parameters(
                {
                    param: action
                    for param, action in zip(self.parameters[0], self._optimal_action)
                }
            )
            duration = sched.duration
            if isinstance(self.backend, DynamicsBackend):
                error = 1.0
                error -= simulate_pulse_schedule(
                    self.backend,
                    sched,
                    target_unitary=Operator(self.target.gate),
                    target_state=Statevector.from_int(0, dims=[2] * self.n_qubits),
                )["gate_fidelity"]["optimal"]

            else:
                error = 1.0 - np.max(self.avg_fidelity_history)
            instruction_prop = InstructionProperties(duration, error, sched)
            self.backend.target.update_instruction_properties(
                self.target.gate.name,
                tuple(self.physical_target_qubits),
                instruction_prop,
            )

            return self.backend.target.get_calibration(
                self.target.gate.name, tuple(self.physical_target_qubits)
            )
        else:
            return self.circuits[0].assign_parameters(
                {self.parameters[0]: self._optimal_action}
            )

    @property
    def benchmark_cycle(self) -> int:
        """
        Cycle at which fidelity benchmarking is performed
        :return:
        """
        return self._benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, step: int) -> None:
        """
        Set cycle at which fidelity benchmarking is performed
        :param step:
        :return:
        """
        assert step >= 0, "Cycle needs to be a positive integer"
        self._benchmark_cycle = step

    def do_benchmark(self) -> bool:
        """
        Check if benchmarking should be performed at current step
        :return:
        """
        if self.config.reward_method == "fidelity":
            return True

        else:
            return self._episode_tracker % self.benchmark_cycle == 0

    def _get_info(self) -> Any:
        step = self._episode_tracker
        if self._episode_ended:
            if self.do_benchmark():
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "circuit fidelity": self.fidelity_history[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "max circuit fidelity": np.max(self.fidelity_history),
                    "arg max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "arg max circuit fidelity": np.argmax(self.fidelity_history),
                    "optimal action": self.optimal_action,
                    "input_state": self.target.input_states[self.trunc_index][
                        self._index_input_state
                    ].circuit.name,
                }
            else:
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "optimal action": self.optimal_action,
                    "input_state": self.target.input_states[self.trunc_index][
                        self._index_input_state
                    ].circuit.name,
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "input_state": self.target.input_states[self.trunc_index][
                    self._index_input_state
                ].circuit.name,
                "truncation_index": self.trunc_index,
            }
        return info

    def __repr__(self):
        string = f"QuantumEnvironment composed of {self.n_qubits} qubits, \n"
        string += (
            f"Defined target: {self.target.target_type} "
            f"({self.target.gate if isinstance(self.target, GateTarget) else self.target.dm})\n"
        )
        string += f"Physical qubits: {self.target.physical_qubits}\n"
        string += f"Backend: {self.backend},\n"
        string += f"Abstraction level: {self.abstraction_level},\n"
        string += f"Run options: N_shots ({self.n_shots}), Sampling_Pauli_space ({self.sampling_Pauli_space}), \n"
        string += f"Batch size: {self.batch_size}, \n"
        return string

    # Properties

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        try:
            assert size > 0 and isinstance(size, int)
            self._batch_size = size
        except AssertionError:
            raise ValueError("Batch size should be positive integer.")

    @property
    def n_qubits(self):
        return self.target.n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        assert (
            isinstance(n_qubits, int) and n_qubits > 0
        ), "n_qubits must be a positive integer"
        self.target.n_qubits = n_qubits

    @property
    def layout(self):
        return self.target.layout

    @layout.setter
    def layout(self, layout: Layout | List[Layout]):
        if isinstance(layout, Layout):
            layout = [layout]
        self.target.layout = layout

    @property
    @abstractmethod
    def parameters(self) -> List[ParameterVector] | ParameterVector:
        """
        Return the Qiskit ParameterVector defining the actions applied on the environment
        """
        raise NotImplementedError("Parameters not implemented")

    @property
    def observables(self) -> SparsePauliOp:
        return self._observables

    @property
    def total_shots(self):
        return self._total_shots
    
    @property
    def hardware_runtime(self):
        return self._hardware_runtime

    @property
    def n_actions(self):
        return self.action_space.shape[-1]

    @property
    def optimal_action(self):
        return self._optimal_action

    @property
    def mean_action(self):
        return self._mean_action

    @mean_action.setter
    def mean_action(self, value):
        self._mean_action = np.array(value)

    @property
    def std_action(self):
        return self._std_action

    @std_action.setter
    def std_action(self, value):
        self._std_action = np.array(value)

    @property
    @abstractmethod
    def trunc_index(self) -> int:
        """
        Index of the truncation to be applied
        """
        raise NotImplementedError("Truncation index not implemented")

    def to_json(self):
        return json.dumps(
            {
                "n_qubits": self.n_qubits,
                "config": asdict(self.config),
                "abstraction_level": self.abstraction_level,
                "sampling_Pauli_space": self.sampling_Pauli_space,
                "n_shots": self.n_shots,
                "target_type": self.target.target_type,
                "target": self.target,
                "c_factor": self.c_factor,
                "reward_history": self.reward_history,
                "action_history": self.action_history,
                "fidelity_history": (
                    self.avg_fidelity_history
                    if self.target.target_type == "gate"
                    else self.state_fidelity_history
                ),
            }
        )

    @classmethod
    def from_json(cls, json_str):
        """Return a MyCustomClass instance based on the input JSON string."""

        class_info = json.loads(json_str)
        abstraction_level = class_info["abstraction_level"]
        target = class_info["target"]
        n_shots = class_info["n_shots"]
        c_factor = class_info["c_factor"]
        sampling_Pauli_space = class_info["sampling_Pauli_space"]
        config = class_info["config"]
        q_env = cls(config)
        q_env.reward_history = class_info["reward_history"]
        q_env.action_history = class_info["action_history"]
        if class_info["target_type"] == "gate":
            q_env.avg_fidelity_history = class_info["fidelity_history"]
        else:
            q_env.state_fidelity_history = class_info["fidelity_history"]
        return q_env

    def compute_channel_reward(self, params, trunc_index):
        pass

    def run_v1_primitive(self, qc: QuantumCircuit, params: np.array):
        if self.config.channel_estimator:
            raise NotImplementedError(
                "Channel estimator not implemented for EstimatorV1"
            )

        job = self.estimator.run(
            circuits=[qc] * self.batch_size,
            observables=[self._observables.apply_layout(qc.layout)] * self.batch_size,
            parameter_values=params,
            shots=int(np.max(self._pauli_shots) * self.n_shots),
        )
        self._total_shots.append(
            int(np.max(self._pauli_shots) * self.n_shots)
            * self.batch_size
            * len(self._observables.group_commuting(qubit_wise=True))
        )
        reward_table = job.result().values / self._observables.size