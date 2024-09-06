"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 05/09/2024
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod

# For compatibility for options formatting between Estimators.
import json
import signal
from dataclasses import asdict, dataclass
from itertools import product
from typing import Optional, List, Callable, Any, Tuple, Literal

from gymnasium import Env
import numpy as np
from gymnasium.core import ObsType
from gymnasium.spaces import Box
from qiskit import schedule, transpile, QiskitError

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    Gate,
    ParameterVector,
    CircuitInstruction,
)

# Qiskit Estimator Primitives: for computing Pauli expectation value sampling easily
from qiskit.primitives import (
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSamplerV2,
    BaseSamplerV1,
)
from qiskit.quantum_info import random_unitary, partial_trace

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info.states.quantum_state import QuantumState, QuantumChannel
from qiskit.quantum_info.operators import (
    SparsePauliOp,
    Operator,
    pauli_basis,
    PauliList,
    Clifford,
)
from qiskit.quantum_info.operators.measures import average_gate_fidelity
from qiskit.quantum_info.random import random_clifford

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
    EstimatorV2 as RuntimeEstimatorV2,
)

from .custom_jax_sim import JaxSolver
from .custom_jax_sim.jax_solver_v2 import JaxSolver as JaxSolverV2
from .helper_functions import (
    retrieve_primitives,
    handle_session,
    simulate_pulse_schedule,
    retrieve_neighbor_qubits,
    substitute_target_gate,
    get_hardware_runtime_single_circuit,
    has_noise_model,
    get_optimal_z_rotation,
    rotate_unitary,
    projected_state,
    qubit_projection,
)
from .qconfig import (
    QiskitConfig,
    QEnvConfig,
    CAFEConfig,
    ChannelConfig,
    ORBITConfig,
    XEBConfig,
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
        layout: Optional[Layout] = None,
    ):
        """
        Initialize the base target for the quantum environment
        :param physical_qubits: Physical qubits on which the target is defined
        :param target_type: Type of the target (state / gate)
        :param tgt_register: Optional existing QuantumRegister for the target
        :param layout: Optional existing layout for the target (used when target touches a subset of
        all physical qubits present in a circuit)

        """
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
        self._layout: Layout = (
            Layout(
                {
                    self._tgt_register[i]: self.physical_qubits[i]
                    for i in range(len(self.physical_qubits))
                }
            )
            if layout is None
            else layout
        )
        self._n_qubits = len(self.physical_qubits)

    @property
    def tgt_register(self):
        return self._tgt_register

    @property
    def layout(self) -> Layout:
        """
        Layout for the target
        """
        return self._layout

    @layout.setter
    def layout(self, layout: Layout):
        if not isinstance(layout, Layout):
            raise ValueError("Layout should be a Layout object")
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
        state: Optional[DensityMatrix | Statevector] = None,
        circuit: Optional[QuantumCircuit] = None,
        physical_qubits: Optional[List[int]] = None,
    ):
        if isinstance(state, DensityMatrix) and state.purity() != 1:
            raise ValueError("Density matrix should be pure")
        if isinstance(state, DensityMatrix):
            state = Statevector(state)
        if state is None and circuit is None:
            raise ValueError("No target provided")
        if state is not None and circuit is not None:
            assert (
                Statevector(circuit) == state
            ), "Provided circuit does not generate the provided state"
        self.dm = DensityMatrix(circuit) if state is None else DensityMatrix(state)
        self.circuit = circuit if isinstance(circuit, QuantumCircuit) else None
        if self.circuit is None:
            qc = QuantumCircuit(self.dm.num_qubits)
            qc.initialize(state)
            self.circuit = qc

        self.Chi = _calculate_chi_target(self.dm)
        super().__init__(
            self.dm.num_qubits if physical_qubits is None else physical_qubits, "state"
        )

    def fidelity(self, state: QuantumState | QuantumCircuit):
        """
        Compute the fidelity of the state with the target
        :param state: State to compare with the target state
        """
        if isinstance(state, QuantumCircuit):
            try:
                state = DensityMatrix(state)
            except Exception as e:
                raise ValueError("Input could not be converted to state") from e
        if not isinstance(state, (Statevector, DensityMatrix)):
            raise ValueError("Input should be a Statevector or DensityMatrix object")
        return state_fidelity(state, self.dm)

    def __repr__(self):
        return f"StateTarget({self.dm} on qubits {self.physical_qubits})"


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
        :param input_circuit: Quantum circuit representing the input state
        :param target_op: Gate to be calibrated
        :param tgt_register: Quantum register for the target gate
        :param n_reps: Number of repetitions of the target gate
        """
        self._n_reps = n_reps
        self._target_op = target_op
        if isinstance(target_op, Gate):
            circ = QuantumCircuit(tgt_register)
            circ.append(target_op, tgt_register)
        else:
            circ = target_op.copy("target")

        circ.compose(input_circuit, inplace=True, front=True)
        super().__init__(circuit=input_circuit)
        for _ in range(self.n_reps - 1):
            if isinstance(target_op, Gate):
                circ.append(target_op, tgt_register)
            else:
                circ.compose(target_op, inplace=True)
        self.target_state = StateTarget(circuit=circ)

    @property
    def layout(self):
        raise AttributeError("Input state does not have a layout")

    @layout.setter
    def layout(self, layout: Layout):
        raise AttributeError("Input state does not have a layout")

    @property
    def tgt_register(self):
        raise AttributeError("Input state does not have a target register")

    @property
    def n_reps(self) -> int:
        return self._n_reps

    @n_reps.setter
    def n_reps(self, n_reps: int):
        assert n_reps > 0, "Number of repetitions should be positive"
        self._n_reps = n_reps
        if isinstance(self._target_op, Gate):
            circ = QuantumCircuit(self.tgt_register)
            circ.append(self._target_op, self.tgt_register)
        else:
            circ = self._target_op.copy("target")

        circ.compose(self.input_circuit, inplace=True, front=True)
        for _ in range(self.n_reps - 1):
            if isinstance(self._target_op, Gate):
                circ.append(self._target_op, self.tgt_register)
            else:
                circ.compose(self._target_op, inplace=True)
        self.target_state = StateTarget(circuit=circ)

    @property
    def input_circuit(self) -> QuantumCircuit:
        """
        Get the input state preparation circuit
        """
        return self.circuit

    @property
    def target_circuit(self) -> QuantumCircuit:
        """
        Get the target circuit for the input state
        """
        return self.target_state.circuit

    @property
    def target_dm(self) -> DensityMatrix:
        """
        Get the target density matrix for the input state
        """
        return self.target_state.dm


class GateTarget(BaseTarget):
    """
    Class to represent the gate target for the quantum environment
    """

    def __init__(
        self,
        gate: Gate,
        physical_qubits: Optional[List[int]] = None,
        n_reps: int = 1,
        target_op: Optional[Gate | QuantumCircuit] = None,
        tgt_register: Optional[QuantumRegister] = None,
        layout: Optional[Layout] = None,
        input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4",
    ):
        """
        Initialize the gate target for the quantum environment
        :param gate: Gate to be calibrated
        :param physical_qubits: Physical qubits forming the target gate
        :param n_reps: Number of repetitions of the target gate / circuit
        :param target_op: Element to be repeated in the calibration (default is the gate to be calibrated)
        :param tgt_register: Specify target QuantumRegister if already declared
        :param layout: Specify layout if already declared
        :param input_states_choice: Type of input states to be used for the calibration
        """
        self.gate = gate
        if target_op is not None:
            if not isinstance(target_op, (QuantumCircuit, Gate)):
                raise ValueError("target_op should be either Gate or QuantumCircuit")
            if isinstance(target_op, Gate):
                target_circuit = QuantumCircuit(target_op.num_qubits)
                target_circuit.append(target_op, range(target_op.num_qubits))
                target_op = target_circuit
        else:
            target_op = QuantumCircuit(gate.num_qubits)
            target_op.append(gate, list(range(gate.num_qubits)))
        self._target_op: QuantumCircuit = target_op
        self._n_reps = n_reps
        n_qubits = self._target_op.num_qubits
        super().__init__(
            gate.num_qubits if physical_qubits is None else physical_qubits,
            "gate",
            tgt_register,
            layout,
        )
        if input_states_choice == "pauli4":
            input_circuits = [
                PauliPreparationBasis().circuit(s)
                for s in product(range(4), repeat=n_qubits)
            ]
        elif input_states_choice == "pauli6":
            input_circuits = [
                Pauli6PreparationBasis().circuit(s)
                for s in product(range(6), repeat=n_qubits)
            ]
        elif input_states_choice == "2-design":  # 2-design

            d = 2**n_qubits
            unitaries = [random_unitary(d) for _ in range(4**n_qubits)]
            circuits = [QuantumCircuit(n_qubits) for _ in range(4**n_qubits)]
            for circ, unitary in zip(circuits, unitaries):
                circ.unitary(unitary, range(n_qubits))
            input_circuits = circuits
        else:
            raise ValueError(
                f"Input states choice {input_states_choice} not recognized. Should be 'pauli4', 'pauli6' or '2-design'"
            )

        self.input_states = [
            InputState(
                input_circuit=circ,
                target_op=self._target_op,
                tgt_register=self.tgt_register,
                n_reps=n_reps,
            )
            for circ in input_circuits
        ]
        if n_qubits <= 3:
            self.Chi = _calculate_chi_target(Operator(self._target_op).power(n_reps))
        else:
            self.Chi = None

    def gate_fidelity(
        self,
        channel: QuantumChannel | Operator | Gate | QuantumCircuit,
        n_reps: int = 1,
    ):
        """
        Compute the average gate fidelity of the gate with the target gate
        If the target has a circuit context, the fidelity is computed with respect to the channel derived from the
        target circuit
        :param channel: channel to compare with the target gate
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if isinstance(channel, (QuantumCircuit, Gate)):
            try:
                channel = Operator(channel)
            except Exception as e:
                raise ValueError("Input could not be converted to channel") from e
        if not isinstance(channel, (Operator, QuantumChannel)):
            raise ValueError("Input should be an Operator object")
        return average_gate_fidelity(channel, Operator(self._target_op).power(n_reps))

    def state_fidelity(self, state: QuantumState, n_reps: int = 1):
        """
        Compute the fidelity of the state with the target state derived from the application of the target gate/circuit
        context to |0...0>
        :param state: State to compare with target state
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if not isinstance(state, (Statevector, DensityMatrix)):
            raise ValueError("Input should be a Statevector or DensityMatrix object")
        return state_fidelity(
            state, Statevector(self._target_op.power(n_reps, True, True))
        )

    def fidelity(self, op: QuantumState | QuantumChannel | Operator, n_reps: int = 1):
        """
        Compute the fidelity of the input op with respect to the target circuit context channel/output state.
        :param op: Object to compare with the target. If QuantumState, computes state fidelity, if QuantumChannel or
            Operator, computes gate fidelity
        :param n_reps: Specify number of repetitions of the target operation (default is 1)
        """
        if isinstance(op, (QuantumCircuit, Gate)):
            try:
                op = Operator(op)
            except Exception as e:
                raise ValueError("Input could not be converted to channel") from e
        if isinstance(op, (Operator, QuantumChannel)):
            return self.gate_fidelity(op, n_reps)
        elif isinstance(op, (Statevector, DensityMatrix)):
            return self.state_fidelity(op, n_reps)
        else:
            raise ValueError(
                f"Input should be a Statevector, DensityMatrix, Operator or QuantumChannel object, "
                f"received {type(op)}"
            )

    @property
    def target_instruction(self):
        """
        Get the target instruction
        """
        return CircuitInstruction(self.gate, (q for q in self.tgt_register))

    @property
    def target_operator(self):
        """
        Get the target unitary operator
        """
        return Operator(self._target_op)

    @property
    def target_circuit(self):
        """
        Get the target circuit
        """
        return self._target_op

    @target_circuit.setter
    def target_circuit(self, target_op: QuantumCircuit):
        """
        Set the target circuit
        """
        self._target_op = target_op
        self.input_states = [
            InputState(input_state.circuit, target_op, self.tgt_register, self.n_reps)
            for input_state in self.input_states
        ]
        if target_op.num_qubits <= 3:
            self.Chi = np.round(
                _calculate_chi_target(Operator(self._target_op).power(self.n_reps)), 5
            )
        else:
            self.Chi = None

    @property
    def has_context(self):
        """
        Check if the target has a circuit context attached or if only composed of the target gate
        """
        gate_qc = QuantumCircuit(self.gate.num_qubits)
        gate_qc.append(self.gate, list(range(self.gate.num_qubits)))

        return Operator(self._target_op) != Operator(gate_qc)

    @property
    def n_reps(self) -> int:
        return self._n_reps

    @n_reps.setter
    def n_reps(self, n_reps: int):
        assert n_reps > 0, "Number of repetitions should be positive"
        self._n_reps = n_reps
        self.Chi = _calculate_chi_target(Operator(self._target_op).power(n_reps))
        for input_state in self.input_states:
            input_state.n_reps = n_reps

    def __repr__(self):
        return (
            f"GateTarget({self.gate.name}, on qubits {self.physical_qubits}"
            f" with{'' if self.has_context else 'out'} context)"
        )


class QiskitBackendInfo:
    """
    Class to store information on Qiskit backend (can also generate some dummy information for the case of no backend)
    """

    def __init__(
        self,
        backend: Optional[BackendV2] = None,
        custom_instruction_durations: Optional[InstructionDurations] = None,
        n_qubits: int = 0,
    ):
        if backend is not None and backend.coupling_map is None:
            raise QiskitError("Backend does not have a coupling map")
        self.backend = backend
        self._instruction_durations = custom_instruction_durations
        self._n_qubits = backend.num_qubits if backend is not None else n_qubits

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
            else CouplingMap.from_full(self._n_qubits)
        )

    @property
    def basis_gates(self):
        """
        Retrieve the basis gates of the backend (default is ['x', 'sx', 'cx', 'rz', 'measure', 'reset'])
        """
        return (
            self.backend.operation_names
            if self.backend is not None
            else ["u", "rzx", "cx", "rz", "measure", "reset"]
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
            and self.backend.instruction_durations.duration_by_name_qubits
            else self._instruction_durations
        )

    @property
    def num_qubits(self):
        return self._n_qubits

    @num_qubits.setter
    def num_qubits(self, n_qubits: int):
        assert n_qubits > 0, "Number of qubits should be positive"
        if self.backend is not None:
            raise ValueError(
                "Number of qubits should not be set if backend is provided"
            )
        self._n_qubits = n_qubits


class BaseQuantumEnvironment(ABC, Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, training_config: QEnvConfig):
        """
        Initialize the quantum environment
        Args:
            training_config: QEnvConfig object containing the training configuration
        """
        self._training_config = training_config
        self._reward_methods = {
            "channel": self.channel_reward_pubs,
            "state": self.state_reward_pubs,
            "cafe": self.cafe_reward_pubs,
            "xeb": self.xeb_reward_pubs,
            "orbit": self.orbit_reward_pubs,
            "fidelity": self.compute_benchmarks,
        }
        if self.config.reward_method not in self._reward_methods:
            raise ValueError(
                f"Reward method {self.config.reward_method} not implemented. Only "
                f"{list(self._reward_methods.keys())} are supported."
            )

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

        self._target, self.circuits, self.baseline_circuits = (
            self.define_target_and_circuits()
        )
        self.abstraction_level = "pulse" if self.circuits[0].calibrations else "circuit"
        self._estimator, self._sampler = retrieve_primitives(
            self.backend,
            self.config.backend_config,
            estimator_options,
            self.circuits[0],
        )

        # self.fidelity_checker = ComputeUncompute(self._sampler)

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
        self.action_history = []
        self.density_matrix_history = []
        self.reward_history = []
        self.qc_history = []
        self._pubs, self._ideal_pubs = [], []
        self._observables, self._pauli_shots = None, None
        self._index_input_state = 0
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        if isinstance(self.target, GateTarget):
            self._input_state = self.target.input_states[self._index_input_state]
        else:
            self._input_state = InputState(
                QuantumCircuit(self.target.n_qubits),
                self.target.circuit,
                self.target.tgt_register,
            )
        self.process_fidelity_history = []
        self.avg_fidelity_history = []
        self.circuit_fidelity_history = []
        self.circuit_fidelity_history_nreps = []
        self.avg_fidelity_history_nreps = []

    @abstractmethod
    def define_target_and_circuits(
        self,
    ) -> tuple[
        GateTarget | StateTarget | List[GateTarget | StateTarget],
        List[QuantumCircuit],
        List[QuantumCircuit],
    ]:
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
    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Benchmark through tomography or through simulation the policy
        Args:
            qc: Quantum circuit to benchmark
            params:

        Returns: Fidelity metric or array of fidelities for all actions in the batch

        """

    def perform_action(self, actions: np.array):
        """
        Send the action batch to the quantum system and retrieve the reward
        :param actions: action vectors to execute on quantum system
        :return: Reward table (reward for each action in the batch)
        """

        trunc_index = self._inside_trunc_tracker
        qc = self.circuits[trunc_index].copy()
        params, batch_size = np.array(actions), actions.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} != {self.batch_size} ")

        # Get the reward method from the configuration
        reward_method = self.config.reward_method
        if self.do_benchmark():  # Benchmarking or fidelity access
            fids = self.compute_benchmarks(qc, params)
            if reward_method == "fidelity":
                self._total_shots.append(0)
                self._hardware_runtime.append(0.0)
                return fids

        # Check if the reward method exists in the dictionary
        self._pubs, total_shots = self._reward_methods[reward_method](qc, params)
        self._total_shots.append(total_shots)
        if self.backend_info.instruction_durations is not None:
            self._hardware_runtime.append(
                get_hardware_runtime_single_circuit(
                    qc,
                    self.backend_info.instruction_durations.duration_by_name_qubits,
                )
                * self.total_shots[-1]
            )
            print("Hardware runtime taken:", sum(self.hardware_runtime))

        counts = (
            self._session_counts
            if isinstance(self.estimator, RuntimeEstimatorV2)
            else trunc_index
        )
        self.estimator = handle_session(
            self.estimator, self.backend, counts, qc, self._input_state.circuit
        )
        print(
            f"Sending {'Estimator' if isinstance(self.primitive, BaseEstimatorV2) else 'Sampler'} job..."
        )
        start = time.time()
        if isinstance(self.estimator, BaseEstimatorV1):
            # TODO: Remove V1 support (once pulse support for V2 is added)
            reward_table = self.run_v1_primitive(qc, params)
        else:  # EstimatorV2
            job = self.primitive.run(pubs=self._pubs)
            pub_results = job.result()
            print("Time for running", time.time() - start)

            if self.config.dfe:
                reward_table = np.sum(
                    [pub_result.data.evs for pub_result in pub_results], axis=0
                ) / len(self._observables)
            else:
                pub_counts = [
                    [pub_result.data.meas.get_counts(i) for i in range(self.batch_size)]
                    for pub_result in pub_results
                ]
                if self.config.reward_method == "xeb":
                    # TODO: Implement XEB reward computation using Sampler
                    reward_table = np.zeros(self.batch_size)
                    raise NotImplementedError(
                        "XEB reward computation not implemented yet"
                    )
                else:
                    survival_probability = [
                        np.array(
                            [
                                count.get("0" * qc.num_qubits, 0) / self.n_shots
                                for count in counts
                            ]
                        )
                        for counts in pub_counts
                    ]
                    reward_table = np.mean(survival_probability, axis=0)

        print(
            f"Finished {'Estimator' if isinstance(self.primitive, BaseEstimatorV2) else 'Sampler'} job"
        )

        return reward_table  # Shape [batch size]

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
                input_states = self.target.input_states
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
        dfe_tuple: Optional[Tuple[float, float]] = None,
    ):
        """
        Retrieve observables to sample for the DFE protocol (PhysRevLett.106.230501) for given target state

        :param target_state: Target state to prepare
        :param qc: Quantum circuit to be executed on quantum system
        :param dfe_tuple: Optional Tuple (Ɛ, δ) from DFE paper
        :return: Observables to sample, number of shots for each observable
        """
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        probabilities = target_state.Chi**2
        full_basis = pauli_basis(qc.num_qubits)
        if not np.isclose(np.sum(probabilities), 1, atol=1e-5):
            print("probabilities sum um to", np.sum(probabilities))
            print("probabilities normalized")
            probabilities = probabilities / np.sum(probabilities)

        sample_size = (
            self.sampling_Pauli_space
            if dfe_tuple is None
            else int(np.ceil(1 / (dfe_tuple[0] ** 2 * dfe_tuple[1])))
        )
        k_samples = np.random.choice(
            len(probabilities), size=sample_size, p=probabilities
        )

        pauli_indices, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = self.c_factor / (
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

    def state_reward_pubs(self, qc: QuantumCircuit, params):
        """
        Retrieve observables and input state to sample for the DFE protocol for a target state
        """

        prep_circuit = qc.copy()
        if isinstance(self.target, GateTarget):
            # Append input state prep circuit to the custom circuit with front composition
            prep_circuit = qc.compose(
                self._input_state.circuit, inplace=False, front=True
            )
            for _ in range(self.n_reps - 1):  # Repeat circuit for noise amplification
                prep_circuit.compose(qc, inplace=True)

        prep_circuit = self.backend_info.custom_transpile(
            prep_circuit,
            initial_layout=self.layout,
            scheduling=False,
        )

        pubs = [
            (
                prep_circuit,
                obs.apply_layout(prep_circuit.layout),
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
        if not isinstance(self.target, GateTarget):
            raise TypeError("Target type should be a gate")
        if not isinstance(self.config.reward_config, ChannelConfig):
            raise TypeError("ChannelConfig object required for channel reward method")
        if qc.num_qubits > 3:
            raise ValueError("Channel reward method is only supported for 1-3 qubits")

        nb_states = self.config.reward_config.num_eigenstates_per_pauli
        if nb_states >= 2**qc.num_qubits:
            raise ValueError(
                f"Number of eigenstates per Pauli should be less than {2 ** qc.num_qubits}"
            )
        d = 2**qc.num_qubits
        probabilities = self.target.Chi**2 / (d**2)
        non_zero_indices = np.nonzero(probabilities)[0]
        non_zero_probabilities = probabilities[non_zero_indices]
        basis = pauli_basis(num_qubits=qc.num_qubits)

        if dfe_precision is not None:
            eps, delta = dfe_precision
            pauli_sampling = int(np.ceil(1 / (eps**2 * delta)))
        else:
            eps, delta = None, None
            pauli_sampling = self.sampling_Pauli_space

        samples, self._pauli_shots = np.unique(
            np.random.choice(
                non_zero_indices, size=pauli_sampling, p=non_zero_probabilities
            ),
            return_counts=True,
        )
        pauli_indices = np.array(
            [np.unravel_index(sample, (d**2, d**2)) for sample in samples], dtype=int
        )

        pauli_prep, pauli_meas = zip(
            *[(basis[p[1]], basis[p[0]]) for p in pauli_indices]
        )
        pauli_prep, pauli_meas = PauliList(pauli_prep), PauliList(pauli_meas)
        reward_factor = [self.c_factor / (d * self.target.Chi[p]) for p in samples]

        self._observables = SparsePauliOp(pauli_meas, reward_factor)

        if dfe_precision is not None:
            self._pauli_shots = np.ceil(
                2
                * np.log(2 / delta)
                / (d * pauli_sampling * eps**2 * self.target.Chi[samples] ** 2)
            )

        pubs, total_shots = [], 0
        used_prep_indices, used_indices, idx = (
            [],
            [],
            0,
        )  # Track used input states to reduce number of PUBs

        for prep, obs, shots in zip(pauli_prep, self._observables, self._pauli_shots):
            max_input_states = d // nb_states
            selected_input_states = np.random.choice(
                d, size=max_input_states, replace=False
            )
            for input_state in selected_input_states:
                prep_indices = []
                dedicated_shots = (
                    shots * self.n_shots // max_input_states
                )  # Number of shots per Pauli eigenstate (divided equally)
                if dedicated_shots == 0:
                    continue
                # Convert input state to Pauli6 basis: preparing pure eigenstates of Pauli_prep
                inputs = np.unravel_index(input_state, (2,) * qc.num_qubits)
                parity = (-1) ** np.sum(inputs)
                for i, pauli_op in enumerate(reversed(prep.to_label())):
                    if pauli_op == "I" or pauli_op == "Z":
                        prep_indices.append(inputs[i])
                    elif pauli_op == "X":
                        prep_indices.append(2 + inputs[i])
                    elif pauli_op == "Y":
                        prep_indices.append(4 + inputs[i])
                if (
                    tuple(prep_indices) not in used_prep_indices
                ):  # If input state not already used, add a PUB
                    used_prep_indices.append(tuple(prep_indices))
                    used_indices.append(idx)
                    idx += 1
                    # Prepare input state in Pauli6 basis (front composition)
                    prep_circuit = qc.compose(
                        Pauli6PreparationBasis().circuit(prep_indices),
                        front=True,
                        inplace=False,
                    )
                    # Repeat circuit for noise amplification
                    for _ in range(self.n_reps - 1):
                        prep_circuit.compose(qc, inplace=True)

                    # Transpile the circuit again to decompose input state preparation
                    prep_circuit = self.backend_info.custom_transpile(
                        prep_circuit,
                        initial_layout=self.layout,
                        scheduling=False,
                        optimization_level=0,
                    )
                    pubs.append(
                        (
                            prep_circuit,
                            parity * obs.apply_layout(prep_circuit.layout),
                            params,
                            1 / np.sqrt(dedicated_shots),
                        )
                    )
                    obs: SparsePauliOp
                    total_shots += (
                        dedicated_shots
                        * self.batch_size
                        * len(obs.group_commuting(qubit_wise=True))
                    )
                else:  # If input state already used, reuse PUB and just update observable and precision
                    pub_ref_index: int = used_indices[
                        used_prep_indices.index(tuple(prep_indices))
                    ]
                    prep_circuit, ref_obs, ref_params, ref_precision = pubs[
                        pub_ref_index
                    ]
                    ref_shots = int(math.ceil(1.0 / ref_precision**2))
                    new_precision = min(ref_precision, 1 / np.sqrt(dedicated_shots))
                    new_shots = int(math.ceil(1.0 / new_precision**2))
                    new_pub = (
                        prep_circuit,
                        ref_obs + parity * obs.apply_layout(prep_circuit.layout),
                        ref_params,
                        new_precision,
                    )
                    pubs[pub_ref_index] = new_pub
                    total_shots -= (
                        ref_shots
                        * self.batch_size
                        * len(ref_obs.group_commuting(qubit_wise=True))
                    )
                    total_shots += (
                        new_shots
                        * self.batch_size
                        * len(new_pub[1].group_commuting(qubit_wise=True))
                    )

        if len(pubs) == 0:  # If nothing was sampled, retry
            pubs, total_shots = self.channel_reward_pubs(qc, params)

        return pubs, total_shots

    def cafe_reward_pubs(self, circuit: QuantumCircuit, params):
        """
        Retrieve PUBs for Context-Aware Fidelity Estimation (CAFE) protocol

        :param circuit: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        """

        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, CAFEConfig
        ), "CAFEConfig object required for CAFE reward method"

        pubs = []
        total_shots = 0
        circuit_ref = self.baseline_circuits[self.trunc_index]
        layout = self.layout

        # samples, shots = np.unique(
        #     np.random.choice(len(input_circuits), self.sampling_Pauli_space),
        #     return_counts=True,
        # )
        # for sample, shot in zip(samples, shots):
        for sample in range(len(self.target.input_states)):
            run_qc = QuantumCircuit.copy_empty_like(
                circuit, name="cafe_circ"
            )  # Circuit with custom target gate
            ref_qc = QuantumCircuit.copy_empty_like(
                circuit_ref, name="cafe_ref_circ"
            )  # Circuit with reference gate

            for qc, context in zip([run_qc, ref_qc], [circuit, circuit_ref]):
                # Bind input states to the circuits
                qc.compose(self.target.input_states[sample].circuit, inplace=True)
                qc.barrier()
                for _ in range(
                    self.n_reps
                ):  # Add custom target gate to run circuit n_reps times
                    qc.compose(context, inplace=True)
                    qc.barrier()

            # Compute inverse unitary for reference circuit
            reverse_unitary = Operator(ref_qc).adjoint()
            reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
            reverse_unitary_qc.unitary(
                reverse_unitary,
                reverse_unitary_qc.qubits,
                label="U_inv",
            )
            reverse_unitary_qc.measure_all()

            reverse_unitary_qc = self.backend_info.custom_transpile(
                reverse_unitary_qc,
                initial_layout=layout,
                scheduling=False,
                optimization_level=3,  # Find smallest circuit implementing inverse unitary
                remove_final_measurements=False,
            )

            # Bind inverse unitary + measurement to run circuit
            for circ, pubs_ in zip([run_qc, ref_qc], [pubs, self._ideal_pubs]):
                transpiled_circuit = self.backend_info.custom_transpile(
                    circ, initial_layout=layout, scheduling=False
                )
                transpiled_circuit.barrier(self.involved_qubits)
                # Add the inverse unitary + measurement to the circuit
                transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                pubs_.append((transpiled_circuit, params, self.n_shots))
                # pubs_.append((transpiled_circuit, params, self.n_shots * shot))
            total_shots += self.batch_size * self.n_shots
            # total_shots += self.batch_size * self.n_shots * shot

        return pubs, total_shots

    def xeb_reward_pubs(self, circuit: QuantumCircuit, params):
        """
        Retrieve PUBs for XEB protocol

        :param circuit: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        """
        # TODO: Complete XEB (will be relevant once pattern for parallel XEB is figured out)
        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        assert isinstance(
            self.config.reward_config, XEBConfig
        ), "XEBConfig object required for XEB reward method"
        layout = self.layout
        circuit_ref = self.baseline_circuits[self.trunc_index]
        pubs = []
        total_shots = 0

        for seq in range(self.sampling_Pauli_space):
            ref_qc = QuantumCircuit.copy_empty_like(
                circuit_ref,
                name="xeb_ref_circ",
            )
            run_qc = QuantumCircuit.copy_empty_like(
                circuit,
                name="xeb_run_circ",
            )
            for l in range(self.n_reps):
                pass

        return pubs, total_shots

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
        layout = self.layout
        circuit_ref = self.baseline_circuits[self.trunc_index]
        pubs = []
        total_shots = 0
        if self.config.reward_config.use_interleaved:  # Interleaved RB
            try:
                Clifford(circuit_ref)
            except QiskitError as e:
                raise ValueError(
                    "Circuit should be a Clifford circuit for using interleaved RB directly"
                ) from e
            ref_element = circuit_ref.to_gate(label="ref_circ")
            custom_element = circuit.to_gate(label="custom_circ")
            exp = InterleavedRB(
                ref_element,
                self.involved_qubits,
                [self.n_reps],
                self.backend,
                self.sampling_Pauli_space,
                self.seed,
                circuit_order="RRRIII",
            )
            # ref_circuits = exp.circuits()[0: self.n_reps]
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
            self._ideal_pubs = [
                (qc, params, self.n_shots) for qc in interleaved_circuits
            ]
        else:
            for seq in range(self.sampling_Pauli_space):
                ref_qc = QuantumCircuit.copy_empty_like(
                    circuit_ref,
                    name="orbit_ref_circ",
                )
                run_qc = QuantumCircuit.copy_empty_like(circuit, name="orbit_run_circ")
                for l in range(self.n_reps):
                    r_cliff = random_clifford(circuit.num_qubits)
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

                for circ, pubs_ in zip([run_qc, ref_qc], [pubs, self._ideal_pubs]):
                    transpiled_circuit = self.backend_info.custom_transpile(
                        circ, initial_layout=layout, scheduling=False
                    )
                    transpiled_circuit.barrier(self.involved_qubits)
                    # Add the inverse unitary + measurement to the circuit
                    transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                    pubs_.append((transpiled_circuit, params, self.n_shots))

                total_shots += self.batch_size * self.n_shots

        return pubs, total_shots

    def simulate_circuit(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        This method should be called only when the abstraction level is "circuit"
        :param qc: QuantumCircuit to execute on quantum system
        :param params: List of Action vectors to execute on quantum system
        :return: Fidelity metric or array of fidelities for all actions in the batch
        """
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        if self.abstraction_level != "circuit":
            raise ValueError(
                "This method should only be called when the abstraction level is 'circuit'"
            )

        names = ["qc_channel", "qc_state", "qc_channel_nreps", "qc_state_nreps"]
        qc_channel, qc_state, qc_channel_nreps, qc_state_nreps = [
            qc.copy(name=name) for name in names
        ]
        returned_fidelity_type = (
            "gate"
            if isinstance(self.target, GateTarget) and qc.num_qubits <= 3
            else "state"
        )
        returned_fidelities = []

        for _ in range(self.n_reps - 1):
            qc_channel_nreps.compose(qc, inplace=True)
            qc_state_nreps.compose(qc, inplace=True)

        print("Starting simulation benchmark...")
        backend = AerSimulator()
        if self.backend is None or (
            isinstance(self.backend, AerSimulator) and not has_noise_model(self.backend)
        ):  # Ideal simulation

            noise_model = None
            qc_channel.save_unitary()
            qc_channel_nreps.save_unitary()
            qc_state.save_statevector()
            qc_state_nreps.save_statevector()
            channel_output = "unitary"
            state_output = "statevector"

        else:  # Noisy simulation
            if isinstance(self.backend, AerSimulator):
                noise_model = self.backend.options.noise_model
            else:
                noise_model = NoiseModel.from_backend(self.backend)
            qc_channel.save_superop()
            qc_channel_nreps.save_superop()
            qc_state.save_density_matrix()
            qc_state_nreps.save_density_matrix()
            channel_output = "superop"
            state_output = "density_matrix"

        basis_gates = backend.operation_names
        if noise_model is not None:
            basis_gates += noise_model.basis_gates
        qc_channel, qc_channel_nreps, qc_state, qc_state_nreps = transpile(
            [qc_channel, qc_channel_nreps, qc_state, qc_state_nreps],
            backend=backend,
            optimization_level=0,
            basis_gates=basis_gates,
        )
        if isinstance(self.parameters, ParameterVector):
            parameters = [self.parameters]
            n_custom_instructions = 1
        else:
            parameters = self.parameters
            n_custom_instructions = len(self.parameters)

        if (
            self.config.reward_method == "fidelity"
        ):  # Return batch of fidelities for each action
            parameter_binds = [
                {
                    parameters[i][j]: params[:, i * self.n_actions + j]
                    for i in range(n_custom_instructions)
                    for j in range(self.n_actions)
                }
            ]
            data_length = self.batch_size
        else:  # Return fidelity for the mean action (policy mean parameter)
            parameter_binds = [
                {
                    parameters[i][j]: [self.mean_action[i * self.n_actions + j]]
                    for i in range(n_custom_instructions)
                    for j in range(self.n_actions)
                }
            ]
            data_length = 1

        for circ, method, fid_array in zip(
            [qc_channel, qc_channel_nreps, qc_state, qc_state_nreps],
            [channel_output] * 2 + [state_output] * 2,
            [
                self.avg_fidelity_history,
                self.avg_fidelity_history_nreps,
                self.circuit_fidelity_history,
                self.circuit_fidelity_history_nreps,
            ],
        ):
            # Avoid channel simulation for more than 3 qubits
            if (method == "superop" or method == "unitary") and circ.num_qubits > 3:
                fidelities = [0.0] * data_length
                n_reps = 1
            else:
                result = backend.run(
                    circ,
                    parameter_binds=parameter_binds,
                    method=method,
                    noise_model=noise_model,
                ).result()
                outputs = [result.data(i)[method] for i in range(data_length)]
                n_reps = self.n_reps if "nreps" in circ.name else 1
                fidelities = [
                    self.target.fidelity(output, n_reps) for output in outputs
                ]
            if (
                (method == "superop" or method == "unitary")
                and returned_fidelity_type == "gate"
                and n_reps == 1
            ):
                returned_fidelities = fidelities
            elif (
                (method == "density_matrix" or method == "statevector")
                and returned_fidelity_type == "state"
                and n_reps == 1
            ):
                returned_fidelities = fidelities

            fid_array.append(np.mean(fidelities))
        print("Fidelity stored", np.mean(returned_fidelities))
        return returned_fidelities

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

    def simulate_pulse_circuit(self, qc: QuantumCircuit, params: np.array):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        This method should be called only when the abstraction level is "pulse"
        """
        if self.abstraction_level != "pulse":
            raise ValueError(
                "This method should only be called when the abstraction level is 'pulse'"
            )
        if not isinstance(self.backend, DynamicsBackend):
            raise ValueError(
                f"Pulse simulation requires a DynamicsBackend; got {self.backend}"
            )
        returned_fidelity_type = (
            "gate"
            if isinstance(self.target, GateTarget) and qc.num_qubits <= 3
            else "state"
        )  # Fidelity type to return (gate or state fidelity metric)
        returned_fidelities = []
        subsystem_dims = list(
            filter(lambda x: x > 1, self.backend.options.subsystem_dims)
        )  # Filter out qubits with dimension 1 (trivial dimension stated for DynamicsBackend)
        n_benchmarks = 1  # Number of benchmarks to run (1 if no n_reps, 2 if n_reps > 1, to benchmark both qc and qc_nreps)
        qc_nreps = None
        if self.n_reps > 1 and isinstance(
            self.target, GateTarget
        ):  # No need to benchmark n_reps for state targets
            qc_nreps = qc.copy("qc_nreps")
            for _ in range(self.n_reps - 1):
                qc_nreps.compose(qc, inplace=True)
            n_benchmarks *= 2

        y0_gate = Operator(
            np.eye(np.prod(subsystem_dims)),
            input_dims=tuple(subsystem_dims),
            output_dims=tuple(subsystem_dims),
        )
        y0_state = Statevector.from_int(0, dims=subsystem_dims)

        if isinstance(self.backend.options.solver, (JaxSolver, JaxSolverV2)):
            # TODO: Handle this case
            raise ValueError("Pulse simulation is not supported with JAX solvers")
        else:  # Standard Dynamics simulation
            if (
                self.config.reward_method == "fidelity"
            ):  # Benchmark all actions in the batch
                circuits = [qc.assign_parameters(p) for p in params]
                circuits_n_reps = (
                    [qc_nreps.assign_parameters(p) for p in params]
                    if qc_nreps is not None
                    else []
                )
                data_length = self.batch_size
            else:  # Benchmark only the mean action (policy mean parameter)
                circuits = [qc.assign_parameters(self.mean_action)]
                circuits_n_reps = (
                    [qc_nreps.assign_parameters(self.mean_action)]
                    if qc_nreps is not None
                    else []
                )
                data_length = 1

            y0_list = (
                [y0_state] * n_benchmarks * data_length
            )  # Initial state for each benchmark
            circuits_list = circuits + circuits_n_reps
            if qc.num_qubits < 3 and isinstance(
                self.target, GateTarget
            ):  # Benchmark channel only for 1-2 qubits
                y0_list += [y0_gate] * n_benchmarks * data_length
                circuits_list += circuits + circuits_n_reps
                n_benchmarks *= (
                    2  # Double the number of benchmarks to include channel fidelity
                )
            # Simulate all circuits
            results = self.backend.solve(circuits_list, y0=y0_list)

        output_data = [result.y[-1] for result in results]
        # Reshape data to isolate benchmarks (Output type can be either State or Channel, and for both qc and qc_nreps)
        output_data = [
            output_data[i * data_length : (i + 1) * data_length]
            for i in range(n_benchmarks)
        ]

        if self.n_reps > 1:  # Benchmark both qc and qc_nreps
            circ_list = [qc, qc_nreps, qc, qc_nreps]
            fid_arrays = [
                self.circuit_fidelity_history,
                self.avg_fidelity_history,
                self.circuit_fidelity_history_nreps,
                self.avg_fidelity_history_nreps,
            ]
        else:  # Benchmark only qc
            circ_list = [qc] * 2
            fid_arrays = [self.circuit_fidelity_history, self.avg_fidelity_history]

        for circ, data, fid_array in zip(circ_list, output_data, fid_arrays):
            n_reps = 1 if "nreps" not in circ.name else self.n_reps

            if isinstance(data[0], (Statevector, DensityMatrix)):
                data = [
                    projected_state(state, subsystem_dims) for state in data
                ]  # Project state to qubit subspace
                if self.target.n_qubits != len(
                    subsystem_dims
                ):  # If state has less qubits than the backend, trace out the rest
                    data = [
                        partial_trace(
                            state,
                            [
                                qubit
                                for qubit in range(state.num_qubits)
                                if qubit not in self.target.physical_qubits
                            ],
                        )
                        for state in data
                    ]
            elif isinstance(data[0], Operator):  # Project channel to qubit subspace
                data = [qubit_projection(op, subsystem_dims) for op in data]

            # Compute fidelities (type of input automatically detected and handled -> state -> state fidelity, channel -> gate fidelity)
            fidelities = [
                (
                    self.target.fidelity(output, n_reps)
                    if n_reps > 1
                    else self.target.fidelity(output)
                )
                for output in data
            ]

            if (
                returned_fidelity_type == "gate"
            ):  # Optimize gate fidelity by finding optimal Z-rotations before and after gate
                fidelities = self._handle_virtual_rotations(
                    data, fidelities, subsystem_dims, n_reps
                )
                if n_reps == 1:
                    returned_fidelities = fidelities
            elif returned_fidelity_type == "state" and n_reps == 1:
                returned_fidelities = fidelities
            fid_array.append(np.mean(fidelities))
        return returned_fidelities

    def _handle_virtual_rotations(self, operations, fidelities, subsystem_dims, n_reps):
        """
        Optimize gate fidelity by finding optimal Z-rotations before and after gate
        """
        best_op = operations[np.argmax(fidelities)]
        res = get_optimal_z_rotation(
            best_op, self.target.target_operator.power(n_reps), len(subsystem_dims)
        )
        rotated_unitaries = [rotate_unitary(res.x, op) for op in operations]
        fidelities = [self.target.fidelity(op, n_reps) for op in rotated_unitaries]

        return fidelities

    def update_gate_calibration(self):
        """
        Update gate calibration parameters
        """
        raise NotImplementedError(
            "Gate calibration not implemented for this environment"
        )

    def modify_environment_params(self, **kwargs):
        """
        Modify environment parameters (can be overridden by subclasses to modify specific parameters)
        """
        pass

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

    @sampler.setter
    def sampler(self, sampler: BaseSamplerV1 | BaseSamplerV2):
        self._sampler = sampler

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
            else self.circuit_fidelity_history
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
        if hasattr(self.estimator, "session"):
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
        self._total_shots.clear()
        self._hardware_runtime.clear()
        self.avg_fidelity_history.clear()
        self.process_fidelity_history.clear()
        self.circuit_fidelity_history.clear()
        self.density_matrix_history.clear()

    @property
    def benchmark_cycle(self) -> int:
        """
        Cycle at which fidelity benchmarking is performed
        :return:
        """
        return self.config.benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, step: int) -> None:
        """
        Set cycle at which fidelity benchmarking is performed
        :param step:
        :return:
        """
        assert step >= 0, "Cycle needs to be a positive integer"
        self.config.benchmark_cycle = step

    def do_benchmark(self) -> bool:
        """
        Check if benchmarking should be performed at current step
        :return:
        """
        if self.config.reward_method == "fidelity":
            return True
        elif self.benchmark_cycle == 0:
            return False
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
                    "input_state": (
                        self.target.input_states[self._index_input_state].circuit.name
                        if isinstance(self.target, GateTarget)
                        else None
                    ),
                }
            else:
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "optimal action": self.optimal_action,
                    "input_state": (
                        self.target.input_states[self._index_input_state].circuit.name
                        if isinstance(self.target, GateTarget)
                        else None
                    ),
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "input_state": (
                    self.target.input_states[self._index_input_state].circuit.name
                    if isinstance(self.target, GateTarget)
                    else None
                ),
                "truncation_index": self.trunc_index,
            }
        return info

    def _ident_str(self):
        """This is a one-line description of the environment with some key parameters."""
        if isinstance(self.target, GateTarget):
            ident_str = f"gate_calibration_{self.target.gate.name}-gate_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        elif isinstance(self.target, StateTarget):
            ident_str = f"state_preparation_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        else:
            raise ValueError("Target type not recognized")
        return ident_str

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
    def target(self) -> GateTarget | StateTarget:
        """
        Return the target object (GateTarget | StateTarget) of the environment
        """
        return self._target

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
    def layout(self, layout: Layout):
        if not isinstance(layout, Layout):
            raise ValueError("Layout should be a Qiskit Layout object")
        self.target.layout = layout

    @property
    @abstractmethod
    def parameters(self) -> List[ParameterVector] | ParameterVector:
        """
        Return the Qiskit ParameterVector defining the actions applied on the environment
        """
        raise NotImplementedError("Parameters not implemented")

    @property
    def involved_qubits(self):
        """
        Return the qubits involved in the calibration task
        """
        return list(self.layout.get_physical_bits().keys())

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
    def ident_str(self):
        return self._ident_str()

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
                    else self.circuit_fidelity_history
                ),
            }
        )

    def run_v1_primitive(self, qc: QuantumCircuit, params: np.array):
        """
        Run the primitive for the EstimatorV1 case (relevant only for DynamicsBackendEstimator call)
        """
        if self.config.reward_method == "channel":
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

        return reward_table
