"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 16/02/2024
"""

from __future__ import annotations
from abc import ABC

# For compatibility for options formatting between Estimators.
import json
import signal
from dataclasses import asdict, dataclass
from itertools import product, chain
from typing import Optional, List, Callable, Any, SupportsFloat

from gymnasium import Env
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box
from qiskit import schedule, transpile

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
from qiskit.quantum_info import partial_trace

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import average_gate_fidelity, state_fidelity
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.transpiler import Layout, InstructionProperties
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Qiskit dynamics for pulse simulation (& benchmarking)
from qiskit_dynamics import DynamicsBackend

# Qiskit Experiments for generating reliable baseline for complex gate calibrations / state preparations
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
    Pauli6PreparationBasis,
)
from qiskit_ibm_runtime import (
    EstimatorV1 as RuntimeEstimatorV1,
    EstimatorV2 as RuntimeEstimatorV2,
    QiskitRuntimeService,
)

from helper_functions import (
    retrieve_primitives,
    Backend_type,
    Estimator_type,
    handle_session,
    qubit_projection,
    retrieve_backend_info,
    simulate_pulse_schedule,
    rotate_unitary,
    get_optimal_z_rotation,
    fidelity_from_tomography,
)
from qconfig import QiskitConfig, QEnvConfig, QuaConfig


# QUA imports
# from qualang_tools.bakery.bakery import baking
# from qm.qua import *
# from qm.QuantumMachinesManager import QuantumMachinesManager


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
    d = 2 ** target.num_qubits
    basis = pauli_basis(num_qubits=target.num_qubits)
    if isinstance(target, DensityMatrix):
        chi = np.real(
            [target.expectation_value(basis[k]) for k in range(d ** 2)]
        ) / np.sqrt(d)
    else:
        dms = [DensityMatrix(pauli).evolve(target) for pauli in basis]
        chi = (
                np.real(
                    [
                        dms[k_].expectation_value(basis[k])
                        for k, k_ in product(range(d ** 2), repeat=2)
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

    def __init__(self, physical_qubits: int | List[int], target_type: str):
        self.physical_qubits = (
            list(range(physical_qubits))
            if isinstance(physical_qubits, int)
            else physical_qubits
        )
        self.target_type = target_type
        self._tgt_register = QuantumRegister(len(self.physical_qubits), "tgt")
        self._layout: List[Layout] = [
            Layout(
                {
                    self._tgt_register[i]: self.physical_qubits[i]
                    for i in range(len(self.physical_qubits))
                }
            )
        ]
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
            input_states: Optional[List[List[InputState]]] = None,
    ):
        """
        Initialize the gate target for the quantum environment
        :param gate: Gate to be calibrated
        :param physical_qubits: Physical qubits forming the target gate
        :param n_reps: Number of repetitions of the target gate
        :param input_states: List of input states to be used for calibration
        """
        self.gate = gate
        self.n_reps = n_reps
        super().__init__(
            gate.num_qubits if physical_qubits is None else physical_qubits, "gate"
        )
        if input_states is None:
            input_circuits = [
                PauliPreparationBasis().circuit(s)
                for s in product(range(4), repeat=self.n_qubits)
            ]
            input_states = [
                [
                    InputState(
                        input_circuit=circ,
                        target_op=gate,
                        tgt_register=self.tgt_register,
                        n_reps=n_reps,
                    )
                    for circ in input_circuits
                ]
            ]
        self.input_states = input_states
        self.Chi = _calculate_chi_target(Operator(gate))


class QiskitBackendInfo:
    """
    Class to store information on Qiskit backend
    """

    def __init__(self, backend: Backend_type, estimator: Estimator_type):
        (
            self.dt,
            self.coupling_map,
            self.basis_gates,
            self.instruction_durations,
        ) = retrieve_backend_info(backend, estimator)
        self.backend = backend

    def custom_transpile(self, qc: QuantumCircuit | List[QuantumCircuit]):
        """
        Custom transpile function to transpile the quantum circuit
        """
        return transpile(
            (
                qc.remove_final_measurements(inplace=False)
                if isinstance(qc, QuantumCircuit)
                else [circ.remove_final_measurements(inplace=False) for circ in qc]
            ),
            backend=self.backend,
            scheduling_method="asap",
            basis_gates=self.basis_gates,
            coupling_map=(self.coupling_map if self.coupling_map.size() != 0 else None),
            instruction_durations=self.instruction_durations,
            optimization_level=0,
            dt=self.dt,
        )


class QuantumEnvironment(Env):
    metadata = {"render_modes": ["human"]}
    check_on_exp = False  # Indicate if fidelity benchmarking should be estimated via experiment or via simulation
    channel_estimator = False  # Indicate if channel estimator should be used (otherwise the state estimator is used)
    fidelity_access = False  # Indicate if fidelity should be accessed as a reward

    def __init__(self, training_config: QEnvConfig):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param training_config: Training configuration, containing all hyperparameters for the environment
        """

        super().__init__()

        self._trunc_index = self._inside_trunc_tracker = (
            0  # Index for circuit truncation (always 0 in this env)
        )
        self.training_config = training_config
        self.action_space = training_config.action_space
        self.n_shots = training_config.n_shots
        self.n_reps = training_config.n_reps
        self.sampling_Pauli_space = (
            training_config.sampling_Paulis
        )  # Number of Pauli observables to sample
        self.c_factor = training_config.c_factor  # Reward scaling factor
        self.training_with_cal = training_config.training_with_cal
        self.batch_size = (
            training_config.batch_size
        )  # Batch size == number of circuits sent in one job
        self._parameters = [
            ParameterVector("a", training_config.action_space.shape[-1])
        ]
        self._tgt_instruction_counts = (
            1  # Number of instructions to calibrate (here, only one gate)
        )
        self._target_instruction_timings = [
            0
        ]  # Timings for each instruction (here, only one gate starting at t=0)
        self._reward_check_max = 1.1

        if "gate" in training_config.target:
            self.target: BaseTarget = GateTarget(
                n_reps=self.n_reps, **training_config.target
            )
        else:
            self.target: BaseTarget = StateTarget(**training_config.target)

        # Qiskit backend
        self.backend: Backend_type = training_config.backend_config.backend
        if self.backend is not None:
            if self.n_qubits > self.backend.num_qubits:
                raise ValueError(
                    f"Target contains more qubits ({self.n_qubits}) than backend ({self.backend.num_qubits})"
                )

        self.parametrized_circuit_func: Callable = (
            training_config.backend_config.parametrized_circuit
        )

        self._func_args = training_config.backend_config.parametrized_circuit_kwargs
        (
            self.circuit_truncations,
            self.baseline_truncations,
        ) = self._generate_circuits()
        self.abstraction_level = (
            "pulse" if self.circuit_truncations[0].calibrations else "circuit"
        )

        estimator_options = training_config.backend_config.estimator_options

        self._estimator, self.fidelity_checker = retrieve_primitives(
            self.backend,
            self.config.backend_config,
            estimator_options,
            self.circuit_truncations[0],
        )
        # Retrieve physical qubits forming the target register (and additional qubits for the circuit context)
        self._physical_target_qubits = self.target.physical_qubits
        self.backend_info = QiskitBackendInfo(self.backend, self._estimator)
        # Retrieve qubits forming the local circuit context (target qubits + nearest neighbor qubits on the chip)
        self._physical_neighbor_qubits = list(
            filter(
                lambda x: x not in self.physical_target_qubits,
                chain(
                    *[
                        list(self.backend_info.coupling_map.neighbors(target_qubit))
                        for target_qubit in self.physical_target_qubits
                    ]
                ),
            )
        )
        self._physical_next_neighbor_qubits = list(
            filter(
                lambda x: x
                          not in self.physical_target_qubits + self.physical_neighbor_qubits,
                chain(
                    *[
                        list(self.backend_info.coupling_map.neighbors(target_qubit))
                        for target_qubit in self.physical_target_qubits
                                            + self.physical_neighbor_qubits
                    ]
                ),
            )
        )

        self._param_values = np.zeros((self.batch_size, self.action_space.shape[-1]))
        self.observation_space = Box(
            low=np.array([0, 0] + [-5] * (2 ** self.n_qubits) ** 2),
            high=np.array([1, 1] + [5] * (2 ** self.n_qubits) ** 2),
            dtype=np.float32,
        )
        self.observation_space = Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32
        )
        # Data storage
        self._seed = self.training_config.seed
        self._session_counts = 0
        self._step_tracker = 0
        self._total_shots = []
        self._max_return = 0
        self._episode_ended = False
        self._episode_tracker = 0
        self._benchmark_cycle = training_config.benchmark_cycle
        self.action_history = []
        self.density_matrix_history = []
        self.reward_history = []
        self.qc_history = []
        self._observables, self._pauli_shots = None, None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        if isinstance(self.target, GateTarget):
            if self.channel_estimator:
                self._pubs = []
            else:
                self._index_input_state = np.random.randint(
                    len(self.target.input_states[0])
                )
                self._input_state = self.target.input_states[0][self._index_input_state]
            self.target_instruction = CircuitInstruction(
                self.target.gate, self.target.tgt_register
            )
            self.process_fidelity_history = []
            self.avg_fidelity_history = []
            self._optimal_action = np.zeros(self.action_space.shape[-1])

        else:
            self.state_fidelity_history = []

        # Check the training_config observation space matches that returned by _get_obs

        self.check_reward()

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
        if isinstance(self.estimator, RuntimeEstimatorV1):
            self.estimator.options.environment["job_tags"] = [
                f"rl_qoc_step{self._step_tracker}"
            ]
        elif isinstance(self.estimator, RuntimeEstimatorV2):
            self.estimator.options.update(job_tags=[f"rl_qoc_step{self._step_tracker}"])
        target_state = None
        if isinstance(self.target, GateTarget):
            if self.channel_estimator:
                pass
            else:
                input_states = self.target.input_states[self.trunc_index]
                self._index_input_state = np.random.randint(len(input_states))
                self._input_state = input_states[self._index_input_state]
                target_state = self._input_state.target_state  # (Gate |input>=|target>)

        else:  # State preparation task
            target_state = self.target

        if target_state is not None:
            self._observables, self._pauli_shots = self.retrieve_observables(
                target_state, self.circuit_truncations[self.trunc_index]
            )

        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"episode": self._episode_tracker, "step": self._step_tracker}

    def _get_obs(self):
        if isinstance(self.target, GateTarget) and not self.channel_estimator:

            return np.array(
                [
                    self._index_input_state
                    / len(self.target.input_states[self.trunc_index]),
                    self._target_instruction_timings[self._inside_trunc_tracker],
                ]
                + list(self._observable_to_observation())
            )
        else:
            return np.array(
                [0, self._target_instruction_timings[self._inside_trunc_tracker]]
            )

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._step_tracker += 1
        if self._episode_ended:
            print("Resetting environment")
            terminated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                terminated,
                False,
                self._get_info(),
            )

        terminated = self._episode_ended = True
        reward = self.perform_action(action)

        # Using Negative Log Error as the Reward
        optimal_error_precision = 1e-6
        max_fidelity = 1.0 - optimal_error_precision
        reward = np.clip(reward, a_min=0.0, a_max=max_fidelity)
        reward = -np.log(1.0 - reward)

        return self._get_obs(), reward, terminated, False, self._get_info()

    def check_reward(self):
        if self.training_with_cal:
            print("Checking reward to adjust C Factor...")
            example_obs, _ = self.reset()
            if example_obs.shape != self.observation_space.shape:
                raise ValueError(
                    f"Training Config observation space ({self.observation_space.shape}) does not "
                    f"match Environment observation shape ({example_obs.shape})"
                )
            sample_action = np.random.normal(
                loc=(self.action_space.low + self.action_space.high) / 2,
                scale=(self.action_space.high - self.action_space.low) / 2,
                size=(self.batch_size, self.action_space.shape[-1]),
            )

            batch_rewards = self.perform_action(sample_action)
            mean_reward = np.mean(batch_rewards)
            if not np.isclose(mean_reward, self.fidelity_history[-1], atol=1e-2):
                self.c_factor *= self.fidelity_history[-1] / mean_reward
                self.c_factor = np.round(self.c_factor, 1)
                print("C Factor adjusted to", self.c_factor)
            self.clear_history()
        else:
            pass

    def perform_action(self, actions: np.array):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :return: Reward table (reward for each run in the batch)
        """
        trunc_index = self._inside_trunc_tracker
        qc = self.circuit_truncations[trunc_index].copy()
        input_state_circ = None
        params, batch_size = np.array(actions), actions.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} != {self.batch_size} ")

        if (
                self.do_benchmark() or self.fidelity_access
        ):  # Benchmarking or fidelity access
            fids = self.store_benchmarks(params)

        if not self.fidelity_access:
            if isinstance(self.target, GateTarget) and not self.channel_estimator:
                # Pick random input state
                input_state_circ = self._input_state.circuit
                for _ in range(self.n_reps - 1):  # Repeat the gate n_reps times
                    qc.compose(self.circuit_truncations[trunc_index], inplace=True)
                qc.compose(
                    input_state_circ, inplace=True, front=True
                )  # Prepend input state preparation circuit

            counts = (
                self._session_counts
                if isinstance(self.estimator, (RuntimeEstimatorV1, RuntimeEstimatorV2))
                else trunc_index
            )
            self.estimator = handle_session(
                self.estimator, self.backend, counts, qc, input_state_circ
            )
            # Append input state prep circuit to the custom circuit with front composition
            full_circ = transpile(
                qc,
                self.backend,
                initial_layout=self.layout[trunc_index],
                optimization_level=1,
            )
            print("Sending Estimator job...")
            if isinstance(self.estimator, BaseEstimatorV1):
                if self.channel_estimator:
                    raise NotImplementedError(
                        "Channel estimator not implemented for EstimatorV1"
                    )
                print(self._observables)
                job = self.estimator.run(
                    circuits=[full_circ] * self.batch_size,
                    observables=[self._observables.apply_layout(full_circ.layout)]
                                * self.batch_size,
                    parameter_values=params,
                    shots=int(np.max(self._pauli_shots) * self.n_shots),
                )
                self._total_shots.append(
                    int(np.max(self._pauli_shots) * self.n_shots)
                    * self.batch_size
                    * len(self._observables.group_commuting(qubit_wise=True))
                )
                reward_table = job.result().values / self._observables.size
            else:  # EstimatorV2
                if self.channel_estimator:
                    self._pubs, total_shots = (
                        self.retrieve_observables_and_input_states(
                            self.circuit_truncations[self.trunc_index], params
                        )
                    )
                    self._total_shots.append(total_shots)
                else:
                    print(self._observables)
                    self._pubs = [
                        (
                            full_circ,
                            observable.apply_layout(full_circ.layout),
                            params,
                            1 / np.sqrt(self.n_shots * pauli_shots),
                        )
                        for observable, pauli_shots in zip(
                            self._observables.group_commuting(qubit_wise=True),
                            self._pauli_shots,
                        )
                    ]
                    self._total_shots.append(
                        self.batch_size * np.sum(self._pauli_shots * self.n_shots)
                    )
                job = self.estimator.run(
                    pubs=self._pubs,
                )
                reward_table = np.sum(
                    [pub_result.data.evs for pub_result in job.result()], axis=0
                ) / len(self._observables)
            print("Finished Estimator job")
        else:
            reward_table = fids

        if np.mean(reward_table) > self._max_return:
            self._max_return = np.mean(reward_table)
            self._optimal_action = np.mean(params, axis=0)
        self.reward_history.append(reward_table)
        assert (
                len(reward_table) == self.batch_size
        ), f"Reward table size mismatch {len(reward_table)} != {self.batch_size} "
        assert not np.any(np.isinf(reward_table)) and not np.any(
            np.isnan(reward_table)
        ), "Reward table contains NaN or Inf values"
        return reward_table  # Shape [batchsize]

    def store_benchmarks(self, params: np.array):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: List of Action vectors to execute on quantum system
        :return: None
        """
        qc = self.circuit_truncations[self.trunc_index].copy()
        n_actions = params.shape[-1]
        if isinstance(self.target, StateTarget):
            key, cls = "dm", DensityMatrix
            benchmark = state_fidelity
            method_str = "statevector" if self.backend is None else "density_matrix"
            method_qc = (
                qc.save_statevector if self.backend is None else qc.save_density_matrix
            )
        else:
            key, cls = "gate", Operator
            benchmark = average_gate_fidelity
            method_str = "unitary" if self.backend is None else "superop"
            method_qc = qc.save_unitary if self.backend is None else qc.save_superop

        target: DensityMatrix | Operator = cls(getattr(self.target, key))

        if self.check_on_exp:
            # Experiment based fidelity estimation
            try:
                qc_list = [qc.assign_parameters(angle_set) for angle_set in params]
                print("Starting tomography...")
                session = (
                    self.estimator.session
                    if hasattr(self.estimator, "session")
                    else None
                )
                fids = fidelity_from_tomography(
                    qc_list,
                    self.backend,
                    target,
                    self.target.physical_qubits,
                    session=session,
                )
                print("Finished tomography")

            except Exception as e:
                self.close()
                raise e
            if isinstance(self.target, StateTarget):
                self.state_fidelity_history.append(np.mean(fids))
            else:
                self.avg_fidelity_history.append(np.mean(fids))

        else:  # Simulation based fidelity estimation (Aer for circuit level, Dynamics for pulse)
            print("Starting simulation benchmark...")
            if self.abstraction_level == "circuit":
                if isinstance(self.backend, AerSimulator):
                    aer_backend = self.backend
                else:
                    noise_model = (
                        NoiseModel.from_backend(self.backend)
                        if self.backend is not None
                        else None
                    )
                    aer_backend = AerSimulator(
                        noise_model=noise_model, method=method_str
                    )

                method_qc()
                circ = transpile(qc, backend=aer_backend, optimization_level=0)
                job = aer_backend.run(
                    circ,
                    False,
                    [
                        {
                            self.parameters[self.trunc_index][j]: params[:, j]
                            for j in range(n_actions)
                        }
                    ],
                )
                result = job.result()
                output_states = [
                    result.data(i)[method_str] for i in range(self.batch_size)
                ]

                fids = [benchmark(state, target) for state in output_states]

                if isinstance(self.target, StateTarget):
                    self.state_fidelity_history.append(np.mean(fids))
                else:
                    self.avg_fidelity_history.append(np.mean(fids))

            else:  # Pulse simulation
                subsystem_dims = list(
                    filter(lambda x: x > 1, self.backend.options.subsystem_dims)
                )
                if isinstance(self.backend, DynamicsBackend):
                    if hasattr(self.backend.options.solver, "unitary_solve"):
                        # Jax compatible pulse simulation
                        unitaries = self.backend.options.solver.unitary_solve(params)[
                                    :, 1, :, :
                                    ]
                    else:
                        qc_list = [
                            qc.assign_parameters(angle_set) for angle_set in params
                        ]
                        scheds = schedule(qc_list, backend=self.backend)
                        dt = self.backend.dt
                        durations = [sched.duration for sched in scheds]
                        results = self.backend.solve(
                            scheds,
                            durations,
                            y0=np.eye(np.prod(subsystem_dims)),
                            convert_results=False,
                        )
                        unitaries = np.array([result.y[-1] for result in results])

                    qubitized_unitaries = [
                        qubit_projection(unitaries[i, :, :], subsystem_dims)
                        for i in range(self.batch_size)
                    ]

                    if self.target.target_type == "state":
                        states = [
                            Statevector.from_int(0, dims=subsystem_dims).evolve(unitary)
                            for unitary in qubitized_unitaries
                        ]
                        density_matrix = DensityMatrix(np.mean(states, axis=0))
                        if target.num_qubits != density_matrix.num_qubits:
                            states = [
                                partial_trace(
                                    state,
                                    [
                                        qubit
                                        for qubit in range(state.num_qubits)
                                        if qubit not in self.target.physical_qubits
                                    ],
                                )
                                for state in states
                            ]
                            density_matrix = partial_trace(
                                density_matrix,
                                [
                                    qubit
                                    for qubit in list(range(density_matrix.num_qubits))
                                    if qubit not in self.target.physical_qubits
                                ],
                            )

                        self.state_fidelity_history.append(
                            state_fidelity(target, density_matrix, validate=False)
                        )
                        fids = [
                            state_fidelity(target, state, validate=False)
                            for state in states
                        ]

                    else:  # Gate calibration task
                        if target.num_qubits < len(subsystem_dims):
                            qc = QuantumCircuit(len(subsystem_dims))
                            qc.append(self.target.gate, self.target.physical_qubits)
                            target = Operator(qc)

                        fids = [
                            average_gate_fidelity(unitary, target)
                            for unitary in qubitized_unitaries
                        ]
                        best_unitary = qubitized_unitaries[np.argmax(fids)]
                        res = get_optimal_z_rotation(
                            best_unitary, target, len(subsystem_dims)
                        )
                        rotated_unitaries = [
                            rotate_unitary(res.x, unitary)
                            for unitary in qubitized_unitaries
                        ]
                        fids = [
                            average_gate_fidelity(unitary, target)
                            for unitary in rotated_unitaries
                        ]
                        avg_fid_batch = np.mean(fids)

                        self.avg_fidelity_history.append(avg_fid_batch)
                else:
                    raise NotImplementedError(
                        "Pulse simulation not yet implemented for this backend"
                    )

            if self.target.target_type == "state":
                print("State fidelity:", self.state_fidelity_history[-1])
            else:
                print("Avg gate fidelity:", self.avg_fidelity_history[-1])
            print("Finished simulation benchmark")
            return fids

    def retrieve_observables(self, target_state: StateTarget, qc: QuantumCircuit):
        """
        Retrieve observables to sample for the DFE protocol for the given target state

        :param target_state: Target state to prepare
        :param qc: Quantum circuit to be executed on quantum system
        :return: Observables to sample, number of shots for each observable
        """
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        probabilities = target_state.Chi ** 2
        full_basis = pauli_basis(qc.num_qubits)
        if not np.isclose(np.sum(probabilities), 1, atol=1e-5):
            print("probabilities sum um to", np.sum(probabilities))
            print("probabilities renormalized")
            probabilities = probabilities / np.sum(probabilities)
        k_samples = np.random.choice(
            len(probabilities), size=self.sampling_Pauli_space, p=probabilities
        )

        pauli_indices, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = self.c_factor / (
                np.sqrt(target_state.dm.dim) * target_state.Chi[pauli_indices]
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

    def retrieve_observables_and_input_states(
            self, qc: QuantumCircuit, params: np.array
    ):
        """
        Retrieve observables and input state to sample for the DFE protocol for a target gate

        :param qc: Quantum circuit to be executed on quantum system
        :param params: Action vectors to execute on quantum system
        :return: Observables to sample, input state to prepare
        """
        assert isinstance(self.target, GateTarget), "Target type should be a gate"
        d = 2 ** qc.num_qubits
        probabilities = self.target.Chi ** 2 / (d ** 2)
        basis = pauli_basis(num_qubits=qc.num_qubits)
        samples, pauli_shots = np.unique(
            np.random.choice(
                len(probabilities), size=self.sampling_Pauli_space, p=probabilities
            ),
            return_counts=True,
        )

        pauli_indices = [np.unravel_index(sample, (d ** 2, d ** 2)) for sample in samples]
        pauli_prep, pauli_meas = zip(
            *[(basis[p[0]], basis[p[1]]) for p in pauli_indices]
        )
        reward_factor = [self.c_factor / (d * self.target.Chi[p]) for p in samples]
        self._observables = [
            SparsePauliOp(pauli_meas[i], reward_factor[i]) for i in range(len(samples))
        ]
        self._pauli_shots = pauli_shots

        pubs = []
        total_shots = 0
        for prep, obs, shot in zip(pauli_prep, self._observables, pauli_shots):
            max_input_states = 2 ** qc.num_qubits // 4
            selected_input_states = np.random.choice(
                2 ** qc.num_qubits, size=max_input_states, replace=False
            )
            for input_state in selected_input_states:
                prep_indices = []
                dedicated_shots = shot // (2 ** qc.num_qubits)
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
                    Pauli6PreparationBasis().circuit(prep_indices), front=True
                )

                pubs.append(
                    (
                        prep_circuit,
                        parity * obs,
                        params,
                        1 / np.sqrt(dedicated_shots * self.n_shots),
                    )
                )
                total_shots += dedicated_shots * self.n_shots * self.batch_size

        return pubs, total_shots

    def _observable_to_observation(self):
        """
        Convert the observable to an observation to be given to the agent
        """
        if not self.channel_estimator:
            n_qubits = self.observables.num_qubits
            d = 2 ** n_qubits
            pauli_to_index = {pauli: i for i, pauli in enumerate(pauli_basis(n_qubits))}
            array_obs = np.zeros(d ** 2)
            for pauli in self.observables:
                array_obs[pauli_to_index[pauli.paulis[0]]] = pauli.coeffs[0]

            array_obs = []
            return array_obs
        else:
            raise NotImplementedError("Channel estimator not yet implemented")

    def _generate_circuits(self):
        """
        Generate circuit to be executed on quantum system
        """
        custom_circuit = QuantumCircuit(self.target.tgt_register)
        ref_circuit = QuantumCircuit(self.target.tgt_register)

        self.parametrized_circuit_func(
            custom_circuit,
            self.parameters[self.trunc_index],
            self.target.tgt_register,
            **self._func_args,
        )
        if isinstance(self.target, GateTarget):
            ref_circuit.append(self.target.gate, self.target.tgt_register)
        elif isinstance(self.target, StateTarget):
            ref_circuit = self.target.dm
        return [custom_circuit], [ref_circuit]

    def clear_history(self):
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

    def close(self) -> None:
        if isinstance(
                self.estimator, (RuntimeEstimatorV1, RuntimeEstimatorV2)
        ) and isinstance(self.estimator.session.service, QiskitRuntimeService):
            self.estimator.session.close()

    def update_gate_calibration(self):
        """
        Update backend target with the optimal action found during training

        :return: Pulse calibration for the target gate
        """
        if not isinstance(self.target, GateTarget):
            raise ValueError("Target type should be a gate for gate calibration task.")

        if self.abstraction_level == "pulse":
            sched = schedule(
                self.circuit_truncations[0], self.backend
            ).assign_parameters(
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
            return self.circuit_truncations[0].assign_parameters(
                {self.parameters[0]: self._optimal_action}
            )

    def episode_length(self, global_step: int):
        """
        Return episode length (here defined as 1 as only one gate is calibrated per episode)
        """
        return 1

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
        if self.fidelity_access:
            return True
        elif self.benchmark_cycle == 0:
            return False
        else:
            return self._episode_tracker % self.benchmark_cycle == 0

    def signal_handler(self, signum, frame):
        """Signal handler for SIGTERM and SIGINT signals."""
        print(f"Received signal {signum}, closing environment...")
        if self.config_type == "Qiskit":
            self.close()
        else:
            pass

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
    def physical_target_qubits(self):
        return self.target.physical_qubits

    @property
    def physical_neighbor_qubits(self):
        return self._physical_neighbor_qubits

    @property
    def physical_next_neighbor_qubits(self):
        return self._physical_next_neighbor_qubits

    @property
    def parameters(self):
        """
        Return the Qiskit ParameterVector defining the actions applied on the environment
        """
        return self._parameters

    @property
    def trunc_index(self):
        """
        Return which truncation of the circuit is currently considered during training.
        Truncations are defined by the number of occurrences of the target gate in the circuit.
        """
        return self._trunc_index

    @property
    def observables(self) -> SparsePauliOp:
        return self._observables

    @property
    def total_shots(self):
        return self._total_shots

    @property
    def optimal_action(self):
        return self._optimal_action

    @property
    def config_type(self):
        return self.training_config.config_type

    @property
    def config(self):
        return self.training_config

    @property
    def estimator(self) -> BaseEstimatorV1 | BaseEstimatorV2:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimatorV1 | BaseEstimatorV2):
        self._estimator = estimator

    @property
    def sampler(self) -> BaseSamplerV1 | BaseSamplerV2:
        return self.fidelity_checker._sampler

    @property
    def tgt_instruction_counts(self):
        return self._tgt_instruction_counts

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

    def to_json(self):
        return json.dumps(
            {
                "n_qubits": self.n_qubits,
                "config": asdict(self.training_config),
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
