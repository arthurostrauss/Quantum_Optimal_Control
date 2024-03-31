"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 16/02/2024
"""

from __future__ import annotations

# For compatibility for options formatting between Estimators.
import json
import signal
from dataclasses import asdict
from itertools import product, chain
from typing import Dict, Optional, List, Callable, Any, SupportsFloat

from gymnasium import Env
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box
from qiskit import schedule, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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
    IBMBackend as RuntimeBackend,
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


def _calculate_chi_target(target: DensityMatrix | Operator):
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
                "Target should be a DensityMatrix or an Operator (Gate) object"
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


def _define_target(target: Dict):
    """
    Define target for the quantum environment
    This function is used to define the target for the quantum environment, and to check that the target is well defined
    It prepares the target for the environment, and returns the necessary information for the environment to be built
    :param target: Dictionary containing target information (gate or state)
    """
    tgt_register = target.get("register", None)
    q_register = None
    layout = None
    if tgt_register is not None:
        if isinstance(tgt_register, List):
            q_register = QuantumRegister(len(tgt_register), "tgt")
            layout = Layout(
                {q_register[i]: tgt_register[i] for i in range(len(tgt_register))}
            )
        elif isinstance(tgt_register, QuantumRegister):  # QuantumRegister or None
            q_register = tgt_register
        else:
            raise TypeError("Register should be of type List[int] or QuantumRegister")

    if "gate" not in target and "circuit" not in target and "dm" not in target:
        raise KeyError(
            "No target provided, need to have one of the following: 'gate' for gate calibration,"
            " 'circuit' or 'dm' for state preparation"
        )
    elif ("gate" in target and "circuit" in target) or (
        "gate" in target and "dm" in target
    ):
        raise KeyError("Cannot have simultaneously a gate target and a state target")
    if "circuit" in target or "dm" in target:  # State preparation task
        target["target_type"] = "state"
        if "circuit" in target:
            assert isinstance(target["circuit"], QuantumCircuit), (
                "Provided circuit is not a qiskit.QuantumCircuit " "object"
            )
            target["dm"] = DensityMatrix(target["circuit"])

        assert (
            "dm" in target
        ), "no DensityMatrix or circuit argument provided to target dictionary"
        assert isinstance(
            target["dm"], DensityMatrix
        ), "Provided dm is not a DensityMatrix object"
        dm: DensityMatrix = target["dm"]
        n_qubits = dm.num_qubits

        if q_register is None:
            q_register = QuantumRegister(n_qubits, "tgt")

        if layout is None:
            layout = Layout.generate_trivial_layout(q_register)
        target["Chi"] = _calculate_chi_target(dm)
        return (
            target,
            "state",
            q_register,
            n_qubits,
            [layout],
        )

    elif "gate" in target:  # Gate calibration task
        target["target_type"] = "gate"
        assert isinstance(
            target["gate"], Gate
        ), "Provided gate is not a qiskit.circuit.Gate operation"
        gate: Gate = target["gate"]
        n_qubits = gate.num_qubits
        if q_register is None:
            q_register = QuantumRegister(n_qubits)
        if layout is None:
            layout = Layout.generate_trivial_layout(q_register)

        assert gate.num_qubits == len(q_register), (
            f"Target gate number of qubits ({gate.num_qubits}) "
            f"incompatible with indicated 'register' ({len(q_register)})"
        )
        if "input_states" not in target:
            target["input_states"] = [
                [
                    {"circuit": PauliPreparationBasis().circuit(s)}
                    for s in product(range(4), repeat=len(tgt_register))
                ]
            ]

        for i, input_state in enumerate(target["input_states"][0]):
            if "circuit" not in input_state:
                raise KeyError("'circuit' key missing in input_state")
            assert isinstance(input_state["circuit"], QuantumCircuit), (
                "Provided circuit is not a" "qiskit.QuantumCircuit object"
            )

            input_circuit: QuantumCircuit = input_state["circuit"]
            input_state["dm"] = DensityMatrix(input_circuit)

            state_target_circuit = QuantumCircuit(q_register)
            state_target_circuit.compose(input_circuit, inplace=True)
            state_target_circuit.append(CircuitInstruction(gate, q_register))

            input_state["target_state"] = {
                "dm": DensityMatrix(state_target_circuit),
                "circuit": state_target_circuit,
                "target_type": "state",
            }
            input_state["target_state"]["Chi"] = _calculate_chi_target(
                input_state["target_state"]["dm"]
            )
        return target, "gate", q_register, n_qubits, [layout]
    else:
        raise KeyError("target type not identified, must be either gate or state")


def retrieve_abstraction_level(qc: QuantumCircuit):
    """
    Retrieve the abstraction level of the quantum circuit
    """
    if qc.calibrations:
        return "pulse"
    else:
        return "circuit"


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
        self.sampling_Pauli_space = (
            training_config.sampling_Paulis
        )  # Number of Pauli observables to sample
        self.c_factor = training_config.c_factor  # Reward scaling factor
        self.training_with_cal = training_config.training_with_cal
        self.batch_size = (
            training_config.batch_size
        )  # Batch size == number of circuits sent in one job
        self._parameters = ParameterVector("a", training_config.action_space.shape[-1])
        self._tgt_instruction_counts = (
            1  # Number of instructions to calibrate (here, only one gate)
        )
        self._target_instruction_timings = [
            0
        ]  # Timings for each instruction (here, only one gate starting at t=0)
        self._reward_check_max = 1.1

        if isinstance(self.training_config.backend_config, QiskitConfig):
            # Qiskit backend
            self._config_type = "qiskit"

            (
                self.target,
                self.target_type,
                self.tgt_register,
                self._n_qubits,
                self._layout,
            ) = _define_target(training_config.target)

            self.backend: Backend_type = training_config.backend_config.backend
            if self.backend is not None:
                if self.n_qubits > self.backend.num_qubits:
                    raise ValueError(
                        f"Target contains more qubits ({self._n_qubits}) than backend ({self.backend.num_qubits})"
                    )

            self.parametrized_circuit_func: Callable = (
                training_config.backend_config.parametrized_circuit
            )

            self._func_args = training_config.backend_config.parametrized_circuit_kwargs
            (
                self.circuit_truncations,
                self.baseline_truncations,
            ) = self._generate_circuits()
            self.abstraction_level = retrieve_abstraction_level(
                self.circuit_truncations[0]
            )

            estimator_options = training_config.backend_config.estimator_options

            self._estimator, self.fidelity_checker = retrieve_primitives(
                self.backend,
                self.layout,
                self.config.backend_config,
                self.abstraction_level,
                estimator_options,
                self.circuit_truncations[0],
            )
            # Retrieve physical qubits forming the target register (and additional qubits for the circuit context)
            self._physical_target_qubits = self.config.target["register"]
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
        if isinstance(self.training_config.backend_config, QuaConfig):
            # QUA backend

            self._config_type = "qua"
            self._channel_mapping = training_config.backend_config.channel_mapping
            raise AttributeError("QUA compatibility not yet implemented")

            # TODO: Add a QUA program

        self._param_values = np.zeros((self.batch_size, self.action_space.shape[-1]))
        self.observation_space = Box(
            low=np.array([0, 0] + [-5] * (2**self.n_qubits) ** 2),
            high=np.array([1, 1] + [5] * (2**self.n_qubits) ** 2),
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
        self._benchmark_cycle = self.training_config.benchmark_cycle
        self.action_history = []
        self.density_matrix_history = []
        self.reward_history = []
        self.qc_history = []
        self._observables, self._pauli_shots = None, None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        if self.target_type == "gate":
            if self.channel_estimator:
                self._chi_gate = [_calculate_chi_target(Operator(self.target["gate"]))]
                self._pubs = self.retrieve_observables_and_input_states(
                    self.circuit_truncations[self._trunc_index]
                )
            else:
                self._index_input_state = np.random.randint(
                    len(self.target["input_states"][0])
                )
            self.target_instruction = CircuitInstruction(
                self.target["gate"], self.tgt_register
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

        if self.target_type == "gate":
            if self.channel_estimator:
                self._pubs = self.retrieve_observables_and_input_states(
                    self.circuit_truncations[self._trunc_index]
                )
                target_state = None
            else:
                self._index_input_state = np.random.randint(
                    len(self.target["input_states"][self._trunc_index])
                )
                input_state = self.target["input_states"][self._trunc_index][
                    self._index_input_state
                ]
                target_state = input_state["target_state"]  # (Gate |input>=|target>)

        else:  # State preparation task
            target_state = self.target

        if target_state is not None:
            self._observables, self._pauli_shots = self.retrieve_observables(
                target_state, self.circuit_truncations[self._trunc_index]
            )

        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"episode": self._episode_tracker, "step": self._step_tracker}

    def _get_obs(self):
        if self.target_type == "gate":
            return np.array(
                [
                    self._index_input_state
                    / len(self.target["input_states"][self._trunc_index]),
                    self._target_instruction_timings[self._inside_trunc_tracker],
                ]
                + list(self._observable_to_observation())
            )
        else:
            return np.array([0, 0])

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
            self.store_benchmarks(sample_action)
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
        qc = self.circuit_truncations[trunc_index]
        input_state_circ = QuantumCircuit(self.tgt_register)

        params, batch_size = np.array(actions), actions.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} != {self.batch_size} ")
        if self.target_type == "gate":
            # Pick random input state from the list of possible input states (forming a tomographically complete set)
            input_state_circ = self.target["input_states"][trunc_index][
                self._index_input_state
            ]["circuit"]
        try:
            if self.do_benchmark():
                print("Starting benchmarking...")
                self.store_benchmarks(params)
                print("Finished benchmarking")
            print("Sending Estimator job...")

            counts = (
                self._session_counts
                if isinstance(self.estimator, (RuntimeEstimatorV1, RuntimeEstimatorV2))
                else trunc_index
            )
            self.estimator = handle_session(
                self.estimator, self.backend, counts, qc, input_state_circ
            )
            # Append input state prep circuit to the custom circuit with front composition
            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=self.backend,
                initial_layout=self.layout[trunc_index],
            )
            full_circ = qc.compose(input_state_circ, inplace=False, front=True)
            full_circ = transpile(
                full_circ,
                self.backend,
                initial_layout=self.layout[trunc_index],
                optimization_level=1,
            )

            if isinstance(self.estimator, BaseEstimatorV1):
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
                    for i in range(len(self._pubs)):
                        self._pubs[i][2] = params
                        self._pubs[i] = tuple(self._pubs[i])
                        self._total_shots.append(
                            self.batch_size
                            * np.sum([pub[3] ** (-2) for pub in self._pubs])
                            * len(self._pubs)
                        )
                else:
                    print(self._observables)
                    self._pubs = [
                        (
                            full_circ,
                            self._observables.apply_layout(full_circ.layout),
                            params,
                            1 / np.sqrt(self.n_shots * int(np.max(self._pauli_shots))),
                        )
                    ]
                    self._total_shots.append(
                        self.batch_size
                        * len(self._observables.group_commuting(qubit_wise=True))
                        * int(np.max(self._pauli_shots))
                        * self.n_shots
                    )
                job = self.estimator.run(
                    pubs=self._pubs,
                )
                reward_table = (
                    np.sum([pub_result.data.evs for pub_result in job.result()], axis=0)
                    / self._observables.size
                )
            print("Finished Estimator job")
        except Exception as e:
            self.close()
            raise e

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
        qc = self.circuit_truncations[0]
        n_actions = self.action_space.shape[-1]
        key, cls = (
            ("dm", DensityMatrix) if self.target_type == "state" else ("gate", Operator)
        )
        target: DensityMatrix | Operator = cls(self.target[key])
        qc_list = [qc.assign_parameters(angle_set) for angle_set in params]

        if self.check_on_exp:
            # Experiment based fidelity estimation
            try:
                print("Starting tomography...")
                session = (
                    self.estimator.session
                    if hasattr(self.estimator, "session")
                    else None
                )
                fid = fidelity_from_tomography(
                    qc_list,
                    self.backend,
                    target,
                    self.target["register"],
                    session=session,
                )
                print("Finished tomography")

            except Exception as e:
                self.close()
                raise e
            if self.target_type == "state":
                self.state_fidelity_history.append(fid)
            else:
                self.avg_fidelity_history.append(fid)
        else:
            # Circuit list for each action of the batch
            print("Starting simulation benchmark...")
            if self.abstraction_level == "circuit":
                if self.target_type == "state":
                    if self.backend is None or (
                        isinstance(self.backend, RuntimeBackend)
                    ):
                        q_state_list = [Statevector(circ) for circ in qc_list]
                        density_matrix = DensityMatrix(
                            np.mean(
                                [
                                    q_state.to_operator().to_matrix()
                                    for q_state in q_state_list
                                ],
                                axis=0,
                            )
                        )
                    else:
                        if isinstance(self.backend, AerSimulator):
                            aer_backend = self.backend
                        else:
                            aer_backend = AerSimulator.from_backend(
                                self.backend, method="density_matrix"
                            )
                        circ = transpile(qc, backend=aer_backend, optimization_level=0)
                        circ.save_density_matrix()
                        states_results = aer_backend.run(
                            circ,
                            parameter_binds=[
                                {
                                    self.parameters[j]: params[:, j]
                                    for j in range(n_actions)
                                }
                            ],
                        ).result()
                        output_states = [
                            states_results.data(i)["density_matrix"]
                            for i in range(self.batch_size)
                        ]
                        density_matrix = DensityMatrix(
                            np.mean(
                                [state.data for state in output_states],
                                axis=0,
                            )
                        )

                    self.state_fidelity_history.append(
                        state_fidelity(target, density_matrix)
                    )
                else:  # Gate calibration task
                    if self.backend is None or (
                        isinstance(self.backend, RuntimeBackend)
                    ):
                        q_process_list = [Operator(circ) for circ in qc_list]
                    else:
                        if isinstance(self.backend, AerSimulator):
                            aer_backend = self.backend
                        else:
                            aer_backend = AerSimulator.from_backend(
                                self.backend, method="superop"
                            )
                        circ = transpile(qc, backend=aer_backend, optimization_level=0)
                        circ.save_superop()
                        process_results = aer_backend.run(
                            circ,
                            parameter_binds=[
                                {
                                    self.parameters[j]: params[:, j]
                                    for j in range(n_actions)
                                }
                            ],
                        ).result()
                        q_process_list = [
                            process_results.data(i)["superop"]
                            for i in range(self.batch_size)
                        ]

                    avg_fidelity = np.mean(
                        [
                            average_gate_fidelity(q_process, target)
                            for q_process in q_process_list
                        ]
                    )
                    self.avg_fidelity_history.append(
                        avg_fidelity
                    )  # Avg gate fidelity over the action batch

            elif self.abstraction_level == "pulse":
                # Pulse simulation
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
                        scheds = schedule(qc_list, backend=self.backend)
                        dt = self.backend.dt
                        max_duration = max([sched.duration for sched in scheds])
                        results = self.backend.solve(
                            scheds,
                            [0, max_duration * dt],
                            y0=np.eye(np.prod(subsystem_dims)),
                            convert_results=False,
                        )
                        unitaries = np.array([result.y[-1] for result in results])

                    qubitized_unitaries = [
                        qubit_projection(unitaries[i, :, :], subsystem_dims)
                        for i in range(self.batch_size)
                    ]

                    if self.target_type == "state":
                        density_matrix = DensityMatrix(
                            np.mean(
                                [
                                    Statevector.from_int(0, dims=target.dims()).evolve(
                                        unitary
                                    )
                                    for unitary in qubitized_unitaries
                                ],
                                axis=0,
                            )
                        )
                        if target.num_qubits != density_matrix.num_qubits:
                            density_matrix = partial_trace(
                                density_matrix,
                                [
                                    qubit
                                    for qubit in list(range(density_matrix.num_qubits))
                                    if qubit not in self.target["register"]
                                ],
                            )

                        self.state_fidelity_history.append(
                            state_fidelity(target, density_matrix, validate=False)
                        )

                    else:  # Gate calibration task
                        if target.num_qubits < len(subsystem_dims):
                            qc = QuantumCircuit(len(subsystem_dims))
                            qc.append(self.target["gate"], self.target["register"])
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
                        avg_fid_batch = np.mean(
                            [
                                average_gate_fidelity(unitary, target)
                                for unitary in rotated_unitaries
                            ]
                        )

                        self.avg_fidelity_history.append(avg_fid_batch)
                        unitaries = rotated_unitaries

                else:
                    raise NotImplementedError(
                        "Pulse simulation not yet implemented for this backend"
                    )
            if self.target_type == "state":
                print("State fidelity:", self.state_fidelity_history[-1])
            else:
                print("Avg gate fidelity:", self.avg_fidelity_history[-1])
            print("Finished simulation benchmark")

    def retrieve_observables(self, target_state: Dict, qc: QuantumCircuit):
        """
        Retrieve observables to sample for the DFE protocol for the given target state

        :param target_state: Target state to prepare
        :param qc: Quantum circuit to be executed on quantum system
        :return: Observables to sample, number of shots for each observable
        """
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        probabilities = target_state["Chi"] ** 2
        if np.sum(probabilities) != 1:
            probabilities = probabilities / np.sum(probabilities)
        k_samples = np.random.choice(
            len(probabilities), size=self.sampling_Pauli_space, p=probabilities
        )

        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = self.c_factor / (
            np.sqrt(target_state["dm"].dim) * target_state["Chi"][pauli_index]
        )

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp(
            pauli_basis(qc.num_qubits)[pauli_index], reward_factor, copy=False
        )
        return observables, pauli_shots

    def retrieve_observables_and_input_states(self, qc: QuantumCircuit):
        """
        Retrieve observables and input state to sample for the DFE protocol for a target gate

        :param qc: Quantum circuit to be executed on quantum system
        :return: Observables to sample, input state to prepare
        """
        assert self.target_type == "gate", "Target type should be a gate"
        d = 2**qc.num_qubits
        probabilities = self._chi_gate[self._trunc_index] ** 2 / (d**2)
        basis = pauli_basis(num_qubits=qc.num_qubits)
        samples, pauli_shots = np.unique(
            np.random.choice(
                len(probabilities), size=self.sampling_Pauli_space, p=probabilities
            ),
            return_counts=True,
        )

        pauli_indices = [np.unravel_index(sample, (d**2, d**2)) for sample in samples]
        pauli_prep, pauli_meas = zip(
            *[(basis[p[0]], basis[p[1]]) for p in pauli_indices]
        )
        prep_circuits, parity = [], 1
        # for prep in pauli_prep:
        #     prep_indices = []
        #     for pauli_op in reversed(prep.to_label()):
        #         input = np.random.randint(2)
        #         parity *= (-1) ** input
        #         if pauli_op == "I" or pauli_op == "Z":
        #             prep_indices.append(input)
        #         elif pauli_op == "X":
        #             prep_indices.append(2 + input)
        #         else:
        #             prep_indices.append(4 + input)
        #     prep_circuits.append(
        #         qc.compose(Pauli6PreparationBasis().circuit(prep_indices), front=True)
        #     )

        reward_factor = [
            self.c_factor / (d * self._chi_gate[self._trunc_index][p]) for p in samples
        ]
        observables = [
            SparsePauliOp(pauli_meas[i], reward_factor[i]) for i in range(len(samples))
        ]
        # pubs = [
        #     [prep_circ, observable, None, 1 / np.sqrt(pauli_shot)]
        #     for prep_circ, observable, pauli_shot in zip(
        #         prep_circuits, observables, pauli_shots
        #     )
        # ]
        pubs = []
        for prep, obs, shot in zip(pauli_prep, observables, pauli_shots):
            for input_state in range(2**qc.num_qubits):
                prep_indices = []
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

                pubs.append([prep_circuit, parity * obs, None, 1 / np.sqrt(shot)])

        return pubs

    def _observable_to_observation(self):
        """
        Convert the observable to an observation to be given to the agent
        """
        if not self.channel_estimator:
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

    def _generate_circuits(self):
        """
        Generate circuit to be executed on quantum system
        """
        custom_circuit = QuantumCircuit(self.tgt_register)
        ref_circuit = QuantumCircuit(self.tgt_register)
        self.parametrized_circuit_func(
            custom_circuit, self._parameters, self.tgt_register, **self._func_args
        )
        if self.target_type == "gate":
            ref_circuit.append(self.target["gate"], self.tgt_register)
        else:
            ref_circuit = self.target["dm"]
        return [custom_circuit], [ref_circuit]

    def clear_history(self):
        self._step_tracker = 0
        self.qc_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        if self.target_type == "gate":
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
        if not self.target_type == "gate":
            raise ValueError("Target type should be a gate for gate calibration task.")
        if self.abstraction_level == "pulse":
            sched = schedule(
                self.circuit_truncations[0], self.backend
            ).assign_parameters(
                {
                    param: action
                    for param, action in zip(self._parameters, self._optimal_action)
                }
            )
            duration = sched.duration
            if isinstance(self.backend, DynamicsBackend):
                error = 1.0
                error -= simulate_pulse_schedule(
                    self.backend,
                    sched,
                    target_unitary=Operator(self.target["gate"]),
                    target_state=Statevector.from_int(0, dims=[2] * self.n_qubits),
                )["gate_fidelity"]["optimal"]

            else:
                error = 1.0 - np.max(self.avg_fidelity_history)
            instruction_prop = InstructionProperties(duration, error, sched)
            self.backend.target.update_instruction_properties(
                self.target["gate"].name,
                tuple(self.physical_target_qubits),
                instruction_prop,
            )

            return self.backend.target.get_calibration(
                self.target["gate"].name, tuple(self.physical_target_qubits)
            )
        else:
            return self.circuit_truncations[0].assign_parameters(
                {self.parameters: self._optimal_action}
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
        else:
            return self._episode_tracker % self.benchmark_cycle == 0

    def signal_handler(self, signum, frame):
        """Signal handler for SIGTERM and SIGINT signals."""
        print(f"Received signal {signum}, closing environment...")
        if self._config_type == "qiskit":
            self.close()
        else:
            pass

    def __repr__(self):
        string = f"QuantumEnvironment composed of {self._n_qubits} qubits, \n"
        string += (
            f"Defined target: {self.target_type} "
            f"({self.target.get('gate', None) if not None else self.target['dm']})\n"
        )
        string += f"Physical qubits: {self.target['register']}\n"
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
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        assert (
            isinstance(n_qubits, int) and n_qubits > 0
        ), "n_qubits must be a positive integer"
        self._n_qubits = n_qubits

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout: Layout):
        self._layout = layout

    @property
    def physical_target_qubits(self):
        return self._physical_target_qubits

    @property
    def physical_neighbor_qubits(self):
        return self._physical_neighbor_qubits

    @property
    def parameters(self):
        return self._parameters

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
        return self._config_type

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
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSamplerV1 | BaseSamplerV2):
        self._sampler = sampler

    @property
    def tgt_instruction_counts(self):
        return self._tgt_instruction_counts

    @property
    def fidelity_history(self):
        return (
            self.avg_fidelity_history
            if self.target_type == "gate"
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
                "target_type": self.target_type,
                "target": self.target,
                "c_factor": self.c_factor,
                "reward_history": self.reward_history,
                "action_history": self.action_history,
                "fidelity_history": (
                    self.avg_fidelity_history
                    if self.target_type == "gate"
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
