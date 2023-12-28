"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
"""
from __future__ import annotations

# For compatibility for options formatting between Estimators.
import json
from dataclasses import asdict
from itertools import product
from typing import Dict, Optional, List, Callable, Any, SupportsFloat

import pandas as pd
from gymnasium import Env
import numpy as np
from gymnasium.core import ObsType, ActType
from qiskit import transpile

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    Gate,
    CircuitInstruction,
    ParameterVector,
)

# Qiskit Estimator Primitives: for computing Pauli expectation value sampling easily
from qiskit.primitives import BaseEstimator

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import (
    average_gate_fidelity,
    state_fidelity,
    process_fidelity,
)
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.transpiler import Layout
from qiskit.algorithms.state_fidelities import ComputeUncompute

# Qiskit dynamics for pulse simulation (& benchmarking)
from qiskit_dynamics import DynamicsBackend
from qiskit_ibm_provider import IBMBackend

# Qiskit Experiments for generating reliable baseline for complex gate calibrations / state preparations
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
)  # , Pauli6PreparationBasis
from qiskit_ibm_runtime import Estimator as RuntimeEstimator

# Tensorflow modules
from tensorflow_probability.python.distributions import Categorical

from custom_jax_sim import JaxSolver
from helper_functions import (
    retrieve_primitives,
    Estimator_type,
    Sampler_type,
    handle_session,
    state_fidelity_from_state_tomography,
    gate_fidelity_from_process_tomography,
    qubit_projection,
)
from qconfig import QiskitConfig, QEnvConfig, QuaConfig


# QUA imports
# from qualang_tools.bakery.bakery import baking
# from qm.qua import *
# from qm.QuantumMachinesManager import QuantumMachinesManager


def _calculate_chi_target_state(target_state: Dict, n_qubits: int):
    """
    Calculate for all P
    :param target_state: Dictionary containing info on target state (name, density matrix)
    :param n_qubits: Number of qubits
    :return: Target state supplemented with appropriate "Chi" key
    """
    assert "dm" in target_state, "No input data for target state, provide DensityMatrix"
    d = 2 ** n_qubits
    Pauli_basis = pauli_basis(num_qubits=n_qubits)
    target_state["Chi"] = np.array(
        [
            np.trace(
                np.array(target_state["dm"].to_operator()) @ Pauli_basis[k].to_matrix()
            ).real
            for k in range(d ** 2)
        ]
    )
    # Real part is taken to convert it in good format,
    # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
    return target_state


def _define_target(target: Dict):
    tgt_register = target.get("register", None)
    q_register = None
    layout = None
    if tgt_register is not None:
        if isinstance(tgt_register, List):
            q_register = QuantumRegister(len(tgt_register))
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
            q_register = QuantumRegister(n_qubits)

        if layout is None:
            layout = Layout.generate_trivial_layout(q_register)

        return (
            _calculate_chi_target_state(target, n_qubits),
            "state",
            q_register,
            n_qubits,
            layout,
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
                {"circuit": PauliPreparationBasis().circuit(s).decompose()}
                for s in product(range(4), repeat=len(tgt_register))
            ]

            # target['input_states'] = [{"dm": Pauli6PreparationBasis().matrix(s),
            #                            "circuit": CircuitOp(Pauli6PreparationBasis().circuit(s).decompose())}
            #                           for s in product(range(6), repeat=len(tgt_register))]

        for i, input_state in enumerate(target["input_states"]):
            if "circuit" not in input_state:
                raise KeyError("'circuit' key missing in input_state")
            assert isinstance(input_state["circuit"], QuantumCircuit), (
                "Provided circuit is not a" "qiskit.QuantumCircuit object"
            )

            input_circuit: QuantumCircuit = input_state["circuit"]
            input_state["dm"] = DensityMatrix(input_circuit)

            state_target_circuit = QuantumCircuit(q_register)
            state_target_circuit.append(input_circuit.to_instruction(), q_register)
            state_target_circuit.append(CircuitInstruction(gate, q_register))

            input_state["target_state"] = {
                "dm": DensityMatrix(state_target_circuit),
                "circuit": state_target_circuit,
                "target_type": "state",
            }
            input_state["target_state"] = _calculate_chi_target_state(
                input_state["target_state"], n_qubits
            )
        return target, "gate", q_register, n_qubits, layout
    else:
        raise KeyError("target type not identified, must be either gate or state")


def retrieve_abstraction_level(qc):
    if qc.calibrations:
        return "pulse"
    else:
        return "circuit"


class QuantumEnvironment(Env):
    metadata = {"render_modes": ["human"]}
    check_on_exp = True  # Indicate if fidelity benchmarking should be estimated via experiment or via simulation

    def __init__(self, training_config: QEnvConfig):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param training_config: Training configuration, containing all hyperparameters for the environment
        """

        super().__init__()
        self.training_config = training_config
        self.action_space = training_config.action_space
        self.observation_space = training_config.observation_space
        self.n_shots = training_config.n_shots
        self.sampling_Pauli_space = training_config.sampling_Paulis
        self.c_factor = training_config.c_factor
        self.training_with_cal = training_config.training_with_cal
        self.batch_size = training_config.batch_size
        self._parameters = ParameterVector("a", training_config.action_space.shape[-1])
        self._tgt_instruction_counts = 1  # Number of instructions to calibrate
        self._reward_check_max = 1.1
        if isinstance(self.training_config.backend_config, QiskitConfig):
            self._config_type = "qiskit"
            if not isinstance(self.training_config.backend_config, QiskitConfig):
                raise ValueError("Config should be of type QiskitConfig")

            (
                self.target,
                self.target_type,
                self.tgt_register,
                self._n_qubits,
                self._layout,
            ) = _define_target(training_config.target)

            self._d = 2 ** self.n_qubits
            self.backend = training_config.backend_config.backend

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
            ) = self._generate_circuit_truncations()
            self.abstraction_level = retrieve_abstraction_level(
                self.circuit_truncations[0]
            )

            estimator_options = training_config.backend_config.estimator_options

            self._estimator, self._sampler = retrieve_primitives(
                self.backend,
                self.layout,
                self.config.backend_config,
                self.abstraction_level,
                estimator_options,
            )
            self._physical_target_qubits = list(self.layout.get_physical_bits().keys())
            self.fidelity_checker = ComputeUncompute(self.sampler)
        elif isinstance(self.training_config.backend_config, QuaConfig):
            raise AttributeError("QUA compatibility not yet implemented")

            # TODO: Add a QUA program

        self._param_values = np.zeros((self.batch_size, self.action_space.shape[-1]))
        # Data storage for plotting
        self._seed = self.training_config.seed

        self._session_counts = 0
        self._step_tracker = 0
        self._episode_ended = False
        self._episode_tracker = 0
        self._benchmark_cycle = self.training_config.benchmark_cycle
        self.action_history = []
        self.density_matrix_history = []
        self.reward_history = []
        self.qc_history = []
        if self.target_type == "gate":
            self._index_input_state = np.random.randint(
                len(self.target["input_states"])
            )
            self.target_instruction = CircuitInstruction(
                self.target["gate"], self.tgt_register
            )
            self.process_fidelity_history = []
            self.avg_fidelity_history = []
            self.built_unitaries = []
        else:
            self.state_fidelity_history = []

        # Check the training_config observation space matches that returned by _get_obs
        example_obs = self._get_obs()
        if example_obs.shape != self.observation_space.shape:
            raise ValueError(
                f"The Training Config observation space: {self.observation_space.shape} does not "
                f"match the Environment observation shape: {example_obs.shape}"
            )
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
        self._episode_tracker += 1
        self._episode_ended = False
        if isinstance(self.estimator, RuntimeEstimator):
            self.estimator.options.environment["job_tags"] = [
                f"rl_qoc_step{self._step_tracker}"
            ]

        if self.target_type == "gate":
            self._index_input_state = np.random.randint(
                len(self.target["input_states"])
            )

        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"episode": self._episode_tracker, "step": self._step_tracker}

    def _get_obs(self):
        return np.array(
            [
                self._index_input_state / len(self.target["input_states"]),
            ]
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
        params, batch_size = np.array(action), action.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} != {self.batch_size} ")

        self._param_values = params
        terminated = self._episode_ended = True
        reward = self.perform_action(self._param_values)

        # Using Negative Log Error as the Reward
        optimal_error_precision = 1e-6
        max_fidelity = 1.0 - optimal_error_precision
        reward = np.clip(reward, a_min=0.0, a_max=max_fidelity)
        reward = -np.log(1.0 - reward)

        return self._get_obs(), reward, terminated, False, self._get_info()

    def check_reward(self):
        if self.training_with_cal:
            sample_action = np.zeros(self.action_space.shape)
            batch_action = np.tile(sample_action, (self.batch_size, 1))
            batch_rewards = self.perform_action(batch_action)
            mean_reward = np.mean(batch_rewards)
            if mean_reward > self._reward_check_max:
                raise ValueError(
                    f"Current Mean Reward with Config Vals is {mean_reward}, try a C Factor around {self.c_factor / mean_reward}"
                )
        else:
            pass

    def episode_length(self, global_step: int):
        assert (
                global_step == self.step_tracker
        ), "Given step not synchronized with internal environment step counter"
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
        if self.benchmark_cycle == 0:
            return False
        else:
            return (
                    self._episode_tracker % self.benchmark_cycle == 0
                    and self._episode_tracker > 1
            )

    def perform_action(self, actions: np.array):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :return: Reward table (reward for each run in the batch)
        """

        qc = self.circuit_truncations[0]
        input_state_circ = QuantumCircuit(self.tgt_register)

        params, batch_size = np.array(actions), self.batch_size
        assert (
                len(params) == batch_size
        ), f"Action size mismatch {len(params)} != {batch_size} "
        self.action_history.append(params)

        if self.target_type == "gate":
            # Pick random input state from the list of possible input states (forming a tomographically complete set)
            index = self._index_input_state
            input_state = self.target["input_states"][index]
            input_state_circ = input_state["circuit"]
            target_state = input_state["target_state"]  # (Gate |input>=|target>)
        else:  # State preparation task
            target_state = self.target

        observables, pauli_shots = self.retrieve_observables(target_state, qc)

        if self.do_benchmark():
            print("Starting benchmarking...")
            self.store_benchmarks(params)
            print("Finished benchmarking")

        try:
            handle_session(
                qc, input_state_circ, self.estimator, self.backend, self._session_counts
            )
            # Append input state prep circuit to the custom circuit with front composition
            full_circ = qc.compose(input_state_circ, inplace=False, front=True)
            print(observables)
            job = self.estimator.run(
                circuits=[full_circ] * batch_size,
                observables=[observables] * batch_size,
                parameter_values=params,
                shots=int(np.max(pauli_shots) * self.n_shots),
            )

            reward_table = job.result().values
        except Exception as e:
            self.close()
            raise
        self.reward_history.append(reward_table)
        assert (
                len(reward_table) == self.batch_size
        ), f"Reward table size mismatch {len(reward_table)} != {self.batch_size} "
        return reward_table  # Shape [batchsize]

    def store_benchmarks(self, params: np.array):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: List of Action vectors to execute on quantum system
        :return: None
        """
        qc = self.circuit_truncations[0]
        # Build reference circuit (ideal)
        if self.target_type == "state":  # State preparation task
            target = self.target["dm"]
        else:  # Gate calibration task
            target = Operator(self.target["gate"])

        qc_list = [qc.assign_parameters(angle_set) for angle_set in params]
        if self.check_on_exp:
            # Experiment based fidelity estimation
            try:
                if self.target_type == "state":  # State preparation task
                    print("Starting state tomography...")
                    self.state_fidelity_history.append(
                        state_fidelity_from_state_tomography(
                            qc_list,
                            self.backend,
                            self.target["register"],
                            target_state=target,
                            session=self.estimator.session
                            if hasattr(self.estimator, "session")
                            else None,
                        )
                    )
                    print("Finished state tomography")
                else:  # Gate calibration task
                    print("Starting process tomography...")
                    self.avg_fidelity_history.append(
                        gate_fidelity_from_process_tomography(
                            qc_list,
                            self.backend,
                            target,
                            self.target["register"],
                            session=self.estimator.session
                            if hasattr(self.estimator, "session")
                            else None,
                        )
                    )
                    print("Finished process tomography")

            except Exception as e:
                self.close()
                raise e
        else:
            # Circuit list for each action of the batch
            print("Starting simulation benchmark...")
            if self.abstraction_level == "circuit":
                if self.target_type == "state":
                    q_state_list = [
                        Statevector.from_instruction(circ) for circ in qc_list
                    ]
                    density_matrix = DensityMatrix(
                        np.mean(
                            [
                                q_state.to_operator().to_matrix()
                                for q_state in q_state_list
                            ],
                            axis=0,
                        )
                    )
                    self.state_fidelity_history.append(
                        state_fidelity(self.target["dm"], density_matrix)
                    )
                else:  # Gate calibration task
                    q_process_list = [Operator(circ) for circ in qc_list]
                    prc_fidelity = np.mean(
                        [
                            process_fidelity(q_process, Operator(self.target["gate"]))
                            for q_process in q_process_list
                        ]
                    )
                    avg_fidelity = np.mean(
                        [
                            average_gate_fidelity(
                                q_process, Operator(self.target["gate"])
                            )
                            for q_process in q_process_list
                        ]
                    )
                    self.avg_fidelity_history.append(
                        avg_fidelity
                    )  # Avg gate fidelity over the action batch

            elif self.abstraction_level == "pulse":
                # Pulse simulation
                if isinstance(self.backend, DynamicsBackend) and isinstance(
                        self.backend.options.solver, JaxSolver
                ):
                    # Jax compatible pulse simulation

                    unitaries = np.array(self.backend.options.solver.unitary_solve(params))[
                                :, 1, :, :
                                ]

                    qubitized_unitaries = [
                        qubit_projection(u, self.backend.options.subsystem_dims)
                        for u in unitaries
                    ]

                    if self.target_type == "state":
                        density_matrix = DensityMatrix(
                            np.mean(
                                [
                                    Statevector.from_int(0, dims=self._d).evolve(
                                        unitary
                                    )
                                    for unitary in qubitized_unitaries
                                ],
                                axis=0,
                            )
                        )
                        self.state_fidelity_history.append(
                            state_fidelity(self.target["dm"], density_matrix)
                        )
                    else:  # Gate calibration task
                        gate = Operator(
                            transpile(self.baseline_truncations[0], self.backend)
                        )
                        self.avg_fidelity_history.append(
                            np.mean(
                                [
                                    average_gate_fidelity(unitary, gate)
                                    for unitary in qubitized_unitaries
                                ]
                            )
                        )
                    self.built_unitaries.append(unitaries)
                else:
                    raise NotImplementedError(
                        "Pulse simulation not yet implemented for this backend"
                    )
            print("Finished simulation benchmark")

    def retrieve_observables(self, target_state, qc):
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = np.round(
            [
                self.c_factor
                * target_state["Chi"][p]
                / (self._d * distribution.prob(p))
                for p in pauli_index
            ],
            5,
        )

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp.from_list(
            [
                (pauli_basis(num_qubits=qc.num_qubits)[p].to_label(), reward_factor[i])
                for i, p in enumerate(pauli_index)
            ]
        )

        return observables, pauli_shots

    def _generate_circuit_truncations(self):
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
            self.built_unitaries.clear()

        else:
            self.state_fidelity_history.clear()
            self.density_matrix_history.clear()

    def close(self) -> None:
        if isinstance(self.estimator, RuntimeEstimator):
            self.estimator.session.close()
        elif isinstance(self.backend, IBMBackend):
            self.backend.cancel_session()

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
        string += f"Batchsize: {self.batch_size}, \n"
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
    def parameters(self):
        return self._parameters

    @property
    def config_type(self):
        return self._config_type

    @property
    def config(self):
        return self.training_config

    @property
    def estimator(self) -> Estimator_type:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        self._estimator = estimator

    @property
    def sampler(self) -> Sampler_type:
        return self._sampler

    @estimator.setter
    def estimator(self, sampler: Sampler_type):
        self._sampler = sampler

    @property
    def tgt_instruction_counts(self):
        return self._tgt_instruction_counts

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
                "fidelity_history": self.avg_fidelity_history
                if self.target_type == "gate"
                else self.state_fidelity_history,
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