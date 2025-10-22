"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate/
execute quantum circuits. The environment is designed to be context-aware, meaning it can focus on a specific
target gate within a larger quantum circuit. This workflow could be extended to multiple target gates in the future.

Author: Arthur Strauss
Created on 26/06/2023
Last modified on 28/04/2025
"""

from __future__ import annotations

import sys
from typing import (
    Optional,
    List,
    Any,
    TypeVar,
    SupportsFloat,
)

import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ParameterVector,
    CircuitInstruction,
    Gate,
    Instruction,
    Parameter,
    Qubit,
)
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.providers import BackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import (
    InstructionProperties,
    PassManager,
)

from qiskit_experiments.library import ProcessTomography

from ..helpers import (
    CustomGateReplacementPass,
    InstructionReplacement,
)
from ..helpers.circuit_utils import get_instruction_timings
from .configuration.qconfig import QEnvConfig
from .base_q_env import (
    GateTarget,
    StateTarget,
    BaseQuantumEnvironment,
)

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def create_array(circ_trunc, batchsize, n_actions):
    """
    Creates a numpy array of objects.

    Args:
        circ_trunc: The size of the first dimension of the array.
        batchsize: The size of the second dimension of the array.
        n_actions: The size of the third dimension of the array.

    Returns:
        A numpy array of objects with the specified dimensions.
    """
    arr = np.empty((circ_trunc,), dtype=object)
    for i in range(circ_trunc):
        arr[i] = np.zeros((i + 1, batchsize, n_actions))
    return arr


def target_instruction_timings(
    circuit_context: QuantumCircuit, target_instruction: CircuitInstruction
) -> tuple[List[int], List[int]]:
    """
    Return the timings of the target instructions in the circuit context.

    Args:
        circuit_context: The circuit context containing the target instruction.
        target_instruction: The target instruction to be found in the circuit context.

    Returns:
        A tuple containing the start times of all operations and the start times of the target instructions.
    """
    try:
        op_start_times = circuit_context.op_start_times
    except AttributeError:
        op_start_times = get_instruction_timings(circuit_context)

    target_instruction_timings = []
    for i, instruction in enumerate(circuit_context.data):
        if instruction == target_instruction:
            target_instruction_timings.append(op_start_times[i])
    return op_start_times, target_instruction_timings


class ContextAwareQuantumEnvironment(BaseQuantumEnvironment):
    """
    A quantum environment for context-aware gate calibration.

    This environment is designed to calibrate a quantum gate within a specific circuit context.
    It uses a reinforcement learning approach to find the optimal parameters for the gate.
    """

    def __init__(
        self,
        config: QEnvConfig,
        **context_kwargs,
    ):
        """
        Initializes the ContextAwareQuantumEnvironment.

        Args:
            config: The configuration for the training environment.
            **context_kwargs: Additional keyword arguments for the environment.
        """

        super().__init__(config)

        self.circ_tgt_register = None
        self.custom_instructions: List[Instruction] = []
        self.new_gates: List[Gate] = []

        if isinstance(self.target, GateTarget) and self.target.context_parameters:
            self.observation_space = DictSpace(
                {
                    p.name: Box(0.0, np.pi, shape=(1,), dtype=np.float32)
                    for p in self.target.context_parameters
                }
            )
        else:
            self.observation_space = Box(
                0.0, 1.0, shape=(1,), dtype=np.float32, seed=self.seed + 98
            )
        self._parameters = [
            [Parameter(f"a_{j}_{i}") for i in range(self.n_actions)]
            for j in range(len(self.target.circuits))
        ]

        self._optimal_actions = [
            np.zeros(self.config.n_actions) for _ in range(len(self.target.circuits))
        ]

        self._pm = [
            PassManager(
                CustomGateReplacementPass(
                    InstructionReplacement(
                        self.target.target_instructions[i],
                        self.parametrized_circuit_func,
                        self.parameters[i],
                        self._func_args,
                    )
                )
            )
            for i in range(len(self.target.circuits))
        ]
        self.circuits = self.define_circuits()

    def define_circuits(self) -> list[QuantumCircuit]:
        """
        Defines the circuits to be used in the environment.

        This method can be overridden by subclasses to provide custom circuit definitions.

        Returns:
            A list of QuantumCircuit objects.
        """
        circuits = []
        for i, circ in enumerate(self.target.circuits):
            custom_circ = self._pm[i].run(circ)
            custom_circ.metadata["baseline_circuit"] = circ.copy(f"baseline_circ_{circ.name}")
            circuits.append(custom_circ)

        return circuits

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Performs a single step in the environment.

        Args:
            action: The action to be performed.

        Returns:
            A tuple containing the observation, reward, whether the episode has ended,
            whether the episode was truncated, and additional information.
        """
        if self._episode_ended:
            terminated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                terminated,
                False,
                self._get_info(),
            )

        params, batch_size = np.array(action), len(np.array(action))
        if batch_size != self.batch_size:
            raise ValueError(
                f"Action batch size {batch_size} does not match environment batch size {self.batch_size}"
            )
        terminated = self._episode_ended = True
        reward = self.perform_action(params)

        obs = self._get_obs()

        # Using Negative Log Error as the Reward
        if np.mean(reward) > self._max_return:
            self._max_return = np.mean(reward)
            self._optimal_actions[self.circuit_choice] = self.mean_action

        assert (
            len(reward) == self.batch_size
        ), f"Reward table size mismatch {len(reward)} != {self.batch_size} "
        assert not np.any(np.isinf(reward)) and not np.any(
            np.isnan(reward)
        ), "Reward table contains NaN or Inf values"
        optimal_error_precision = 1e-6
        max_fidelity = 1.0 - optimal_error_precision
        if self._fit_function is not None:
            reward = self._fit_function(reward, self.n_reps)
        reward = np.clip(reward, a_min=0.0, a_max=max_fidelity)
        self.reward_history.append(reward)
        reward = -np.log10(1.0 - reward)

        return obs, reward, terminated, False, self._get_info()

    def _get_obs(self) -> ObsType:
        """
        Returns the observation of the environment.

        This method can be overridden by subclasses to provide custom observations.

        Returns:
            The observation of the environment.
        """

        if isinstance(self.observation_space, DictSpace):
            return {p.name: val for p, val in self.target.context_parameters.items()}
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def compute_benchmarks(
        self, qc: QuantumCircuit, params: np.ndarray, update_env_history=True
    ) -> np.ndarray:
        """
        Computes benchmarks for the given circuit and parameters.

        Args:
            qc: The quantum circuit to benchmark.
            params: The parameters for the circuit.
            update_env_history: Whether to update the environment's history.

        Returns:
            An array of fidelity values.
        """
        if (
            self.config.check_on_exp
        ):  # Perform real experiments to retrieve from measurement data fidelities
            raise NotImplementedError(
                "Direct Fidelity Estimation is not yet supported in the context-aware environment"
            )

        else:  # Perform simulation at circuit or pulse level
            print("Starting simulation benchmark...")
            if not self.config.reward_method == "fidelity":
                params = np.array([self.mean_action])  # Benchmark policy only through mean action
            if self.abstraction_level == "circuit":
                fids = self.simulate_circuit(qc, params, update_env_history)
            else:  # Pulse simulation
                fids = self.simulate_pulse_circuit(qc, params, update_env_history)
            print("Finished simulation benchmark \n")

            fidelity_type = "Gate" if self.target.causal_cone_size <= 3 else "State"
            if len(fids) == 1:
                print(f"{fidelity_type} Fidelity (per Cycle): ", fids[0])
            else:
                print(f"{fidelity_type} Fidelities (per Cycle): ", np.mean(fids))
            return fids

    @property
    def parameters(self) -> List[ParameterVector] | List[List[Parameter]]:
        """The parameters of the environment."""
        return self._parameters

    @property
    def target(self) -> GateTarget | StateTarget:
        """The current target to be calibrated."""
        return self.config.target

    @property
    def virtual_target_qubits(self) -> List[Qubit] | QuantumRegister:
        """The virtual target qubits used in the circuit context."""
        if isinstance(self.config.target, GateTarget):
            return self.config.target.virtual_target_qubits
        else:
            return self.config.target.tgt_register

    @property
    def optimal_action(self):
        """The optimal action for the current gate instance."""
        return self._optimal_actions[self.circuit_choice]

    def optimal_actions(self, indices: Optional[int | List[int]] = None):
        """
        Returns the optimal actions for the selected circuit truncation index.

        Args:
            indices: The indices of the optimal actions to return. If None, all optimal actions are returned.

        Returns:
            A list of optimal actions.
        """
        if isinstance(indices, int):
            return self._optimal_actions[indices]
        elif isinstance(indices, List):
            return [self._optimal_actions[index] for index in indices]
        else:
            return self._optimal_actions

    def episode_length(self, global_step: int) -> int:
        """
        Returns the length of an episode.

        Args:
            global_step: The current global step.

        Returns:
            The length of the episode.
        """
        return 1

    def clear_history(self) -> None:
        """Resets all counters related to training."""
        super().clear_history()

    def set_env_params(self, **kwargs):
        """
        Sets the environment parameters.

        Args:
            **kwargs: The keyword arguments to set.
        """
        if "circuit_choice" in kwargs:
            self.circuit_choice = kwargs.pop("circuit_choice")
        if "parameters" in kwargs:
            assert isinstance(self.config.target, GateTarget), "Target must be a gate target"
            assert isinstance(kwargs["parameters"], dict), "Parameters must be a dictionary"
            self.config.target.clear_parameters()
            assert all(
                p in self.config.target.circuit.parameters
                or p in [p_.name for p_ in self.config.target.circuit.parameters]
                for p in kwargs["parameters"]
            ), "Parameters must be in the circuit parameters"
            self.config.target.bind_parameters(kwargs.pop("parameters"))
            self.circuits = self.define_circuits()

        super().set_env_params(**kwargs)

    @property
    def circuit_context(self) -> QuantumCircuit:
        """The current circuit context."""
        return self.target.circuit

    @property
    def circuit_choice(self) -> int:
        """The index of the circuit choice in the context-aware environment."""
        return super().circuit_choice

    @circuit_choice.setter
    def circuit_choice(self, value: int):
        """
        Sets the index of the circuit choice in the context-aware environment.

        Args:
            value: The new index.
        """
        self.config.target.circuit_choice = value

    @property
    def has_context(self) -> bool:
        """
        Checks if the environment has a circuit context.

        Returns:
            True if the environment has a circuit context, False otherwise.
        """
        return self.target.has_context

    def update_gate_calibration(self, gate_names: Optional[List[str]] = None):
        """
        Updates the gate calibrations with the optimal actions.

        Args:
            gate_names: The names of the custom gates to be created.
        """
        try:
            from qiskit import schedule
            from qiskit_dynamics import DynamicsBackend
            from ..helpers.pulse_utils import (
                simulate_pulse_input,
            )
        except ImportError as e:
            raise ImportError(
                "Pulse calibration requires Qiskit Pulse, Qiskit Dynamics and Qiskit Experiments below 0.10."
                "Please set your Qiskit version to 1.x to use this feature."
            )
        if self.abstraction_level == "pulse":
            if gate_names is not None and len(gate_names) != len(self.custom_instructions):
                raise ValueError(
                    "Number of gate names does not match number of custom instructions"
                )
            else:
                gate_names = [
                    f"{gate.name}_{i}_opt" for i, gate in enumerate(self.custom_instructions)
                ]

            value_dicts = [{} for _ in range(self.tgt_instruction_counts)]
            for i, custom_circ in enumerate(self.circuits):
                for param in custom_circ.parameters:
                    if (
                        isinstance(param, ParameterVectorElement)
                        and param.vector in self.parameters
                    ):
                        vector_index = self.parameters.index(param.vector)
                        param_index = param.index
                        value_dicts[i][param] = self.optimal_actions(vector_index)[param_index]

                    elif param.name.startswith("a_"):
                        vector_index = int(param.name.split("_")[1])
                        param_index = int(param.name.split("_")[2])
                        value_dicts[i][param] = self.optimal_actions(vector_index)[param_index]

            contextual_schedules = schedule(self.circuits, self.backend)

            gate_qc = [
                QuantumCircuit(self.circ_tgt_register) for _ in range(self.tgt_instruction_counts)
            ]
            schedules, durations = [], []
            for i, gate in enumerate(self.custom_instructions):
                baseline_circ = self.baseline_circuits[i]
                custom_circ = self.circuits[i].assign_parameters(value_dicts[i], inplace=False)

                gate_qc[i].append(gate, self.circ_tgt_register)
                gate_qc[i].assign_parameters(self.optimal_actions(i), inplace=True)

                def _define(self2):
                    qc = QuantumCircuit(len(self.physical_target_qubits))
                    # Sort the qubits to ensure the gate is applied on the correct qubits ordering
                    sorted_indices = sorted(self.physical_target_qubits)
                    index_map = {value: i for i, value in enumerate(sorted_indices)}
                    new_indices = [index_map[value] for value in self.physical_target_qubits]
                    qc.append(self.target.gate, new_indices)
                    self2._definition = qc

                def array(self2, dtype=None):
                    return self.target.gate.to_matrix()

                def new_init(self2):
                    Gate.__init__(self2, gate_names[i], self.target.gate.num_qubits, [])

                new_gate_methods = {
                    "_define": _define,
                    "__array__": array,
                    "__init__": new_init,
                }
                new_gate_cls = type(f"{gate.name.capitalize()}_{i}", (Gate,), new_gate_methods)
                new_gate = new_gate_cls()
                self.new_gates.append(new_gate)
                cal_sched = schedule(gate_qc[i], self.backend)
                duration = cal_sched.duration
                schedules.append(cal_sched)
                durations.append(duration)
                contextual_schedules[i].assign_parameters(value_dicts[i], inplace=True)
                if isinstance(self.backend, DynamicsBackend):
                    sim_result = simulate_pulse_input(
                        self.backend,
                        contextual_schedules[i],
                        target=Operator(self.baseline_circuits[i]),
                    )
                    error = 1.0 - sim_result["gate_fidelity"]["raw"]

                else:
                    exp = ProcessTomography(
                        custom_circ,
                        self.backend,
                        self.involved_qubits_list[i],
                        target=Operator(baseline_circ),
                    )
                    exp_data = exp.run(shots=10000).block_for_results()
                    process_matrix = exp_data.analysis_results("state").value
                    process_fid = exp_data.analysis_results("process_fidelity").value
                    dim, _ = process_matrix.dim
                    avg_gate_fid = (dim * process_fid + 1) / (dim + 1)
                    error = 1.0 - avg_gate_fid

                instruction_prop = InstructionProperties(duration, error, cal_sched)

                self.backend.target.add_instruction(
                    new_gate, {tuple(self.physical_target_qubits): instruction_prop}
                )

    @property
    def involved_qubits_list(self):
        """
        Returns a list of lists of physical qubits involved in each circuit truncation.

        Returns:
            A list of lists of physical qubits.
        """
        involved_qubits = []
        for target in self._target:
            involved_qubits.extend(list(target.layout.get_physical_bits().keys()))
        return involved_qubits
