"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 05/09/2024
"""

from __future__ import annotations

from typing import List, Any, SupportsFloat, Tuple, Optional
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box

# Qiskit imports
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator
from qiskit.transpiler import InstructionProperties


from .base_q_env import (
    BaseQuantumEnvironment,
    GateTarget,
    StateTarget,
)

from ..helpers.circuit_utils import fidelity_from_tomography
from .configuration.qconfig import QEnvConfig


class QuantumEnvironment(BaseQuantumEnvironment):
    """
    A quantum environment for reinforcement learning.

    This environment is designed to be used with PyTorch and Qiskit. It allows an agent to interact with a quantum
    system and learn to perform a specific task, such as gate calibration or state preparation.
    """

    def __init__(self, training_config: QEnvConfig):
        """
        Initializes the QuantumEnvironment.

        Args:
            training_config: The configuration for the training environment.
        """
        self._parameters = [Parameter(f"a_{i}") for i in range(training_config.n_actions)]

        super().__init__(training_config)

        self.circuits = self.define_circuits()
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)

    @property
    def parameters(self):
        """The parameters of the environment."""
        return self._parameters

    @property
    def circuit_choice(self) -> int:
        """The index of the circuit to be used."""
        return 0

    def episode_length(self, global_step: int) -> int:
        """
        Determines the length of an episode.

        Args:
            global_step: The current step in the training loop.

        Returns:
            The length of the episode.
        """
        return 1

    def define_circuits(
        self,
    ) -> List[QuantumCircuit]:
        """
        Defines the circuits to be used in the environment.

        Returns:
            A list of quantum circuits.
        """

        custom_circuit = QuantumCircuit(self.config.target.tgt_register, name="custom_circuit")

        self.parametrized_circuit_func(
            custom_circuit,
            self.parameters,
            self.config.target.tgt_register,
            **self._func_args,
        )

        custom_circuit.metadata["baseline_circuit"] = self.config.target.circuits[0].copy(
            "baseline_circuit"
        )
        return [custom_circuit]

    def _get_obs(self):
        """
        Returns the observation of the environment.

        Returns:
            The observation of the environment.
        """
        if isinstance(self.target, GateTarget) and self.config.reward_method == "state":
            return np.array(
                [
                    0.0,
                    0.0,
                ]
                + list(self._observable_to_observation())
            )
        else:
            return np.array([0, 0])

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
        params, batch_size = np.array(action), len(np.array(action))
        if params.shape != (batch_size, self.n_actions):
            raise ValueError(
                f"Action shape mismatch: {params.shape} != {(batch_size, self.n_actions)}"
            )
        reward = self.perform_action(action)

        if np.mean(reward) > self._max_return:
            self._max_return = np.mean(reward)
            self._optimal_action = self.mean_action
        self.reward_history.append(reward)
        assert (
            len(reward) == self.batch_size
        ), f"Reward table size mismatch {len(reward)} != {self.batch_size} "
        assert not np.any(np.isinf(reward)) and not np.any(
            np.isnan(reward)
        ), "Reward table contains NaN or Inf values"
        optimal_error_precision = 1e-6
        max_fidelity = 1.0 - optimal_error_precision
        reward = np.clip(reward, a_min=0.0, a_max=max_fidelity)
        reward = -np.log10(1.0 - reward)
        return self._get_obs(), reward, terminated, False, self._get_info()

    def compute_benchmarks(
        self, qc: QuantumCircuit, params: np.array, update_env_history=True
    ) -> np.array:
        """
        Computes benchmarks for the given circuit and parameters.

        Args:
            qc: The quantum circuit to benchmark.
            params: The parameters for the circuit.
            update_env_history: Whether to update the environment's history.

        Returns:
            An array of fidelity values.
        """

        if self.config.check_on_exp:

            try:
                if not self.config.reward_method == "fidelity":
                    if self.config.benchmark_config.benchmark_batch_size > 1:
                        angle_sets = np.clip(
                            np.random.normal(
                                self.mean_action,
                                self.std_action,
                                size=(self.config.benchmark_config.benchmark_batch_size, self.n_actions),
                            ),
                            self.action_space.low,
                            self.action_space.high,
                        )
                    else:
                        angle_sets = [self.mean_action]
                else:
                    if len(params.shape) == 1:
                        params = np.expand_dims(params, axis=0)
                    angle_sets = params

                qc_input = [qc.assign_parameters(angle_set) for angle_set in angle_sets]
                print("Starting tomography...")
                fids = fidelity_from_tomography(
                    qc_input,
                    self.backend,
                    self.target.physical_qubits,
                    (
                        self.target.target_operator
                        if isinstance(self.target, GateTarget)
                        else self.target.dm
                    ),
                    analysis=self.config.tomography_analysis,
                    sampler=self.sampler,
                )
                print("Finished tomography")

            except Exception as e:
                self.close()
                raise e
            if isinstance(self.target, StateTarget):
                self.circuit_fidelity_history.append(np.mean(fids))
            else:
                self.avg_fidelity_history.append(np.mean(fids))

        else:
            print("Starting simulation benchmark...")
            if not self.config.reward_method == "fidelity":
                params = np.array([self.mean_action])
            if self.abstraction_level == "circuit":
                fids = self.simulate_circuit(qc, params, update_env_history)
            else:
                fids = self.simulate_pulse_circuit(qc, params, update_env_history)
            if self.target.target_type == "state":
                print("State fidelity:", fids)
            else:
                print("Avg gate fidelity:", fids)
            print("Finished simulation benchmark")
        return fids

    def update_gate_calibration(self, gate_name: Optional[str] = None):
        """
        Updates the gate calibration with the optimal actions.

        Args:
            gate_name: The name of the custom gate to be created.

        Returns:
            The pulse calibration for the custom gate.
        """
        try:
            from qiskit import schedule, pulse
            from qiskit_dynamics import DynamicsBackend
            from ..helpers.pulse_utils import (
                simulate_pulse_input,
                get_optimal_z_rotation,
            )
        except ImportError as e:
            raise ImportError(
                "Pulse calibration requires Qiskit Pulse, Qiskit Dynamics and Qiskit Experiments below 0.10."
                "Please set your Qiskit version to 1.x to use this feature."
            )
        if not isinstance(self.target, GateTarget):
            raise ValueError("Target type should be a gate for gate calibration task.")

        if self.abstraction_level == "pulse":
            assert self.backend is not None, "Backend must be set for pulse calibration"
            qc = self.circuits[0].assign_parameters(self.optimal_action, inplace=False)
            schedule_ = schedule(qc, self.backend)
            duration = schedule_.duration
            if isinstance(self.backend, DynamicsBackend):
                sim_data = simulate_pulse_input(
                    self.backend,
                    schedule_,
                    target=Operator(self.target.gate),
                )
                error = 1.0 - sim_data["gate_fidelity"]["optimal"]
                optimal_rots = sim_data["gate_fidelity"]["rotations"]
            else:
                exp = ProcessTomography(qc, self.backend, self.target.physical_qubits)
                exp_data = exp.run(shots=10000).block_for_results()
                process_matrix = exp_data.analysis_results("state").value
                opt_res = get_optimal_z_rotation(
                    process_matrix, self.target.target_operator, self.n_qubits
                )
                optimal_rots = opt_res.x
                error = 1.0 - opt_res.fun

            with pulse.build(self.backend) as sched:
                for i in range(self.n_qubits):
                    rz_cal = self.backend.target.get_calibration("rz", (i,))
                    pulse.call(
                        rz_cal,
                        value_dict={parameter: optimal_rots[i] for parameter in rz_cal.parameters},
                    )
                pulse.call(schedule_)
                for i in range(self.n_qubits):
                    rz_cal = self.backend.target.get_calibration("rz", (i,))
                    pulse.call(
                        rz_cal,
                        value_dict={
                            parameter: optimal_rots[-1 - i] for parameter in rz_cal.parameters
                        },
                    )

            instruction_prop = InstructionProperties(duration, error, sched)
            if gate_name is None:
                gate_name = self.target.gate.name

            if gate_name not in self.backend.operation_names:
                self.backend.target.add_instruction(
                    self.target.gate,
                    {tuple(self.physical_target_qubits): instruction_prop},
                    name=gate_name,
                )

            else:
                self.backend.target.update_instruction_properties(
                    gate_name,
                    tuple(self.physical_target_qubits),
                    instruction_prop,
                )

            return self.backend.target.get_calibration(
                gate_name, tuple(self.physical_target_qubits)
            )
        else:
            return self.circuits[0].assign_parameters({self.parameters[0]: self.optimal_action})
