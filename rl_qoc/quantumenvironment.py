"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 05/09/2024
"""

from __future__ import annotations

# For compatibility for options formatting between Estimators.
from typing import List, Any, SupportsFloat, Tuple
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
)

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.states import DensityMatrix, Statevector

from .base_q_env import (
    BaseQuantumEnvironment,
    GateTarget,
    StateTarget,
)
from .helper_functions import (
    qubit_projection,
    rotate_unitary,
    get_optimal_z_rotation,
    fidelity_from_tomography,
)
from .qconfig import QEnvConfig


class QuantumEnvironment(BaseQuantumEnvironment):

    def __init__(self, training_config: QEnvConfig):
        """
        Initialize the Quantum Environment
        Args:
            training_config: QEnvConfig object containing the training configuration
        """
        self._parameters = ParameterVector("θ", training_config.n_actions)

        super().__init__(training_config)

        # self.observation_space = Box(
        #     low=np.array([0, 0] + [-5] * (2 ** self.n_qubits) ** 2),
        #     high=np.array([1, 1] + [5] * (2 ** self.n_qubits) ** 2),
        #     dtype=np.float32,
        # )
        self.observation_space = Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32
        )

    @property
    def parameters(self):
        return self._parameters

    @property
    def trunc_index(self) -> int:
        return 0

    @property
    def tgt_instruction_counts(self) -> int:
        return 1

    def episode_length(self, global_step: int) -> int:
        return 1

    def define_target_and_circuits(
        self,
    ) -> Tuple[
        GateTarget | StateTarget,
        List[QuantumCircuit],
        List[QuantumCircuit | DensityMatrix],
    ]:
        """
        Define the target to be used in the environment
        Returns:
            target: GateTarget or StateTarget object
        """
        if "gate" in self.config.target:
            target = GateTarget(
                n_reps=self.config.n_reps,
                **self.config.target,
                **self.config.reward_config.reward_args,
            )
        else:
            target = StateTarget(**self.config.target)

        custom_circuit = QuantumCircuit(target.tgt_register, name="custom_circuit")
        ref_circuit = QuantumCircuit(target.tgt_register, name="baseline_circuit")

        self.parametrized_circuit_func(
            custom_circuit,
            self.parameters,
            target.tgt_register,
            **self._func_args,
        )
        if isinstance(target, GateTarget):
            ref_circuit.append(target.gate, target.tgt_register)
        elif isinstance(target, StateTarget):
            ref_circuit = target.dm
        return target, [custom_circuit], [ref_circuit]

    def _get_obs(self):
        if isinstance(self.target, GateTarget) and self.config.reward_method == "state":
            return np.array(
                [
                    self._index_input_state / len(self.target.input_states),
                    0.0,
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

    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: List of Action vectors to execute on quantum system
        :return: None
        """

        if self.config.check_on_exp:

            # Experiment based fidelity estimation
            try:
                angle_sets = np.clip(
                    np.random.normal(
                        self.mean_action,
                        self.std_action,
                        size=(self.config.benchmark_batch_size, self.n_actions),
                    ),
                    self.action_space.low,
                    self.action_space.high,
                )

                qc_list = [qc.assign_parameters(angle_set) for angle_set in angle_sets]
                print("Starting tomography...")
                session = (
                    self.estimator.session
                    if hasattr(self.estimator, "session")
                    else None
                )
                fids = fidelity_from_tomography(
                    qc_list,
                    self.backend,
                    (
                        Operator(self.target.gate)
                        if isinstance(self.target, GateTarget)
                        else Statevector(self.target.dm)
                    ),
                    self.target.physical_qubits,
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

        else:  # Simulation based fidelity estimation (Aer for circuit level, Dynamics for pulse)
            print("Starting simulation benchmark...")
            if self.abstraction_level == "circuit":  # Circuit simulation
                fids = self.simulate_circuit(qc, params)
            else:  # Pulse simulation
                fids = self.simulate_pulse_circuit(qc, params)
            if self.target.target_type == "state":
                print("State fidelity:", self.circuit_fidelity_history[-1])
            else:
                print("Avg gate fidelity:", self.avg_fidelity_history[-1])
            print("Finished simulation benchmark")
        return fids
