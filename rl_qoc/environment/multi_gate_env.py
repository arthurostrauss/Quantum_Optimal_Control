"""
MultiGateEnv: Environment for joint calibration of multiple gate targets.

This environment extends BaseQuantumEnvironment to support simultaneous calibration
of multiple GateTargets within a circuit context using MultiTarget.

Author: Auto-generated
Created on: 2025
"""

from __future__ import annotations

from typing import Optional, List, Any, Dict, SupportsFloat, TypeVar
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector

from .base_q_env import BaseQuantumEnvironment
from .configuration.multi_target_qconfig import MultiTargetQEnvConfig
from .target import MultiTarget

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MultiGateEnv(BaseQuantumEnvironment):
    """
    Environment for joint calibration of multiple gate targets.
    
    This environment supports simultaneous calibration of multiple GateTargets
    that are defined within the same circuit context(s). The action space is
    automatically inferred from all InstructionReplacements in the MultiTarget.
    """

    def __init__(
        self,
        training_config: MultiTargetQEnvConfig,
        **kwargs,
    ):
        """
        Initialize the MultiGateEnv.
        
        Args:
            training_config: MultiTargetQEnvConfig containing the training configuration
            **kwargs: Additional keyword arguments
        """
        # Temporarily set target to first gate target for base class initialization
        # We'll override this after initialization
        original_target = training_config.target
        if len(original_target.gate_targets) > 0:
            temp_target = original_target.gate_targets[0]
        else:
            raise ValueError("MultiTarget must contain at least one GateTarget")
        
        # Create a temporary config with single target for base class
        from .configuration.qconfig import QEnvConfig
        temp_config = QEnvConfig(
            target=temp_target,
            backend_config=training_config.backend_config,
            action_space=training_config.action_space,
            execution_config=training_config.execution_config,
            reward=training_config.reward,
            benchmark_config=training_config.benchmark_config,
            env_metadata=training_config.env_metadata,
        )
        
        super().__init__(temp_config)
        
        # Now restore the MultiTarget
        self._env_config = training_config
        self._multi_target = original_target
        
        # Reference parameters directly from MultiTarget circuits
        self._parameters = self._multi_target.circuit_parameters
        
        # Set up observation space
        self._setup_observation_space()
        
        # Define circuits (custom contexts produced by MultiTarget)
        self.circuits = self.define_circuits()
        
        # Circuit choice for observation
        self._circuit_choice = 0


    def _setup_observation_space(self):
        """Set up the observation space similarly to the context-aware environment."""
        param_names = self._multi_target.action_parameter_names
        if param_names:
            self.observation_space = DictSpace(
                {
                    name: Box(0.0, np.pi, shape=(1,), dtype=np.float32)
                    for name in param_names
                }
            )
        else:
            # No symbolic parameters - use constant observation
            self.observation_space = Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def define_circuits(self) -> List[QuantumCircuit]:
        """
        Retrieve the custom circuits prepared by the MultiTarget definition.
        """
        return [circ.copy() for circ in self._multi_target.custom_circuits]

    def _get_obs(self) -> ObsType:
        """Return the observation of the environment."""
        if isinstance(self.observation_space, DictSpace):
            obs = {
                name: np.zeros((1,), dtype=np.float32)
                for name in self.observation_space.spaces
            }
            return obs
        else:
            # Constant observation
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def compute_benchmarks(
        self, qc: QuantumCircuit, params: np.ndarray, update_env_history=True
    ) -> np.ndarray:
        """
        Benchmark through simulation the policy.
        For MultiTarget, we compute fidelities for each individual GateTarget.
        """
        # For now, delegate to base class behavior
        # In the future, this could compute separate fidelities for each target
        return super().compute_benchmarks(qc, params, update_env_history)

    def episode_length(self, global_step: int) -> int:
        """Return episode length (always 1 for this environment)."""
        return 1

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Randomly select a circuit context
        if len(self._multi_target.circuit_contexts) > 1:
            self._circuit_choice = self.np_random.integers(0, len(self._multi_target.circuit_contexts))
        else:
            self._circuit_choice = 0
        self._multi_target.circuit_choice = self._circuit_choice
        
        return self._get_obs(), self._get_info()

    @property
    def config(self) -> MultiTargetQEnvConfig:
        """Return the environment configuration."""
        return self._env_config

    @property
    def target(self) -> MultiTarget:
        """Return the MultiTarget object."""
        return self._multi_target

    @property
    def parameters(
        self,
    ) -> List[ParameterVector | List[Parameter]] | ParameterVector | List[Parameter]:
        """Return the Qiskit Parameter(s) for all targets."""
        return self._parameters

    @property
    def circuit_choice(self) -> int:
        """Return the index of the current circuit context."""
        return self._circuit_choice

    @circuit_choice.setter
    def circuit_choice(self, value: int):
        """Set the index of the current circuit context."""
        if 0 <= value < len(self._multi_target.circuit_contexts):
            self._circuit_choice = value
            self._multi_target.circuit_choice = value
        else:
            raise ValueError(f"Circuit choice {value} out of range [0, {len(self._multi_target.circuit_contexts)})")

    @property
    def circuit(self):
        """Return the current circuit."""
        return self.circuits[self.circuit_choice]

    @property
    def baseline_circuit(self):
        """Return the baseline circuit for the current circuit choice."""
        return self._multi_target.baseline_circuit(self.circuit_choice)

    def perform_action(self, actions: np.ndarray, update_env_history: bool = True):
        """
        Send the action batch to the quantum system and retrieve reward.
        Overrides base class to use MultiTarget-specific reward methods.
        
        :param actions: action vectors to execute on quantum system
        :param update_env_history: Boolean to update the environment history
        :return: Reward table (reward for each action in the batch)
        """
        if not actions.shape[-1] == self.n_actions:
            raise ValueError(f"Action shape mismatch: {actions.shape[-1]} != {self.n_actions}")
        qc = self.circuit.copy()
        params, batch_size = np.array(actions), actions.shape[0]
        if len(params.shape) == 1:
            params = np.expand_dims(params, axis=0)

        # Get the reward method from the configuration
        rewarder = self.config.reward

        if self.do_benchmark():  # Benchmarking or fidelity access
            fids = self.compute_benchmarks(qc, params, update_env_history)

        # Check if the reward method exists in the dictionary
        if self.config.execution_config.n_reps_mode == "sequential":
            # Use MultiTarget-specific reward method if available
            if hasattr(rewarder, 'get_reward_data_multi_target'):
                reward_data = rewarder.get_reward_data_multi_target(
                    qc,
                    params,
                    self.config,
                )
            else:
                # Fallback to standard method (may not work correctly for MultiTarget)
                reward_data = rewarder.get_reward_data(
                    qc,
                    params,
                    self.config,
                )
            total_shots = reward_data.total_shots
            if update_env_history:
                self.update_env_history(qc, total_shots)
            self._pubs = reward_data.pubs
            self._reward_data = reward_data
            
            # Use MultiTarget-specific reward extraction if available
            if hasattr(rewarder, 'get_reward_with_primitive_multi_target'):
                target_rewards = rewarder.get_reward_with_primitive_multi_target(reward_data, self.primitive)
                # target_rewards is now shape (num_targets, batch_size)
                if target_rewards is not None and target_rewards.size > 0:
                    # For now, return the average reward across all targets
                    # In the future, this could return individual rewards or a combined metric
                    reward = np.mean(target_rewards, axis=0)  # Shape: [batch_size]
                    print(f"Reward (avg across {target_rewards.shape[0]} targets):", np.mean(reward), "Std:", np.std(reward))
                else:
                    reward = np.zeros(batch_size)
                    print("Warning: No target rewards returned")
            else:
                # Fallback to standard method
                reward = rewarder.get_reward_with_primitive(reward_data, self.primitive)
                print("Reward (avg):", np.mean(reward), "Std:", np.std(reward))

            return reward  # Shape [batch size]
        else:
            raise NotImplementedError("Only sequential mode is supported for now")
