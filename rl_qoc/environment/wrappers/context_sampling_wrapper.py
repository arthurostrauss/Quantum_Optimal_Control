from curses import raw
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from ..context_aware_quantum_environment import ContextAwareQuantumEnvironment
from gymnasium.wrappers import RescaleObservation
from gymnasium.spaces import Box, Dict as DictSpace
from abc import ABC, abstractmethod

@dataclass
class ContextSamplingWrapperConfig:
    num_warmup_updates: int = 20
    context_buffer_size: int = 500
    sampling_prob: float = 0.3
    eviction_strategy: str = "hybrid"
    evict_best_prob: float = 0.2
    anneal_noise: bool = True
    initial_noise_scale: float = 0.15
    final_noise_scale: float = 0.05

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ContextSamplingWrapperConfig":
        return cls(**config_dict)


class ContextSamplingWrapper(gym.Wrapper, ABC):
    """
    A Gymnasium wrapper that implements reward-guided context sampling.

    This wrapper manages a buffer of (context, reward, mean_action) tuples.
    On `reset()`, it samples a new context from this buffer (or randomly)
    and passes it to the underlying environment. On `step()`, it logs the
    resulting reward to update its buffer.
    """

    def __init__(
        self,
        env: ContextAwareQuantumEnvironment,
        config: ContextSamplingWrapperConfig | Dict[str, Any],
    ):
        super().__init__(env)
        # Store hyperparameters
        self.context_config = config if isinstance(config, ContextSamplingWrapperConfig) else ContextSamplingWrapperConfig.from_dict(config)

        # Buffers for context sampling logic
        self.context_buffer = []
        self.context_rewards = []
        self.mean_action_buffer = []

        # State tracking
        self.current_context = None

        if not isinstance(self.observation_space, DictSpace):
            raise ValueError("Observation space must be a Dict space")
        for key, obs_space in self.observation_space.items():
            if not isinstance(obs_space, Box):
                raise ValueError(f"Observation space for key {key} must be a Box space")

    @property
    def observation_space(self) -> DictSpace:
        """
        Return the observation space of the environment.
        """
        return super().observation_space

    def _add_to_buffer(
        self, context:Dict[str, Any], reward: float, mean_action: np.ndarray
    ):
        """Adds context, reward, and action to buffers with eviction logic."""
        if context is None or mean_action is None:
            return  # Don't add if context/action aren't set

        self.context_buffer.append(context)
        self.context_rewards.append(reward)
        self.mean_action_buffer.append(mean_action)

        if len(self.context_buffer) > self.context_config.context_buffer_size:
            if (
                self.context_config.eviction_strategy == "hybrid"
                and self.np_random.random() < self.context_config.evict_best_prob
            ):
                idx_to_remove = np.argmax(self.context_rewards)  # Evict highest reward
            else:
                idx_to_remove = 0  # Evict oldest (FIFO)

            self.context_buffer.pop(idx_to_remove)
            self.context_rewards.pop(idx_to_remove)
            self.mean_action_buffer.pop(idx_to_remove)

    @abstractmethod
    def sample_context(self) -> Dict[str, Any]:
        """Samples a context for the environment to reset to.
        This method should be overridden by the subclass.
        The output should be a dictionary with keys corresponding to environment parameters
        that can be updated at each reset. The keys should be environment attributes, or it can also be "circuit_choice"
        if there are multiple circuit choices in the Target of the environment. There can also be a "parameters" key,
        which is a dictionary of Qiskit Parameters that can be bound to the parametrized circuit context.
        """
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Resets the environment with a sampled context."""
        # Sample a new context and store it for the upcoming episode
        self.current_context = self.sample_context() if options is None else options

        # Reset the underlying environment, passing the specific angles
        return self.env.reset(options=self.current_context, seed=seed)

    def step(self, action):
        """Takes a step and logs the result to the context buffer."""
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # The reward is of shape (batch_size, 1), we take the mean
        mean_reward = np.mean(reward)

        # Add the experience from this step to the buffer
        self._add_to_buffer(self.current_context, mean_reward, self.env.mean_action)

        # If the episode ends, clear the context. A new one will be sampled on reset.
        if terminated or truncated:
            self.current_context = None

        return next_obs, reward, terminated, truncated, info
