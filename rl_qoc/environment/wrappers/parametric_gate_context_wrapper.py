from .context_sampling_wrapper import ContextSamplingWrapper
from gymnasium.spaces import Box, Dict as DictSpace
from typing import Dict, Any
from ..context_aware_quantum_environment import ContextAwareQuantumEnvironment
import numpy as np


class ParametricGateContextWrapper(ContextSamplingWrapper):
    """
    A Gymnasium wrapper that implements an observation space consisting of the parameters of the original
    parametric gate to be calibrated in a quantum environment.
    This wrapper is designed to work with environments that are instances of `ContextAwareQuantumEnvironment`
    and have a `target` attribute of type `GateTarget`.
    """

    @property
    def observation_space(self) -> DictSpace:
        """
        Return the observation space of the environment.
        The observation space is a dictionary where each key corresponds to a parameter name
        and the value is a Box space representing the range of that parameter.
        """
        return DictSpace(
            {
                p.name: Box(low=0.0, high=np.pi, shape=(1,))
                for p in self.env.target.context_parameters.keys()
            }
        )

    def sample_context(self) -> Dict[str, Any]:
        """
        Sample a new context by sampling new values for each parameter in the target circuit.
        :return: A dictionary mapping parameter names to their sampled values.
        """
        is_warmup = self.env.step_tracker <= self.context_config.num_warmup_updates
        context = {key: np.zeros(space.shape) for key, space in self.observation_space.items()}
        if is_warmup or len(self.context_buffer) == 0:
            for key, space in self.observation_space.items():
                context[key] = self.env.np_random.uniform(space.low, space.high, size=space.shape)

            return {"parameters": context}

        if self.np_random.random() < self.context_config.sampling_prob:
            # Rank-based prioritized replay
            rewards = np.array(self.context_rewards)
            ranks = np.argsort(np.argsort(rewards)) + 1  # Rank starts from 1 (lowest reward)
            prob_weights = 1.0 / ranks
            prob_weights /= prob_weights.sum()

            idx = self.np_random.choice(len(self.context_buffer), p=prob_weights)
            context = self.context_buffer[idx]["parameters"]

            # Anneal noise added to the sampled context
            if self.context_config.anneal_noise and self.env.total_updates is not None:
                progress = (
                    self.env.unwrapped.step_tracker - self.context_config.num_warmup_updates
                ) / (self.env.total_updates - self.context_config.num_warmup_updates)
                progress = np.clip(progress, 0.0, 1.0)
                current_noise_scale = (
                    self.context_config.initial_noise_scale * (1 - progress)
                    + self.context_config.final_noise_scale * progress
                )
            else:
                current_noise_scale = self.context_config.initial_noise_scale

            noise = {
                key: self.np_random.normal(0, current_noise_scale, size=space.shape)
                for key, space in self.observation_space.items()
            }
            for key, val in noise.items():
                context[key] = np.clip(
                    context[key] + val,
                    self.observation_space[key].low,
                    self.observation_space[key].high,
                )

        return {"parameters": context}
