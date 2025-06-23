from __future__ import annotations
import numpy as np
from typing import Optional, Dict
from gymnasium import Wrapper, ActionWrapper

# Torch imports for building RL agent and framework
import torch
from torch.distributions import Normal

from ..agent import CustomPPO, PPOConfig
from ..environment.base_q_env import BaseQuantumEnvironment as QuantumEnvironment
from qiskit_qm_provider import FixedPoint

import sys
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def lcg(seed, a, c, m):
    """Linear Congruential Generator (32-bit version with wrap-around)."""
    seed = np.int32(seed)
    a = np.int32(a)
    c = np.int32(c)
    return (a * seed + c) % m


def lcg_int(seed, a, c, m, upper_bound):
    """Generates a random integer between 0 and upper_bound-1 (inclusive)."""
    seed = lcg(seed, a, c, m)
    return np.int32(seed / 2**28 * upper_bound), seed  # Return the updated seed


def lcg_fixed_point(seed, a, c, m):
    """Generates a 4.28 fixed-point number between 0 and 1."""
    seed = lcg(seed, a, c, m)
    # Extract fractional part (assuming 4.28 fixed-point format)
    fractional_part = seed & 0x0FFFFFFF  # Mask to keep only the lower 28 bits
    fixed_point_value = fractional_part / float(2**28)
    return fixed_point_value, seed  # Return the updated seed


def lcg_int_sequence(seed, a, c, m, n, upper_bound):
    """Generates a sequence of n random integers between 0 and upper_bound-1 (inclusive)."""
    sequence = []
    for _ in range(n):
        int_value, seed = lcg_int(seed, a, c, m, upper_bound)  # Update seed
        sequence.append(int_value)
    return sequence


def lcg_fixed_point_sequence(seed, a, c, m, n):
    """Generates a sequence of n 4.28 fixed-point numbers between 0 and 1."""
    sequence = []
    for _ in range(n):
        fixed_point_value, seed = lcg_fixed_point(seed, a, c, m)  # Update seed
        sequence.append(fixed_point_value)
    return sequence


a = 137939405
c = 12345
m = 2**28
rounding = 8


class CustomQMPPO(CustomPPO):
    def __init__(
        self,
        agent_config: Dict | PPOConfig,
        env: QuantumEnvironment | Wrapper,
        chkpt_dir: Optional[str] = "tmp/agent",
        chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
    ):
        """
        Initializes the PPO algorithm with the given hyperparameters
        :param agent_config: Dictionary containing all the hyperparameters for the PPO algorithm
        :param env: Quantum Environment on which the algorithm is trained
        :param chkpt_dir: Directory where the policy network is saved
        :param chkpt_dir_critic: Directory where the critic network is saved

        """

        super().__init__(agent_config, env, chkpt_dir, chkpt_dir_critic)

    def process_action(self, probs: Normal):
        """
        Process the action to be taken by the agent
        :param mean_action: Mean of the action distribution
        :param std_action: Standard deviation of the action distribution
        :param probs: Probabilities of the action distribution
        :return: The action to be taken by the agent
        """
        mean_action = probs.mean
        std_action = probs.stddev
        batch_size = mean_action.size(0)
        if isinstance(self.env, ActionWrapper):
            self.unwrapped_env.mean_action = self.env.action(mean_action[0].cpu().numpy())
        else:
            self.unwrapped_env.mean_action = mean_action[0].cpu().numpy()
        self.unwrapped_env.std_action = std_action[0].cpu().numpy()
        μ = self.unwrapped_env.mean_action
        σ = std_action[0].cpu().numpy()
        μ_f = [FixedPoint(μ[i]) for i in range(self.n_actions)]
        σ_f = [FixedPoint(σ[i]) for i in range(self.n_actions)]
        action = np.zeros((batch_size, self.n_actions))
        seed = self.seed + self.global_step - 1
        n_lookup = 512
        cos_array = [FixedPoint(np.cos(2 * np.pi * x / n_lookup)) for x in range(n_lookup)]
        ln_array = [
            FixedPoint(np.sqrt(-2 * np.log(x / (n_lookup + 1)))) for x in range(1, n_lookup + 1)
        ]
        for b in range(0, batch_size, 2):
            for j in range(self.n_actions):
                uniform_sample, seed = lcg_fixed_point(seed, a, c, m)
                uniform_sample = FixedPoint(uniform_sample)
                u1 = (uniform_sample >> 19).to_unsafe_int()
                u2 = uniform_sample.to_unsafe_int() & ((1 << 19) - 1)
                temp_action1 = μ_f[j] + σ_f[j] * ln_array[u1] * cos_array[u2 & (n_lookup - 1)]
                temp_action2 = (
                    μ_f[j]
                    + σ_f[j] * ln_array[u1] * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)]
                )
                action[b][j] = temp_action1.to_float()
                action[b + 1][j] = temp_action2.to_float()

        action = np.clip(action, self.min_action, self.max_action)
        torch_action = torch.tensor(action, device=self.device)
        logprob = probs.log_prob(torch_action).sum(1)
        return torch_action, logprob

    def post_process_action(self, probs: Normal, action: torch.Tensor, logprob: torch.Tensor):
        """
        Post-process the action taken by the agent
        :param probs: Probabilities of the action distribution
        :param action: Action taken by the agent
        :param logprob: Log probabilities of the action distribution
        :return: The action to be taken by the agent
        """

        return action, logprob
