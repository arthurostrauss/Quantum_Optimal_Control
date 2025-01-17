from __future__ import annotations

import time
import numpy as np
from typing import Optional, Dict
import tqdm
import warnings
from IPython.display import clear_output
from gymnasium import Wrapper

# Torch imports for building RL agent and framework
from gymnasium.spaces import Box
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.distributions import Normal
from ..agent import CustomPPO
from ..environment import QuantumEnvironment

import sys
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


class CustomQMPPO(CustomPPO):
    def __init__(
        self,
        agent_config: Dict,
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

    def process_action(self, mean_action, std_action, probs):
        """
        Process the action to be taken by the agent
        :param mean_action: Mean of the action distribution
        :param std_action: Standard deviation of the action distribution
        :param probs: Probabilities of the action distribution
        :return: The action to be taken by the agent
        """

        action = torch.normal(mean_action, std_action)
        action = torch.clamp(
            action, self.env.action_space.low[0], self.env.action_space.high[0]
        )
        return action

    def post_process_action(
        self, probs: Normal, action: torch.Tensor, logprob: torch.Tensor
    ):
        """
        Post-process the action taken by the agent
        :param probs: Probabilities of the action distribution
        :param action: Action taken by the agent
        :param logprob: Log probabilities of the action distribution
        :return: The action to be taken by the agent
        """

        return action, logprob
