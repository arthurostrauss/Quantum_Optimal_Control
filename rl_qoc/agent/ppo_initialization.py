from __future__ import annotations

from typing import List

import torch.nn as nn
from gymnasium.spaces import Box
from gymnasium import Wrapper
from .agent import ActorNetwork, CriticNetwork, Agent
from .ppo_utils import *
from ..environment.base_q_env import BaseQuantumEnvironment


def initialize_environment(env: BaseQuantumEnvironment | Wrapper) -> tuple:
    """
    Initializes the environment by extracting necessary information.

    Args:
        env: The environment object.

    Returns:
        A tuple containing the following information:
        - seed: The seed value of the environment.
        - min_action: The minimum action value in the environment.
        - max_action: The maximum action value in the environment.

    Raises:
        ValueError: If the environment action space is not a Box.
    """
    seed = env.unwrapped.seed

    if hasattr(env, "min_action") and hasattr(env, "max_action"):
        min_action = env.min_action
        max_action = env.max_action
    elif isinstance(env.action_space, Box):
        min_action = env.action_space.low
        max_action = env.action_space.high
    else:
        raise ValueError("Environment action space is not a Box")

    return seed, min_action, max_action


def initialize_networks(
    observation_space,
    hidden_units: List[int],
    n_actions: int,
    activation_functions: List[nn.Module | str],
    include_critic: bool,
    input_activation_function: nn.Module | str,
    output_activation_mean: nn.Module | str,
    output_activation_std: nn.Module | str,
    chkpt_dir: str,
    chkpt_dir_critic: str,
):
    """
    Initializes the actor and critic networks for the Proximal Policy Optimization (PPO) algorithm.

    Args:
        observation_space (Space): The observation space of the environment.
        hidden_units (List[int]): A list containing the number of units in each hidden layer.
        n_actions (int): The number of actions in the environment.
        activation_functions (List[nn.Module|str]): A list containing the activation functions for each hidden layer.
        include_critic (bool): A boolean flag indicating whether to include a critic network.
        input_activation_function (nn.Module|str): The activation function for the input layer.
        output_activation_mean (nn.Module|str): The activation function for the mean output layer.
        output_activation_std (nn.Module|str): The activation function for the standard deviation output layer.
        chkpt_dir (str): The directory where the actor network checkpoint is saved.
        chkpt_dir_critic (str): The directory where the critic network checkpoint is saved.

    Returns:
        agent (Agent): An instance of the Agent class that encapsulates the actor and critic networks.

    """
    actor_net = ActorNetwork(
        observation_space,
        hidden_units,
        n_actions,
        input_activation_function,
        activation_functions,
        output_activation_mean,
        output_activation_std,
        include_critic,
        chkpt_dir,
    )
    critic_net = CriticNetwork(
        observation_space,
        hidden_units,
        input_activation_function,
        activation_functions,
        chkpt_dir_critic,
    )

    if include_critic:
        agent = Agent(actor_net, critic_net=None)
    else:
        agent = Agent(actor_net, critic_net=critic_net)

    return agent


def initialize_optimizer(agent: Agent, agent_config):
    """
    Initializes the optimizer for the agent.

    Args:
        agent: The agent for which the optimizer is initialized.
        agent_config: A dictionary containing the configuration parameters for the agent.

    Returns:
        The initialized optimizer.

    """
    optim_name = agent_config["OPTIMIZER"]
    optim_eps = 1e-5
    optimizer = get_optimizer(optim_name)(
        agent.parameters(), lr=agent_config["LR"], eps=optim_eps
    )

    return optimizer
