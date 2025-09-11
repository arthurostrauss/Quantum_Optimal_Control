from typing import Sequence, Optional

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from .ppo_utils import get_module
from numpy import sqrt


class ActorNetwork(nn.Module):
    def __init__(
        self,
        observation_space: Box,
        hidden_layers: Sequence[int],
        n_actions: int,
        input_activation_function: nn.Module | str = "identity",
        hidden_activation_functions: Sequence[nn.Module | str] | nn.Module | str = "tanh",
        output_activation_mean: nn.Module | str = "tanh",
        output_activation_std: Optional[nn.Module | str] = None,
        include_critic=True,
        chkpt_dir: str = "tmp/agent",
    ):
        """
        Initialize the Actor Network
        :param observation_space: Observation space of the environment
        :param hidden_layers: List of sizes of the hidden layers
        :param n_actions: Number of actions
        :param input_activation_function: Activation function for the input layer
        :param hidden_activation_functions: Activation functions for the hidden layers
        :param output_activation_mean: Activation function for the mean output
        :param output_activation_std: Activation function for the standard deviation output (if None, std is a parameter)
        :param include_critic: Whether to include a critic network
        :param chkpt_dir: Directory to save the checkpoint
        """
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        # Define a list to hold the layer sizes including input and output sizes
        input_activation_function = get_module(input_activation_function)
        output_activation_mean = get_module(output_activation_mean)
        input_size = observation_space.shape[0]  # TODO: Check if it's not shape[-1] that we actually need
        hidden_sizes = list(hidden_layers)

        if isinstance(hidden_activation_functions, nn.Module) or isinstance(
            hidden_activation_functions, str
        ):
            hidden_activation_functions = [
                get_module(hidden_activation_functions) for _ in range(len(hidden_sizes))
            ]
        else:
            if len(hidden_activation_functions) != len(hidden_sizes):
                raise ValueError(
                    "Number of hidden layers and hidden activation functions must be the same"
                )
            hidden_activation_functions = [
                get_module(activation) for activation in hidden_activation_functions
            ]

        # Define a list to hold the layers of the network
        layers = []
        # Handle the input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(input_activation_function)

        # Handle the hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(hidden_activation_functions[i])

        # Create the actor network using Sequential container
        self.base_layers = layers
        self.mean_action = nn.Linear(hidden_layers[-1], n_actions)
        self.mean_activation = get_module(output_activation_mean)
        if output_activation_std is not None:
            self.std_action = nn.Linear(hidden_layers[-1], n_actions)
            self.std_activation = get_module(output_activation_std)
        else:
            self.std_action = nn.Parameter(torch.zeros(1, n_actions))
            self.std_activation = None

        self.include_critic = include_critic
        self.critic_output = nn.Linear(hidden_layers[-1], 1)

        self.base_network = nn.Sequential(*layers)

        # Initialize the weights of the network
        for layer in self.base_network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.base_network(x)
        mean_action = self.mean_action(x)
        mean_action = self.mean_activation(mean_action)
        if self.std_activation is not None:
            std_action = self.std_action(x)
            std_action = self.std_activation(std_action)
        else:
            std_action = self.std_action.expand_as(mean_action)
            std_action = torch.exp(std_action)  # log std -> std

        critic_output = self.critic_output(x)

        if self.include_critic:
            return mean_action, std_action, critic_output
        else:
            return mean_action, std_action

    def get_value(self, x):
        x = self.base_network(x)
        x = self.critic_output(x)
        return x

    def save_checkpoint(self):
        torch.save(self, self.checkpoint_dir)

    def load_checkpoint(self):
        torch.load(self.checkpoint_dir)

    def to(self, device):
        self.base_network.to(device)
        self.mean_action.to(device)
        self.critic_output.to(device)
        if self.std_activation is not None:
            self.std_action.to(device)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        observation_space: Box,
        hidden_layers: Sequence[int],
        input_activation_function: nn.Module | str = "identity",
        hidden_activation_functions: Sequence[nn.Module | str] | nn.Module | str = "tanh",
        chkpt_dir: str = "tmp/critic_ppo",
    ):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir

        # Define a list to hold the layer sizes including input and output sizes
        input_activation_function = get_module(input_activation_function)
        input_size = observation_space.shape[
            0
        ]  # TODO: Check if it's not shape[-1] that we actually need
        hidden_sizes = list(hidden_layers)

        if isinstance(hidden_activation_functions, nn.Module) or isinstance(
            hidden_activation_functions, str
        ):
            hidden_activation_functions = [
                get_module(hidden_activation_functions) for _ in range(len(hidden_sizes))
            ]
        else:
            if len(hidden_activation_functions) != len(hidden_sizes):
                raise ValueError(
                    "Number of hidden layers and hidden activation functions must be the same"
                )
            hidden_activation_functions = [
                get_module(activation) for activation in hidden_activation_functions
            ]

        # Define a list to hold the layers of the network
        layers = []
        # Handle the input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(input_activation_function)

        # Handle the hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(hidden_activation_functions[i])

        # Create the actor network using Sequential container
        self.base_layers = layers
        self.critic_output = nn.Linear(hidden_layers[-1], 1)
        self.base_layers.append(self.critic_output)
        self.critic_network = nn.Sequential(*self.base_layers)

        # Initialize the weights of the network
        for layer in self.critic_network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.critic_network(x)

    def save_checkpoint(self):
        torch.save(self, self.checkpoint_dir)

    def load_checkpoint(self):
        torch.load(self.checkpoint_dir)

    def to(self, device):
        self.critic_network.to(device)


class Agent(nn.Module):
    def __init__(self, actor_net: ActorNetwork, critic_net: Optional[CriticNetwork] = None):
        super().__init__()

        self.actor_net = actor_net
        self.critic_net = critic_net

        if self.critic_net is not None:
            assert not self.actor_net.include_critic, "Critic already included in Actor Network"

    def forward(self, x):
        if self.actor_net.include_critic:
            return self.actor_net(x)
        else:
            assert (
                self.critic_net is not None
            ), "Critic Network not provided and not included in ActorNetwork"
            mean_action, std_action = self.actor_net(x)
            value = self.critic_net(x)
            return mean_action, std_action, value

    def get_value(self, x):
        if self.actor_net.include_critic:
            return self.actor_net.get_value(x)
        else:
            assert (
                self.critic_net is not None
            ), "Critic Network not provided and not included in ActorNetwork"
            return self.critic_net(x)

    def save_checkpoint(self):
        self.actor_net.save_checkpoint()
        if self.critic_net is not None:
            self.critic_net.save_checkpoint()

    def to(self, device):
        self.actor_net.to(device)
        if self.critic_net is not None:
            self.critic_net.to(device)
