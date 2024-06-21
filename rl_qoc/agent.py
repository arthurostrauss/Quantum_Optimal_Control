from typing import Sequence, Optional

import torch
import torch.nn as nn
from gymnasium import Space


class ActorNetwork(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        hidden_layers: Sequence[int],
        n_actions: int,
        hidden_activation_functions: Optional[Sequence[nn.Module]] = None,
        include_critic=True,
        chkpt_dir: str = "tmp/ppo",
    ):
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        # Define a list to hold the layer sizes including input and output sizes
        layer_sizes = [observation_space.shape[0]] + list(hidden_layers)
        if hidden_activation_functions is None:
            hidden_activation_functions = [nn.ReLU() for _ in range(len(layer_sizes))]

        assert len(hidden_activation_functions) == len(layer_sizes)
        # Define a list to hold the layers of the network
        layers = []

        # Iterate over the layer sizes to create the network layers
        for i in range(len(layer_sizes) - 1):
            # Add a linear layer with the current and next layer sizes
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            layers.append(hidden_activation_functions[i])

        # Create the actor network using Sequential container
        self.layers = layers
        self.mean_action = nn.Linear(hidden_layers[-1], n_actions)
        self.mean_activation = nn.Tanh()
        self.std_action = nn.Linear(hidden_layers[-1], n_actions)
        self.std_activation = nn.Sigmoid()

        self.include_critic = include_critic
        self.critic_output = nn.Linear(hidden_layers[-1], 1)

        self.base_network = nn.Sequential(*layers)

        # Initialize the weights of the network
        for layer in self.base_network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.base_network(x)
        mean_action = self.mean_action(x)
        mean_action = self.mean_activation(mean_action)
        std_action = self.std_action(x)
        std_action = self.std_activation(std_action)
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


class CriticNetwork(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        hidden_layers: Sequence[int],
        hidden_activation_functions: Optional[Sequence[nn.Module]] = None,
        chkpt_dir: str = "tmp/critic_ppo",
    ):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        # Define a list to hold the layer sizes including input and output sizes
        layer_sizes = [observation_space.shape[0]] + list(hidden_layers)
        if hidden_activation_functions is None:
            hidden_activation_functions = [nn.ReLU() for _ in range(len(layer_sizes))]

        assert len(hidden_activation_functions) == len(layer_sizes)
        # Define a list to hold the layers of the network
        layers = []

        # Iterate over the layer sizes to create the network layers
        for i in range(len(layer_sizes) - 1):
            # Add a linear layer with the current and next layer sizes
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add a ReLU activation function for all layers except the output layer

            layers.append(hidden_activation_functions[i])

        # Create the actor network using Sequential container
        self.layers = layers
        self.critic_output = nn.Linear(hidden_layers[-1], 1)
        self.layers.append(self.critic_output)
        self.critic_network = nn.Sequential(*self.layers)

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


class Agent(nn.Module):
    def __init__(
        self, actor_net: ActorNetwork, critic_net: Optional[CriticNetwork] = None
    ):
        super().__init__()

        self.actor_net = actor_net
        self.critic_net = critic_net

        if self.critic_net is not None:
            assert (
                not self.actor_net.include_critic
            ), "Critic already included in Actor Network"

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
