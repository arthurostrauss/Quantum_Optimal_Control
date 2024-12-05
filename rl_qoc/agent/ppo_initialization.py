import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from gymnasium import Wrapper
from .agent import ActorNetwork, CriticNetwork, Agent
from ..environment.base_q_env import BaseQuantumEnvironment
from torch.utils.tensorboard import SummaryWriter


def get_module_from_str(module_str):
    """
    Converts a string representation of a module to the corresponding PyTorch module class.

    Args:
        module_str (str): The string representation of the module.

    Returns:
        torch.nn.Module: The PyTorch module class corresponding to the input string.

    Raises:
        ValueError: If the input string does not match any of the supported module names.
    """
    module_dict = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
        "none": nn.ReLU,
        "softmax": nn.Softmax,
        "log_softmax": nn.LogSoftmax,
        "gelu": nn.GELU,
    }
    if module_str not in module_dict:
        raise ValueError(
            f"Agent Config `ACTIVATION` needs to be one of {module_dict.keys()}"
        )
    return module_dict[module_str]


def get_optimizer_from_str(optim_str):
    """
    Returns the optimizer class corresponding to the given optimizer string.

    Args:
        optim_str (str): The optimizer string.

    Returns:
        torch.optim.Optimizer: The optimizer class.

    Raises:
        ValueError: If the optimizer string is not valid.
    """
    optim_dict = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "rmsprop": optim.RMSprop,
        "rprop": optim.Rprop,
        "sgd": optim.SGD,
    }
    if optim_str not in optim_dict:
        raise ValueError(
            f"Agent Config `OPTIMIZER` needs to be one of {optim_dict.keys()}"
        )

    return optim_dict[optim_str]


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


def initialize_agent_config(agent_config, batchsize):
    """
    Initializes the agent configuration.

    Args:
        agent_config (dict): A dictionary containing the agent configuration parameters.
        batchsize (int): The batch size.

    Returns:
        tuple: A tuple containing the following elements:
            - hidden_units (list): A list of hidden units.
            - activation_functions (list): A list of activation functions.
            - include_critic (bool): A boolean indicating whether to include the critic.
            - minibatch_size (int): The minibatch size.
            - writer (SummaryWriter): A SummaryWriter object for writing TensorBoard logs.
    """
    hidden_units = agent_config["N_UNITS"]
    activation_fn = agent_config["ACTIVATION"]
    activation_functions = [
        get_module_from_str(activation_fn)() for _ in range(len(hidden_units) + 1)
    ]
    include_critic = agent_config["INCLUDE_CRITIC"]
    minibatch_size = agent_config["MINIBATCH_SIZE"]

    if batchsize % minibatch_size != 0:
        raise ValueError(
            f"The current minibatch size of {minibatch_size} does not evenly divide the batch size of {batchsize}"
        )

    return hidden_units, activation_functions, include_critic, minibatch_size


def initialize_rl_params(agent_config):
    """
    Initializes the parameters for the reinforcement learning agent.

    Args:
        agent_config (dict): A dictionary containing the configuration parameters for the agent.

    Returns:
        tuple: A tuple containing the initialized parameters for the agent.

    """
    n_epochs = agent_config["N_EPOCHS"]
    lr = agent_config["LR"]
    ppo_epsilon = agent_config["CLIP_RATIO"]
    critic_loss_coef = agent_config["V_COEF"]
    gamma = agent_config["GAMMA"]
    gae_lambda = agent_config["GAE_LAMBDA"]
    clip_vloss = agent_config["CLIP_VALUE_LOSS"]
    grad_clip = agent_config["GRADIENT_CLIP"]
    clip_coef = agent_config["CLIP_VALUE_COEF"]
    normalize_advantage = agent_config["NORMALIZE_ADVANTAGE"]
    ent_coef = agent_config["ENT_COEF"]

    return (
        n_epochs,
        lr,
        ppo_epsilon,
        critic_loss_coef,
        gamma,
        gae_lambda,
        clip_vloss,
        grad_clip,
        clip_coef,
        normalize_advantage,
        ent_coef,
    )


def initialize_networks(
    observation_space,
    hidden_units,
    n_actions,
    activation_functions,
    include_critic,
    chkpt_dir,
    chkpt_dir_critic,
):
    """
    Initializes the actor and critic networks for the Proximal Policy Optimization (PPO) algorithm.

    Args:
        observation_space (gym.spaces): Observation space of the environment.
        hidden_units (list): A list of integers specifying the number of units in each hidden layer of the networks.
        n_actions (int): The number of possible actions in the environment.
        activation_functions (list): A list of activation functions to be used in the networks.
        include_critic (bool): Whether to include a critic network or not.
        chkpt_dir (str): The directory where the actor network checkpoints will be saved.
        chkpt_dir_critic (str): The directory where the critic network checkpoints will be saved.

    Returns:
        agent (Agent): An instance of the Agent class that encapsulates the actor and critic networks.

    """
    actor_net = ActorNetwork(
        observation_space,
        hidden_units,
        n_actions,
        activation_functions,
        include_critic,
        chkpt_dir,
    )
    critic_net = CriticNetwork(
        observation_space, hidden_units, activation_functions, chkpt_dir_critic
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
    optimizer = get_optimizer_from_str(optim_name)(
        agent.parameters(), lr=agent_config["LR"], eps=optim_eps
    )

    return optimizer
