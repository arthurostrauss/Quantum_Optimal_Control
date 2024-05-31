from __future__ import annotations

import time
import numpy as np
from typing import Optional, Dict, List
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
from agent import ActorNetwork, CriticNetwork, Agent
from quantumenvironment import QuantumEnvironment

import sys
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


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


def initialize_environment(env):
    """
    Initializes the environment by extracting necessary information.

    Args:
        env: The environment object.

    Returns:
        A tuple containing the following information:
        - seed: The seed value of the environment.
        - n_actions: The number of actions in the environment.
        - batchsize: The batch size of the environment.
        - num_time_steps: The number of time steps in the environment.
        - min_action: The minimum action value in the environment.
        - max_action: The maximum action value in the environment.

    Raises:
        ValueError: If the environment action space is not a Box.
    """
    seed = env.unwrapped.seed
    n_actions = env.action_space.shape[-1]
    batchsize = env.unwrapped.batch_size
    num_time_steps = env.unwrapped.tgt_instruction_counts

    if hasattr(env, "min_action") and hasattr(env, "max_action"):
        min_action = env.min_action
        max_action = env.max_action
    elif isinstance(env.action_space, Box):
        min_action = env.action_space.low
        max_action = env.action_space.high
    else:
        raise ValueError("Environment action space is not a Box")

    return seed, n_actions, batchsize, num_time_steps, min_action, max_action


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
            f"The current minibatch size of {minibatch_size} does not evenly divide the batchsize of {batchsize}"
        )

    run_name = agent_config["RUN_NAME"]
    writer = SummaryWriter(f"runs/{run_name}")

    return hidden_units, activation_functions, include_critic, minibatch_size, writer


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
    env,
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
        env (gym.Env): The environment for which the networks are being initialized.
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
        env.observation_space,
        hidden_units,
        n_actions,
        activation_functions,
        include_critic,
        chkpt_dir,
    )
    critic_net = CriticNetwork(
        env.observation_space, hidden_units, activation_functions, chkpt_dir_critic
    )

    if include_critic:
        agent = Agent(actor_net, critic_net=None)
    else:
        agent = Agent(actor_net, critic_net=critic_net)

    return agent


def initialize_optimizer(agent, agent_config):
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


def plot_curves(env):
    """
    Plots the reward history and fidelity history of the environment
    """
    fidelity_range = [
        i * env.unwrapped.benchmark_cycle
        for i in range(len(env.unwrapped.fidelity_history))
    ]
    plt.plot(np.mean(env.reward_history, axis=1) ** (1 / env.n_reps), label="Reward")
    plt.plot(
        fidelity_range,
        env.fidelity_history,
        label="Circuit Fidelity",
    )

    plt.title("Reward History")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.show()


def get_empty_tensors(env, num_time_steps, batchsize):
    """
    Create empty tensors for observations, actions, log probabilities, rewards, dones, values,
    average reward, fidelities, standard actions, and average action history.
    """
    obs = torch.zeros((num_time_steps, batchsize) + env.observation_space.shape)
    actions = torch.zeros((num_time_steps, batchsize) + env.action_space.shape)
    logprobs = torch.zeros((num_time_steps, batchsize))
    rewards = torch.zeros((num_time_steps, batchsize))
    dones = torch.zeros((num_time_steps, batchsize))
    values = torch.zeros((num_time_steps, batchsize))

    avg_reward = []
    fidelities = []
    std_actions = []
    avg_action_history = []
    return (
        obs,
        actions,
        logprobs,
        rewards,
        dones,
        values,
        avg_reward,
        fidelities,
        std_actions,
        avg_action_history,
    )


def reset_env(env, seed, global_step, batchsize, dones):
    """
    Resets the environment and returns the initial observation, number of steps, batch observations, and batch done flags.

    Args:
        env (object): The environment object.
        seed (int): The seed for the environment reset.
        global_step (int): The global step count.
        batchsize (int): The size of the batch.
        dones (list): A list of done flags for each environment instance.

    Returns:
        next_obs (object): The initial observation after resetting the environment.
        num_steps (int): The number of steps in the episode.
        batch_obs (object): The batch observations.
        batch_done (object): The batch done flags.
    """
    next_obs, _ = env.reset(seed=seed)
    num_steps = env.unwrapped.episode_length(global_step)  # num_time_steps
    batch_obs = torch.tile(torch.Tensor(next_obs), (batchsize, 1))
    batch_done = torch.zeros_like(dones[0])
    return next_obs, num_steps, batch_obs, batch_done


def take_step(
    step,
    global_step,
    batchsize,
    num_steps,
    obs,
    dones,
    actions,
    logprobs,
    rewards,
    values,
    batch_obs,
    batch_done,
    min_action,
    max_action,
    agent,
    env,
    writer,
):
    """
    Takes a step in the environment using the PPO algorithm.

    Args:
        step (int): The current step index.
        global_step (int): The global step index.
        batchsize (int): The size of the batch.
        num_steps (int): The total number of steps.
        obs (list): List to store the observations.
        dones (list): List to store the done flags.
        actions (list): List to store the actions.
        logprobs (list): List to store the log probabilities.
        rewards (list): List to store the rewards.
        values (list): List to store the critic values.
        batch_obs (torch.Tensor): Batch of observations.
        batch_done (torch.Tensor): Batch of done flags.
        min_action (list): List of minimum action values.
        max_action (list): List of maximum action values.
        agent (object): The agent model.
        env (object): The environment.
        writer (object): The writer object for logging.

    Returns:
        torch.Tensor: The next observations.
        torch.Tensor: The next done flags.
        torch.Tensor: The mean action.
        torch.Tensor: The standard deviation of the action distribution.
    """
    global_step += 1
    obs[step] = batch_obs
    dones[step] = batch_done

    with torch.no_grad():
        mean_action, std_action, critic_value = agent(batch_obs)
        probs = Normal(mean_action, std_action)
        action = torch.clip(
            probs.sample(),
            torch.Tensor(min_action),
            torch.Tensor(max_action),
        )
        logprob = probs.log_prob(action).sum(1)
        values[step] = critic_value.flatten()

    actions[step] = action
    logprobs[step] = logprob

    next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
    next_obs = torch.Tensor(next_obs)
    done = int(np.logical_or(terminated, truncated))
    reward = torch.Tensor(reward)
    rewards[step] = reward

    batch_obs = torch.tile(next_obs, (batchsize, 1))
    next_done = done * torch.ones_like(dones[0])
    obs[step] = batch_obs
    dones[step] = next_done

    # print(f"global_step={global_step}, episodic_return={np.mean(reward)}")
    writer.add_scalar("charts/episodic_return", np.mean(reward.numpy()), global_step)
    writer.add_scalar("charts/episodic_length", num_steps, global_step)

    return next_obs, next_done, mean_action, std_action


def do_bootstrap(
    next_obs, next_done, num_steps, rewards, dones, values, gamma, gae_lambda, agent
):
    """
    Calculates advantages and returns for the Proximal Policy Optimization (PPO) algorithm.

    Args:
        next_obs (torch.Tensor): The next observation tensor.
        next_done (torch.Tensor): The tensor indicating whether the next state is terminal.
        num_steps (int): The number of steps in the trajectory.
        rewards (torch.Tensor): The tensor containing the rewards for each step.
        dones (torch.Tensor): The tensor indicating whether each step is terminal.
        values (torch.Tensor): The tensor containing the predicted values for each step.
        gamma (float): The discount factor.
        gae_lambda (float): The Generalized Advantage Estimation (GAE) lambda parameter.
        agent (Agent): The agent used for value prediction.

    Returns:
        torch.Tensor: The advantages for each step.
        torch.Tensor: The estimated returns for each step.
    """
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
    return advantages, returns


def flatten_batch(env, obs, logprobs, actions, advantages, returns, values):
    """
    Flattens the batch of observations, log probabilities, actions, advantages, returns, and values.
    """
    b_obs = obs.reshape((-1,) + env.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + env.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values


def optimize_policy_and_value_network(
    b_obs,
    b_logprobs,
    b_actions,
    b_advantages,
    b_returns,
    b_values,
    b_inds,
    start,
    minibatch_size,
    agent,
    optimizer,
    ppo_epsilon,
    ent_coef,
    critic_loss_coef,
    clip_vloss,
    clip_coef,
    grad_clip,
    normalize_advantage,
    clipfracs,
):
    """
    Optimizes the policy and value network using Proximal Policy Optimization (PPO) algorithm.

    Args:
        b_obs (tensor): Batch of observations.
        b_logprobs (tensor): Batch of log probabilities of actions.
        b_actions (tensor): Batch of actions.
        b_advantages (tensor): Batch of advantages.
        b_returns (tensor): Batch of returns.
        b_values (tensor): Batch of predicted values.
        b_inds (tensor): Batch indices.
        start (int): Start index of the minibatch.
        minibatch_size (int): Size of the minibatch.
        agent (object): Policy and value network agent.
        optimizer (object): Optimizer for updating the agent's parameters.
        ppo_epsilon (float): PPO epsilon value for clipping the ratio.
        ent_coef (float): Coefficient for the entropy loss.
        critic_loss_coef (float): Coefficient for the value loss.
        clip_vloss (bool): Whether to clip the value loss.
        clip_coef (float): Coefficient for clipping the value loss.
        grad_clip (float): Maximum gradient norm for clipping gradients.
        normalize_advantage (bool): Whether to normalize the advantages.
        clipfracs (list): List to store the clipping fractions.

    Returns:
        tuple: Tuple containing the policy loss, entropy loss, value loss, approximate KL divergence,
               old approximate KL divergence, and the list of clipping fractions.
    """
    end = start + minibatch_size
    mb_inds = b_inds[start:end]
    new_mean, new_sigma, new_value = agent(b_obs[mb_inds])
    new_dist = Normal(new_mean, new_sigma)
    new_logprob, entropy = new_dist.log_prob(b_actions[mb_inds]).sum(
        1
    ), new_dist.entropy().sum(1)
    logratio = new_logprob - b_logprobs[mb_inds]
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs += [((ratio - 1.0).abs() > ppo_epsilon).float().mean().item()]

    mb_advantages = b_advantages[mb_inds]
    if normalize_advantage:  # Normalize advantage
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - ppo_epsilon, 1 + ppo_epsilon)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = new_value.view(-1)
    if clip_vloss:
        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds],
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * critic_loss_coef

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
    optimizer.step()

    return pg_loss, entropy_loss, v_loss, approx_kl, old_approx_kl, clipfracs


def print_debug_info(env, mean_action, std_action, b_returns, b_advantages):
    """
    Print debug information for the training process.
    """
    print("mean", mean_action[0])
    print("sigma", std_action[0])
    print(
        "DFE Rewards Mean:",
        np.mean(env.unwrapped.reward_history, axis=1)[-1],
    )
    print(
        "DFE Rewards standard dev",
        np.std(env.unwrapped.reward_history, axis=1)[-1],
    )
    print("Returns Mean:", np.mean(b_returns.numpy()))
    print("Returns standard dev:", np.std(b_returns.numpy()))
    print("Advantages Mean:", np.mean(b_advantages.numpy()))
    print("Advantages standard dev", np.std(b_advantages.numpy()))


def write_to_tensorboard(
    writer,
    global_step,
    optimizer,
    v_loss,
    env,
    clipfracs,
    entropy_loss,
    old_approx_kl,
    approx_kl,
    pg_loss,
    explained_var,
):
    """
    Writes various metrics and losses to TensorBoard.
    """
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar(
        "losses/avg_return",
        np.mean(env.unwrapped.reward_history, axis=1)[-1],
        global_step,
    )
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)


def check_convergence_std_actions(std_action, std_actions_eps):
    """
    Check if the standard deviation of actions has converged to a specified value.

    Args:
        std_action (Tensor): The current standard deviation of actions.
        std_actions_eps (float): The desired value for the standard deviation of actions.

    Returns:
        bool: True if the standard deviation of actions has converged to the desired value, False otherwise.
    """
    if np.allclose(np.mean(std_action.numpy()), std_actions_eps):
        logging.warning(
            "Standard deviation of actions converged to {}".format(std_actions_eps)
        )
        logging.warning("Stop training")
        return True
    return False


def perform_training_iteration(
    env,
    seed,
    agent,
    n_epochs,
    optimizer,
    writer,
    global_step,
    batchsize,
    minibatch_size,
    dones,
    obs,
    actions,
    logprobs,
    rewards,
    values,
    min_action,
    max_action,
    gamma,
    gae_lambda,
    ppo_epsilon,
    ent_coef,
    critic_loss_coef,
    clip_vloss,
    clip_coef,
    grad_clip,
    normalize_advantage,
    num_prints,
    print_debug,
    plot_real_time,
):
    """
    Perform a single training iteration of the Proximal Policy Optimization (PPO) algorithm.

    Args:
        env: The environment used for training.
        seed: The random seed for reproducibility.
        agent: The agent used for interaction with the environment.
        n_epochs: The number of optimization epochs.
        optimizer: The optimizer used for updating the agent's policy and value networks.
        writer: The writer object for logging to TensorBoard.
        global_step: The global step counter.
        batchsize: The size of the training batch.
        minibatch_size: The size of each minibatch used for optimization.
        dones: The list of done flags indicating episode terminations.
        obs: The observations collected during the training iteration.
        actions: The actions taken during the training iteration.
        logprobs: The log probabilities of the actions taken.
        rewards: The rewards received during the training iteration.
        values: The estimated values of the observations.
        min_action: The minimum value of the action space.
        max_action: The maximum value of the action space.
        gamma: The discount factor for future rewards.
        gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
        ppo_epsilon: The clipping parameter for PPO.
        ent_coef: The coefficient for the entropy loss term.
        critic_loss_coef: The coefficient for the critic loss term.
        clip_vloss: Whether to clip the value loss during optimization.
        clip_coef: The coefficient for the clipping term in the PPO loss.
        grad_clip: The maximum gradient norm for gradient clipping.
        normalize_advantage: Whether to normalize the advantages during optimization.
        num_prints: The number of training iterations between printing debug information.
        print_debug: Whether to print debug information during training.
        plot_real_time: Whether to plot training curves in real time.

    Returns:
        mean_action: The mean action taken during the training iteration.
        std_action: The standard deviation of the action taken during the training iteration.
    """
    next_obs, num_steps, batch_obs, batch_done = reset_env(
        env, seed, global_step, batchsize, dones
    )

    for step in range(num_steps):
        next_obs, next_done, mean_action, std_action = take_step(
            step,
            global_step,
            batchsize,
            num_steps,
            obs,
            dones,
            actions,
            logprobs,
            rewards,
            values,
            batch_obs,
            batch_done,
            min_action,
            max_action,
            agent,
            env,
            writer,
        )

    advantages, returns = do_bootstrap(
        next_obs, next_done, num_steps, rewards, dones, values, gamma, gae_lambda, agent
    )
    b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = flatten_batch(
        env, obs, logprobs, actions, advantages, returns, values
    )

    b_inds = np.arange(batchsize)
    clipfracs = []
    for epoch in range(n_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batchsize, minibatch_size):
            (
                pg_loss,
                entropy_loss,
                v_loss,
                approx_kl,
                old_approx_kl,
                clipfracs,
            ) = optimize_policy_and_value_network(
                b_obs,
                b_logprobs,
                b_actions,
                b_advantages,
                b_returns,
                b_values,
                b_inds,
                start,
                minibatch_size,
                agent,
                optimizer,
                ppo_epsilon,
                ent_coef,
                critic_loss_coef,
                clip_vloss,
                clip_coef,
                grad_clip,
                normalize_advantage,
                clipfracs,
            )

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    if print_debug:
        print_debug_info(env, mean_action, std_action, b_returns, b_advantages)

    if global_step % num_prints == 0:
        clear_output(wait=True)
        if plot_real_time:
            plot_curves(env)

    write_to_tensorboard(
        writer,
        global_step,
        optimizer,
        v_loss,
        env,
        clipfracs,
        entropy_loss,
        old_approx_kl,
        approx_kl,
        pg_loss,
        explained_var,
    )

    return mean_action, std_action


def update_metric_lists(
    env,
    mean_action,
    std_action,
    avg_reward,
    fidelities,
    avg_action_history,
    std_actions,
):
    """
    Update the metric lists with the latest values.
    """
    avg_reward.append(np.mean(env.unwrapped.reward_history, axis=1)[-1])
    if len(env.unwrapped.fidelity_history) > 0:
        fidelities.append(env.unwrapped.fidelity_history[-1])
    avg_action_history.append(mean_action[0].numpy())
    std_actions.append(std_action[0].numpy())


def learning_rate_annealing(
    env,
    hardware_constraint_use_case,
    optimizer,
    lr,
    total_updates,
    iteration: Optional[int] = None,
    max_hardware_runtime: Optional[int | float] = None,
):
    """
    Adjusts the learning rate based on the given parameters.

    Args:
        env: The environment object.
        hardware_constraint_use_case: A boolean indicating whether to use hardware constraints.
        optimizer: The optimizer object.
        lr: The initial learning rate.
        total_updates: The total number of updates.
        iteration: The current iteration number (optional).
        max_hardware_runtime: The maximum hardware runtime (optional).
    """
    if not hardware_constraint_use_case:
        frac = 1.0 - (iteration - 1.0) / total_updates
    else:
        frac = 1.0 - (np.sum(env.unwrapped.hardware_runtime) / max_hardware_runtime)
    lrnow = frac * lr
    optimizer.param_groups[0]["lr"] = lrnow


def update_fidelity_info(
    fidelity_info,
    fidelities,
    target_fidelities,
    lookback_window,
    env,
    update_step,
    mean_action,
    std_action,
    start_time,
):
    """
    Update the fidelity information based on the current fidelities and target fidelities.
    """
    if len(fidelities) > lookback_window:
        for fidelity in target_fidelities:
            info = fidelity_info[fidelity]
            # Sliding window lookback to check if target fidelity has been surpassed
            if (
                not info["achieved"]
                and np.mean(fidelities[-lookback_window:]) > fidelity
            ):
                info.update(
                    {
                        "achieved": True,
                        "update_at": update_step,
                        "mean_action": mean_action[0].numpy(),
                        "std_action": std_action.numpy()[0],
                        "hardware_runtime": np.sum(env.unwrapped.hardware_runtime),
                        "simulation_train_time": time.time() - start_time,
                        "shots_used": np.cumsum(env.unwrapped.total_shots)[
                            update_step - 1
                        ],
                        "shots_per_updates": int(
                            np.ceil(
                                np.cumsum(env.unwrapped.total_shots)[update_step - 1]
                                / update_step
                            )
                        ),
                    }
                )
                logging.warning(
                    f"Target fidelity {fidelity} surpassed at update {update_step}"
                )


def log_fidelity_info_summary(fidelity_info, max_hardware_runtime):
    """
    Logs a summary of fidelity information.
    """
    for fidelity, info in fidelity_info.items():
        if info["achieved"]:
            logging.warning(
                f"Target fidelity {fidelity} achieved: Update {info['update_at']}, Hardware Runtime: {round(info['hardware_runtime'], 2)} sec, Simulation Train Time: {round(info['simulation_train_time']/60, 4)} mins, Shots Used {info['shots_used']:,}"
            )
        else:
            logging.warning(
                f"Target fidelity {fidelity} not achieved within max hardware runtime of {max_hardware_runtime}s."
            )


def update_training_results(
    env,
    avg_reward,
    std_actions,
    fidelities,
    avg_action_history,
    total_updates,
    fidelity_info,
):
    return {
        "avg_reward": avg_reward,
        "std_action": std_actions,
        "fidelity_history": fidelities,
        "hardware_runtime": env.unwrapped.hardware_runtime,
        "action_history": avg_action_history,
        "best_action_vector": env.unwrapped.optimal_action,
        "total_shots": env.unwrapped.total_shots,
        "total_updates": total_updates,
        "n_reps": env.unwrapped.n_reps,
        "fidelity_info": fidelity_info,
    }


def unpack_training_config(config: Dict):
    """
    Unpacks the training configuration dictionary and returns the extracted values.

    Args:
        config (Dict): The training configuration dictionary.

    Returns:
        Tuple: A tuple containing the extracted values in the following order:
            - training_mode (str): The training mode.
            - total_updates (int or None): The total number of updates for normal training mode.
            - target_fidelities (List[float]): The target fidelities.
            - lookback_window (int): The lookback window.
            - max_hardware_runtime (float or None): The maximum hardware runtime for hardware_constraint_use_case mode.
            - anneal_learning_rate (bool): Flag indicating whether to anneal the learning rate.
            - std_actions_eps (float): The epsilon value for standard actions.
    
    Raises:
        ValueError: If the required keys are not present in the configuration dictionary.
    """
    # Mandatory keys
    training_mode = config.get("training_mode")
    training_details = config.get("training_details", {})

    if training_mode == "hardware_constraint_use_case":
        max_hardware_runtime = training_details.get("max_hardware_runtime")
        total_updates = None
        if max_hardware_runtime is None:
            raise ValueError(
                "max_hardware_runtime must be set for hardware_constraint_use_case mode"
            )
    else:
        total_updates = training_details.get("total_updates")
        max_hardware_runtime = None
        if total_updates is None:
            raise ValueError("total_updates must be set for normal training mode")

    # Optional keys with default values
    target_fidelities = training_details.get("target_fidelities", [0.99, 0.999, 0.9999])
    lookback_window = training_details.get("lookback_window", 10)
    anneal_learning_rate = training_details.get("anneal_learning_rate", False)
    std_actions_eps = training_details.get("std_actions_eps", 1e-2)

    return (
        training_mode,
        total_updates,
        target_fidelities,
        lookback_window,
        max_hardware_runtime,
        anneal_learning_rate,
        std_actions_eps,
    )


def make_train_ppo(
    agent_config: Dict,
    env: QuantumEnvironment,
    chkpt_dir: Optional[str] = "tmp/ppo",
    chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
):
    """
    Factory function that creates a training function for the Proximal Policy Optimization (PPO) algorithm.

    Args:
        agent_config (Dict): Configuration parameters for the PPO agent.
        env (QuantumEnvironment): The quantum environment for training.
        chkpt_dir (Optional[str], optional): Directory to save the PPO agent's checkpoint files. Defaults to "tmp/ppo".
        chkpt_dir_critic (Optional[str], optional): Directory to save the critic network's checkpoint files. Defaults to "tmp/critic_ppo".

    Returns:
        train (function): The training function for the PPO algorithm.
    """
    # Initialize environment parameters
    (
        seed,
        n_actions,
        batchsize,
        num_time_steps,
        min_action,
        max_action,
    ) = initialize_environment(env)

    # Initialize agent configuration
    (
        hidden_units,
        activation_functions,
        include_critic,
        minibatch_size,
        writer,
    ) = initialize_agent_config(agent_config, batchsize)

    # Initialize RL parameters
    (
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
    ) = initialize_rl_params(agent_config)

    # Initialize networks
    agent = initialize_networks(
        env,
        hidden_units,
        n_actions,
        activation_functions,
        include_critic,
        chkpt_dir,
        chkpt_dir_critic,
    )

    # Initialize optimizer
    optimizer = initialize_optimizer(agent, agent_config)

    def train(
        training_config: Dict,
        plot_real_time: bool = False,
        print_debug: Optional[bool] = False,
        num_prints: Optional[int] = 40,
    ):
        """
        Trains the model using Proximal Policy Optimization (PPO) algorithm.

        Args:
            training_config (Dict): A dictionary containing the training configuration parameters.
            plot_real_time (bool, optional): Whether to plot the training progress in real time. Defaults to False.
            print_debug (bool, optional): Whether to print debug information during training. Defaults to False.
            num_prints (int, optional): The number of times to print training progress. Defaults to 40.

        Returns:
            Dict: A dictionary containing the training results, including average reward and fidelity history.
        """
        (
            training_mode,
            total_updates,
            target_fidelities,
            lookback_window,
            max_hardware_runtime,
            anneal_lr,
            std_actions_eps,
        ) = unpack_training_config(training_config)
        hardware_constraint_use_case = training_mode == "hardware_constraint_use_case"

        try:
            if hardware_constraint_use_case or target_fidelities is not None:
                fidelity_info = {
                    fidelity: {
                        "achieved": False,
                        "update_at": None,
                        "train_time": None,
                        "shots_used": None,
                    }
                    for fidelity in target_fidelities
                }

            env.unwrapped.clear_history()
            global_step = 0

            (
                obs,
                actions,
                logprobs,
                rewards,
                dones,
                values,
                avg_reward,
                fidelities,
                std_actions,
                avg_action_history,
            ) = get_empty_tensors(env, num_time_steps, batchsize)

            start_time = time.time()

            ### Training Loop ###

            # Normal Mode: No hardware constraints
            if not hardware_constraint_use_case:
                logging.warning("Training in normal mode")
                for iteration in tqdm.tqdm(range(1, total_updates + 1)):
                    if anneal_lr:
                        learning_rate_annealing(
                            env,
                            hardware_constraint_use_case,
                            optimizer,
                            lr,
                            total_updates,
                            iteration=iteration,
                        )

                    mean_action, std_action = perform_training_iteration(
                        env,
                        seed,
                        agent,
                        n_epochs,
                        optimizer,
                        writer,
                        global_step,
                        batchsize,
                        minibatch_size,
                        dones,
                        obs,
                        actions,
                        logprobs,
                        rewards,
                        values,
                        min_action,
                        max_action,
                        gamma,
                        gae_lambda,
                        ppo_epsilon,
                        ent_coef,
                        critic_loss_coef,
                        clip_vloss,
                        clip_coef,
                        grad_clip,
                        normalize_advantage,
                        num_prints,
                        print_debug,
                        plot_real_time,
                    )

                    update_metric_lists(
                        env,
                        mean_action,
                        std_action,
                        avg_reward,
                        fidelities,
                        avg_action_history,
                        std_actions,
                    )
                    if target_fidelities is not None:
                        update_fidelity_info(
                            fidelity_info,
                            fidelities,
                            target_fidelities,
                            lookback_window,
                            env,
                            iteration,
                            mean_action,
                            std_action,
                            start_time,
                        )

                    training_results = update_training_results(
                        env,
                        avg_reward,
                        std_actions,
                        fidelities,
                        avg_action_history,
                        total_updates,
                        fidelity_info,
                    )

                    if check_convergence_std_actions(std_action, std_actions_eps):
                        break

            else:  # Hardware Constraint Mode: Train until hardware runtime exceeds maximum
                logging.warning("Training in hardware constraint use case mode")
                iteration = 0
                while np.sum(env.unwrapped.hardware_runtime) < max_hardware_runtime:
                    iteration += 1
                    if anneal_lr:
                        learning_rate_annealing(
                            env,
                            hardware_constraint_use_case,
                            optimizer,
                            lr,
                            total_updates,
                            max_hardware_runtime=max_hardware_runtime,
                        )

                    mean_action, std_action = perform_training_iteration(
                        env,
                        seed,
                        agent,
                        n_epochs,
                        optimizer,
                        writer,
                        global_step,
                        batchsize,
                        minibatch_size,
                        dones,
                        obs,
                        actions,
                        logprobs,
                        rewards,
                        values,
                        min_action,
                        max_action,
                        gamma,
                        gae_lambda,
                        ppo_epsilon,
                        ent_coef,
                        critic_loss_coef,
                        clip_vloss,
                        clip_coef,
                        grad_clip,
                        normalize_advantage,
                        num_prints,
                        print_debug,
                        plot_real_time,
                    )

                    update_metric_lists(
                        env,
                        mean_action,
                        std_action,
                        avg_reward,
                        fidelities,
                        avg_action_history,
                        std_actions,
                    )
                    update_fidelity_info(
                        fidelity_info,
                        fidelities,
                        target_fidelities,
                        lookback_window,
                        env,
                        iteration,
                        mean_action,
                        std_action,
                        start_time,
                    )

                    training_results = update_training_results(
                        env,
                        avg_reward,
                        std_actions,
                        fidelities,
                        avg_action_history,
                        total_updates,
                        fidelity_info,
                    )

                    if check_convergence_std_actions(std_action, std_actions_eps):
                        break

            env.unwrapped.close()
            writer.close()

            if hardware_constraint_use_case:
                log_fidelity_info_summary(fidelity_info, max_hardware_runtime)

            return training_results

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise
            return {
                "avg_reward": -1.0,
                "fidelity_history": [0] * total_updates,
            }

    return train