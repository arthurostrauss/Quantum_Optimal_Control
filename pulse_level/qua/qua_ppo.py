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
from rl_qoc.agent import ActorNetwork, CriticNetwork, Agent
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


class Custom_QUA_PPO:
    def __init__(
        self,
        agent_config: Dict,
        env: QuantumEnvironment | Wrapper,
        chkpt_dir: Optional[str] = "tmp/ppo",
        chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
    ):
        """
        Initializes the PPO algorithm with the given hyperparameters
        :param agent_config: Dictionary containing all the hyperparameters for the PPO algorithm
        :param env: Quantum Environment on which the algorithm is trained
        :param chkpt_dir: Directory where the policy network is saved
        :param chkpt_dir_critic: Directory where the critic network is saved
        """
        self.agent_config = agent_config
        self.env = env
        self.chkpt_dir = chkpt_dir
        self.chkpt_dir_critic = chkpt_dir_critic
        self.seed = env.unwrapped.seed
        self.n_actions = env.unwrapped.action_space.shape[-1]
        self.batchsize = env.unwrapped.batch_size
        self.num_time_steps = env.unwrapped.tgt_instruction_counts
        if hasattr(env, "min_action") and hasattr(env, "max_action"):
            self.min_action = env.min_action
            self.max_action = env.max_action
        else:
            if isinstance(env.action_space, Box):
                self.min_action = env.action_space.low
                self.max_action = env.action_space.high
            else:
                raise ValueError("Environment action space is not a Box")

        self.hidden_units = agent_config["N_UNITS"]
        self.activation_fn = agent_config["ACTIVATION"]
        self.activation_functions = [
            get_module_from_str(self.activation_fn)()
            for _ in range(len(self.hidden_units) + 1)
        ]
        self.include_critic = agent_config["INCLUDE_CRITIC"]
        self.minibatch_size = agent_config["MINIBATCH_SIZE"]
        if self.batchsize % self.minibatch_size != 0:
            raise ValueError(
                f"The current minibatch size of {self.minibatch_size} does not evenly divide the batchsize of {self.batchsize}"
            )

        self.run_name = agent_config["RUN_NAME"]
        self.writer = SummaryWriter(f"runs/{self.run_name}")

        # General RL Params
        self.n_epochs = agent_config["N_EPOCHS"]
        self.lr = agent_config["LR"]

        # PPO Specific Params
        self.ppo_epsilon = agent_config["CLIP_RATIO"]
        self.critic_loss_coef = agent_config["V_COEF"]
        self.gamma = agent_config["GAMMA"]
        self.gae_lambda = agent_config["GAE_LAMBDA"]

        # Clipping
        self.clip_vloss = agent_config["CLIP_VALUE_LOSS"]
        self.grad_clip = agent_config["GRADIENT_CLIP"]
        self.clip_coef = agent_config["CLIP_VALUE_COEF"]
        self.normalize_advantage = agent_config["NORMALIZE_ADVANTAGE"]
        self.ent_coef = agent_config["ENT_COEF"]

        self.actor_net = ActorNetwork(
            env.observation_space,
            self.hidden_units,
            self.n_actions,
            self.activation_functions,
            self.include_critic,
            self.chkpt_dir,
        )
        self.critic_net = CriticNetwork(
            env.observation_space,
            self.hidden_units,
            self.activation_functions,
            self.chkpt_dir_critic,
        )
        if not self.include_critic:
            self.agent = Agent(self.actor_net, critic_net=self.critic_net)
        else:
            self.agent = Agent(self.actor_net, critic_net=None)

        self.optim_name = agent_config["OPTIMIZER"]
        self.optim_eps = 1e-5
        self.optimizer = get_optimizer_from_str(self.optim_name)(
            self.agent.parameters(), lr=self.lr, eps=self.optim_eps
        )

    def plot_curves(self):
        """
        Plots the reward history and fidelity history of the environment
        """
        fidelity_range = [
            i * self.env.unwrapped.benchmark_cycle
            for i in range(len(self.env.unwrapped.fidelity_history))
        ]
        plt.plot(np.mean(self.reward_history, axis=1), label="Reward")
        plt.plot(
            fidelity_range,
            self.fidelity_history,
            label="Fidelity",
        )

        plt.title("Reward History")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.show()

    def train(
        self,
        total_updates: int,
        print_debug=True,
        num_prints=40,
        clear_history=False,
        plot_real_time=True,
    ):
        """
        Training function for PPO algorithm
        :param total_updates: Total number of updates to perform
        :param print_debug: If True, then print debug statements
        :param num_prints: Number of times to print debug statements
        :param clear_history: If True, then clear the history of the environment
        :param plot_real_time: If True, then plot the reward history in real time
        """
        if clear_history:
            self.env.unwrapped.clear_history()
            start = time.time()
            global_step = 0
        else:
            global_step = self.env.unwrapped.step_tracker

        obs = torch.zeros(
            (self.num_time_steps, self.batchsize) + self.env.observation_space.shape
        )
        actions = torch.zeros(
            (self.num_time_steps, self.batchsize) + self.env.action_space.shape
        )
        logprobs = torch.zeros((self.num_time_steps, self.batchsize))
        rewards = torch.zeros((self.num_time_steps, self.batchsize))
        dones = torch.zeros((self.num_time_steps, self.batchsize))
        values = torch.zeros((self.num_time_steps, self.batchsize))

        # Starting Learning
        for _ in tqdm.tqdm(range(1, total_updates + 1)):
            next_obs, _ = self.env.reset(seed=self.seed)
            num_steps = self.env.unwrapped.episode_length(global_step)
            batch_obs = torch.tile(torch.Tensor(next_obs), (self.batchsize, 1))
            batch_done = torch.zeros_like(dones[0])

            # print("episode length:", num_steps)

            for step in range(num_steps):
                global_step += 1
                obs[step] = batch_obs
                dones[step] = batch_done

                with torch.no_grad():
                    mean_action, std_action, critic_value = self.agent(batch_obs)
                    probs = Normal(mean_action, std_action)
                    if isinstance(self.env.unwrapped.estimator, QuaEstimator):
                        self.env.unwrapped.estimator.mean_action = mean_action
                        self.env.unwrapped.estimator.std_action = std_action

                    # TODO: The action and log_prob must be retrieved from the QUA program (sampled on the fly)
                    action = torch.clip(
                        probs.sample(),
                        torch.Tensor(self.min_action),
                        torch.Tensor(self.max_action),
                    )
                    logprob = probs.log_prob(action).sum(1)
                    values[step] = critic_value.flatten()

                actions[step] = action
                logprobs[step] = logprob
                start_time = time.time()
                next_obs, reward, terminated, truncated, infos = self.env.step(
                    action.cpu().numpy()
                )
                print("Time taken", time.time() - start_time)
                next_obs = torch.Tensor(next_obs)
                done = int(np.logical_or(terminated, truncated))
                reward = torch.Tensor(reward)
                rewards[step] = reward

                batch_obs = torch.tile(next_obs, (self.batchsize, 1))
                next_done = done * torch.ones_like(dones[0])
                obs[step] = batch_obs
                dones[step] = next_done

                # print(f"global_step={global_step}, episodic_return={np.mean(reward)}")
                self.writer.add_scalar(
                    "charts/episodic_return", np.mean(reward.numpy()), global_step
                )
                # self.writer.add_scalar("charts/episodic_length", num_steps, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batchsize)
            clipfracs = []
            for epoch in range(self.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batchsize, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    new_mean, new_sigma, new_value = self.agent(b_obs[mb_inds])
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
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.ppo_epsilon)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.normalize_advantage:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = new_value.view(-1)

                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.ent_coef * entropy_loss
                        + v_loss * self.critic_loss_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
                    self.optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            if print_debug:
                print("mean", mean_action[0])
                print("sigma", std_action[0])
                print(
                    "DFE Rewards Mean:",
                    np.mean(self.env.unwrapped.reward_history, axis=1)[-1],
                )
                print(
                    "DFE Rewards standard dev",
                    np.std(self.env.unwrapped.reward_history, axis=1)[-1],
                )
                print("Returns Mean:", np.mean(b_returns.numpy()))
                print("Returns standard dev", np.std(b_returns.numpy()))
                print("Advantages Mean:", np.mean(b_advantages.numpy()))
                print("Advantages standard dev", np.std(b_advantages.numpy()))
                # print(np.mean(env.unwrapped.reward_history, axis =1)[-1])
                # print("Circuit fidelity:", env.unwrapped.circuit_fidelity_history[-1])

            if global_step % num_prints == 0:
                clear_output(wait=True)
                if plot_real_time:
                    self.plot_curves()

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/avg_return",
                np.mean(self.env.unwrapped.reward_history, axis=1)[-1],
                global_step,
            )
            if self.env.unwrapped.do_benchmark():
                if hasattr(self.env.unwrapped, "circuit_fidelity_history"):
                    self.writer.add_scalar(
                        "losses/circuit_fidelity",
                        self.env.unwrapped.circuit_fidelity_history[-1],
                        global_step,
                    )
                else:
                    self.writer.add_scalar(
                        f"losses/avg_{self.env.unwrapped.target.target_type}_fidelity",
                        self.env.unwrapped.fidelity_history[-1],
                        global_step,
                    )
            # self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            # self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # self.writer.add_scalar(
            #     "losses/old_approx_kl", old_approx_kl.item(), global_step
            # )
            # self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            # self.writer.add_scalar("losses/explained_var", explained_var, global_step)
            # self.writer.add_scalar(
            #     "losses/advantage", np.mean(b_advantages.numpy()), global_step
            # )
            # self.writer.add_scalar(
            #     "losses/advantage_std", np.std(b_advantages.numpy()), global_step
            # )

        self.env.unwrapped.close()
        self.writer.close()

    @property
    def reward_history(self):
        return self.env.unwrapped.reward_history

    @property
    def fidelity_history(self):
        return self.env.unwrapped.fidelity_history


def make_train_ppo(
    agent_config: Dict,
    env: QuantumEnvironment,
    chkpt_dir: Optional[str] = "tmp/ppo",
    chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
):
    """
    Creates a training function for PPO algorithm.
    :param agent_config: Dictionary containing all the hyperparameters for the PPO algorithm
    :param env: Quantum Environment on which the algorithm is trained
    :param chkpt_dir: Directory where the policy network is saved
    :param chkpt_dir_critic: Directory where the critic network is saved
    :return: Training function for PPO algorithm
    """
    seed = env.unwrapped.seed
    n_actions = env.action_space.shape[-1]
    batchsize = env.unwrapped.batch_size
    num_time_steps = env.unwrapped.tgt_instruction_counts
    if hasattr(env, "min_action") and hasattr(env, "max_action"):
        min_action = env.min_action
        max_action = env.max_action
    else:
        if isinstance(env.action_space, Box):
            min_action = env.action_space.low
            max_action = env.action_space.high
        else:
            raise ValueError("Environment action space is not a Box")

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

    # General RL Params
    n_epochs = agent_config["N_EPOCHS"]
    lr = agent_config["LR"]

    # PPO Specific Params
    ppo_epsilon = agent_config["CLIP_RATIO"]
    critic_loss_coef = agent_config["V_COEF"]
    gamma = agent_config["GAMMA"]
    gae_lambda = agent_config["GAE_LAMBDA"]

    # Clipping
    clip_vloss = agent_config["CLIP_VALUE_LOSS"]
    grad_clip = agent_config["GRADIENT_CLIP"]
    clip_coef = agent_config["CLIP_VALUE_COEF"]
    normalize_advantage = agent_config["NORMALIZE_ADVANTAGE"]
    ent_coef = agent_config["ENT_COEF"]

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
    if not include_critic:
        agent = Agent(actor_net, critic_net=critic_net)
    else:
        agent = Agent(actor_net, critic_net=None)

    optim_name = agent_config["OPTIMIZER"]
    optim_eps = 1e-5
    optimizer = get_optimizer_from_str(optim_name)(
        agent.parameters(), lr=lr, eps=optim_eps
    )

    def train(
        total_updates: int,
        print_debug: Optional[bool] = True,
        num_prints: Optional[int] = 40,
    ):
        """
        Training function for PPO algorithm
        :param total_updates: Total number of updates to perform
        :param print_debug: If True, then print debug statements
        :param num_prints: Number of times to print debug statements
        """
        try:
            env.unwrapped.clear_history()
            start = time.time()
            global_step = 0

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

            ### Starting Learning ###
            for _ in tqdm.tqdm(range(1, total_updates + 1)):
                next_obs, _ = env.reset(seed=seed)
                num_steps = env.unwrapped.episode_length(global_step)  # num_time_steps
                batch_obs = torch.tile(torch.Tensor(next_obs), (batchsize, 1))
                batch_done = torch.zeros_like(dones[0])

                # print("episode length:", num_steps)

                for step in range(num_steps):
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

                    next_obs, reward, terminated, truncated, infos = env.step(
                        action.cpu().numpy()
                    )
                    next_obs = torch.Tensor(next_obs)
                    done = int(np.logical_or(terminated, truncated))
                    reward = torch.Tensor(reward)
                    rewards[step] = reward

                    batch_obs = torch.tile(next_obs, (batchsize, 1))
                    next_done = done * torch.ones_like(dones[0])
                    obs[step] = batch_obs
                    dones[step] = next_done

                    # print(f"global_step={global_step}, episodic_return={np.mean(reward)}")
                    writer.add_scalar(
                        "charts/episodic_return", np.mean(reward.numpy()), global_step
                    )
                    writer.add_scalar("charts/episodic_length", num_steps, global_step)

                # bootstrap value if not done
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
                        delta = (
                            rewards[t]
                            + gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                        )
                    returns = advantages + values

                # flatten the batch
                b_obs = obs.reshape((-1,) + env.observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + env.action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(batchsize)
                clipfracs = []
                for epoch in range(n_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batchsize, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        new_mean, new_sigma, new_value = agent(b_obs[mb_inds])
                        new_dist = Normal(new_mean, new_sigma)
                        new_logprob, entropy = new_dist.log_prob(
                            b_actions[mb_inds]
                        ).sum(1), new_dist.entropy().sum(1)
                        logratio = new_logprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [
                                ((ratio - 1.0).abs() > ppo_epsilon)
                                .float()
                                .mean()
                                .item()
                            ]

                        mb_advantages = b_advantages[mb_inds]
                        if normalize_advantage:  # Normalize advantage
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                            )

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1 - ppo_epsilon, 1 + ppo_epsilon
                        )
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
                        loss = (
                            pg_loss
                            - ent_coef * entropy_loss
                            + v_loss * critic_loss_coef
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
                        optimizer.step()

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
                if print_debug:
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
                    # print(np.mean(env.unwrapped.reward_history, axis =1)[-1])
                    # print("Circuit fidelity:", env.unwrapped.circuit_fidelity_history[-1])

                if global_step % num_prints == 0:
                    clear_output(wait=True)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(
                    "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar(
                    "losses/avg_return",
                    np.mean(env.unwrapped.reward_history, axis=1)[-1],
                    global_step,
                )
                # writer.add_scalar("losses/avg_gate_fidelity", env.unwrapped.avg_fidelity_history[-1], global_step)
                # writer.add_scalar("losses/circuit_fidelity", env.unwrapped.circuit_fidelity_history[-1], global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar(
                    "losses/old_approx_kl", old_approx_kl.item(), global_step
                )
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar(
                    "losses/explained_variance", explained_var, global_step
                )

                # Collect results
                avg_reward.append(np.mean(env.unwrapped.reward_history, axis=1)[-1])
                (
                    fidelities.append(env.unwrapped.fidelity_history[-1])
                    if len(env.unwrapped.fidelity_history) > 0
                    else None
                )
                avg_action_history.append(mean_action[0].numpy())
                std_actions.append(std_action[0].numpy())

            env.unwrapped.close()
            writer.close()

            return {
                "avg_reward": avg_reward,
                "std_action": std_actions,
                "fidelity_history": fidelities,
                "action_history": avg_action_history,
                "best_action_vector": env.unwrapped.optimal_action,
            }
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            return {
                "avg_reward": -1.0,
                "fidelity_history": [0] * total_updates,
            }

    return train
