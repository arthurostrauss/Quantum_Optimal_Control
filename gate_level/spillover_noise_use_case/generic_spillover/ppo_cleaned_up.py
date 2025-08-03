import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

# from gymnasium.wrappers import ActionWrapper
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
from typing import Optional, Dict


# Adapted from CleanRL ppo_continuous_action.py
from rl_qoc.environment.base_q_env import BaseQuantumEnvironment


def process_action(self, probs: Normal):
    """
    Decide how actions should be processed before being sent to environment.
    For certain environments such as QUA, policy parameters should be streamed to the environment directly
    and actions are sampled within the environment (in real time).
    """
    action = probs.sample()
    logprob = probs.log_prob(action).sum(1)
    mean_action = probs.mean
    std_action = probs.stddev

    if isinstance(self.env, ActionWrapper):
        self.unwrapped_env.mean_action = self.env.action(mean_action[0].cpu().numpy())
    else:
        self.unwrapped_env.mean_action = mean_action[0].cpu().numpy()

    self.unwrapped_env.std_action = std_action[0].cpu().numpy()
    return action, logprob


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def plot_curves(env: BaseQuantumEnvironment):
    """
    Plots the reward history, fidelity history, and (for moving_discrete) marginal reward and probability distributions for each qubit.
    """
    fidelity_range = [i * env.benchmark_cycle for i in range(len(env.fidelity_history))]
    num_qubits = len(env.unwrapped.applied_qubits)  # Number of qubits
    num_plots = 2 + (
        2 * num_qubits
        if env.unwrapped.circuit_param_distribution == "moving_discrete"
        else 0
    )
    fig, ax = plt.subplots(num_plots, 1, figsize=(8.0, 6.0 * num_plots), squeeze=False)
    ax = ax.flatten()

    ax[0].plot(np.mean(env.reward_history, axis=1), label="Reward")
    ax[0].plot(fidelity_range, env.fidelity_history, label="Circuit Fidelity")
    ax[0].set_title("Reward History")
    ax[0].legend()
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Reward")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[1].plot(-np.log10(1.0 - np.mean(env.reward_history, axis=1)), label="RL Reward")
    ax[1].set_title("RL Reward History")
    ax[1].legend()
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("RL Reward")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    if env.unwrapped.circuit_param_distribution == "moving_discrete":
        single_qubit_vals = env.unwrapped.single_qubit_discrete_obs_vals_raw
        angles = env.unwrapped.obs_raw_to_angles(single_qubit_vals)
        num_vals = len(single_qubit_vals)

        for qubit_idx in range(num_qubits):
            marginal_rewards = np.zeros(num_vals)
            marginal_std = np.zeros(num_vals)
            for i, val in enumerate(single_qubit_vals):
                mask = np.isclose(
                    env.unwrapped.discrete_obs_vals_raw[:, qubit_idx],
                    val,
                    rtol=1e-5,
                    atol=1e-8,
                )
                marginal_rewards[i] = (
                    np.mean(env.unwrapped.discrete_reward_history[:, mask])
                    if np.any(mask)
                    else 0
                )
                marginal_std[i] = (
                    np.std(env.unwrapped.discrete_reward_history[:, mask])
                    if np.any(mask)
                    else 0
                )

            ax[2 + qubit_idx * 2].errorbar(
                angles,
                marginal_rewards,
                yerr=marginal_std,
                fmt="-o",
                label=f"Qubit {qubit_idx} Reward History",
            )
            ax[2 + qubit_idx * 2].set_title(
                f"Marginal Reward History (Qubit {qubit_idx})"
            )
            ax[2 + qubit_idx * 2].legend()
            ax[2 + qubit_idx * 2].set_xlabel("Observation Value (Angles)")
            ax[2 + qubit_idx * 2].set_ylabel("Reward")

            marginal_probs = np.zeros(num_vals)
            for i, val in enumerate(single_qubit_vals):
                mask = np.isclose(
                    env.unwrapped.discrete_obs_vals_raw[:, qubit_idx],
                    val,
                    rtol=1e-5,
                    atol=1e-8,
                )
                marginal_probs[i] = (
                    np.sum(env.unwrapped.prob_weights[mask]) if np.any(mask) else 0
                )

            ax[3 + qubit_idx * 2].scatter(
                angles, marginal_probs, label=f"Qubit {qubit_idx} Probability Weights"
            )
            ax[3 + qubit_idx * 2].set_title(
                f"Marginal Probability Weights (Qubit {qubit_idx})"
            )
            ax[3 + qubit_idx * 2].legend()
            ax[3 + qubit_idx * 2].set_xlabel("Observation Value (Angles)")
            ax[3 + qubit_idx * 2].set_ylabel("Probability")

    plt.show()


class Agent(nn.Module):
    def __init__(
        self,
        env,
        use_combined_networks,
        activation_function_str,
        layer_size,
        robust_ppo,
    ):
        super().__init__()
        self.use_combined_networks = use_combined_networks
        self.robust_ppo = robust_ppo
        if activation_function_str == "tanh":
            self.activation_fn = nn.Tanh
        elif activation_function_str == "relu":
            self.activation_fn = nn.ReLU
        elif activation_function_str == "gelu":
            self.activation_fn = nn.GELU
        elif activation_function_str == "leaky_relu":
            self.activation_fn = nn.LeakyReLU
        elif activation_function_str == "elu":
            self.activation_fn = nn.ELU

        if not self.use_combined_networks:
            self.critic = nn.Sequential(
                layer_init(
                    nn.Linear(np.array(env.observation_space.shape).prod(), layer_size)
                ),
                self.activation_fn(),
                layer_init(nn.Linear(layer_size, layer_size)),
                self.activation_fn(),
                layer_init(nn.Linear(layer_size, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(
                    nn.Linear(np.array(env.observation_space.shape).prod(), layer_size)
                ),
                self.activation_fn(),
                layer_init(nn.Linear(layer_size, layer_size)),
                self.activation_fn(),
                layer_init(
                    nn.Linear(layer_size, np.prod(env.action_space.shape)), std=0.01
                ),
            )
            self.actor_logstd = nn.Parameter(
                torch.zeros(1, np.prod(env.action_space.shape))
            )

        if self.use_combined_networks:
            self.main_network = nn.Sequential(
                layer_init(
                    nn.Linear(np.array(env.observation_space.shape).prod(), layer_size)
                ),
                self.activation_fn(),
                layer_init(nn.Linear(layer_size, layer_size)),
                self.activation_fn(),
            )
            self.actor_mean = nn.Sequential(
                self.main_network,
                layer_init(
                    nn.Linear(layer_size, np.prod(env.action_space.shape)), std=0.01
                ),
            )
            self.critic = nn.Sequential(
                self.main_network, layer_init(nn.Linear(layer_size, 1), std=1.0)
            )
            self.actor_logstd = nn.Parameter(
                torch.zeros(1, np.prod(env.action_space.shape))
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPO:
    def __init__(
        self,
        agent_config: Dict,
        env: gym.Env,  # BaseQuantumEnvironment,
        chkpt_dir: Optional[str] = "tmp/ppo",
        chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
    ):
        self.agent_config = agent_config
        self.exp_name = agent_config["exp_name"]
        self.seed = agent_config["seed"]
        self.torch_deterministic = agent_config["torch_deterministic"]
        self.cuda = agent_config["cuda"]
        self.track = agent_config["track"]
        self.wandb_project_name = agent_config["wandb_project_name"]
        self.wandb_entity = agent_config["wandb_entity"]
        self.save_model = agent_config["save_model"]
        self.plot_real_time = agent_config["plot_real_time"]
        self.print_freq = agent_config["print_freq"]

        self.num_updates = agent_config["total_updates"]
        self.learning_rate = agent_config["learning_rate"]
        self.num_envs = agent_config["num_envs"]
        self.num_steps = agent_config["num_steps"]
        self.anneal_lr = agent_config["anneal_lr"]
        self.anneal_num_updates = agent_config["anneal_num_updates"]
        self.exp_anneal_lr = agent_config["exp_anneal_lr"]
        self.exp_update_time = agent_config["exp_update_time"]
        self.plateau_lr = agent_config["plateau_lr"]
        self.gamma = agent_config["gamma"]
        self.gae_lambda = agent_config["gae_lambda"]
        self.num_minibatches = agent_config["num_minibatches"]
        self.update_epochs = agent_config["update_epochs"]
        self.norm_adv = agent_config["norm_adv"]
        self.activation_function_str = agent_config["activation_function_str"]
        self.use_combined_networks = agent_config["use_combined_networks"]
        self.layer_size = agent_config["layer_size"]
        self.clip_coef = agent_config["clip_coef"]
        self.clip_vloss = agent_config["clip_vloss"]
        self.ent_coef = agent_config["ent_coef"]
        self.vf_coef = agent_config["vf_coef"]
        self.max_grad_norm = agent_config["max_grad_norm"]
        self.target_kl = agent_config["target_kl"]
        self.robust_ppo = agent_config["robust_ppo"]

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        assert (
            self.anneal_lr + self.exp_anneal_lr
        ), "Learning Rate Combination doesn't work"

        self.run_name = f"PPO__{self.exp_name}__{self.seed}__{int(time.time())}"

        if self.track:
            import wandb

            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                config=self.agent_config,
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in self.agent_config.items()]
                )
            ),
        )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )
        self.env = env

        # Pass total updates to the wrapper for noise annealing calculations
        if hasattr(self.env, "set_total_updates"):
            self.env.set_total_updates(self.num_updates)

        self.agent = Agent(
            self.env,
            self.use_combined_networks,
            self.activation_function_str,
            self.layer_size,
            self.robust_ppo,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate, eps=1e-5
        )

    def run_training(self):
        self.obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.observation_space.shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.action_space.shape
        ).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        global_step = 0
        start_time = time.time()

        # Initial reset is now simpler and uses the seed
        next_obs, _ = self.env.reset(seed=self.seed)
        next_obs = np.tile(next_obs, (self.num_envs, 1))
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        for update in range(1, self.num_updates + 1):
            # Tell the wrapper that a PPO update step has occurred
            if hasattr(self.env, "increment_update_step"):
                self.env.increment_update_step()

            if self.anneal_lr or self.exp_anneal_lr:
                if self.anneal_lr:
                    frac = 1.0 - (update - 1.0) / self.anneal_num_updates
                    lrnow = (
                        np.clip(frac, a_min=0.0, a_max=None) * self.learning_rate
                        + self.plateau_lr
                    )
                    self.optimizer.param_groups[0]["lr"] = lrnow
                if self.exp_anneal_lr:
                    lrnow = (
                        self.learning_rate * np.exp(-update / self.exp_update_time)
                        + self.plateau_lr
                    )
                    self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                cpu_action = action.cpu().numpy()
                # The mean action over the parallel environments for this step
                mean_action_for_step = np.mean(cpu_action, axis=0)

                # Pass the mean action to the wrapper before stepping
                if hasattr(self.env, "set_mean_action"):
                    self.env.set_mean_action(mean_action_for_step)

                try:
                    self.env.unwrapped.mean_action = self.env.action(
                        mean_action_for_step
                    )
                except AttributeError:
                    self.env.unwrapped.mean_action = mean_action_for_step

                next_obs, reward, terminations, truncations, infos = self.env.step(
                    cpu_action
                )

                mean_reward = np.mean(reward)

                next_obs = np.tile(next_obs, (self.num_envs, 1))
                next_done = np.logical_or(terminations, truncations)

                if next_done.any():
                    next_obs, _ = self.env.reset()
                    next_obs = np.tile(next_obs, (self.num_envs, 1))

                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(next_done).to(self.device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
                            )
                            self.writer.add_scalar(
                                "charts/episodic_return",
                                info["episode"]["r"],
                                global_step,
                            )
                            self.writer.add_scalar(
                                "charts/episodic_length",
                                info["episode"]["l"],
                                global_step,
                            )

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = (
                        self.rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + self.values

            b_obs = self.obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
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
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            self.writer.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar(
                "losses/explained_variance", explained_var, global_step
            )
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

            # Plotting calls are now directed to the wrapped environment
            if self.plot_real_time and (
                update % (self.print_freq) == 0
                or update == self.num_updates
            ):
                print("Plotting")
                from IPython.display import clear_output

                clear_output(wait=True)

                # This plot uses the unwrapped env's history
                if hasattr(self.env.unwrapped, "reward_history"):
                    print("Plotting Curves")
                    plot_curves(self.env.unwrapped)

                # These plots use the wrapper's buffer
                plot_args_2 = {
                    "gamma_matrix": self.env.unwrapped.gamma_matrix,
                    "spillover_qubits": self.env.unwrapped.applied_qubits,  # [0, 1, 2, 3, 4, 5],
                    "target_qubit": 2,
                    "action_scale": np.mean(self.env.unwrapped.action_space.high),
                }
                plot_args_3 = {
                    "gamma_matrix": self.env.unwrapped.gamma_matrix,
                    "spillover_qubits": self.env.unwrapped.applied_qubits,  # [0, 1, 2, 3, 4, 5],
                    "target_qubit": 3,
                    "action_scale": np.mean(self.env.unwrapped.action_space.high),
                }
                if hasattr(self.env, "plot_buffer_reward_distribution"):
                    self.env.plot_buffer_reward_distribution()
                if hasattr(self.env, "plot_policy_behaviour"):
                    self.env.plot_policy_behaviour(**plot_args_2)
                    self.env.plot_policy_behaviour(**plot_args_3)
                if hasattr(self.env, "plot_action_comparison"):
                    self.env.plot_action_comparison(**plot_args_2)
                    self.env.plot_action_comparison(**plot_args_3)

        if self.save_model:
            model_path = f"runs/{self.run_name}/{self.exp_name}.cleanrl_model"
            torch.save(self.agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        self.env.close()
        self.writer.close()
