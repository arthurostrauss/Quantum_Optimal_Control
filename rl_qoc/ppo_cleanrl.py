# Adapted from CleanRL ppo_continuous_action.py
import random
import time
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from rl_qoc.base_q_env import BaseQuantumEnvironment


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.zeros_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def plot_curves(env: BaseQuantumEnvironment):
    """
    Plots the reward history and fidelity history of the environment
    """
    fidelity_range = [i * env.benchmark_cycle for i in range(len(env.fidelity_history))]
    plt.plot(np.mean(env.reward_history, axis=1), label="Reward")
    plt.plot(
        fidelity_range,
        env.fidelity_history,
        label=f"Circuit Fidelity",
    )

    plt.title("Reward History")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    # Ensure integer ticks on the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


class Agent(nn.Module):
    def __init__(self, env, use_combined_networks, activation_function_str, layer_size):
        super().__init__()
        self.use_combined_networks = use_combined_networks
        if activation_function_str == "tanh":
            self.activation_fn = nn.Tanh
        elif activation_function_str == "relu":
            self.activation_fn = nn.ReLU
        elif activation_function_str == "gelu":
            self.activation_fn = nn.GELU
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
                self.activation_fn(),
            )
            self.actor_std = nn.Sequential(
                self.main_network,
                layer_init(
                    nn.Linear(layer_size, np.prod(env.action_space.shape)), std=0.01
                ),
                nn.Sigmoid(),
            )
            self.critic = nn.Sequential(
                self.main_network,
                layer_init(nn.Linear(layer_size, 1), std=1.0),
            )
            self.actor_logstd = nn.Parameter(
                torch.zeros(1, np.prod(env.action_space.shape))
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        # action_logstd = self.actor_logstd.expand_as(action_mean)
        # action_std = torch.exp(action_logstd)
        if self.use_combined_networks:
            action_std = self.actor_std(x)
        else:
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


class PPO_CleanRL:
    def __init__(
        self,
        agent_config: Dict,
        env: BaseQuantumEnvironment,
        chkpt_dir: Optional[str] = "tmp/ppo",
        chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
    ):
        # General Run Params
        self.agent_config = agent_config
        self.exp_name = self.agent_config["exp_name"]
        self.seed = self.agent_config["seed"]
        self.torch_deterministic = self.agent_config["torch_deterministic"]
        self.cuda = self.agent_config["cuda"]
        self.track = self.agent_config["track"]
        self.wandb_project_name = self.agent_config["wandb_project_name"]
        self.wandb_entity = self.agent_config["wandb_entity"]
        self.save_model = self.agent_config["save_model"]
        self.plot_real_time = self.agent_config["plot_real_time"]
        self.num_prints = self.agent_config["num_prints"]

        # PPO Specific Params
        self.num_updates = self.agent_config["total_updates"]
        self.learning_rate = self.agent_config["learning_rate"]
        self.num_envs = self.agent_config["num_envs"]
        self.num_steps = self.agent_config["num_steps"]
        self.anneal_lr = self.agent_config["anneal_lr"]
        self.gamma = self.agent_config["gamma"]
        self.gae_lambda = self.agent_config["gae_lambda"]
        self.num_minibatches = self.agent_config["num_minibatches"]
        self.update_epochs = self.agent_config["update_epochs"]
        self.norm_adv = self.agent_config["norm_adv"]
        self.activation_function_str = self.agent_config["activation_function_str"]
        self.use_combined_networks = self.agent_config["use_combined_networks"]
        self.layer_size = self.agent_config["layer_size"]
        self.clip_coef = self.agent_config["clip_coef"]
        self.clip_vloss = self.agent_config["clip_vloss"]
        self.ent_coef = self.agent_config["ent_coef"]
        self.vf_coef = self.agent_config["vf_coef"]
        self.max_grad_norm = self.agent_config["max_grad_norm"]
        self.target_kl = self.agent_config["target_kl"]

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.run_name = f"PPO_CleanRL__{self.exp_name}__{self.seed}__{int(time.time())}"

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

        # TRY NOT TO MODIFY: seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        self.env = env

        self.agent = Agent(
            self.env,
            self.use_combined_networks,
            self.activation_function_str,
            self.layer_size,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate, eps=1e-5
        )

    def run_training(self):
        # ALGO Logic: Storage setup
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

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.env.reset(seed=self.seed)
        next_obs = np.tile(next_obs, (self.num_envs, 1))
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        for update in range(1, self.num_updates + 1):
            # Annealing the Learning Rate
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                cpu_action = action.cpu().numpy()
                cpu_action = np.clip(
                    cpu_action,
                    self.env.action_space.low,
                    self.env.action_space.high,
                )

                next_obs, reward, terminations, truncations, infos = self.env.step(
                    cpu_action
                )
                next_obs = np.tile(next_obs, (self.num_envs, 1))
                next_done = np.logical_or(terminations, truncations)
                if next_done:
                    next_obs, _ = self.env.reset(seed=self.seed)
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

            # bootstrap value if not done
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

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
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
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
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

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
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

            # TRY NOT TO MODIFY: record rewards for plotting purposes
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

            if global_step % self.num_prints == 0:
                clear_output(wait=True)
                if self.plot_real_time:
                    plot_curves(self.env.unwrapped)

        if self.save_model:
            model_path = f"runs/{self.run_name}/{self.exp_name}.cleanrl_model"
            torch.save(self.agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        self.env.close()
        self.writer.close()
