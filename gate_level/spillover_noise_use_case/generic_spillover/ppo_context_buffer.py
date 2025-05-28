import uuid
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from gymnasium.spaces import Box
from typing import Dict, Optional, Any
import collections

# Assuming these are defined elsewhere
from your_environment import GeneralAngleSpilloverEnv, BaseQuantumEnvironment
from your_utils import layer_init, plot_curves


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


class PPO_ContextBuffer:
    def __init__(
        self,
        agent_config: Dict,
        env: BaseQuantumEnvironment,
        chkpt_dir: Optional[str] = "tmp/ppo",
        chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
        num_warmup_updates: int = 100,
        context_buffer_size: int = 1000,
        noise_scale: float = 0.1,
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
        self.anneal_num_updates = self.agent_config["anneal_num_updates"]
        self.exp_anneal_lr = self.agent_config["exp_anneal_lr"]
        self.exp_update_time = self.agent_config["exp_update_time"]
        self.plateau_lr = self.agent_config["plateau_lr"]
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
        self.robust_ppo = self.agent_config["robust_ppo"]

        # Context Buffer Params
        self.num_warmup_updates = num_warmup_updates
        self.context_buffer_size = context_buffer_size
        self.noise_scale = noise_scale
        self.context_buffer = collections.deque(maxlen=context_buffer_size)
        self.context_rewards = []

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        assert (
            self.anneal_lr + self.exp_anneal_lr
        ), "Learning Rate Combination doesn't work"

        self.run_name = (
            f"PPO_ContextBuffer__{self.exp_name}__{self.seed}__{int(time.time())}"
        )

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

        # Seeding
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
            self.robust_ppo,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate, eps=1e-5
        )

    def add_to_context_buffer(self, context, reward):
        """Add context-reward pair to buffer and maintain size"""
        self.context_buffer.append(context)
        self.context_rewards.append(reward)
        if len(self.context_buffer) > self.context_buffer_size:
            self.context_buffer.popleft()
            self.context_rewards.pop(0)

    def sample_context(self, warmup=False):
        """Sample context from buffer or randomly during warmup"""
        if warmup or len(self.context_buffer) == 0:
            return np.random.uniform(
                self.env.observation_space.low,
                self.env.observation_space.high,
                self.env.observation_space.shape,
            )
        else:
            # Create probability distribution based on inverse rewards
            rewards = np.array(self.context_rewards)
            # Avoid division by zero and handle negative rewards
            rewards = np.clip(rewards, a_min=1e-6, a_max=None)
            prob_weights = 1.0 / rewards
            prob_weights = prob_weights / np.sum(prob_weights)

            # Sample context from buffer
            idx = np.random.choice(len(self.context_buffer), p=prob_weights)
            context = self.context_buffer[idx]

            # Add noise to sampled context
            noise = np.random.normal(0, self.noise_scale, context.shape)
            noisy_context = context + noise
            # Clip to observation space bounds
            noisy_context = np.clip(
                noisy_context,
                self.env.observation_space.low,
                self.env.observation_space.high,
            )

            return noisy_context

    def run_training(self):
        # Storage setup
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

        # Start training
        global_step = 0
        start_time = time.time()

        for update in range(1, self.num_updates + 1):
            # Determine if in warmup phase
            is_warmup = update <= self.num_warmup_updates

            # Sample new context
            context = self.sample_context(warmup=is_warmup)

            # Reset environment with sampled context
            next_obs, _ = self.env.reset(debug_obs=context)
            next_obs = np.tile(next_obs, (self.num_envs, 1))
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.zeros(self.num_envs).to(self.device)

            # Annealing the Learning Rate
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

                # Action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # Execute step
                cpu_action = action.cpu().numpy()
                mean_action = np.mean(cpu_action, axis=0)
                self.env.unwrapped.mean_action = self.env.action(mean_action)

                next_obs, reward, terminations, truncations, infos = self.env.step(
                    cpu_action
                )
                print(f"Mean Action: {np.mean(cpu_action, axis=0)}")

                # Add to context buffer
                mean_reward = np.mean(reward)
                self.add_to_context_buffer(context, mean_reward)

                next_obs = np.tile(next_obs, (self.num_envs, 1))
                next_done = np.logical_or(terminations, truncations)
                if next_done.any():
                    context = self.sample_context(warmup=is_warmup)
                    next_obs, _ = self.env.reset(debug_obs=context)
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

            # Bootstrap value
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

            # Flatten batch
            b_obs = self.obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimize policy and value network
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

            # Log metrics
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
                from IPython.display import clear_output

                clear_output(wait=True)
                if self.plot_real_time:
                    plot_curves(self.env.unwrapped)

        if self.save_model:
            model_path = f"runs/{self.run_name}/{self.exp_name}.cleanrl_model"
            torch.save(self.agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        self.env.close()
        self.writer.close()
