from typing import Optional, Dict
from IPython.display import clear_output
from gymnasium import ActionWrapper

# Torch imports for building RL agent and framework
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Optimizer

from .agent import Agent
from .ppo_config import TotalUpdates, TrainFunctionSettings, TrainingConfig, PPOConfig
from .ppo_initialization import initialize_networks
from .ppo_logging import *

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


class CustomPPO:
    def __init__(
        self,
        agent_config: Dict | PPOConfig | str,
        env: BaseQuantumEnvironment | ActionWrapper,
        chkpt_dir: Optional[str] = "tmp/ppo",
        chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
        save_data: Optional[bool] = False,
    ):
        """
        Custom Agent implementing Proximal Policy Optimization (PPO) using PyTorch.
        We define a custom class because the existing PPO implementation in stable-baselines3 does not support
        the submission of batches of actions to the environment, which is the typical use case for quantum control.

        Args:
            agent_config (Dict|PPOConfig|str): Configuration parameters for the PPO agent (can be a dictionary, PPOConfig object, or a file path to a YAML file).
            env (QuantumEnvironment): The quantum environment for training.
            chkpt_dir (Optional[str], optional): Directory to save the PPO agent's checkpoint files. Defaults to "tmp/ppo".
            chkpt_dir_critic (Optional[str], optional): Directory to save the critic network's checkpoint files. Defaults to "tmp/critic_ppo"
        """
        if isinstance(agent_config, str):
            agent_config = PPOConfig.from_yaml(agent_config)
        elif isinstance(agent_config, Dict):
            agent_config = PPOConfig.from_dict(agent_config)

        self.agent_config = agent_config
        self.env = env
        self.chkpt_dir = chkpt_dir
        self.chkpt_dir_critic = chkpt_dir_critic

        self._training_config = self.agent_config.training_config
        self._training_results = {
            "avg_reward": [],
            "fidelity_history": [],
            "hardware_runtime": [],
            "total_shots": [],
            "total_updates": [],
        }
        for i in range(self.unwrapped_env.n_actions):
            self._training_results[f"clipped_mean_action_{i}"] = []
            self._training_results[f"mean_action_{i}"] = []
            self._training_results[f"std_action_{i}"] = []

        self._train_function_settings = self.agent_config.train_function_settings
        if save_data:
            run_name = self.agent_config.run_name
            writer = SummaryWriter(f"runs/{run_name}")

            if self.agent_config.wandb_config.enabled:
                wandb.login(key=self.agent_config.wandb_config.api_key, verify=True)

        else:
            writer = None

        self.writer = writer
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize networks
        self.agent = initialize_networks(
            self.unwrapped_env.observation_space,
            agent_config.hidden_layers,
            self.unwrapped_env.n_actions,
            agent_config.hidden_activation_functions,
            agent_config.include_critic,
            agent_config.input_activation_function,
            agent_config.output_activation_mean,
            agent_config.output_activation_std,
            self.chkpt_dir,
            self.chkpt_dir_critic,
        )
        self.agent.to(self.device)
        # Initialize optimizer
        self.optimizer: Optimizer = self.agent_config.optimizer(
            self.agent.parameters(), lr=self.agent_config.learning_rate, eps=1e-5
        )

    def close(self):
        """
        Close all resources used by the agent.
        """
        self.unwrapped_env.close()
        if self.save_data:
            self.writer.close()
            wandb.finish()

    def process_action(self, probs: Normal):
        """
        Decide how actions should be processed before being sent to environment.
        For certain environments such as QUA, policy parameters should be streamed to the environment directly
        and actions are sampled within the environment (in real time).
        """

        # action = torch.clip(
        #     probs.sample(),
        #     torch.Tensor(self.min_action),
        #     torch.Tensor(self.max_action),
        # )
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

    def process_std(self, std_action):
        return std_action

    def train(
        self,
        training_config: Optional[TrainingConfig] = None,
        train_function_settings: Optional[TrainFunctionSettings] = None,
    ):
        """
        Trains the model using Proximal Policy Optimization (PPO) algorithm.

        Args:
            training_config (Optional[TrainingConfig], optional): Training configuration parameters. Defaults to None.
            train_function_settings (Optional[TrainFunctionSettings], optional): Training function settings. Defaults to None.
        Returns:
            Dict: A dictionary containing the training results, including average reward and fidelity history.
        """
        if training_config is not None:
            self._training_config = training_config
        if train_function_settings is not None:
            self._train_function_settings = train_function_settings

        if self.agent_config.wandb_config.enabled:
            wandb.init(
                project=self.agent_config.wandb_config.project,
                config=self.agent_config.as_dict() | self.unwrapped_env.config.as_dict(),
                name=self.agent_config.run_name,
                sync_tensorboard=True,
            )

        fidelity_info = {
            fidelity: {
                "achieved": False,
                "update_at": None,
                "train_time": None,
                "shots_used": None,
            }
            for fidelity in self.target_fidelities
        }

        # Ease access to all the parameters
        env, agent, optimizer = self.env, self.agent, self.optimizer
        u_env = self.unwrapped_env
        gamma, gae_lambda, n_epochs = self.gamma, self.gae_lambda, self.n_epochs
        num_time_steps, batch_size, minibatch_size = (
            self.num_time_steps,
            self.batch_size,
            self.minibatch_size,
        )
        ppo_epsilon, ent_coef, critic_loss_coef = (
            self.ppo_epsilon,
            self.ent_coef,
            self.critic_loss_coef,
        )
        clip_vloss, grad_clip = self.clip_vloss, self.grad_clip
        clip_coef, normalize_advantage = self.clip_coef, self.normalize_advantage
        obs_space = env.observation_space.shape
        action_space = env.action_space.shape

        obs = torch.zeros((num_time_steps, batch_size) + obs_space, device=self.device)
        actions = torch.zeros((num_time_steps, batch_size) + action_space, device=self.device)
        logprobs = torch.zeros((num_time_steps, batch_size), device=self.device)
        rewards = torch.zeros((num_time_steps, batch_size), device=self.device)
        dones = torch.zeros((num_time_steps, batch_size), device=self.device)
        values = torch.zeros((num_time_steps, batch_size), device=self.device)

        # avg_reward = fidelities = std_actions = avg_action_history = []
        start_time = time.time()

        # Clear the history of the environment if required
        if self.clear_history or self.hpo_mode:
            u_env.clear_history()
            self.global_step = 0
        try:
            ### Training Loop ###
            if isinstance(self.training_constraint, TotalUpdates):
                end_condition = self.training_constraint.total_updates
                step_tracker = lambda x: x
            elif isinstance(self.training_constraint, HardwareRuntime):
                end_condition = self.training_constraint.hardware_runtime
                step_tracker = lambda x: np.sum(self.unwrapped_env.hardware_runtime)
            else:
                raise ValueError(
                    "Invalid training constraint. Please provide either TotalUpdates or HardwareRuntime."
                )

            iteration = 0
            while step_tracker(iteration) < end_condition:
                if self.anneal_learning_rate:  # Anneal learning rate
                    self.learning_rate_annealing(iteration=iteration)

                # Reset the environment
                next_obs, _ = env.reset()
                batch_obs = torch.tile(torch.Tensor(next_obs, device=self.device), (batch_size, 1))
                batch_done = torch.zeros_like(dones[0], device=self.device)

                for step in range(num_time_steps):
                    self.global_step += 1
                    obs[step] = batch_obs
                    dones[step] = batch_done

                    with torch.no_grad():
                        mean_action, std_action, critic_value = agent(batch_obs)
                        std_action = self.process_std(std_action)
                        probs = Normal(mean_action, std_action)
                        action, logprob = self.process_action(probs)
                        values[step] = critic_value.flatten()

                    next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
                    next_obs = torch.Tensor(next_obs, device=self.device)
                    done = int(np.logical_or(terminated, truncated))
                    reward = torch.Tensor(reward, device=self.device)
                    rewards[step] = reward

                    actions[step], logprobs[step] = action, logprob

                    batch_obs = torch.tile(next_obs, (batch_size, 1))
                    next_done = done * torch.ones_like(dones[0], device=self.device)
                    obs[step] = batch_obs
                    dones[step] = next_done

                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards, device=self.device)
                    lastgaelam = 0
                    for t in reversed(range(num_time_steps)):
                        if t == num_time_steps - 1:
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

                # Flatten batches
                b_obs = obs.reshape((-1,) + env.observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + env.action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                b_inds = np.arange(batch_size)
                clipfracs = []

                # Optimization loop
                for epoch in range(n_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        new_mean, new_sigma, new_value = agent(b_obs[mb_inds])
                        new_dist = Normal(new_mean, new_sigma)
                        new_logprob = new_dist.log_prob(b_actions[mb_inds]).sum(1)
                        entropy = new_dist.entropy().sum(1)
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
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1 - ppo_epsilon, 1 + ppo_epsilon
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        new_value = new_value.view(-1)
                        if clip_vloss:
                            v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                new_value - b_values[mb_inds],
                                -clip_coef,
                                clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - ent_coef * entropy_loss + v_loss * critic_loss_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
                        optimizer.step()

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                mean_action_np = mean_action[0].cpu().numpy()
                std_action_np = std_action[0].cpu().numpy()
                # Print debug information
                if self.print_debug:
                    print_debug_info(u_env, mean_action, std_action, b_returns, b_advantages)

                if self.target_fidelities is not None:
                    update_fidelity_info(
                        fidelity_info,
                        u_env.fidelity_history,
                        self.target_fidelities,
                        self.lookback_window,
                        u_env,
                        iteration,
                        mean_action_np,
                        std_action_np,
                        start_time,
                    )

                if self.global_step % self.num_prints == 0:
                    clear_output(wait=False)
                    if self.plot_real_time:
                        plt.close()
                        plot_curves(u_env)
                        plt.draw()
                if self.save_data:
                    write_to_tensorboard(
                        self.writer,
                        self.global_step,
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
                summary = {
                    "n_reps": u_env.n_reps,
                    "training_constraint": self.training_constraint.constraint_name,
                    "env_ident_str": u_env.ident_str,
                    "reward_method": u_env.config.reward.reward_method,
                }
                training_results = {
                    "avg_reward": np.mean(u_env.reward_history, axis=1)[-1].tolist(),
                    "total_shots": int(u_env.reward_data.total_shots),
                }
                if u_env.backend_info.instruction_durations is not None:
                    training_results["hardware_runtime"] = u_env.hardware_runtime[-1]
                if u_env.do_benchmark():
                    training_results["fidelity_history"] = u_env.fidelity_history[-1]

                for i in range(u_env.n_actions):
                    training_results[f"clipped_mean_action_{i}"] = (
                        env.action(mean_action_np)[i].tolist()
                        if isinstance(env, ActionWrapper)
                        else mean_action_np[i].tolist()
                    )
                    training_results[f"mean_action_{i}"] = mean_action_np[i].tolist()
                    training_results[f"std_action_{i}"] = std_action_np[i].tolist()

                if self.agent_config.wandb_config.enabled:
                    write_to_wandb(summary, training_results)
                for key, value in training_results.items():
                    self._training_results[key].append(value)
                self._training_results["mean_action"] = mean_action_np.tolist()
                self._training_results["std_action"] = std_action_np.tolist()
                self._training_results["clipped_mean_action"] = env.action(mean_action_np).tolist()
                iteration += 1

                if check_convergence_std_actions(std_action, self.std_actions_eps):
                    break
            if self.clear_history:
                self.close()
            log_fidelity_info_summary(self.training_constraint, fidelity_info)
            return self.training_results
        except KeyboardInterrupt:
            self.close()
            return self.training_results

        except Exception as e:
            if self.hpo_mode:  # Return a default value for HPO
                logging.error(f"An error occurred during training: {e}")
                return {
                    "avg_reward": -1.0,
                    "fidelity_history": [0] * int(self.training_constraint.constraint_value),
                }
            else:  # Raise the error for debugging in the normal mode
                raise

    def learning_rate_annealing(
        self,
        iteration: Optional[int] = None,
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
        constraint_val = self.training_constraint.constraint_value
        if isinstance(self.training_constraint, TotalUpdates):
            frac = 1.0 - (iteration - 1.0) / constraint_val
        elif isinstance(self.training_constraint, HardwareRuntime):
            frac = 1.0 - (np.sum(self.unwrapped_env.hardware_runtime) / constraint_val)
        else:
            raise ValueError(
                "Invalid training constraint. Please provide either TotalUpdates or HardwareRuntime."
            )
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow

    @property
    def batch_size(self):
        """
        The size of the batch
        """
        return self.unwrapped_env.batch_size

    @property
    def num_time_steps(self):
        """
        The number of time steps in the environment (number of times the target gate is applied)
        """
        return self.unwrapped_env.episode_length(self.global_step)

    @property
    def n_actions(self):
        """
        The number of actions in the environment
        """
        return self.unwrapped_env.n_actions

    @property
    def global_step(self):
        """
        The global step count
        """
        return self.unwrapped_env.step_tracker

    @global_step.setter
    def global_step(self, value: int):
        if not isinstance(value, int):
            raise ValueError("global_step must be an integer")
        self.unwrapped_env.step_tracker = value

    @property
    def seed(self):
        """
        The seed of the environment
        """
        return self.unwrapped_env.seed

    @property
    def min_action(self):
        """
        The minimum action value in the environment
        """
        if hasattr(self.unwrapped_env, "min_action"):
            return self.unwrapped_env.min_action
        return self.unwrapped_env.action_space.low

    @property
    def max_action(self):
        """
        The maximum action value in the environment
        """
        if hasattr(self.unwrapped_env, "max_action"):
            return self.unwrapped_env.max_action
        return self.unwrapped_env.action_space.high

    @property
    def training_results(self):
        return self._training_results

    @property
    def training_config(self):
        return self._training_config

    @property
    def train_function_settings(self):
        return self._train_function_settings

    @property
    def target_fidelities(self):
        return self.training_config.target_fidelities

    @property
    def training_constraint(self):
        return self.training_config.training_constraint

    @property
    def lookback_window(self):
        """
        The lookback window for checking if the learning has converged to a stable value.
        """
        return self.training_config.lookback_window

    @property
    def std_actions_eps(self):
        """
        The standard deviation of actions to which the training should converge.
        """
        return self.training_config.std_actions_eps

    @property
    def anneal_learning_rate(self):
        """
        Whether to anneal the learning rate during training
        """
        return self.training_config.anneal_learning_rate

    @property
    def plot_real_time(self):
        """
        Whether to plot the training progress in real time.
        """
        return self.train_function_settings.plot_real_time

    @plot_real_time.setter
    def plot_real_time(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("plot_real_time must be a boolean")
        self.train_function_settings.plot_real_time = value

    @property
    def print_debug(self):
        return self.train_function_settings.print_debug

    @print_debug.setter
    def print_debug(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("print_debug must be a boolean")
        self.train_function_settings.print_debug = value

    @property
    def num_prints(self):
        return self.train_function_settings.num_prints

    @num_prints.setter
    def num_prints(self, value: int):
        if not isinstance(value, int):
            raise ValueError("num_prints must be an integer")
        self.train_function_settings.num_prints = value

    @property
    def hpo_mode(self):
        return self.train_function_settings.hpo_mode

    @property
    def clear_history(self):
        """
        Whether to clear the history of the environment after training.
        """
        return self.train_function_settings.clear_history

    @clear_history.setter
    def clear_history(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("clear_history must be a boolean")
        self.train_function_settings.clear_history = value

    @property
    def unwrapped_env(self) -> BaseQuantumEnvironment:
        return self.env.unwrapped

    @property
    def save_data(self):
        return self.writer is not None

    @property
    def n_epochs(self):
        """
        The number of epochs for training
        """
        return self.agent_config.n_epochs

    @property
    def lr(self):
        """
        The learning rate for training
        """
        return self.agent_config.learning_rate

    @property
    def ppo_epsilon(self):
        """
        The PPO epsilon value
        """
        return self.agent_config.clip_ratio

    @property
    def critic_loss_coef(self):
        """
        The coefficient for the critic loss
        """
        return self.agent_config.value_loss_coef

    @property
    def gamma(self):
        """
        The discount factor
        """
        return self.agent_config.gamma

    @property
    def gae_lambda(self):
        """
        The GAE lambda value
        """
        return self.agent_config.gae_lambda

    @property
    def clip_vloss(self):
        """
        Whether to clip the value loss
        """
        return self.agent_config.clip_value_loss

    @property
    def grad_clip(self):
        """
        The gradient clipping value
        """
        return self.agent_config.gradient_clip

    @property
    def clip_coef(self):
        """
        The clipping coefficient
        """
        return self.agent_config.clip_value_coef

    @property
    def normalize_advantage(self):
        """
        Whether to normalize the advantage
        """
        return self.agent_config.normalize_advantage

    @property
    def ent_coef(self):
        """
        The entropy coefficient
        """
        return self.agent_config.entropy_coef

    @property
    def include_critic(self):
        """
        Whether to include a critic network
        """
        return self.agent_config.include_critic

    @property
    def minibatch_size(self):
        """
        The size of the minibatch
        """
        return self.agent_config.minibatch_size


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
    agent: Agent,
    env: BaseQuantumEnvironment | ActionWrapper,
    writer,
):
    """
    Takes a step in the environment using the PPO algorithm.

    Args:
        step (int): current step index.
        global_step (int): global step index.
        batchsize (int): Size of the batch.
        num_steps (int): Total number of steps.
        obs (list): List to store observations.
        dones (list): List to store done flags.
        actions (list): List to store actions.
        logprobs (list): List to store log probabilities.
        rewards (list): List to store rewards.
        values (list): List to store critic values.
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
        env.unwrapped.mean_action = env.action(mean_action.cpu().numpy())[0]
        env.unwrapped.std_action = std_action[0]
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
    if writer is not None:
        writer.add_scalar("charts/episodic_return", np.mean(reward.numpy()), global_step)
        writer.add_scalar("charts/episodic_length", num_steps, global_step)
        for i in range(env.unwrapped.mean_action.shape[0]):
            writer.add_scalar(
                f"charts/clipped_mean_action_{i}",
                env.unwrapped.mean_action[i],
                global_step,
            )
            writer.add_scalar(f"charts/std_action_{i}", np.array(std_action)[i], global_step)
            writer.add_scalar(f"charts/unclipped_action_{i}", np.array(action)[i], global_step)

    return next_obs, next_done, mean_action, std_action
