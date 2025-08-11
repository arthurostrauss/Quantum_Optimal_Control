import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Any
from dataclasses import dataclass
from ..context_aware_quantum_environment import ContextAwareQuantumEnvironment
from gymnasium.wrappers import RescaleObservation
from gymnasium.spaces import Box, Dict as DictSpace
from abc import ABC, abstractmethod

@dataclass
class ContextSamplingWrapperConfig:
    num_warmup_updates: int = 20
    context_buffer_size: int = 500
    sampling_prob: float = 0.3
    eviction_strategy: str = "hybrid"
    evict_best_prob: float = 0.2
    anneal_noise: bool = True
    initial_noise_scale: float = 0.15
    final_noise_scale: float = 0.05

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ContextSamplingWrapperConfig":
        return cls(**config_dict)
    
@dataclass
class SpilloverConfig:
    spillover_qubits: list
    target_subsystem: list
    discrete_history_length: int
    



    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SpilloverConfig":
        return cls(**config_dict)


class ContextSamplingWrapper(gym.Wrapper, ABC):
    """
    A Gymnasium wrapper that implements reward-guided context sampling.

    This wrapper manages a buffer of (context, reward, mean_action) tuples.
    On `reset()`, it samples a new context from this buffer (or randomly)
    and passes it to the underlying environment. On `step()`, it logs the
    resulting reward to update its buffer.
    """

    def __init__(
        self,
        env: ContextAwareQuantumEnvironment | gym.Wrapper,
        config: ContextSamplingWrapperConfig | Dict[str, Any],
    ):
        super().__init__(env)
        # Store hyperparameters
        self.config = ContextSamplingWrapperConfig.from_dict(config) if isinstance(config, dict) else  config

        # Buffers for context sampling logic
        self.context_buffer = []
        self.context_rewards = []
        self.mean_action_buffer = []

        # State tracking
        self.current_context = None
        self.total_updates_for_annealing = None  # Set by the PPO agent

        if not isinstance(self.observation_space, DictSpace):
            raise ValueError("Observation space must be a Dict space")
        for key, obs_space in self.observation_space.items():
            if not isinstance(obs_space, Box):
                raise ValueError(f"Observation space for key {key} must be a Box space")

    @property
    def observation_space(self) -> DictSpace:
        """
        Return the observation space of the environment.
        """
        return super().observation_space

    def _add_to_buffer(
        self, context:Dict[str, Any], reward: float, mean_action: np.ndarray
    ):
        """Adds context, reward, and action to buffers with eviction logic."""
        if context is None or mean_action is None:
            return  # Don't add if context/action aren't set

        self.context_buffer.append(context)
        self.context_rewards.append(reward)
        self.mean_action_buffer.append(mean_action)

        if len(self.context_buffer) > self.config.context_buffer_size:
            if (
                self.config.eviction_strategy == "hybrid"
                and self.np_random.random() < self.config.evict_best_prob
            ):
                idx_to_remove = np.argmax(self.context_rewards)  # Evict highest reward
            else:
                idx_to_remove = 0  # Evict oldest (FIFO)

            self.context_buffer.pop(idx_to_remove)
            self.context_rewards.pop(idx_to_remove)
            self.mean_action_buffer.pop(idx_to_remove)

    @abstractmethod
    def _sample_context(self) -> Dict[str, Any]:
        """Samples a context for the environment to reset to."""
        pass

    def reset(self, **kwargs):
        """Resets the environment with a sampled context."""
        # Sample a new context and store it for the upcoming episode
        self.current_context = self._sample_context()

        # Reset the underlying environment, passing the specific angles
        return self.env.reset(options=self.current_context)

    def step(self, action):
        """Takes a step and logs the result to the context buffer."""
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # The reward is of shape (batch_size, 1), we take the mean
        mean_reward = np.mean(reward)

        # Add the experience from this step to the buffer
        self._add_to_buffer(self.current_context), mean_reward, self.env.mean_action)

        # If the episode ends, clear the context. A new one will be sampled on reset.
        if terminated or truncated:
            self.current_context = None

        return next_obs, reward, terminated, truncated, info

class SpilloverContextSamplingWrapper(ContextSamplingWrapper):

    def _sample_context(self) -> Dict[str, Any]:
        """Samples a context for the environment to reset to."""
        is_warmup = self.env.unwrapped.step_tracker <= self.config.num_warmup_updates

        if is_warmup or len(self.context_buffer) == 0:
            return {
                "parameters": {key: self.np_random.uniform(
                obs_space.low,
                obs_space.high,
                obs_space.shape,
            ) for key, obs_space in self.observation_space.items()}}

        if self.np_random.random() < self.config.sampling_prob:
            # Rank-based prioritized replay
            rewards = np.array(self.context_rewards)
            ranks = np.argsort(np.argsort(rewards)) + 1  # Ranks from 1 (lowest reward)
            prob_weights = 1.0 / ranks
            prob_weights /= np.sum(prob_weights)

            idx = self.np_random.choice(len(self.context_buffer), p=prob_weights)
            context = self.context_buffer[idx]

            # Anneal noise added to the sampled context
            if self.config.anneal_noise and self.total_updates_for_annealing is not None:
                progress = (self.env.unwrapped.step_tracker - self.config.num_warmup_updates) / (
                    self.total_updates_for_annealing - self.config.num_warmup_updates
                )
                progress = np.clip(progress, 0.0, 1.0)
                current_noise_scale = (
                    self.config.initial_noise_scale * (1 - progress)
                    + self.config.final_noise_scale * progress
                )
            else:
                current_noise_scale = self.config.initial_noise_scale

            noise = self.np_random.normal(0, current_noise_scale, context.shape)
            noisy_context = context + noise

            return {
                "parameters": {key: np.clip(
                    noisy_context[i], obs_space.low, obs_space.high
                ) for i, (key, obs_space) in enumerate(self.observation_space.items())}
            }
        else:
            # Random exploration
            return {
                "parameters": {key: self.np_random.uniform(
                obs_space.low,
                obs_space.high,
                obs_space.shape,
            ) for key, obs_space in self.observation_space.items()}}
        
    # --- Plotting functions that depend on the buffer ---
    def plot_buffer_reward_distribution(self, **kwargs):
        if not self.context_rewards:
            print("No rewards in buffer to plot.")
            return
        plt.figure(figsize=(10, 6))
        plt.hist(
            self.context_rewards,
            bins=30,
            density=True,
            alpha=0.7,
            label="Buffer Rewards",
        )
        plt.xlabel("Reward")
        plt.ylabel("Density")
        plt.title("Distribution of Rewards in Context Buffer")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_policy_behaviour(
        self,
        gamma_matrix: np.ndarray,
        spillover_qubits: list,
        target_qubit: int,
        action_scale: float,
        filename="policy_behaviour.png",
    ):
        """
        Plots the agent's reward against the optimal action required to
        counteract spillover noise.

        A robust agent should show a flat reward profile, while a naive
        agent's reward will decrease as the optimal action magnitude increases.
        """
        # if len(self.context_buffer) < 100:
        #     print("Not enough data in buffer to plot policy behaviour.")
        #     return

        # 1. Convert buffers to numpy arrays for calculation
        obs_arr = np.array(self.context_buffer)
        rewards_arr = np.array(self.context_rewards)

        # 2. Calculate the optimal action based on the provided formula
        # Assumes the dimensions of obs_arr correspond to the spillover_qubits
        source_angles = 0.5 * (obs_arr + 1) * np.pi
        spillover_factors = gamma_matrix[spillover_qubits, target_qubit]

        # Calculate the total spillover angle for each context in the buffer
        total_spillover_angle = np.sum(
            source_angles * spillover_factors.reshape(1, -1), axis=-1
        )

        # The optimal action is the one that perfectly cancels this spillover
        optimal_action = -total_spillover_angle / action_scale

        # 3. Create the plot
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 7), dpi=100)

        # Scatter plot of raw data points
        plt.scatter(
            optimal_action,
            rewards_arr,
            alpha=0.1,
            label="Raw Data Points",
            color="royalblue",
        )

        # Calculate and plot a binned average to show the trend clearly
        try:
            bins = np.linspace(optimal_action.min(), optimal_action.max(), 15)
            bin_indices = np.digitize(optimal_action, bins)
            binned_rewards_mean = [
                rewards_arr[bin_indices == i].mean() for i in range(1, len(bins))
            ]
            binned_rewards_std = [
                rewards_arr[bin_indices == i].std() for i in range(1, len(bins))
            ]
            bin_centers = (bins[:-1] + bins[1:]) / 2

            plt.errorbar(
                bin_centers,
                binned_rewards_mean,
                yerr=binned_rewards_std,
                fmt="-o",
                color="red",
                markersize=8,
                capsize=5,
                linewidth=2.5,
                label="Binned Average Reward",
            )
        except Exception as e:
            print(f"Could not compute binned average for policy behaviour plot: {e}")

        plt.title("Agent Reward vs. Optimal Corrective Action")
        plt.xlabel("Optimal Action Magnitude (to counteract spillover)")
        plt.ylabel("Achieved Reward")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        # plt.savefig(f"runs/{self.run_name}/{filename}")
        plt.show()

    def plot_action_comparison(
        self,
        gamma_matrix: np.ndarray,
        spillover_qubits: list,
        target_qubit: int,
        action_scale: float,
        filename="action_comparison.png",
    ):
        """
        Plots the agent's actual output action against the analytically
        calculated optimal action.
        """
        # if len(self.context_buffer) < 100:
        #     print("Not enough data in buffer to plot action comparison.")
        #     return

        # 1. Get data from buffers
        obs_arr = np.array(self.context_buffer)
        agent_actions = np.array(self.mean_action_buffer)

        # 2. Calculate the optimal action for each observation
        source_angles = 0.5 * (obs_arr + 1) * np.pi
        spillover_factors = gamma_matrix[spillover_qubits, target_qubit]
        total_spillover_angle = np.sum(
            source_angles * spillover_factors.reshape(1, -1), axis=-1
        )
        optimal_action = -total_spillover_angle / action_scale

        # 3. Create the plot
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 7), dpi=100)

        plot_action_num = 0 if target_qubit == 2 else 3

        # Plot each dimension of the agent's action
        for i in range(agent_actions.shape[1]):
            if i == plot_action_num:
                plt.scatter(
                    optimal_action,
                    agent_actions[:, i],
                    alpha=0.2,
                    label=f"Agent Action Dim {i}",
                )

        # Plot the "perfect" policy line for reference
        plt.plot(
            optimal_action,
            optimal_action,
            "r--",
            linewidth=2.5,
            label="Analytical Optimal (y=x)",
        )

        plt.title("Agent's Learned Action vs. Optimal Action")
        plt.xlabel("Optimal Corrective Action (Calculated)")
        plt.ylabel("Actual Action (From Agent Policy)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--")
        # plt.axis('equal') # Ensures the y=x line is at a 45-degree angle
        plt.show()