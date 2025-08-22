from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.transpiler import PassManager
from rl_qoc.environment.wrappers import ContextSamplingWrapper, ContextSamplingWrapperConfig
from gymnasium.spaces import Dict as DictSpace, Box
from rl_qoc.environment.context_aware_quantum_environment import ContextAwareQuantumEnvironment
import matplotlib.pyplot as plt
from gate_level.spillover_noise_use_case.generic_spillover.spillover_effect_on_subsystem import noisy_backend, LocalSpilloverNoiseAerPass, numpy_to_hashable
from rl_qoc.helpers.transpiler_passes import CausalConePass

def obs_dict_to_array(obs_dict: Dict[str, Any]) -> np.ndarray:
    """
    Convert a dictionary of observations to a numpy array.
    """
    return np.array([obs_dict[key] for key in obs_dict.keys()])

@dataclass
class SpilloverConfig:
    gamma_matrix: np.ndarray
    target_subsystem: Tuple[int, int]
    spillover_qubits: List[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SpilloverConfig":
        return cls(**config_dict)
    
    def __post_init__(self):
        if self.spillover_qubits is None:
            self.spillover_qubits = list(range(len(self.gamma_matrix)))


class SpilloverContextSamplingWrapper(ContextSamplingWrapper):
    """
    A wrapper for environments that sample contexts based on spillover noise
    """
    def __init__(self, env: ContextAwareQuantumEnvironment,
                 spillover_config: SpilloverConfig,
                 context_config: ContextSamplingWrapperConfig | Dict[str, Any] = None):
        super().__init__(env, context_config)
        self.spillover_config = SpilloverConfig.from_dict(spillover_config) if isinstance(spillover_config, dict) else spillover_config
        keys = list(self.observation_space.keys())
        spillover_keys = [keys[i] for i in self.spillover_config.spillover_qubits]
        self.spillover_space = DictSpace({key: self.observation_space[key] for key in spillover_keys})
        self.custom_passes = [LocalSpilloverNoiseAerPass(numpy_to_hashable(self.spillover_config.gamma_matrix),
                                                         self.spillover_config.target_subsystem),
                                                         CausalConePass(self.spillover_config.target_subsystem)]

    def sample_context(self) -> Dict[str, Any]:
        """Samples a context for the environment to reset to."""
        is_warmup = self.env.step_tracker <= self.context_config.num_warmup_updates
        context =  {key: np.zeros(space.shape) for key, space in self.observation_space.items()}
        if is_warmup or len(self.context_buffer) == 0:
            for key, space in self.spillover_space.items():
                if isinstance(space, Box):
                    context[key] = self.np_random.uniform(space.low, space.high, space.shape)
                else:
                    raise ValueError(f"Spillover space for key {key} must be a Box space")
            
            return {"parameters": context}

        if self.np_random.random() < self.context_config.sampling_prob:
            # Rank-based prioritized replay
            rewards = np.array(self.context_rewards)
            ranks = np.argsort(np.argsort(rewards)) + 1  # Ranks from 1 (lowest reward)
            prob_weights = 1.0 / ranks
            prob_weights /= np.sum(prob_weights)

            idx = self.np_random.choice(len(self.context_buffer), p=prob_weights)
            context = obs_dict_to_array(self.context_buffer[idx]["parameters"])

            # Anneal noise added to the sampled context
            if self.context_config.anneal_noise and self.env.total_updates is not None:
                progress = (self.env.unwrapped.step_tracker - self.context_config.num_warmup_updates) / (
                    self.env.total_updates - self.context_config.num_warmup_updates
                )
                progress = np.clip(progress, 0.0, 1.0)
                current_noise_scale = (
                    self.context_config.initial_noise_scale * (1 - progress)
                    + self.context_config.final_noise_scale * progress
                )
            else:
                current_noise_scale = self.context_config.initial_noise_scale

            noise = {key:self.np_random.normal(0, current_noise_scale, space.shape) for key, space in self.spillover_space.items()}
            for key, val in noise.items():
                if isinstance(self.spillover_space[key], Box):
                    context[key] = np.clip(context[key] + val, self.spillover_space[key].low, self.spillover_space[key].high)
                else:
                    raise ValueError(f"Spillover space for key {key} must be a Box space")

            return {
                "parameters": context
            }
        else:
            # Random exploration
            for key, space in self.spillover_space.items():
                if isinstance(space, Box):
                    context[key] = self.np_random.uniform(space.low, space.high, space.shape)
                else:
                    raise ValueError(f"Spillover space for key {key} must be a Box space")
            return {
                "parameters": context
            }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        backend = noisy_backend(
            self.env.target.circuit, self.spillover_config.gamma_matrix, self.spillover_config.target_subsystem
        )
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        custom_translation_pass = PassManager(self.custom_passes)
        custom_translation_pass.append(pm.translation.to_flow_controller().passes)
        pm.translation = custom_translation_pass
        self.env.set_env_params(backend=backend, 
                                pass_manager=pm)
        
        return obs, info
        
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
        target_qubit: int,
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
        spillover_factors = self.spillover_config.gamma_matrix[self.spillover_config.spillover_qubits, target_qubit]

        # Calculate the total spillover angle for each context in the buffer
        total_spillover_angle = np.sum(
            source_angles * spillover_factors.reshape(1, -1), axis=-1
        )

        # The optimal action is the one that perfectly cancels this spillover
        optimal_action = -total_spillover_angle / np.mean(self.env.action_space.high)

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
        target_qubit: int,
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
        spillover_factors = self.spillover_config.gamma_matrix[self.spillover_config.spillover_qubits, target_qubit]
        total_spillover_angle = np.sum(
            source_angles * spillover_factors.reshape(1, -1), axis=-1
        )
        optimal_action = -total_spillover_angle / np.mean(self.env.action_space.high)

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

    def plot_all(self):
        """
        Plot all relevant plots for the spillover use case
        """

        self.plot_buffer_reward_distribution()
        for tgt_qubit in self.spillover_config.target_subsystem:
            self.plot_policy_behaviour(tgt_qubit)
        for tgt_qubit in self.spillover_config.target_subsystem:
            self.plot_action_comparison(tgt_qubit)

    