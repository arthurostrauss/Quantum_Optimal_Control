import time

import wandb
from matplotlib.ticker import MaxNLocator
from ..environment.base_q_env import BaseQuantumEnvironment
import matplotlib.pyplot as plt
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from .ppo_config import HardwareRuntime


# def plot_curves(env: BaseQuantumEnvironment):
#     """
#     Plots the reward history and fidelity history of the environment
#     """
#     fidelity_range = [i * env.benchmark_cycle for i in range(len(env.fidelity_history))]
#     plt.plot(np.mean(env.reward_history, axis=1), label="Reward")
#     if env.target.target_type == "gate" and env.target.causal_cone_size < 3:
#         fidelity_type = "Avg gate"
#     else:
#         fidelity_type = "State"

#     plt.plot(
#         fidelity_range,
#         env.fidelity_history,
#         label=f"{fidelity_type} Fidelity",
#     )

#     plt.title("Reward History")
#     plt.legend()
#     plt.xlabel("Iteration")
#     plt.ylabel("Reward")
#     # Ensure integer ticks on the x-axis
#     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.show()


def plot_curves(env: BaseQuantumEnvironment):
    """
    Plots the reward history, fidelity history, and (for moving_discrete) marginal reward and probability distributions for each qubit.
    """
    fidelity_range = [i * env.benchmark_cycle for i in range(len(env.fidelity_history))]
    num_qubits = len(env.unwrapped.applied_qubits)  # Number of qubits
    # Number of subplots: 2 for reward/fidelity, plus 2 per qubit for moving_discrete
    num_plots = 2 + (
        2 * num_qubits
        if env.unwrapped.circuit_param_distribution == "moving_discrete"
        else 0
    )
    fig, ax = plt.subplots(num_plots, 1, figsize=(8.0, 6.0 * num_plots), squeeze=False)
    ax = ax.flatten()

    # Plot average reward and fidelity history
    ax[0].plot(np.mean(env.reward_history, axis=1), label="Reward")
    ax[0].plot(
        fidelity_range,
        env.fidelity_history,
        label="Circuit Fidelity",
    )
    ax[0].set_title("Reward History")
    ax[0].legend()
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Reward")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot RL reward history
    ax[1].plot(np.mean(env.reward_history, axis=1), label="RL Reward")
    ax[1].set_title("RL Reward History")
    ax[1].legend()
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("RL Reward")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    if env.unwrapped.circuit_param_distribution == "moving_discrete":
        # Single-qubit discrete values and their corresponding angles
        single_qubit_vals = env.unwrapped.single_qubit_discrete_obs_vals_raw
        angles = env.unwrapped.obs_raw_to_angles(single_qubit_vals)
        num_vals = len(single_qubit_vals)

        for qubit_idx in range(num_qubits):
            # Compute marginal reward distribution for this qubit
            marginal_rewards = np.zeros(num_vals)
            marginal_std = np.zeros(num_vals)
            for i, val in enumerate(single_qubit_vals):
                # Find indices where this qubit has the given value
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

            # Plot marginal reward history with error bars
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

            # Compute marginal probability distribution
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

            # Plot marginal probability weights
            ax[3 + qubit_idx * 2].scatter(
                angles,
                marginal_probs,
                label=f"Qubit {qubit_idx} Probability Weights",
            )
            ax[3 + qubit_idx * 2].set_title(
                f"Marginal Probability Weights (Qubit {qubit_idx})"
            )
            ax[3 + qubit_idx * 2].legend()
            ax[3 + qubit_idx * 2].set_xlabel("Observation Value (Angles)")
            ax[3 + qubit_idx * 2].set_ylabel("Probability")

    plt.show()


def log_fidelity_info_summary(training_constraint, fidelity_info):
    """
    Logs a summary of fidelity information.
    """

    for fidelity, info in fidelity_info.items():
        if info["achieved"]:
            logging.info(
                f"Target fidelity {fidelity} achieved: "
                f"Update {info['update_at']}, "
                f"Hardware Runtime: {round(info['hardware_runtime'], 2)} sec,"
                f" Simulation Train Time: {round(info['simulation_train_time'] / 60, 4)} mins,"
                f" Shots Used {info['shots_used']:,}"
            )
        else:
            logging.info(
                f"Target fidelity {fidelity} not achieved "
                f"within {training_constraint}"
                f"{'s' if isinstance(training_constraint, HardwareRuntime) else ''}."
            )


def write_to_tensorboard(
    writer: SummaryWriter,
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


def write_to_wandb(summary, training_results):
    """
    Writes the training results to Weights and Biases.
    """
    for key, value in summary.items():
        wandb.run.summary[key] = value

    wandb.define_metric("fidelity_history", summary="max")
    wandb.define_metric("avg_reward", summary="max")
    wandb.log(training_results)


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


def update_fidelity_info(
    fidelity_info,
    fidelities,
    target_fidelities,
    lookback_window,
    env: BaseQuantumEnvironment,
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
                        "hardware_runtime": np.sum(env.hardware_runtime),
                        "simulation_train_time": time.time() - start_time,
                        "shots_used": (
                            np.cumsum(env.total_shots)[update_step - 1]
                            if env.total_shots
                            else 0
                        ),
                        "shots_per_updates": (
                            int(
                                np.ceil(
                                    np.cumsum(env.total_shots)[update_step - 1]
                                    / update_step
                                )
                            )
                            if env.total_shots
                            else 0
                        ),
                    }
                )
                logging.warning(
                    f"Target fidelity {fidelity} surpassed at update {update_step}"
                )


def print_debug_info(
    env: BaseQuantumEnvironment, mean_action, std_action, b_returns, b_advantages
):
    """
    Print debug information for the training process.
    """
    print("mean", mean_action[0].numpy())
    print("sigma", std_action[0].numpy())
    print(
        "DFE Rewards Mean:",
        np.mean(env.reward_history, axis=1)[-1],
    )
    print(
        "DFE Rewards standard dev",
        np.std(env.reward_history, axis=1)[-1],
    )
    print("Returns Mean:", np.mean(b_returns.numpy()))
    print("Returns standard dev:", np.std(b_returns.numpy()))
    print("Advantages Mean:", np.mean(b_advantages.numpy()))
    print("Advantages standard dev", np.std(b_advantages.numpy()))
