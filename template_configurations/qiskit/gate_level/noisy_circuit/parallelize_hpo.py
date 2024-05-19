import sys
import os
import numpy as np
from itertools import product
import multiprocessing

from get_nm_cmaes_ideal_actions_noisy_circ import get_optimized_params

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
module_path = os.path.abspath(
    os.path.join(
        "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control"
    )
)
if module_path not in sys.path:
    sys.path.append(module_path)

from correlated_noise_q_env_config_function import setup_quantum_environment

# from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from hyperparameter_optimization_resource_constraint import HyperparameterOptimizer
from gymnasium.spaces import Box
from gymnasium.wrappers import RescaleAction, ClipAction

from helper_functions import load_from_pickle

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def scale_action_space(
    phi_gamma_tuple,
    initial_phi=np.pi,
    initial_gamma=0.05,
    initial_space=Box(-0.1, 0.1, (7,), np.float32),
):
    phi, gamma = phi_gamma_tuple

    if gamma <= initial_gamma:
        logging.warning(
            "Do not scale down the action space for smaller noise gamma<0.05 and return initial action space."
        )
        return initial_space

    # Calculate the initial and new ratios
    initial_ratio = initial_phi * initial_gamma
    new_ratio = phi * gamma

    # Scale the action space based on the ratio
    scale_factor = new_ratio / initial_ratio
    new_low = initial_space.low * scale_factor
    new_high = initial_space.high * scale_factor

    # Create a new action space with the scaled values
    new_space = Box(new_low, new_high, initial_space.shape, initial_space.dtype)

    return new_space


def perform_hpo_noisy_single_arg(phi_gamma_tuple):
    gate_q_env_config, circuit_context, _ = setup_quantum_environment(
        phi_gamma_tuple=phi_gamma_tuple
    )
    # gate_q_env_config.action_space = scale_action_space(phi_gamma_tuple=phi_gamma_tuple)

    optimal_noise_free_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
    backend = gate_q_env_config.backend_config.backend
    print(backend)

    # action_optimization_result = get_optimized_params(
    #     optimal_noise_free_params=optimal_noise_free_params,
    #     phi_val=phi_gamma_tuple[0],
    #     gamma_val=phi_gamma_tuple[1],
    #     backend=backend,
    # )

    # Retrieve the values to tailor the new action space
    action_optimization_result = load_from_pickle('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/qiskit/gate_level/noisy_circuit/optimization_results_NM_CMAES.pickle')
    action_deviations = action_optimization_result["nelder_mead"]["optimal_deviations"]
    action_space_borders = round(
        1.2 * max(np.abs(action_deviations)), 3
    )  # use 20% margin
    action_space_borders = min(action_space_borders, np.pi)
    gate_q_env_config.action_space = Box(
        -action_space_borders, action_space_borders, (7,), np.float32
    )

    print(
        f"Action space for phi={phi_gamma_tuple[0]} and gamma={phi_gamma_tuple[1]}: {gate_q_env_config.action_space.low} to {gate_q_env_config.action_space.high}"
    )

    q_env = ContextAwareQuantumEnvironment(gate_q_env_config, circuit_context)
    q_env = ClipAction(q_env)
    q_env = RescaleAction(q_env, -1.0, 1.0)

    path_agent_config = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml"  # os.path.join(os.path.dirname(grand_parent_dir), "agent_config.yaml")
    path_hpo_config = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/qiskit/gate_level/noisy_circuit/noise_hpo_config.yaml"  # os.path.join(current_dir, "noise_hpo_config.yaml")
    save_results_path = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/qiskit/gate_level/noisy_circuit/hpo_results/resource_constraint"  # os.path.join("hpo_results", "resource_constraint")

    target_fidelities = [1.0 - infid for infid in [1e-3, 1e-4, 1e-5]]

    # TODO: Set these values to the desired values
    max_runtime_per_trial = 600
    num_hpo_trials = 20
    lookback_window = 8

    experimental_penalty_weights = {
        "penalty_n_shots": 0.01,
        "penalty_per_missed_fidelity": 1e4,
        "fidelity_reward": 2 * 1e4,
    }

    optimizer = HyperparameterOptimizer(
        q_env=q_env,
        path_agent_config=path_agent_config,
        path_hpo_config=path_hpo_config,
        save_results_path=save_results_path,
        saving_mode="all",
        experimental_penalty_weights=experimental_penalty_weights,
        log_progress=False,
    )

    _ = optimizer.optimize_hyperparameters(
        num_hpo_trials=num_hpo_trials,
        phi_gamma_tuple=phi_gamma_tuple,
        target_fidelities=target_fidelities,
        lookback_window=lookback_window,
        max_runtime=max_runtime_per_trial,
    )


# %%
if __name__ == "__main__":
    phis = np.pi * np.array([0.25, 0.5, 0.75, 1.0])
    gammas = np.array([0.1, 0.05, 0.10, 0.15])  # np.logspace(-4, -2, 3)

    # Create all combinations of phis and gammas
    combinations = list(product(phis, gammas))

    # Use multiprocessing to parallelize across multiple CPU cores
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(perform_hpo_noisy_single_arg, combinations)
    pool.close()
    pool.join()
