# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
module_path = os.path.abspath(
    os.path.join(
        "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control"
    )
)
if module_path not in sys.path:
    sys.path.append(module_path)

from correlated_noise_q_env_config import (
    q_env_config as gate_q_env_config,
    circuit_context,
)
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from hyperparameter_optimization import HyperparameterOptimizer
from gymnasium.wrappers import RescaleAction, ClipAction

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

# %%
q_env = ContextAwareQuantumEnvironment(gate_q_env_config, circuit_context)
q_env = ClipAction(q_env)
q_env = RescaleAction(q_env, -1.0, 1.0)

q_env.unwrapped.backend

# %%
q_env.unwrapped.circuit_truncations[0].draw("mpl")

# %%
q_env.unwrapped.circuit_truncations[0].decompose().draw("mpl")

# %%
current_dir = os.getcwd()
grand_parent_dir = os.path.dirname(os.path.dirname(current_dir))

path_agent_config = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml"  # os.path.join(os.path.dirname(grand_parent_dir), "agent_config.yaml")
path_hpo_config = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/qiskit/gate_level/noisy_circuit/noise_hpo_config.yaml"  # os.path.join(grand_parent_dir, "noise_hpo_config.yaml")
save_results_path = "hpo_results"

# %%
experimental_penalty_weights = {
    "runtime": 0.01,
    "n_shots": 0.01,
    "batchsize": 0.01,
    "sample_paulis": 0.015,
}

# %%
optimizer = HyperparameterOptimizer(
    q_env=q_env,
    path_agent_config=path_agent_config,
    path_hpo_config=path_hpo_config,
    save_results_path=save_results_path,
    experimental_penalty_weights=experimental_penalty_weights,
    log_progress=False,
)

# %%
best_trial = optimizer.optimize_hyperparameters(num_hpo_trials=2)

# %%
best_trial
