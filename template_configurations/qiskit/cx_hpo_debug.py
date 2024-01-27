# %% [markdown]
# ## CX Calibration with HPO

# %% [markdown]
# #### Imports

# %%
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)

from template_configurations import gate_q_env_config
from quantumenvironment import QuantumEnvironment
from hyperparameter_optimization import HyperparameterOptimizer
from gymnasium.wrappers import RescaleAction, ClipAction

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s", # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

# %% [markdown]
# Which gate is to be calibrated?

# %%
gate_q_env_config.target

# %% [markdown]
# ### Perform HPO

# %% [markdown]
# Set path to the files specifying the RL agent and where to store the HPO results

# %%
current_dir = os.getcwd()

path_agent_config = '/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/agent_config.yaml' # os.path.join(os.path.dirname(current_dir), 'agent_config.yaml')
path_hpo_config = '/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/qiskit/config_yamls/hpo_config.yaml' # os.path.join(current_dir, 'config_yamls', 'hpo_config.yaml')
save_results_path = 'hpo_results'

# %% [markdown]
# #### Create the QuantumEnvironment, clip and rescale the action space

# %%
# Define the original action space
print('Initial loower bounds:', gate_q_env_config.action_space.low)
print('Initial upper bounds:', gate_q_env_config.action_space.high)

q_env = QuantumEnvironment(gate_q_env_config)

# Apply the RescaleAction wrapper
q_env = ClipAction(q_env)
q_env = RescaleAction(q_env, min_action=-1.0, max_action=1.0)

# Confirm the rescale box dimensions
print('Rescaled lower bounds:', q_env.action_space.low)
print('Rescaled upper bounds:', q_env.action_space.high)

# %%
optimizer = HyperparameterOptimizer(
    q_env=q_env,
    path_agent_config=path_agent_config,
    path_hpo_config=path_hpo_config, 
    save_results_path=save_results_path, 
    log_progress=True
)
best_trial = optimizer.optimize_hyperparameters(num_hpo_trials=2)

# %% [markdown]
# #### Quick Summary of HPO Task

# %%
optimizer.target_gate

# %%
optimizer.hyperparams

# %%
optimizer.num_hpo_trials

# %%
optimizer.best_hpo_configuration

# %%
import pickle
file_path = "/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/template_configurations/qiskit/hpo_results/reward_0.99126.pickle"

with open(file_path, "rb") as file:
    data = pickle.load(file)

# %%
data

# %%
import matplotlib.pyplot as plt

# %%
plt.plot(data['avg_reward'], label='Avg. Reward')
plt.plot(data['fidelity_history'], label='Fidelity')
plt.xlabel('Updates')
plt.title(f'Training History of {gate_q_env_config.targetp["gate"].name}-gate')
plt.legend()

# %%



