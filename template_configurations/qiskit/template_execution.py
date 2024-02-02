from template_configurations.qiskit.pulse_level.pulse_config import (
    q_env_config as pulse_q_env_config,
)
from quantumenvironment import QuantumEnvironment
from ppo import make_train_ppo
import yaml

with open(
    "/Users/arthurostrauss/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Coding_projects/Quantum_Optimal_Control/template_configurations/agent_config.yaml",
    "r",
) as f:
    agent_config = yaml.safe_load(f)

q_env = QuantumEnvironment(pulse_q_env_config)

ppo_agent = make_train_ppo(agent_config, q_env)
ppo_agent(total_updates=2, print_debug=True, num_prints=1)

print(q_env.avg_fidelity_history)
