from rl_qoc.helper_functions import load_from_yaml_file
from rl_qoc.quantumenvironment import QuantumEnvironment
from rl_qoc.context_aware_quantum_environment import ContextAwareQuantumEnvironment
from gymnasium.wrappers import RescaleAction, ClipAction
from ppo import CustomPPO
from qiskit_serverless import save_result, get_arguments
import numpy as np

arguments = get_arguments()
use_context = arguments.get("use_context")
abstraction_level = arguments.get("abstraction_level")
if abstraction_level == "pulse":
    from pulse_config import (
        q_env_config as pulse_q_env_config,
        circuit_context as pulse_circuit_context,
    )

    config = pulse_q_env_config
    circuit_context = pulse_circuit_context
else:
    from q_env_config import (
        q_env_config as gate_q_env_config,
        circuit_context as gate_circuit_context,
    )

    config = gate_q_env_config
    circuit_context = gate_circuit_context

if use_context:
    q_env = ContextAwareQuantumEnvironment(
        config, circuit_context, training_steps_per_gate=250
    )
else:
    q_env = QuantumEnvironment(config)
rescaled_env = RescaleAction(ClipAction(q_env), -1.0, 1.0)
agent_config = load_from_yaml_file("agent_config.yaml")
ppo_agent = CustomPPO(agent_config, rescaled_env)
ppo_agent.train(
    total_updates=arguments.get("num_updates"),
    print_debug=False,
    num_prints=0,
    clear_history=False,
    plot_real_time=False,
)

reward_history = np.array(q_env.reward_history)
mean_rewards = np.mean(reward_history, axis=-1)
fid = q_env.fidelity_history

results = {
    "rewards": mean_rewards.to_list(),
    "fidelities": fid,
    "optimal_action": q_env.optimal_action,
}

save_result(results)
