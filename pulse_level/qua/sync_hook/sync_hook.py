import json

from rl_qoc.agent import PPOConfig, TrainingConfig, TrainFunctionSettings, TotalUpdates
from rl_qoc.qua import QMEnvironment, CustomQMPPO, QMConfig
from iqcc_cloud_client.runtime import get_qm_job
from rl_qoc import (
    RescaleAndClipAction,
    GateTarget,
    StateTarget,
    ChannelReward,
    ExecutionConfig,
    QEnvConfig,
    BenchmarkConfig,
)
import numpy as np
from gymnasium.spaces import Box

job = get_qm_job()
physical_qubits = (0,)
target_gate = "x"
gate_target = GateTarget(gate=target_gate, physical_qubits=physical_qubits)
reward = ChannelReward()

# Action space specification
param_bounds = [(-1.98, 2.0)]  # Can be any number of bounds


# Environment execution parameters
seed = 1203  # Master seed to make training reproducible
batch_size = 32  # Number of actions to evaluate per policy evaluation
n_shots = 100  # Minimum number of shots per fiducial evaluation
pauli_sampling = 100  # Number of fiducials to compute for fidelity estimation (DFE only)
n_reps = 1  # Number of repetitions of the cycle circuit
num_updates = TotalUpdates(50)

input_type = "INPUT_STREAM"
backend_config = QMConfig(
    num_updates=num_updates.total_updates,
    input_type=input_type,
    verbosity=2,
)

execution_config = ExecutionConfig(
    batch_size=batch_size,
    sampling_paulis=pauli_sampling,
    n_shots=n_shots,
    n_reps=n_reps,
    seed=seed,
)


def create_action_space(param_bounds):
    param_bounds = np.array(param_bounds, dtype=np.float32)
    lower_bound, upper_bound = param_bounds.T
    return Box(low=lower_bound, high=upper_bound, shape=(len(param_bounds),), dtype=np.float32)


action_space = create_action_space(param_bounds)
q_env_config = QEnvConfig(
    target=gate_target,
    backend_config=backend_config,
    action_space=action_space,
    execution_config=execution_config,
    reward=reward,
    benchmark_config=BenchmarkConfig(0),  # No benchmark for now)
)
q_env = QMEnvironment(q_env_config, job=job)
rescaled_env = RescaleAndClipAction(q_env, -1.0, 1.0)
ppo_config = PPOConfig()

ppo_agent = CustomQMPPO(ppo_config, rescaled_env)

ppo_training = TrainingConfig(num_updates)
ppo_settings = TrainFunctionSettings(plot_real_time=True, print_debug=True)

results = ppo_agent.train(ppo_training, ppo_settings)

print(json.dumps(results))
