import base64
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, Gate
from typing import List, Optional, Union, Dict, Any, Tuple
from gymnasium.spaces import Box
from rl_qoc.qua import QMEnvironment, QMConfig
from iqcc_calibration_tools.quam_config.components import Quam, Transmon
from qiskit_qm_provider import (
    FluxTunableTransmonBackend,
    QMInstructionProperties,
    InputType,
    ParameterPool,
)
from qiskit_qm_provider.backend.backend_utils import add_basic_macros_to_machine
from rl_qoc.agent.ppo_config import (
    TotalUpdates,
    TrainingConfig,
)
from rl_qoc import (
    RescaleAndClipAction,
    ChannelReward,
    StateReward,
    CAFEReward,
    ExecutionConfig,
    QEnvConfig,
    BenchmarkConfig,
    StateTarget,
    GateTarget,
    PPOConfig,
)
from rl_qoc.helpers import add_custom_gate
from rl_qoc.helpers import load_from_yaml_file
from iqcc_cloud_client import IQCC_Cloud
import json
import os
from pathlib import Path

from rl_qoc.qua.pi_pulse_reward.pi_pulse_reward import PiPulseReward
from rl_qoc.qua.qua_ppo import CustomQMPPO
from rl_qoc.rewards.base_reward import Reward
from rl_qoc.qua.iqcc import generate_sync_hook, get_machine_from_iqcc

# Set your quantum computer backend
path = Path.home() / "iqcc_token.json"
with open(path, "r") as f:
    iqcc_config = json.load(f)

backend_name = "gilboa"

machine, iqcc = get_machine_from_iqcc(backend_name, iqcc_config[backend_name])

add_basic_macros_to_machine(machine)
backend = FluxTunableTransmonBackend(machine)
path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
ppo_config = load_from_yaml_file(path)


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: List[Parameter], q_reg: QuantumRegister, **kwargs
):

    physical_qubits: List[int] = kwargs["physical_qubits"]
    backend: FluxTunableTransmonBackend = kwargs["backend"]

    # TODO: Enter your custom parametric QUA macro here
    def qua_macro(amp):
        qubit: Transmon = backend.get_qubit(physical_qubits[0])
        qubit.xy.play("x180", amplitude_scale=amp)

    # Create a custom gate with the QUA macro
    custom_x = Gate("x_cal", 1, params)
    instruction_prop = QMInstructionProperties(qua_pulse_macro=qua_macro)
    qc = add_custom_gate(qc, custom_x, q_reg, params, physical_qubits, backend, instruction_prop)
    return qc


physical_qubits = (0,)

target_name = "x"
target = GateTarget(gate=target_name, physical_qubits=physical_qubits)
reward = ChannelReward()

target_name = "1"
target = StateTarget(state=target_name, physical_qubits=physical_qubits)
reward = PiPulseReward()


# Action space specification
param_bounds = [(-1.98, 2.0)]  # Can be any number of bounds

# Environment execution parameters
seed = 1203  # Master seed to make training reproducible
batch_size = 32  # Number of actions to evaluate per policy evaluation
n_shots = 50  # Minimum number of shots per fiducial evaluation
pauli_sampling = 100  # Number of fiducials to compute for fidelity estimation (DFE only)
n_reps = 1  # Number of repetitions of the cycle circuit
num_updates = TotalUpdates(50)
input_type = InputType.INPUT_STREAM


def create_action_space(param_bounds):
    param_bounds = np.array(param_bounds, dtype=np.float32)
    lower_bound, upper_bound = param_bounds.T
    return Box(low=lower_bound, high=upper_bound, shape=(len(param_bounds),), dtype=np.float32)


action_space = create_action_space(param_bounds)

backend_config = QMConfig(
    parametrized_circuit=apply_parametrized_circuit,
    backend=backend,
    input_type=input_type,
    verbosity=2,
    parametrized_circuit_kwargs={"physical_qubits": physical_qubits, "backend": backend},
    num_updates=num_updates.total_updates,
)
execution_config = ExecutionConfig(
    batch_size=batch_size,
    sampling_paulis=pauli_sampling,
    n_shots=n_shots,
    n_reps=n_reps,
    seed=seed,
)
q_env_config = QEnvConfig(
    target=target,
    backend_config=backend_config,
    action_space=action_space,
    execution_config=execution_config,
    reward=reward,
    benchmark_config=BenchmarkConfig(0),
)  # No benchmark for now

q_env = QMEnvironment(training_config=q_env_config)
rescaled_env = RescaleAndClipAction(q_env, -1.0, 1.0)

# Générer le fichier sync_hook.py avant l'exécution
sync_hook_path = generate_sync_hook(
    target=target,
    reward=reward,
    param_bounds=param_bounds,
    seed=seed,
    batch_size=batch_size,
    n_shots=n_shots,
    pauli_sampling=pauli_sampling,
    n_reps=n_reps,
    num_updates=num_updates,
    input_type=input_type,
    backend_config=backend_config,
    ppo_config=ppo_config,
)
print(f"Sync hook file generated at: {sync_hook_path}")


if hasattr(q_env.real_time_circuit, "calibrations") and q_env.real_time_circuit.calibrations:
    backend.update_calibrations(qc=q_env.real_time_circuit, input_type=input_type)
backend.update_compiler_from_target()
prog = q_env.rl_qoc_training_qua_prog(num_updates=num_updates.total_updates)
run_data = iqcc.execute(
    prog,
    backend.qm_config,
    terminal_output=True,
    options={"sync_hook": sync_hook_path, "timeout": 600, "profiling": False},
)
# base64_str = run_data["result"]["__sync_hook"]["cprofile"]
# binary_data = base64.b64decode(base64_str.encode("utf-8"))

# with open("profile.dat", "wb") as output_file:
#     output_file.write(binary_data)
print("Job submitted successfully.")
# print(f"Run data: {run_data}")
