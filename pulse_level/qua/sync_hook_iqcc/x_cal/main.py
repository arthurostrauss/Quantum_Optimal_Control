import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, Gate
from typing import List
from gymnasium.spaces import Box
from rl_qoc.qua import QMEnvironment, QMConfig
from quam_libs.components import QuAM, Transmon
from qiskit_qm_provider import (
    FluxTunableTransmonBackend,
    QMInstructionProperties,
    InputType,
    ParameterPool,
)
from qiskit_qm_provider.backend.backend_utils import add_basic_macros_to_machine
from rl_qoc.agent.ppo_config import (
    TotalUpdates,
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

backend_name = "gilboa"
iqcc = IQCC_Cloud(quantum_computer_backend=backend_name)

# Get the latest state and wiring files
latest_wiring = iqcc.state.get_latest("wiring")
latest_state = iqcc.state.get_latest("state")

# Get the state folder path from environment variable
quam_state_folder_path = os.environ["QUAM_STATE_PATH"]

# Save the files
with open(os.path.join(quam_state_folder_path, "wiring.json"), "w") as f:
    json.dump(latest_wiring.data, f, indent=4)

with open(os.path.join(quam_state_folder_path, "state.json"), "w") as f:
    json.dump(latest_state.data, f, indent=4)

machine = QuAM.load()
if not machine.active_qubits[0].macros:
    add_basic_macros_to_machine(machine)
backend = FluxTunableTransmonBackend(machine)

ppo_config = load_from_yaml_file("agent_config.yaml")


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: List[Parameter], q_reg: QuantumRegister, **kwargs
):

    physical_qubits: List[int] = kwargs["physical_qubits"]
    backend: FluxTunableTransmonBackend = kwargs["backend"]

    # TODO: Enter your custom parametric QUA macro here
    def qua_macro(amp):
        qubit: Transmon = backend.get_qubit(physical_qubits[0])
        qubit.xy.play("x180_DragCosine", amplitude_scale=amp)

    # Create a custom gate with the QUA macro
    custom_x = Gate("x_cal", 1, params)
    instruction_prop = QMInstructionProperties(qua_pulse_macro=qua_macro)
    qc = add_custom_gate(qc, custom_x, q_reg, params, physical_qubits, backend, instruction_prop)
    return qc


physical_qubits = (0,)
target_name = "1"

# target = GateTarget(gate=target_name, physical_qubits=physical_qubits)
target = StateTarget(state=target_name, physical_qubits=physical_qubits)
reward = StateReward()

# Action space specification
param_bounds = [(-1.98, 2.0)]  # Can be any number of bounds

# Environment execution parameters
seed = 1203  # Master seed to make training reproducible
batch_size = 32  # Number of actions to evaluate per policy evaluation
n_shots = 100  # Minimum number of shots per fiducial evaluation
pauli_sampling = 100  # Number of fiducials to compute for fidelity estimation (DFE only)
n_reps = 1  # Number of repetitions of the cycle circuit
num_updates = TotalUpdates(5)
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


# Génération dynamique du fichier sync_hook.py
def generate_sync_hook():
    # Déterminer le type de cible pour la génération du code approprié
    target_initialization_code = ""

    if isinstance(target, GateTarget):
        target_initialization_code = f"""physical_qubits = {physical_qubits}
target_name = "{target.gate.name}"
target = GateTarget(gate=target_name, physical_qubits=physical_qubits)
"""
    elif isinstance(target, StateTarget):
        # Convertir l'état cible en chaîne de caractères pour le code généré
        state_str = np.array2string(
            target.dm.data, precision=8, separator=", ", suppress_small=True
        )
        target_initialization_code = f"""
physical_qubits = {physical_qubits}
target_state = np.array({state_str}, dtype=complex)
target = StateTarget(state=target_state, physical_qubits=physical_qubits)
"""
    sync_hook_code = f"""# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
# This file was automatically generated by main.py

import json
from rl_qoc.agent import PPOConfig, TrainingConfig, TrainFunctionSettings, TotalUpdates
from rl_qoc.qua import QMEnvironment, CustomQMPPO, QMConfig
from iqcc_cloud_client.runtime import get_qm_job
from rl_qoc import (
    RescaleAndClipAction,
    GateTarget,
    StateTarget,
    ChannelReward,
    StateReward,
    CAFEReward,
    ExecutionConfig,
    QEnvConfig,
    BenchmarkConfig,
)
import numpy as np
from gymnasium.spaces import Box

job = get_qm_job()
{target_initialization_code}
reward = {reward.__class__.__name__}() 

# Action space specification
param_bounds = {param_bounds}  # Can be any number of bounds

# Environment execution parameters
seed = {seed}  # Master seed to make training reproducible
batch_size = {batch_size}  # Number of actions to evaluate per policy evaluation
n_shots = {n_shots}  # Minimum number of shots per fiducial evaluation
pauli_sampling = {pauli_sampling}  # Number of fiducials to compute for fidelity estimation (DFE only)
n_reps = {n_reps}  # Number of repetitions of the cycle circuit
num_updates = TotalUpdates({num_updates.total_updates})
input_type = "{str(input_type)}"
backend_config = QMConfig(
    num_updates=num_updates.total_updates,
    input_type=input_type,
    verbosity=2,
    timeout={backend_config.timeout}
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
    target=target,
    backend_config=backend_config,
    action_space=action_space,
    execution_config=execution_config,
    reward=reward,
    benchmark_config=BenchmarkConfig(0),  # No benchmark for now)
)
q_env = QMEnvironment(q_env_config, job=job)
rescaled_env = RescaleAndClipAction(q_env, -1.0, 1.0)
ppo_config = PPOConfig.from_dict({ppo_config})

ppo_agent = CustomQMPPO(ppo_config, rescaled_env)

ppo_training = TrainingConfig(num_updates)
ppo_settings = TrainFunctionSettings(plot_real_time=True, print_debug=True)

results = ppo_agent.train(ppo_training, ppo_settings)

print(json.dumps(results))
"""

    # Écrire le contenu généré dans le fichier sync_hook.py
    sync_hook_path = os.path.join(os.path.dirname(__file__), "sync_hook.py")
    with open(sync_hook_path, "w") as f:
        f.write(sync_hook_code)

    return sync_hook_path


# Générer le fichier sync_hook.py avant l'exécution
sync_hook_path = generate_sync_hook()
print(f"Sync hook file generated at: {sync_hook_path}")

prog = q_env.rl_qoc_training_qua_prog(num_updates=num_updates.total_updates)
if input_type == InputType.DGX:
    ParameterPool.configure_stream()
if hasattr(q_env.real_time_circuit, "calibrations") and q_env.real_time_circuit.calibrations:
    backend.update_calibrations(qc=q_env.real_time_circuit, input_type=input_type)
backend.update_compiler_from_target()

run_data = iqcc.execute(
    prog, backend.qm_config, terminal_output=True, options={"sync_hook": sync_hook_path}
)

print("Job submitted successfully.")
print(f"Run data: {run_data}")