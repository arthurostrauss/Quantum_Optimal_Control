# %% md
# # Gate calibration with RL with the Quantum Orchestration Platform
#
# This notebook is the template workflow enabling you to run gate calibration leveraging all the low-level capabilities of the QOP for maximum efficiency.
# %%
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, Gate
from typing import List
from gymnasium.spaces import Box
from rl_qoc.qua import QMEnvironment, QMConfig, CustomQMPPO
from quam_libs.components import QuAM, Transmon
from qiskit_qm_provider import FluxTunableTransmonBackend, QMInstructionProperties, InputType
from qiskit_qm_provider.backend import add_basic_macros_to_machine
from rl_qoc.agent.ppo_config import (
    WandBConfig,
    PPOConfig,
    TrainingConfig,
    TrainFunctionSettings,
    TotalUpdates,
)
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

# %% md
# ## Accessing the device
# %%
from iqcc_cloud_client import IQCC_Cloud
import json
import os

# Set your quantum computer backend
quantum_computer_backend = "gilboa"  # for example qc_qwfix
iqcc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)
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
# %%
machine = QuAM.load()
if not machine.active_qubits[0].macros:
    add_basic_macros_to_machine(machine)
backend = FluxTunableTransmonBackend(machine)
print(backend.target)
# %%
print("Available qubits: ", backend.qubit_dict)
print("Available qubit pairs: ", backend.qubit_pair_dict)
# %%
using_vpn = True
if using_vpn:
    machine.network["cloud"] = False
    machine.network["port"] = 9510


# %% md
# ## Create a custom gate to edit
#
# We add at the Qiskit level the custom gate we want to calibrate, and we then specify the QUA macro in charge of
# implementing the parametrized pulse representation of this template parametrized gate.
# %%
def apply_parametrized_circuit(
    qc: QuantumCircuit, params: List[Parameter], q_reg: QuantumRegister, **kwargs
):

    try:
        physical_qubits: List[int] = kwargs["physical_qubits"]
        backend: FluxTunableTransmonBackend = kwargs["backend"]
    except KeyError:
        raise KeyError(
            "Missing one of the required keys for the parametric circuit ('backend' and 'physical_qubits'"
        )
    custom_x = Gate("x_cal", 1, params)
    qc.append(custom_x, [q_reg[0]])

    def qua_macro(amp):
        qubit: Transmon = backend.get_qubit(physical_qubits[0])
        qubit.xy.play("x180", amplitude_scale=amp)

    x_duration = backend.target["x"][tuple(physical_qubits)].duration
    instruction_prop = QMInstructionProperties(duration=x_duration,
                                               qua_pulse_macro=qua_macro)
    backend.target.add_instruction(custom_x, {tuple(physical_qubits): instruction_prop})
    return qc


# %% md
# ## Create Environment configuration
#
# 1. Create target instance: Specify which gate you want and on which qubit it should be applied
# 2. Choose which reward scheme you want. You can choose among three methods that are carefully implemented within the QOP: Direct Channel/State Fidelity Estimation (DFE), and Context-Aware Fidelity Estimation (CAFE)
# %%
physical_qubits = (0,)
# target_gate = "x"
# target = GateTarget(physical_qubits, gate=target_gate)
target_state = '1'
target = StateTarget(state=target_state, physical_qubits=physical_qubits)
reward = StateReward()
# %% md
# 3. Decide which action space to create
# 4. Decide how the parameters should be passed to the QOP (Choose between Input Stream, DGX Quantum, IO variables)
# 5. To fix the QUA program duration, we also can pass to the configuration the number of training updates expected (it should be the same variable used when declaring the agent).
# 6. Set up training hyperparameters on the environment side
# %%
# Action space specification
param_bounds = [(-1.98, 2.0)]  # Can be any number of bounds


# Environment execution parameters
seed = 1203  # Master seed to make training reproducible
batch_size = 10  # Number of actions to evaluate per policy evaluation
n_shots = 10  # Minimum number of shots per fiducial evaluation
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
    num_updates=TotalUpdates.total_updates,
)
execution_config = ExecutionConfig(
    batch_size=batch_size,
    sampling_paulis=pauli_sampling,
    n_shots=n_shots,
    n_reps=n_reps,
    seed=seed,
    control_flow_enabled=True,
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
# %%
ppo_config = PPOConfig.from_yaml("agent_config.yaml")
ppo_agent = CustomQMPPO(ppo_config, rescaled_env)

ppo_training = TrainingConfig(num_updates)
ppo_settings = TrainFunctionSettings(plot_real_time=False, print_debug=True)
# %%
job = q_env.start_program()
# %%
print(job.status)
# %%
results = ppo_agent.train()
# %%
q_env.close()
# %%
# %%
print(json.dumps(results, indent=4))
