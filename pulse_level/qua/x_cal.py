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
    ChannelReward,
    StateReward,
    CAFEReward,
    ExecutionConfig,
    QEnvConfig,
    BenchmarkConfig,
    GateTarget,
    StateTarget,
)
import cProfile
import pstats
import io

# %% md
# ## Accessing the device
#
# Here, we fetch the latest calibration data from one of the IQCC available devices. For now, the two available ones are Gilboa ("gilboa") and Arbel ("arbel").
# %%
from iqcc_cloud_client import IQCC_Cloud
import json
import os
from pathlib import Path

# Set your quantum computer backend
path = Path.home() / "iqcc_token.json"
with open(path, "r") as f:
    iqcc_config = json.load(f)

quantum_computer_backend = "arbel"  # for example qc_qwfix
iqcc = IQCC_Cloud(
    quantum_computer_backend=quantum_computer_backend,
    api_token=iqcc_config[quantum_computer_backend],
)
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
# %% md
# ## Creating a Qiskit Backend interface from Quam.
#
# We have created a powerful Qiskit to QUA compiler that can enable the compilation (and execution) of almost all Qiskit QuantumCircuits, as long as they are transpiled against the backend object created from the Quam that we define below.
# %%
machine = QuAM.load()

add_basic_macros_to_machine(machine)
backend = FluxTunableTransmonBackend(machine, name=quantum_computer_backend)
backend.set_options(timeout=100)
print(backend.target)
# %%
print(backend.options)
# %%
print("Available qubits: ", backend.qubit_dict)
print("Available qubit pairs: ", backend.qubit_pair_dict)
# %%
using_vpn = True
if using_vpn:
    machine.network["cloud"] = False
    machine.network["port"] = 9510
else:
    backend.qmm = iqcc
# %%
machine.network
# %% md
# ## Create a custom gate to edit
#
# We add at the Qiskit level the custom gate we want to calibrate, and we then specify the QUA macro in charge of
# implementing the parametrized pulse representation of this template parametrized gate.
#
# The way we specify such gate is by implementing a function (think about a Pennylane qnode), that modifies an input QuantumCircuit, and adds a custom Gate of your choice that is parametrized in a certain fashion.
# The choice of the parametrization is defined by the action space specification (see below), and the shape of the QUA macro you decide to insert as a physical implementation of your gate.
# For now, parametrization is constrained to parameters that can be used to update the pulses in real-time, e.g. amplitude, phase, frequency, duration. For more advanced composite pulses, we can also think about baking some precomposite pulses, that could then be passed as keyword arguments to the parametrized circuit function.
# %%
from rl_qoc.helpers import add_custom_gate


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
    x_duration = backend.target["x"][(physical_qubits[0],)].duration
    instruction_prop = QMInstructionProperties(duration=x_duration, qua_pulse_macro=qua_macro)
    qc = add_custom_gate(qc, custom_x, q_reg, params, physical_qubits, backend, instruction_prop)
    return qc


# %% md
# ## Create Environment configuration
#
# 1. Create target instance: Specify which gate you want and on which qubit it should be applied
# 2. Choose which reward scheme you want. You can choose among three methods that are carefully implemented within the QOP: Direct Channel/State Fidelity Estimation (DFE), and Context-Aware Fidelity Estimation (CAFE)
# %%
physical_qubits = (0,)
target_gate = "x"
target_state = "1"
# We show the two possible targets:
# 1. A gate target, i.e. a gate that we want to calibrate
gate_target = GateTarget(physical_qubits=physical_qubits, gate=target_gate)
# 2. A state target, i.e. a state that we want to reach
state_target = StateTarget(state=target_state, physical_qubits=physical_qubits)
# Choose the target
target = state_target
# Initialize the reward scheme to be used in the environment
reward = StateReward()
# %%
"""
It is possible to define custom reward methods, by inheriting from the Reward class. 
The two main methods to implement are:
- get_reward_data: this method is called to generate the necessary data to pass to the OPX to compute the reward (e.g., observables, input states)
- rl_qoc_training_qua_prog: this method is called to generate the QUA program that will be executed on the device to compute the reward.
- qm_step: this method is called to compute the reward for a given epoch.
"""
from rl_qoc.qua.pi_pulse_reward import PiPulseReward

# reward = PiPulseReward()


# %% md
# 3. Decide which action space to create
# 4. Decide how the parameters should be passed to the QOP (Choose between Input Stream, DGX Quantum, IO variables)
# 5. To fix the QUA program duration, we also can pass to the configuration the number of training updates expected (it should be the same variable used when declaring the agent).
# 6. Set up training hyperparameters on the environment side
# %%
# Action space specification
param_bounds = [(-1.98, 2.0)]  # Can be any number of bounds


# Environment execution parameters
seed = 98  # Master seed to make training reproducible
batch_size = 8 * 4  # Number of actions to evaluate per policy evaluation
n_shots = 10  # Minimum number of shots per fiducial evaluation
pauli_sampling = 10  # Number of fiducials to compute for fidelity estimation (DFE only)
n_reps = 1  # Number of repetitions of the cycle circuit
num_updates = TotalUpdates(5)

input_type = InputType.INPUT_STREAM
test_mode = True


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
    timeout=40,
    test_mode=test_mode,
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
rescaled_env = RescaleAndClipAction(q_env, np.float32(-1.0), np.float32(1.0))
print("Env config: ", q_env.config.as_dict())
# %%
q_env.real_time_circuit.draw("mpl", style="iqp-dark", fold=25)
# %%
q_env.real_time_transpiled_circuit.draw("mpl", fold=32, style="iqp-dark")
# %%
# Print the OpenQASM3 program
print(backend.oq3_exporter.dumps(q_env.real_time_transpiled_circuit))
# %%
from qm import generate_qua_script

# Print the QUA program
print(generate_qua_script(q_env.rl_qoc_training_qua_prog(num_updates.total_updates)))
# %%
reward.get_reward_data(
    q_env.circuit, np.zeros((q_env.batch_size, q_env.n_actions)), q_env.target, q_env.config
)
# %% md
# # Defining the Agent and launching training
#
# To push efficiency, we have introduced a custom version of the PPO algorithm whose particularity is that the sampling of the actions is in fact replicating a sampling mechanism done in real-time within the OPX. This way, we can through seed sharing share the actions between the two entities without having to introduce any communication latency to perform training.
# %%
path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
ppo_config = PPOConfig.from_yaml(path)
ppo_agent = CustomQMPPO(ppo_config, rescaled_env)
ppo_training = TrainingConfig(num_updates)
ppo_settings = TrainFunctionSettings(plot_real_time=True, print_debug=True, num_prints=10)
# %%
running_jobs = backend.qmm.get_jobs(status="Running")
print("Running jobs:", running_jobs)
pr = cProfile.Profile()

# Enable profiling
pr.enable()
if not running_jobs:
    q_env.clear_history()
    job = q_env.start_program()
    results = ppo_agent.train(ppo_training, ppo_settings)

pr.disable()

s = io.StringIO()
sortby = "cumulative"  # Sort by cumulative time
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()

# Print the profiling results to the console
print(s.getvalue())

# Optional: Save the profiling results to a file for later analysis
with open("profile_results_with_startup.txt", "w") as f:
    f.write(s.getvalue())
