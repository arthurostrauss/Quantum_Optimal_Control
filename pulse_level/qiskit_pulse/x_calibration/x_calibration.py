# %% md
# # Pulse Level Learning
#
# In this notebook, we will train a reinforcement learning agent to learn pulse level control of a quantum system. The configuration is drawn from two dedicated yaml files respectively describing the quantum environment and the agent. The quantum environment is a `QuantumEnvironment` object, which is a subclass of `gym.Env ` and is designed to be compatible with standard RL libraries. The agent on the other hand, is specifically hard coded for our need because of the need to be able to send to the same resource (the Estimator primitive) a batch of quantum circuits (in contrast with usual RL frameworks where actions can be evaluated in parallel).
#
# For this notebook, we take all necessary inputs from the `pulse_config.py` file. This python file contains all elements necessary to create a pulse-level quantum environment. The file contains the following elements:
# - a parametrized gate function, coded through Qiskit; leveraging behind the scenes a custom parametrized pulse schedule, to be inserted in the quantum circuit. The function needs to modify an input parametrized quantum circuit, by appending the parametrized gate to it.
# - a Qiskit backend object, which is the quantum device or simulator on which the quantum circuits will be executed. The backend is retrieved through another template function called get_backend.
# - A circuit context, which is a `QuantumCircuit` object that contains the quantum circuit in which the target gate operation should be optimized. The context is used to create the `ContextAwareQuantumEnvironment` object, which is a subclass of `BaseQuantumEnvironment` that takes into account the context of the quantum circuit in which the gate is to be optimized. Note that this mode is optional, as one could just focus on optimizing the gate operation in a standalone manner (i.e., without any context).
#
# We provide a dedicated template for IBM devices working through ECR, X, and SX basis gates. The user can adapt this to the platform and basis gates of his choice.
# %%

from pulse_level.qiskit_pulse.x_calibration.pulse_config import (
    apply_parametrized_circuit,
)
from rl_qoc import ContextAwareQuantumEnvironment, QuantumEnvironment, CustomPPO
from gymnasium.wrappers import RescaleAction, ClipAction
from qiskit.circuit import QuantumCircuit
from rl_qoc.helpers import (
    simulate_pulse_input,
    load_from_yaml_file,
    get_q_env_config,
)
import numpy as np
import os
import matplotlib.pyplot as plt

# %% md
# # QuantumEnvironment setup
# We first define our custom `DynamicsBackend` object describing our quantum system. We use some helper functions to create it and make our job easier
# %%
# Backend definition
from pulse_level.qiskit_pulse.dynamics_backends import fixed_frequency_transmon_backend
from rl_qoc.helpers import (
    perform_standard_calibrations,
    custom_experiment_result_function,
)

dims = [3]
freqs = [4.86e9]
anharmonicities = [-0.33e9]
rabi_freqs = [0.22e9]
couplings = None

backend = fixed_frequency_transmon_backend(
    dims,
    freqs,
    anharmonicities,
    rabi_freqs,
    couplings,
    experiment_result_function=custom_experiment_result_function,
    # Custom experiment result function returning the statevector on top of the original counts
)

# Add calibrations for standard single qubit basis gates (X, SX, RZ, and other phase gates)
cals, exps = perform_standard_calibrations(backend)
# Backend now contains the standard calibrations
# %%
print(backend.target)
# %%
current_dir = os.getcwd()
config_file_name = "q_env_pulse_config.yml"
config_file_address = os.path.join(current_dir, config_file_name)

q_env_config = get_q_env_config(config_file_address, apply_parametrized_circuit, backend)

# Set the parametrized circuit kwargs to be used in the parametrized circuit function
q_env_config.backend_config.parametrized_circuit_kwargs = {
    "target": q_env_config.target,
    "backend": q_env_config.backend,
}
# %% md
# We can define an optional circuit context in which we would like to calibrate the gate.
# If the context is not provided, the gate will be calibrated in a standalone manner.
#
# Note: If no context is provided, then the user should use the `QuantumEnvironment` class. Conversely, if a context is provided, the user should use the `ContextAwareQuantumEnvironment` class.
# %%
# Set the use_context flag to True if you want to use the context of the quantum circuit
# TODO: Fill up your circuit context
circuit_context = None

if circuit_context is not None:
    fig = circuit_context.draw("mpl", interactive=True)

    q_env = ContextAwareQuantumEnvironment(
        q_env_config, circuit_context, training_steps_per_gate=250
    )

else:
    q_env = QuantumEnvironment(q_env_config)
    fig = q_env.circuits[0].draw("mpl", interactive=True)

# Rescale the action space to [-1, 1] and clip the actions to the valid range
# This step is dependent on the output layer of the agent for the mean of the policy
# If the output layer is a tanh layer (default), then rescaling is necessary

rescaled_env = ClipAction(RescaleAction(q_env, q_env.action_space.low, q_env.action_space.high))

print("Circuit subject to calibration:")
fig
# %%
q_env.backend.target.get_calibration("x", (0,)).draw()
# %%
from qiskit import schedule

print("Parametrized circuit")
schedule(q_env.circuits[0], q_env.backend).draw()
# %%
from qiskit.visualization import plot_circuit_layout

print("Available operations", q_env.backend.operation_names)
try:
    plot_circuit_layout(circuit_context, backend=q_env.backend)
except:
    print("No layout available")
# %% md
# # Agent setup
#
# We now define the agent counterpart.
# %%
from pathlib import Path

# Change the file_name to the name of the agent configuration file and specify the file location
file_name = "agent_config.yaml"
file_location = Path.cwd() / file_name

agent_config = load_from_yaml_file(file_location)
# %%
ppo_agent = CustomPPO(agent_config, rescaled_env)
# %%
from rl_qoc.agent import TotalUpdates, TrainFunctionSettings, TrainingConfig

total_updates = TotalUpdates(500)
# hardware_runtime = HardwareRuntime(300)
training_config = TrainingConfig(
    training_constraint=total_updates,
    target_fidelities=[0.999, 0.9999],
    lookback_window=10,
    anneal_learning_rate=True,
    std_actions_eps=1e-2,
)

train_function_settings = TrainFunctionSettings(
    plot_real_time=True,
    print_debug=True,
    num_prints=1,
    hpo_mode=False,
    clear_history=True,
)
# %%
ppo_agent.agent_config.ent_coef = 0.01
# %%
ppo_agent.train(training_config=training_config, train_function_settings=train_function_settings)
# %%
import matplotlib.pyplot as plt

plt.plot(ppo_agent.training_results["std_action"])
plt.xlabel("Updates")
plt.ylabel("Standard Deviation of Actions")
plt.title("Standard Deviation of Actions during Training")
# %%
plt.plot(ppo_agent.training_results["action_history"])
plt.xlabel("Updates")
plt.ylabel("Actions")
plt.title("Actions during Training")
# %%
from rl_qoc import GateTarget

reward_history = np.array(q_env.reward_history)
mean_rewards = np.mean(reward_history, axis=-1)
max_mean = int(np.max(mean_rewards) * 1e4) / 1e4
n_epochs = len(mean_rewards)
if q_env.benchmark_cycle != 0:
    fidelity_range = np.arange(0, n_epochs, q_env.benchmark_cycle)
plt.plot(
    # fidelity_range,
    np.array(q_env.fidelity_history),
    label=f"Average {q_env.target.target_type} Fidelity",
)
plt.plot(mean_rewards, label=f"Mean Batch Rewards, max: {max_mean}")

plt.xlabel("Updates")
plt.ylabel("Reward")
plt.title(
    f"{q_env.target.gate.name if isinstance(q_env.target, GateTarget) else ''} Learning Curve"
)
plt.legend()
plt.show()
# %%
print("Optimal action", q_env.optimal_action)
print("Best fidelity:", np.max(q_env.fidelity_history))
# %%
initial_amp = q_env.backend.target.get_calibration("x", (0,)).instructions[0][1].pulse.amp
initial_amp
# %%
# Update calibration in the backend target and retrieve the calibration
optimal_calibration = q_env.update_gate_calibration()
optimal_calibration.draw()
# %%
# Plot the fidelity as a function of the amplitude (for X gate)
from pulse_level.qiskit_pulse.x_calibration.pulse_config import custom_schedule

amp_range = np.linspace(-1, 1, 500)
fidelity_range = []
cals = []
for amp in amp_range:
    cal = custom_schedule(
        q_env.backend, q_env.physical_target_qubits, q_env.parameters
    ).assign_parameters({q_env.parameters: [amp]})
    cals.append(cal)
data = simulate_pulse_input(q_env.backend, cals, target=q_env.target.target_operator)
fidelity_range = data["gate_fidelity"]["raw"]
plt.plot(amp_range, fidelity_range)
plt.xlabel("Amplitude")
plt.ylabel("Gate Fidelity")
plt.title("Rabi experiment")
# %%
from qiskit.quantum_info import Operator, Statevector

data = simulate_pulse_input(
    q_env.backend,
    optimal_calibration,
    target=Operator(q_env.target.gate),
)

print(data)
# %%
# Testing gate in a quantum circuit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(1)
qc.x(0)
qc.measure_all()
qc.draw("mpl")

basic_simulator = BasicSimulator()
basic_result = basic_simulator.run(qc).result()

pulse_circ = transpile(qc, q_env.backend)
q_env.backend.options.shots = 1000
pulse_results = q_env.backend.run(pulse_circ).result()

plot_histogram([basic_result.get_counts(), pulse_results.get_counts()], legend=["Ideal", "Pulse"])
# %%
pulse_circ.draw("mpl")
# %%
# Testing the pulse schedule
from qiskit import schedule

pulse_schedule = schedule(pulse_circ.remove_final_measurements(inplace=False), q_env.backend)
pulse_schedule.draw()
# %%
# Testing the pulse schedule
pulse_sim_results = simulate_pulse_input(q_env.backend, pulse_circ)
print(pulse_sim_results)
