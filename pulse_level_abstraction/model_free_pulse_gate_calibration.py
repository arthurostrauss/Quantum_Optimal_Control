"""
Code for model-free gate calibration in IBM device using Reinforcement Learning tools to manipulate the pulse level

 Author: Arthur Strauss
 Created on 18/04/2023
"""

import numpy as np

from quantumenvironment import QuantumEnvironment
from helper_functions import (
    select_optimizer,
    generate_model,
    get_control_channel_map,
    get_solver_and_freq_from_backend,
)
from qconfig import QiskitConfig

# qiskit imports for building RL environment (circuit level)
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2
from qiskit.providers import QubitProperties, BackendV1, BackendV2
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
from qiskit import pulse
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit_dynamics.array import Array
from qiskit.circuit import ParameterVector, QuantumCircuit, Gate, QuantumRegister
from qiskit.extensions import XGate, RXGate

from qiskit.opflow import H, I, X, Z
from qiskit.quantum_info import Operator

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag

# Additional imports
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Union, List

# configure jax to use 64 bit mode
import jax

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
# import Array and set default backend

Array.set_default_backend("jax")

""" 
-----------------------------------------------------------------------------------------------------
We set a RL agent based on PPO and an actor critic network to perform gate calibration based on
scheme proposed in the appendix of Vladimir Sivak paper. Here, each gate is decomposed at the circuit level to enable
optimal control pulse shaping.
In there, we design classes for the environment (Quantum Circuit simulated on IBM Q simulator) and the agent 
-----------------------------------------------------------------------------------------------------
"""


def custom_pulse_schedule(
    backend: Union[BackendV1, BackendV2],
    qubit_tgt_register: Union[List[int], QuantumRegister],
    params: ParameterVector,
    default_schedule: Optional[Union[pulse.ScheduleBlock, pulse.Schedule]] = None,
):
    """
    Define parametrization of the pulse schedule characterizing the target gate
        :param backend: IBM Backend on which schedule shall be added
        :param qubit_tgt_register: Qubit register on which
        :param params: Parameters of the Schedule
        :param default_schedule:  baseline from which one can customize the pulse parameters

        :return: Parametrized Schedule
    """

    if default_schedule is None:  # No baseline pulse, full waveform builder
        pass
    else:
        # Look here for the pulse features to specifically optimize upon, for the x gate here, simply retrieve relevant
        # parameters for the Drag pulse
        pulse_ref = default_schedule.instructions[0][1].pulse

        with pulse.build(
            backend=backend, name="param_schedule"
        ) as parametrized_schedule:
            pulse.play(
                pulse.Drag(
                    duration=pulse_ref.duration,
                    amp=params[0],
                    sigma=pulse_ref.sigma,
                    beta=pulse_ref.beta,
                    angle=pulse_ref.angle,
                ),
                channel=pulse.drive_channel(qubit_tgt_register[0]),
            )

        return parametrized_schedule


def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :return:
    """
    # qc.num_qubits
    global n_actions, fake_backend, backend, qubit_tgt_register, target

    # x_pulse = backend.defaults().instruction_schedule_map.get('x', (qubit_tgt_register,)).instructions[0][1].pulse
    params = ParameterVector("theta", n_actions)

    # original_calibration = backend.instruction_schedule_map.get(target["name"])

    parametrized_gate = Gate(
        f"custom_{target['gate'].name}", len(qubit_tgt_register), params=[params[0]]
    )
    if isinstance(backend, BackendV1):
        instruction_schedule_map = backend.defaults().instruction_schedule_map
    else:
        instruction_schedule_map = backend.target.instruction_schedule_map()
    default_schedule = instruction_schedule_map.get(
        target["gate"].name, qubit_tgt_register
    )
    parametrized_schedule = custom_pulse_schedule(
        backend=backend,
        qubit_tgt_register=qubit_tgt_register,
        params=params,
        default_schedule=default_schedule,
    )
    qc.add_calibration(parametrized_gate, qubit_tgt_register, parametrized_schedule)
    qc.append(parametrized_gate, qubit_tgt_register)


"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""
abstraction_level = "pulse"
qubit_tgt_register = [0]
n_qubits = 1
sampling_Paulis = 10
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
n_actions = 1  # Choose how many control parameters in pulse/circuit parametrization
n_epochs = 200  # Number of epochs
batchsize = 50  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
"""
Choose your backend: Here we deal with pulse level implementation. In qiskit, there are only way two ways
to run this code. If simulation, use DynamicsBackend and use BackendEstimator for implementing the 
primitive enabling Pauli expectation value sampling. If real device, use qiskit Runtime backend and set a 
Runtime Service 
"""


"""Real backend initialization"""
backend_name = "ibm_perth"
estimator_options = {"resilience_level": 0}
# service = QiskitRuntimeService(channel='ibm_quantum')
# runtime_backend = service.get_backend(backend_name)
# control_channel_map = {**{qubits: runtime_backend.control_channel(qubits)[0].index
#                           for qubits in runtime_backend.coupling_map}}

"""Fake Backend initialization"""
fake_backend, fake_backend_v2 = FakeJakarta(), FakeJakartaV2()
control_channel_map = get_control_channel_map(fake_backend, qubit_tgt_register)
dt = fake_backend_v2.target.dt

dynamics_options = {
    "seed_simulator": None,  # "configuration": fake_backend.configuration(),
    "control_channel_map": control_channel_map,
    # Control channels to play CR tones, should match connectivity of device
    "solver_options": {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax": dt},
}
dynamics_backend = DynamicsBackend.from_backend(
    fake_backend, subsystem_list=qubit_tgt_register, **dynamics_options
)
target = dynamics_backend.target
target.qubit_properties = fake_backend_v2.qubit_properties(qubit_tgt_register)

# Extract channel frequencies and Solver instance from backend to provide a pulse level simulation enabling
# fidelity benchmarking
channel_freq, solver = get_solver_and_freq_from_backend(
    backend=fake_backend,
    subsystem_list=qubit_tgt_register,
    rotating_frame="auto",
    evaluation_mode="dense",
    rwa_cutoff_freq=None,
    static_dissipators=None,
    dissipator_channels=None,
    dissipator_operators=None,
)
calibration_files = None


"""
Custom Hamiltonian model for building DynamicsBackend
"""
r = 0.1

# Frequency of the qubit transition in GHz.
w = 5.0

# Sample rate of the backend in ns.
dt = 2.2222222e-10

# Define gaussian envelope function to have a pi rotation.
amp = 1.0
area = 1
sig = area * 0.399128 / r / amp
T = 4 * sig
duration = int(T / dt)
beta = 2.0

drift = 2 * np.pi * w * Operator(Z) / 2
operators = [2 * np.pi * r * Operator(X) / 2]

hamiltonian_solver = Solver(
    static_hamiltonian=drift,
    hamiltonian_operators=operators,
    rotating_frame=drift,
    rwa_cutoff_freq=2 * 5.0,
    hamiltonian_channels=["d0"],
    channel_carrier_freqs={"d0": w},
    dt=dt,
)

custom_backend = DynamicsBackend(hamiltonian_solver, **dynamics_options)

# Choose among defined backends {runtime_backend, dynamics_backend, custom_backend}
backend = dynamics_backend

# Define target gate
X_tgt = {"gate": XGate("X"), "register": qubit_tgt_register}

target = X_tgt

# Wrap all info in one dict Qiskit_setup
Qiskit_setup = QiskitConfig(
    parametrized_circuit=apply_parametrized_circuit,
    backend=backend,
    channel_freq=channel_freq,
    solver=solver,
    calibration_files=calibration_files,
)

q_env = QuantumEnvironment(
    target=target,
    abstraction_level=abstraction_level,
    Qiskit_config=Qiskit_setup,
    sampling_Pauli_space=sampling_Paulis,
    n_shots=N_shots,
    c_factor=0.5,
)

"""
-----------------------------------------------------------------------------------------------------
Hyperparameters for RL agent
-----------------------------------------------------------------------------------------------------
"""
# Hyperparameters for the agent

opti = "Adam"
eta = 0.001  # Learning rate for policy update step
eta_2 = None  # Learning rate for critic (value function) update step

use_PPO = True
epsilon = 0.1  # Parameter for clipping value (PPO)
grad_clip = 0.01
critic_loss_coeff = 0.5
optimizer = select_optimizer(
    lr=eta, optimizer=opti, grad_clip=grad_clip, concurrent_optimization=True, lr2=eta_2
)
sigma_eps = 1e-3  # for numerical stability

"""
-----------------------------------------------------------------------------------------------------
Policy parameters
-----------------------------------------------------------------------------------------------------
"""
# Policy parameters
N_in = (
    n_qubits + 1
)  # One input for each measured qubit state (0 or 1 input for each neuron)
hidden_units = [10]  # List containing number of units in each hidden layer

network = generate_model((N_in,), hidden_units, n_actions, actor_critic_together=True)
network.summary()
init_msmt = np.zeros(
    (1, N_in)
)  # Here no feedback involved, so measurement sequence is always the same

"""
-----------------------------------------------------------------------------------------------------
Training loop
-----------------------------------------------------------------------------------------------------
"""
do_benchmark = False
mu_old = tf.Variable(initial_value=network(init_msmt)[0][0], trainable=False)
sigma_old = tf.Variable(initial_value=network(init_msmt)[1][0], trainable=False)

policy_params_str = "Policy params:"

for i in tqdm(range(n_epochs)):
    Old_distrib = MultivariateNormalDiag(
        loc=mu_old, scale_diag=sigma_old, validate_args=True, allow_nan_stats=False
    )

    with tf.GradientTape(persistent=True) as tape:
        mu, sigma, b = network(init_msmt, training=True)
        mu = tf.squeeze(mu, axis=0)
        sigma = tf.squeeze(sigma, axis=0)
        b = tf.squeeze(b, axis=0)

        Policy_distrib = MultivariateNormalDiag(
            loc=mu, scale_diag=sigma, validate_args=True, allow_nan_stats=False
        )

        action_vector = tf.stop_gradient(
            tf.clip_by_value(Policy_distrib.sample(batchsize), -0.5, 0.5)
        )

        # Adjust the action vector according to params physical significance

        reward = q_env.perform_action(action_vector, do_benchmark=do_benchmark)
        advantage = reward - b

        if use_PPO:
            ratio = Policy_distrib.prob(action_vector) / (
                tf.stop_gradient(Old_distrib.prob(action_vector)) + 1e-6
            )
            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    advantage * ratio,
                    advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon),
                )
            )
        else:  # REINFORCE algorithm
            actor_loss = -tf.reduce_mean(
                advantage * Policy_distrib.log_prob(action_vector)
            )

        critic_loss = tf.reduce_mean(advantage**2)
        combined_loss = actor_loss + critic_loss_coeff * critic_loss

    grads = tape.gradient(combined_loss, network.trainable_variables)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        mu_old.assign(mu)
        sigma_old.assign(sigma)

    print("\n Epoch", i)
    print(f"{policy_params_str:#<100}")
    print("mu_vec:", np.array(mu))
    print("sigma_vec:", np.array(sigma))
    print("baseline:", np.array(b))
    print("Average reward", np.mean(q_env.reward_history[i]))
    if do_benchmark:
        print("Average Gate Fidelity:", q_env.avg_fidelity_history[i])

    # Apply gradients
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

if isinstance(q_env.estimator, Estimator):
    q_env.estimator.session.close()
"""
-----------------------------------------------------------------------------------------
Plotting tools
-----------------------------------------------------------------------------------------
"""


def plot_examples(fig, ax, reward_table):
    """
    Helper function to plot data with associated colormap, used for plotting the reward per each epoch and each episode
    (From original repo associated to the paper https://github.com/v-sivak/quantum-control-rl)
    """

    im = ax.imshow(np.transpose(reward_table))
    ax.set_ylabel("Episode")
    ax.set_xlabel("Epoch")
    fig.colorbar(im, ax=ax, label="Reward")


figure, (ax1, ax2) = plt.subplots(1, 2)
#  Plot return as a function of epochs
ax1.plot(np.mean(q_env.reward_history, axis=1), "-.", label="Reward")
# ax1.plot(data["baselines"], '-.', label='baseline')
# ax1.plot(q_env.process_fidelity_history, '-o', label='Process Fidelity')
# ax1.plot(q_env.avg_fidelity_history, '-.', label='Average Gate Fidelity')

# adapted_epoch = np.arange(0, n_epochs, window_size)
# ax1.plot(adapted_epoch, moving_average_reward, '-.', label='Reward')
# # ax1.plot(data["baselines"], '-.', label='baseline')
# ax1.plot(adapted_epoch, moving_average_process_fidelity, '-o', label='Process Fidelity')
# ax1.plot(adapted_epoch, moving_average_gate_fidelity, '-.', label='Average Gate Fidelity')

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Expected reward")
ax1.legend()
# plot_examples(figure, ax2, q_env.reward_history)
plt.show()
