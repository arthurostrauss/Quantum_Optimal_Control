"""
Code for model-free gate calibration in IBM device using Reinforcement Learning tools to manipulate the pulse level

 Author: Arthur Strauss
 Created on 18/04/2023
"""

import numpy as np

from quantumenvironment import QuantumEnvironment, get_solver_and_freq_from_backend
from helper_functions import select_optimizer, generate_model, custom_pulse_schedule

# Qiskit imports for building RL environment (circuit level)
from qiskit import pulse, transpile
from qiskit.providers.options import Options
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2
from qiskit.providers import QubitProperties
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_dynamics import DynamicsBackend
from qiskit_dynamics.array import Array
from qiskit.circuit import ParameterVector, Parameter, ParameterExpression, QuantumCircuit, Gate
from qiskit.extensions import CXGate, XGate, SXGate, RZGate
from qiskit.opflow import H, I, X, S
from qiskit_ibm_provider import IBMBackend, IBMProvider

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag

from tf_agents.specs import array_spec, tensor_spec

# Additional imports
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
# configure jax to use 64 bit mode
import jax

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
# import Array and set default backend

Array.set_default_backend('jax')

""" 
-----------------------------------------------------------------------------------------------------
We set a RL agent based on PPO and an actor critic network to perform arbitrary state preparation based on
scheme proposed in the appendix of Vladimir Sivak paper. 
In there, we design classes for the environment (Quantum Circuit simulated on IBM Q simulator) and the agent 
-----------------------------------------------------------------------------------------------------
"""


def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :return:
    """
    # qc.num_qubits
    global n_actions, fake_backend, backend, qubit_tgt_register, target

    # x_pulse = backend.defaults().instruction_schedule_map.get('x', (qubit_tgt_register,)).instructions[0][1].pulse
    params = ParameterVector('theta', n_actions)

    # original_calibration = backend.instruction_schedule_map.get(target["name"])

    parametrized_gate = Gate(f"custom_{target['gate'].name}", 1, params=[params[0]])
    default_schedule = fake_backend.defaults().instruction_schedule_map.get(target["gate"].name, qubit_tgt_register)
    parametrized_schedule = custom_pulse_schedule(backend=backend, target=target, qubit_tgt_register=qubit_tgt_register,
                                                  params=params, default_schedule=default_schedule)
    qc.add_calibration(parametrized_gate, qubit_tgt_register, parametrized_schedule)
    qc.append(parametrized_gate, qubit_tgt_register)


"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""
qubit_tgt_register = [0]
n_qubits = 1
sampling_Paulis = 10
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
n_actions = 1  # Choose how many control parameters in pulse/circuit parametrization
time_steps = 1  # Number of time steps within an episode (1 means you do one readout and assign right away the reward)

"""
Choose your backend: Here we deal with pulse level implementation. In Qiskit, there are only way two ways
to run this code. If simulation, use DynamicsBackend and use BackendEstimator for implementing the 
primitive enabling Pauli expectation value sampling. If real device, use Qiskit Runtime backend and set a 
Runtime Service 
"""
estimator_options = {'resilience_level': 0}

"""Real backend initialization"""
backend_name = 'ibm_perth'
# provider = IBMProvider()
# service = QiskitRuntimeService(channel='ibm_quantum')
# runtime_backend = service.get_backend(backend_name)
# real_backend = provider.get_backend(backend_name)
# control_channel_map = {**{qubits: real_backend.control_channel(qubits)[0].index
#                           for qubits in real_backend.coupling_map}}

service = None
fake_backend = FakeJakarta()
fake_backend_v2 = FakeJakartaV2()
control_channel_map = {}
control_channel_map_backend = {
    **{qubits: fake_backend.configuration().control_channels[qubits][0].index for qubits in
       fake_backend.configuration().control_channels}}
for qubits in control_channel_map_backend:
    if qubits[0] in qubit_tgt_register and qubits[1] in qubit_tgt_register:
        control_channel_map[qubits] = control_channel_map_backend[qubits]

print(control_channel_map)
dynamics_options = {'seed_simulator': 5000,  # "configuration": fake_backend.configuration(),
                    'control_channel_map': control_channel_map
                    # Control channels to play CR tones, should match connectivity of device
                    }
dynamics_backend = DynamicsBackend.from_backend(fake_backend, subsystem_list=qubit_tgt_register, **dynamics_options)
target = dynamics_backend.target
target.qubit_properties = fake_backend_v2.qubit_properties(qubit_tgt_register)
# Choose here among the above backends, if simulator or dynamics_backend, set service to None
backend = dynamics_backend

# Extract channel frequencies and Solver instance from backend to provide a pulse level simulation enabling
# fidelity benchmarking
channel_freq, solver = get_solver_and_freq_from_backend(
    backend=backend,
    subsystem_list=qubit_tgt_register,
    rotating_frame="auto",
    evaluation_mode="dense",
    rwa_cutoff_freq=None,
    static_dissipators=None,
    dissipator_channels=None,
    dissipator_operators=None
)
# Define target gate
X_tgt = {
    "target_type": 'gate',
    "gate": XGate("X"),
    "register": qubit_tgt_register,
    "input_states": [
        {"name": '|0>',
         "circuit": I},

        {"name": '|1>',
         "circuit": X},

        {"name": '|+>',
         "circuit": H},
        {"name": '|->',
         "circuit": H @ X},
    ]

}

target = X_tgt

Qiskit_setup = {
    "backend": backend,
    "parametrized_circuit": apply_parametrized_circuit,
    "estimator_options": estimator_options,
    "service": service,
    "target_register": qubit_tgt_register,
    "channel_freq": channel_freq,
    "solver": solver
}

action_spec = tensor_spec.BoundedTensorSpec(shape=(n_actions,), dtype=tf.float32, minimum=-1., maximum=1.)
observation_spec = array_spec.ArraySpec(shape=(time_steps,), dtype=np.int32)

q_env = QuantumEnvironment(n_qubits=n_qubits, target=target, abstraction_level="pulse",
                           action_spec=action_spec, observation_spec=observation_spec,
                           Qiskit_config=Qiskit_setup,
                           sampling_Pauli_space=sampling_Paulis, n_shots=N_shots, c_factor=0.5)

"""
-----------------------------------------------------------------------------------------------------
Hyperparameters for RL agent
-----------------------------------------------------------------------------------------------------
"""
# Hyperparameters for the agent
n_epochs = 200  # Number of epochs
batchsize = 1  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
opti = "Adam"
eta = 0.001  # Learning rate for policy update step
eta_2 = None  # Learning rate for critic (value function) update step

use_PPO = True
epsilon = 0.1  # Parameter for clipping value (PPO)
grad_clip = 0.01
critic_loss_coeff = 0.5
optimizer = select_optimizer(lr=eta, optimizer=opti, grad_clip=grad_clip, concurrent_optimization=True, lr2=eta_2)
sigma_eps = 1e-3  # for numerical stability
grad_update_number = 20
# class Agent:
#     def __init__(self, epochs:int, batchsize:int, optimizer, lr: float, lr2: Optional[float], grad_clip:):
#         pass


"""
-----------------------------------------------------------------------------------------------------
Policy parameters
-----------------------------------------------------------------------------------------------------
"""
# Policy parameters
N_in = n_qubits + 1  # One input for each measured qubit state (0 or 1 input for each neuron)
hidden_units = [82, 82]  # List containing number of units in each hidden layer

network = generate_model((N_in,), hidden_units, n_actions, actor_critic_together=True)
network.summary()
init_msmt = np.zeros((1, N_in))  # Here no feedback involved, so measurement sequence is always the same

"""
-----------------------------------------------------------------------------------------------------
Training loop
-----------------------------------------------------------------------------------------------------
"""

mu_old = tf.Variable(initial_value=network(init_msmt)[0][0], trainable=False)
sigma_old = tf.Variable(initial_value=network(init_msmt)[1][0], trainable=False)

policy_params_str = 'Policy params:'
print('Neural net output', network(init_msmt), type(network(init_msmt)))
print("mu_old", mu_old)
print("sigma_old", sigma_old)

for i in tqdm(range(n_epochs)):

    Old_distrib = MultivariateNormalDiag(loc=mu_old, scale_diag=sigma_old,
                                         validate_args=True, allow_nan_stats=False)

    with tf.GradientTape(persistent=True) as tape:

        mu, sigma, b = network(init_msmt, training=True)
        mu = tf.squeeze(mu, axis=0)
        sigma = tf.squeeze(sigma, axis=0)
        b = tf.squeeze(b, axis=0)

        Policy_distrib = MultivariateNormalDiag(loc=mu, scale_diag=sigma,
                                                validate_args=True, allow_nan_stats=False)

        action_vector = tf.stop_gradient(tf.clip_by_value(Policy_distrib.sample(batchsize), 0, 1.))

        # Adjust the action vector according to params physical significance

        reward = q_env.perform_action(action_vector)
        advantage = reward - b

        if use_PPO:
            ratio = Policy_distrib.prob(action_vector) / (tf.stop_gradient(Old_distrib.prob(action_vector)) + 1e-6)
            actor_loss = - tf.reduce_mean(tf.minimum(advantage * ratio,
                                                     advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
        else:  # REINFORCE algorithm
            actor_loss = - tf.reduce_mean(advantage * Policy_distrib.log_prob(action_vector))

        critic_loss = tf.reduce_mean(advantage ** 2)
        combined_loss = actor_loss + critic_loss_coeff * critic_loss

    grads = tape.gradient(combined_loss, network.trainable_variables)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        mu_old.assign(mu)
        sigma_old.assign(sigma)

    print('\n Epoch', i)
    print("Average reward", np.mean(q_env.reward_history[i]))
    # print("Average Gate Fidelity:", q_env.avg_fidelity_history[i])

    # Apply gradients
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

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
    ax.set_ylabel('Episode')
    ax.set_xlabel('Epoch')
    fig.colorbar(im, ax=ax, label='Reward')


figure, (ax1, ax2) = plt.subplots(1, 2)
#  Plot return as a function of epochs
ax1.plot(np.mean(q_env.reward_history, axis=1), '-.', label='Reward')
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
