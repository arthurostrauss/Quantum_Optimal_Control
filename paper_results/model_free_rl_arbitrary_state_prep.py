"""
Code for arbitrary state preparation based on scheme described in Appendix D.2b  of paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 11/11/2022
"""

import numpy as np
from quantumenvironment import QuantumEnvironment
from helper_functions import select_optimizer

# Qiskit imports for building RL environment (circuit level)
from qiskit import IBMQ
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit_ibm_runtime import QiskitRuntimeService

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
import tensorflow_probability as tfp

from tf_agents.specs import array_spec, tensor_spec
# from tensorflow.python.keras.callbacks import TensorBoard
# from tensorboard.plugins.hparams import api as hp
# from tensorflow.python.keras.callbacks import TensorBoard

# Additional imports
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
# import csv

tfpl = tfp.layers
tfd = tfp.distributions


""" 
-----------------------------------------------------------------------------------------------------
Here we set a RL agent based on PPO and an actor critic network to perform arbitrary state preparation based on
scheme proposed in the appendix of Vladimir Sivak paper. 
In there, we design classes for the environment (Quantum Circuit simulated on IBM Q simulator) and the agent 
-----------------------------------------------------------------------------------------------------
"""

# IBMQ.save_account(TOKEN)
IBMQ.load_account()  # Load account from disk
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')


def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    :param qc: Quantum Circuit instance to add the gates on
    :return:
    """
    # qc.num_qubits
    params = ParameterVector('theta', 1)
    qc.ry(2 * np.pi * params[0], 0)
    qc.cx(0, 1)
    # qc.u(angle[0][0], angle[0][1], angle[0, 2], 0)
    # qc.u(angle[1][0], angle[1][1], angle[1, 2], 1)
    # qc.ecr(0, 1)


# action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=tf.float32, minimum=-1., maximum=1.)
action_spec = tensor_spec.BoundedTensorSpec(shape=(1,), dtype=tf.float32, minimum=-1., maximum=1.)

"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""

service = QiskitRuntimeService(channel='ibm_quantum')
seed = 3590  # Seed for action sampling
backend = service.backends(simulator=True)[0]  # Simulation backend (mock quantum computer)
options = {"seed_simulator": None, 'resilience_level': 0}
n_qubits = 2
sampling_Paulis = 100
N_shots = 100  # Number of shots for sampling the quantum computer for each action vector

# Target state: Bell state
ket0, ket1 = np.array([[1.], [0]]), np.array([[0.], [1.]])
ket00, ket11 = np.kron(ket0, ket0), np.kron(ket1, ket1)
bell_state = (ket00 + ket11) / np.sqrt(2)
bell_dm = bell_state @ bell_state.conj().T
bell_tgt = {"dm": DensityMatrix(bell_dm)}
print(bell_tgt)
target_state = bell_tgt

Qiskit_setup = {
    "backend": backend,
    "service": service,
    "parametrized_circuit": apply_parametrized_circuit,
    "options": options
}
q_env = QuantumEnvironment(n_qubits=n_qubits, target_state=bell_tgt, abstraction_level="circuit",
                           action_spec=action_spec,
                           Qiskit_setup=Qiskit_setup,
                           sampling_Pauli_space=sampling_Paulis, n_shots=N_shots, c_factor=2)
# q_env.perform_action(np.array([[0.25], [0.25]]))


"""
-----------------------------------------------------------------------------------------------------
Hyperparameters for RL agent
-----------------------------------------------------------------------------------------------------
"""
# Hyperparameters for the agent
n_epochs = 200  # Number of epochs
batchsize = 100  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
opti = "Adam"
eta = 0.05  # Learning rate for policy update step
eta_2 = 0.1  # Learning rate for critic (value function) update step

use_PPO = True
epsilon = 0.2  # Parameter for clipping value (PPO)
grad_clip = 0.3
critic_loss_coeff = 0.5
optimizer = select_optimizer(lr=eta, optimizer=opti, grad_clip=grad_clip)
sigma_eps = 1e-6  # for numerical stability

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
n_actions = 1  # Choose how many control parameters in pulse/circuit parametrization
hidden_units = [5, 7]  # List containing number of units in each hidden layer

input_layer = Input(shape=(N_in,))

Net = Dense(hidden_units[0], activation='relu', input_shape=(N_in,),
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{0}")(input_layer)
for i in range(1, len(hidden_units)):
    Net = Dense(hidden_units[i], activation='relu', kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{i}")(Net)

# actor_output = tfpl.IndependentNormal(N_out)(Net)
mean_param = Dense(n_actions, activation=None, name='mean_vec')(Net)
sigma_param = Dense(n_actions, activation="relu", name="sigma_vec")(Net)
critic_output = Dense(1, activation=None, name="critic_output")(Net)

network = Model(inputs=input_layer, outputs=[mean_param, sigma_param, critic_output])
network.summary()
init_msmt = np.zeros((1, N_in))

#  Keep track of variables
data = {
    "means": np.zeros(n_epochs + 1),
    "stds": np.zeros(n_epochs + 1),
    "amps": np.zeros([n_epochs, batchsize, 1]),
    "rewards": np.zeros([n_epochs, batchsize]),
    "baselines": np.zeros(n_epochs + 1),
    "fidelity": np.zeros(n_epochs),
    "params": {
        "learning_rate": eta,
        "seed": seed,
        "clipping_PPO": epsilon,
        "n_epochs": n_epochs,
        "batchsize": batchsize,
        "target_state": target_state,
        "PPO?": use_PPO,
    }
}

"""
-----------------------------------------------------------------------------------------------------
Training loop
-----------------------------------------------------------------------------------------------------
"""
# TODO: Use TF-Agents PPO Agent
mu_old = tf.Variable(initial_value=network(init_msmt)[0][0], trainable=False)
sigma_old = tf.Variable(initial_value=network(init_msmt)[1][0] + sigma_eps, trainable=False)

policy_params_str = 'Policy params:'
print('Neural net output', network(init_msmt), type(network(init_msmt)))
print("mu_old", mu_old)
print("sigma_old", sigma_old)

for i in tqdm(range(n_epochs)):

    with tf.GradientTape(persistent=True) as tape:

        mu, sigma, b = network(init_msmt)
        print(mu, sigma, b)
        mu = tf.clip_by_value(tf.squeeze(mu, axis=0), -1., 1.)
        sigma = tf.squeeze(sigma, axis=0) + sigma_eps
        b = tf.squeeze(b, axis=0)
        print('\n Epoch', i)
        print(f"{policy_params_str:#<100}")
        print('mu_vec', mu)
        print('sigma_vec', sigma)
        print('baseline', b)
        Old_distrib = tfd.MultivariateNormalDiag(loc=mu_old, scale_diag=sigma_old,
                                                 validate_args=True, allow_nan_stats=False)
        Policy_distrib = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma,
                                                    validate_args=True, allow_nan_stats=False)

        action_vector = Policy_distrib.sample(batchsize, seed=seed)
        # print('action_vec', action_vector)
        reward = q_env.perform_action(action_vector)
        advantage = reward - b

        if use_PPO:
            print('prob', Policy_distrib.prob(action_vector))
            print('old prob', Old_distrib.prob(action_vector))
            ratio = Policy_distrib.prob(action_vector) / (Old_distrib.prob(action_vector) + sigma_eps)
            actor_loss = - tf.reduce_mean(tf.minimum(advantage * ratio,
                                                     advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
        else:  # REINFORCE algorithm
            actor_loss = - tf.reduce_mean(advantage * Policy_distrib.log_prob(action_vector))

        critic_loss = tf.reduce_mean(advantage ** 2)
        combined_loss = actor_loss + critic_loss_coeff * critic_loss

    grads = tape.gradient(combined_loss, network.trainable_variables)
    # print('grads', grads)
    # grads = tf.clip_by_value(combined_grads, -grad_clip, grad_clip)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        mu_old.assign(mu)
        sigma_old.assign(sigma)

    data["rewards"][i] = reward
    data["means"][i] = np.array(mu)
    data["stds"][i] = np.array(sigma)
    data["baselines"][i] = np.array(b)
    # print('dm', q_env.density_matrix)
    data["fidelity"][i] = state_fidelity(target_state["dm"], q_env.density_matrix_history[i])

    # Apply gradients
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

"""
-----------------------------------------------------------------------------------------
Plotting tools
-----------------------------------------------------------------------------------------
"""


#  Plotting results
def plot_examples(fig, ax, reward_table):
    """
    Helper function to plot data with associated colormap, used for plotting the reward per each epoch and each episode
    (From original repo associated to the paper https://github.com/v-sivak/quantum-control-rl)
    """

    im = ax.imshow(np.transpose(reward_table))
    ax.set_ylabel('Episode')
    ax.set_xlabel('Epoch')
    fig.colorbar(im, ax=ax, label='Reward')
    plt.show()


number_of_steps = 10
x = np.linspace(-1., 1., 100)
figure, (ax1, ax2, ax3) = plt.subplots(1, 3)
# Plot probability density associated to updated parameters for a few steps
for i in range(0, n_epochs + 1, number_of_steps):
    ax1.plot(x, norm.pdf(x, loc=data["means"][i], scale=np.abs(data["stds"][i])), '-o', label=f'{i}')

ax1.set_xlabel("Action, a")
ax1.set_ylabel("Probability density")
ax1.set_ylim(0., 20)
#  Plot return as a function of epochs
ax2.plot(np.mean(data["rewards"], axis=1), '-.', label='Reward')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Expected reward")
# ax2.plot(data["baselines"], '-.', label='baseline')
ax2.plot(data["fidelity"], '-o', label='Fidelity')
ax2.legend()
ax1.legend()
plot_examples(figure, ax3, data["rewards"])
