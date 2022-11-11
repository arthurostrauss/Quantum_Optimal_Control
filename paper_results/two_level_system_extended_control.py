"""
Code for arbitrary state preparation based on scheme described in Appendix D.2b  of paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 11/11/2022
"""
# Qiskit imports for building RL environment (circuit level)
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV1
from qiskit_aer.backends.qasm_simulator import QasmSimulator
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, state_fidelity

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_probability.python.distributions import MultivariateNormalDiag, Categorical
from tensorflow.python.keras.callbacks import TensorBoard
# Additional imports
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Dict
import csv
from itertools import product
from copy import deepcopy

""" Here we set a RL agent based on PPO and an actor critic network to perform arbitrary state preparation based on
scheme proposed in the appendix of Vladimir Sivak paper. 
In there, we design classes for the environment (Quantum Circuit simulated on IBM Q simulator) and the agent 
"""


def rotation_gate(pauli_string: str):
    Op_list = []
    for index in range(len(pauli_string)):
        if pauli_string[index] == "I":
            pass
        elif pauli_string[index] == "X":
            Op_list.append(f"H{index}")
        elif pauli_string[index] == "Y":
            Op_list.append(f"H{index}")
            Op_list.append(f"S{index}")
        elif pauli_string[index] == "Z":
            pass
        else:
            raise NameError('Letter does not correspond to a Pauli (I, X, Y, Z)')
    return Op_list


def generate_pauli_ops(n_qubits: int):
    Pauli_ops = [
        {"string": ''.join(s),
         "matrix": Pauli(''.join(s)).to_matrix(),
         "rotation_gates": rotation_gate(''.join(s))}
        for s in product(["I", "X", "Y", "Z"], repeat=n_qubits)
    ]
    return Pauli_ops


def apply_parametrized_circuit(qc, angle):
    pass


class QuantumEnvironment:
    def __init__(self, n_qubits: int,
                 target_state: Dict[str, Union[str, DensityMatrix]],
                 abstraction_level: str,
                 backend: BackendV1 = QasmSimulator(method="statevector"),
                 projector_reward_scheme: bool = False,
                 sampling_Pauli_space: int = 10,
                 n_shots: int = 1,
                 c_factor: float = 0.5):
        """

        :param n_qubits: Number of qubits in quantum system
        :param abstraction_level: Circuit or pulse level parametrization of action space
        :param backend: Quantum backend, QASM simulator by default
        :param target_state: control target of interest
        :param projector_reward_scheme: Indicate which kind of reward circuit to execute
        :param sampling_Pauli_space: Number of samples to build a fidelity estimator for one action
        :param n_shots: Number of shots to sample for one specific computation (action/Pauli expectation sampling)
        """

        self.c_factor = c_factor
        assert abstraction_level == 'circuit' or abstraction_level == 'pulse', 'Abstraction layer parameter can be' \
                                                                               'only pulse or circuit'
        self.abstraction_level = abstraction_level
        if abstraction_level == 'circuit':
            self.qc = QuantumCircuit(n_qubits, n_qubits)
            self.backend = backend
        else:
            # TODO: Define pulse level (Schedule most likely, cf Qiskit Pulse doc)
            pass
        self.time_step = 0
        self.Pauli_ops = generate_pauli_ops(n_qubits)
        self.rotation_ops = [rotation_gate(pauli['name']) for pauli in self.Pauli_ops]
        self.d = 2 ** n_qubits  # Dimension of Hilbert space
        self.density_matrix = np.zeros([d, d], dtype='complex128')
        self.sampling_Pauli_space = sampling_Pauli_space
        self.n_shots = n_shots
        self.target_state = self.calculate_chi_target_state(target_state)

    def calculate_chi_target_state(self, target_state: Dict):
        """
        Calculatte for all P
        :param target_state: Dictionary containing info on target state (name, density matrix)
        :return: target state, initializes self.target_state argument
        """
        target_state["Chi"] = np.zeros(self.d ** 2)
        for k in range(self.d ** 2):
            target_state["Chi"][k] = 1 / np.sqrt(self.d) * np.trace(np.array(target_state["dm"].to_operator())
                                                                    @ self.Pauli_ops[k]["matrix"])
        return target_state

    def perform_action(self, amps: Union[tf.Tensor, np.array],
                       dfe_params: Optional[Tuple] = None):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param amps: amplitude parameter, provided as an array of shape [batchsize, 3]
        :param dfe_params: Parameters (epsilon, delta) for performing direct fidelity estimation (epsilon is the desired
        additive error, delta is the failure probability)

        :return: Reward table (reward for each run in the batch), observations (measurement outcomes),
        obtained density matrix
        """

        angles, batchsize = np.array(amps), len(np.array(amps))
        outcome = np.zeros([batchsize, 2])
        reward_table = np.zeros(batchsize)
        expected_value = np.zeros(self.d ** 2)
        reward_factor = np.zeros(self.d ** 2)
        for j, angle in enumerate(angles):  # Iterate over batch of actions

            # Direct fidelity estimation protocol for one qubit  (https://doi.org/10.1103/PhysRevLett.106.230501)
            distribution = Categorical(probs=self.target_state["Chi"] ** 2)
            k_samples = distribution.sample(self.sampling_Pauli_space)
            pauli_index, idx, pauli_shots = tf.unique_with_counts(k_samples)

            if self.abstraction_level == 'circuit':
                # Perform actions, followed by relevant expectation value sampling for reward calculation
                for k in range(len(pauli_index)):  # Iterate over Pauli observables to sample expectation values

                    # Apply parametrized quantum circuit (action)
                    apply_parametrized_circuit(self.qc, angle)

                    # Keep track of state for benchmarking purpose only
                    q_state = Statevector.from_instruction(self.qc)
                    self.density_matrix += np.array(q_state.to_operator()) / len(angles)

                    # Apply relevant qubit rotations to measure in corresponding Pauli eigenbasis
                    for op in self.Pauli_ops[int(pauli_index[k])]["rotation_gates"]:
                        gate, qubit_index = op[0], int(op[-1])
                        if gate == "H":
                            self.qc.h(qubit_index)
                        if gate == "S":
                            self.qc.s(qubit_index)
                    self.qc.measure(0, 0)  # Measure the qubit
                    total_shots = self.n_shots * int(pauli_shots[k])
                    job = self.backend.run(self.qc, shots=total_shots)
                    result = job.result()
                    counts = result.get_counts(self.qc)
                    self.qc.clear()

                    # As we rotated the states and measure in sigma_z basis, we calculate the full sigma_z exp value
                    expected_value[k] = np.sum([(-1) ** c.count("1") * counts[c] / self.n_shots for c in counts.keys()])
                    reward_factor[k] = self.c_factor * self.target_state["Chi"][int(pauli_index[k])] * \
                                       expected_value[k] / (self.d * float(distribution.prob(pauli_index[k])))
                reward_table[j] = tf.reduce_sum(reward_factor)

                # TODO: Convert everything to Tensors

        return reward_table, outcome  # Shape [batchsize]


class Agent:
    def __init__(self):
        pass


# Variables to define environment
seed = 3590  # Seed for action sampling
qasm = QasmSimulator(method="statevector")  # Simulation backend (mock quantum computer)
n_qubits = 1
d = 2 ** n_qubits

dim_factor = 1 / np.sqrt(d)  # Factor for computing expectation values of different Pauli ops
target_state = {
    "|1>": {
        "dm": DensityMatrix(np.array([[0.], [1.]]) @ np.array([[0., 1.]])),
        "Chi": np.zeros(d ** 2)
    },
    "|->": {
        "dm": DensityMatrix(0.5 * np.array([[1.], [-1.]]) @ np.array([[1., -1.]])),
        "Chi": np.zeros(d ** 2)
    }
}

tgt_string = "|->"
print(target_state)
# Hyperparameters for the agent
insert_baseline = True  # Indicate if you want the actor-critic version (True) or simple REINFORCE (False)
use_PPO = True
epsilon = 0.2  # Parameter for clipping value (PPO)
n_epochs = 50  # Number of epochs
batchsize = 50  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
eta = 0.1  # Learning rate for policy update step
eta_2 = 0.1  # Learning rate for critic (value function) update step

concurrent_optimization = True  # Fix if optimization of actor and critic should be done by same optimizer or separately

if insert_baseline:
    if concurrent_optimization:
        # Choose optimizer of your choice by commenting irrelevant line
        optimizer = Adam(learning_rate=eta)
        # optimizer = SGD(learning_rate=eta)
    else:
        # Choose optimizer of your choice by commenting irrelevant line
        optimizer_actor, optimizer_critic = Adam(learning_rate=eta), Adam(learning_rate=eta_2)
        # optimizer_actor, optimizer_critic = SGD(learning_rate=eta), SGD(learning_rate=eta_2)
else:
    # Choose optimizer of your choice by commenting irrelevant line
    optimizer = Adam(learning_rate=eta)
    # optimizer = SGD(learning_rate=eta)

# Policy parameters
N_in = 2  # One input neuron indicates how many times |0> was measured, the other how many |1>
N_out = 7  # 3 output neurons for the mean, 3 for the diagonal covariance (vector of 3 angles shall be drawn),
# 1 for critic
layers = [10, 10]  # List containing the number of neurons in each hidden layer

input_layer = Input(shape=(2,), batch_size=1)
hidden = Sequential([Dense(layer, activation='relu') for layer in layers])(input_layer)
actor_output = Dense(N_out - 1, activation=None)(hidden)
critic_output = Dense(1, activation=None)(hidden)
network = Model(inputs=input_layer, outputs=[actor_output, critic_output])
initial_action = np.zeros([batchsize, 3])
_, initial_measurement_outcome, _ = perform_action(amps=initial_action, tgt_string=tgt_string, dfe_params=(0.1, 0.1))
# TODO: Put one hot encoding of time step in most general setting
sigma_eps = 1e-6  # for numerical stability

#  Keep track of variables (when script will be functional, do some saving to external file)
data = {
    "means": np.zeros(n_epochs + 1),
    "stds": np.zeros(n_epochs + 1),
    "amps": np.zeros([n_epochs, batchsize]),
    "rewards": np.zeros([n_epochs, batchsize]),
    "baselines": np.zeros(n_epochs + 1),
    "fidelity": np.zeros(n_epochs),
    "params": {
        "learning_rate": eta,
        "seed": seed,
        "clipping_PPO": epsilon,
        "n_epochs": n_epochs,
        "batchsize": batchsize,
        "target_state": (tgt_string, target_state[tgt_string]),
        "critic?": insert_baseline,
        "PPO?": use_PPO,
        "Concurrent optimization?": concurrent_optimization
    }
}
log_probs_old = None

for i in tqdm(range(n_epochs)):

    with tf.GradientTape(persistent=True) as tape:

        """
        Calculate return (to be maximized, therefore the minus sign placed in front of the loss 
        since applying gradients minimize the loss), E[R*log(proba(amp)] where proba is the gaussian
        probability density (cf paper of reference, educational example).
        In case of the PPO, loss function is slightly changed.
        """

        # Sample action from policy (Gaussian distribution with parameters mu and sigma)

        policy_params, b = network(initial_measurement_outcome)
        mu, sigma = policy_params[:3], policy_params[3:]
        print(sigma)
        Distribution = MultivariateNormalDiag(loc=mu, scale_diag=sigma + sigma_eps)
        action_vector, log_probs = Distribution.experimental_sample_and_log_prob([batchsize], seed=seed)

        # Run quantum circuit to retrieve rewards (in this example, only one time step)
        reward, _, dm_observed = perform_action(action_vector, shots=1)

        advantage = reward - b  # If not using the critic (baseline), then b=0, and we are left with the reward
        if use_PPO:
            if i == 0:
                log_probs_old = log_probs
            ratio = tf.exp(log_probs - log_probs_old)

            actor_loss = - tf.reduce_mean(tf.minimum(advantage * ratio,
                                                     advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
        else:  # REINFORCE algorithm
            actor_loss = - tf.reduce_mean(advantage * log_probs)

        if insert_baseline:
            # loss2 = MSE(reward, b)  # Loss for the critic (Mean square error between return and the baseline)
            critic_loss = tf.reduce_mean(advantage ** 2)
        combined_loss = actor_loss + 0.5 * critic_loss

    # Compute gradients
    grad_clip = 0.001

    combined_grads = tape.gradient(combined_loss, network.trainable_variables)
    # combined_grads = tf.clip_by_value(grad3, -grad_clip, grad_clip)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        log_probs_old = log_probs

    data["amps"][i] = np.array(action_vector)
    data["rewards"][i] = reward
    data["means"][i] = np.array(mu)
    data["stds"][i] = np.array(sigma)
    data["baselines"][i] = np.array(b)
    data["fidelity"][i] = state_fidelity(target_state[tgt_string]["dm"], dm_observed)

    # Apply gradients
    optimizer.apply_gradients(zip(combined_grads, network.trainable_variables))

data["means"][-1] = np.array(mu)
data["stds"][-1] = np.array(sigma)
data["baselines"][-1] = np.array(b)
print(data)

# open file for writing, "w" is writing
w = csv.writer(open("output.csv", "w"))

# loop over dictionary keys and values
for key, val in data.items():
    # write every key and value to file
    w.writerow([key, val])

"""
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
Plotting tools
"""


#  Plotting results
def plot_examples(ax, reward_table):
    """
    Helper function to plot data with associated colormap, used for plotting the reward per each epoch and each episode
    (From original repo associated to the paper https://github.com/v-sivak/quantum-control-rl)
    """

    vals = np.where(reward_table == 1, 0.6, -0.9)

    ax.pcolormesh(np.transpose(vals), cmap='RdYlGn', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(0, vals.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(0, vals.shape[1], 1), minor=True)
    ax.grid(which='both', color='w', linestyle='-')
    ax.set_aspect('equal')
    ax.set_ylabel('Episode')
    ax.set_xlabel('Epoch')
    plt.show()


number_of_steps = 10
x = np.linspace(-1., 1., 100)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
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
ax2.plot(data["baselines"], '-.', label='baseline')
ax2.plot(data["fidelity"], '-o', label='Fidelity')
ax2.legend()
ax1.legend()
plot_examples(ax3, data["rewards"])
