"""
Code example reproducing Educational Example described in Appendix A of the paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 05/08/2022
"""
# Qiskit imports for building RL environment (circuit level)
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, state_fidelity

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, RNN, Input
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
# from tensorflow.python.keras.losses import MSE
from tensorflow_probability.python.distributions import MultivariateNormalDiag, Categorical

# Additional imports
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
import csv

"""This code sets the simplest RL algorithm (Policy Gradient) for solving a quantum control problem. The goal is the 
following: We have access to a quantum computer (here a simulator provided by IBM Q) containing one qubit. The qubit 
originally starts in the state |0>, and we would like to apply a quantum gate (operation) to bring it to the |1> 
state. To do so, we have access to a gate parametrized with an angle, and the RL agent must find the optimal angle 
that maximizes the probability of measuring the qubit in the |1> state. Optimal value for amplitude amp (angle/2π) is 
0.5. The RL agent chooses its actions (that is picks a random value for amp) by drawing a number from a Gaussian 
distribution, of mean mu and standard deviation sigma. The trainable parameters are therefore those two latter 
variables (we expect the mean to be close to 0.5 and the variance very low). The reward is a binary number obtained 
upon measurement of the circuit produced (only two possible outcomes can be measured). 

"""


def perform_action(amps: Union[tf.Tensor, np.array], tgt_string: str, shots: Optional[int],
                   dfe_params: Optional[Tuple]):
    """
    Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
    :param amps: amplitude parameter, provided as an array of shape [batchsize, 3]
    :param tgt_string: String indicator of desired target state: will look for corresponding probability distribution
    to perform direct fidelity estimation (should be a key of the target_state dictionary)
    :param shots: Fixed number of shots to calculate estimation of reward on quantum computer
    :param dfe_params: Parameters (epsilon, delta) for performing direct fidelity estimation (epsilon is the desired
    additive error, and delta is the failure probability)

    :return: Reward table (reward for each run in the batch)
    """
    global qc, qasm, seed
    target = target_state[tgt_string]
    angles = np.array(amps)
    density_matrix = np.zeros([2, 2], dtype='complex128')
    outcome = np.zeros([batchsize, 2])
    reward_table = np.zeros(np.shape(angles))
    for j, angle in enumerate(angles):
        qc.u(angle[0], angle[1], angle[2], 0)
        q_state = Statevector.from_instruction(qc)
        density_matrix += np.array(q_state.to_operator()) / len(angles)
        # Direct fidelity estimation protocol for one qubit  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(target["Chi"] ** 2)

        if shots is not None and dfe_params is None:
            k_samples = distribution.sample(shots)
            shots_Pauli_X, shots_Pauli_Y = len(tf.where(k_samples == 1)), len(tf.where(k_samples == 2))

        else:
            l = np.ceil(1/(dfe_params[0]**2*dfe_params[1]))
            k_samples = distribution.sample(l)
            m = np.ceil(2 * np.log(2/dfe_params[1])/(d * target["Chi"][k_samples]**2 * l * dfe_params[0]**2))
            shots_Pauli_X = np.sum(np.where(k_samples == 1, m, 0))
            shots_Pauli_Y = np.sum(np.where(k_samples == 2, m, 0))

        shots_Pauli_Z = shots - (shots_Pauli_X + shots_Pauli_Y)
        qc_x = qc.copy('qc_x')
        qc_y = qc.copy('qc_y')

        qc_x.ry(np.pi / 2, 0)
        qc_y.rx(np.pi / 2, 0)

        qc.measure(0, 0)  # Measure the qubit
        qc_x.measure(0, 0)
        qc_y.measure(0, 0)
        job_x = qasm.run(qc, shots=shots_Pauli_X, seed_simulator=seed)
        job_y = qasm.run(qc, shots=shots_Pauli_Y, seed_simulator=seed)
        job_z = qasm.run(qc, shots=shots_Pauli_Z, seed_simulator=seed)
        result_x = job_x.result()
        result_y = job_y.result()
        result_z = job_z.result()
        counts_x, counts_y, counts_z = result_x.get_counts(qc_x), result_y.get_counts(qc_y), result_z.get_counts(qc)


        if '0' not in counts:
            outcome[j][0] = 0.
            outcome[j][1] = 1.
        elif '1' not in counts:
            outcome[j][0] = 1.
            outcome[j][1] = 0.
        else:
            outcome[j][0], outcome[j][1] = counts['0'] / shots, counts['1'] / shots

        expectation_estimate = outcome[:, -1] - outcome[:, 0]

        qc.clear()
    return reward_table, outcome, DensityMatrix(density_matrix)  # Shape [batchsize]


# Variables to define environment
seed = 3590  # Seed for action sampling
seed2 = 3000  # Seed for QASM simulator
tf.random.set_seed(seed)
np.random.seed(seed)
qc = QuantumCircuit(1, 1, name="qc")  # Two-level system of interest, 1 qubit
qasm = QasmSimulator(method="statevector")  # Simulation backend (mock quantum computer)

Pauli_ops = [Pauli(s).to_matrix() for s in ["I", "X", "Y", "Z"]]
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

for tgt in target_state.keys():
    for k in range(d ** 2):
        target_state[tgt]["Chi"][k] = dim_factor * np.trace(np.array(target_state[tgt]["dm"].to_operator())
                                                            * Pauli_ops[k]).real
print(target_state)
tgt_string = "|->"

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
_, measurement_outcome, _ = perform_action(amps=initial_action, shots=N_shots)

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

        policy_params, b = network(measurement_outcome)
        mu, sigma = policy_params[:3], policy_params[3:]
        print(sigma)
        Distribution = MultivariateNormalDiag(loc=mu, scale_diag=sigma + sigma_eps)
        action_vector, log_probs = Distribution.experimental_sample_and_log_prob([batchsize], seed=seed)

        # Run quantum circuit to retrieve rewards (in this example, only one time step)
        reward, measurement_outcome, dm_observed = perform_action(action_vector, shots=1)

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
