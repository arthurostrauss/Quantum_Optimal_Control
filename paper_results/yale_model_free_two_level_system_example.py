"""
Code example reproducing Educational Example described in Appendix A of the paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 05/08/2022
"""

from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.assembler import RunConfig
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tqdm import tqdm
import time
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
import tensorflow as tf
from tensorflow.python.keras.losses import MSE
import tensorflow.python.ops.math_ops as math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""This code sets the simplest RL algorithm (Policy Gradient) for solving amp quantum control problem. The goal is the 
following: We have access to amp quantum computer (here amp simulator provided by IBM Q) containing one qubit. The qubit 
originally starts in the state |0>, and we would like to apply amp quantum gate (operation) to bring it to the |1> 
state. To do so, we have access to amp gate parametrized with an angle, and the RL agent must find the optimal angle 
that maximizes the probability of measuring the qubit in the |1> state. Optimal value for amplitude amp (angle/2π) is 
0.5. The RL agent chooses its actions (that is picks amp random value for amp) by drawing amp number from amp Gaussian 
distribution, of mean mu and standard deviation sigma. The trainable parameters are therefore those two latter 
variables (we expect the mean to be close to 0.5 and the variance very low). The reward is amp binary number obtained 
upon measurement of the circuit produced (only two possible outcomes can be measured). 

"""


def perform_action(amp: tf.Tensor, shots=1):
    """
    Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
    :param amp: amplitude parameter
    :param shots: number of evaluations to be done on the quantum computer (for simplicity stays to 1)
    :return: Reward table (reward for each run in the batch)
    """
    global qc, qasm
    angles = np.array(amp)
    assert len(np.shape(angles)) == 1, f'What happens : {np.shape(angles)}'

    reward_table = np.zeros(np.shape(angles))
    for i, angle in enumerate(angles):
        qc.rx(2 * np.pi * angle, 0)  # Add parametrized gate for each amplitude in the batch
        qc.measure(0, 0)  # Measure the qubit
        job = qasm.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)  # Returns dictionary with keys '0' and '1' with number of counts for each key

        #  Calculate reward
        if '1' in counts and '0' in counts:
            reward_table[i] += np.mean(np.array([1] * counts['1'] + [-1] * counts['0']))
        elif '0' in counts:
            reward_table[i] += np.mean([-1] * counts['0'])
        else:
            reward_table[i] += np.mean([1] * counts['1'])
        qc.clear()

    return reward_table  # Shape [batchsize]


# Variables to define environment
seed = 2024
config = RunConfig(shots=1, seed_simulator=seed)
qc = QuantumCircuit(1, 1, name="qc")  # Two-level system of interest, 1 qubit
qasm = QasmSimulator(method="statevector")  # Simulation backend (mock quantum computer)

# Hyperparameters for the agent
insert_baseline = True  # Indicate if you want the actor-critic version (True) or simple REINFORCE (False)
use_PPO = True
epsilon = 0.2
n_epochs = 150
batchsize = 100
eta = 0.1
eta_2 = 0.1

tf.random.set_seed(seed)
optimizer_actor = Adam(learning_rate=eta)
optimizer_critic = Adam(learning_rate=eta_2)
# optimizer = SGD(learning_rate=eta)
# Policy parameters

mu = tf.Variable(initial_value=np.random.uniform(0., 1.), trainable=True, name="µ")
sigma = tf.Variable(initial_value=np.random.uniform(1., 2.), trainable=True, name="sigma")
mu_old = tf.Variable(initial_value=mu, trainable=False)
sigma_old = tf.Variable(initial_value=sigma, trainable=False)
# Critic parameter (single state-independent baseline)

if insert_baseline:
    b = tf.Variable(initial_value=np.random.uniform(0., 1.), trainable=True, name="baseline")
else:
    b = tf.Variable(initial_value=0., trainable=True, name="baseline")

losses, means, stds, amps, rewards = np.zeros(n_epochs), np.zeros(n_epochs + 1), np.zeros(n_epochs + 1), np.zeros(
    n_epochs), np.zeros([n_epochs, batchsize])
baselines = np.zeros(n_epochs + 1)
grad_list = [None] * n_epochs


def normal_distrib(amp, mean, std):
    # return math.divide(math.exp(math.divide(math.pow(amp - mu, 2), -2 * math.pow(sigma, 2))),
    #                    math.sqrt(2 * np.pi * math.pow(sigma, 2)))
    return math.exp(-(amp - mean) ** 2 / (2 * std ** 2)) / math.sqrt(2 * np.pi * std ** 2)


for i in tqdm(range(n_epochs)):
    a = tf.random.normal([batchsize], mean=mu, stddev=sigma)
    reward = perform_action(a)

    with tf.GradientTape(persistent=True) as tape:

        # Calculate expected return (to be maximized, therefore the minus sign placed in the function below will be
        # useful when applying gradients which minimize the loss), E[R*log(proba(amp)] where proba is amp gaussian
        # probability density (cf paper of reference, educational example)
        advantage = reward - b
        if use_PPO:
            ratio = normal_distrib(a, mu, sigma) / normal_distrib(a, mu_old, sigma_old)
            loss = - tf.reduce_mean(math.minimum(advantage * ratio,
                                                 advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
        else:
            loss = -tf.reduce_mean(
                advantage * math.log(normal_distrib(a, mu, sigma)))
            # loss = tf.reduce_mean(-reward * math.log(normal_distrib(amp, mu, sigma)))

        loss2 = MSE(reward, b)  # Loss for the critic

    amps[i] = np.array(tf.reduce_mean(a))
    losses[i] = np.array(loss)
    rewards[i] = reward
    means[i] = np.array(mu)
    stds[i] = np.array(sigma)
    baselines[i] = np.array(b)

    # Compute gradients
    grads = tape.gradient(loss, [mu, sigma])
    grads2 = tape.gradient(loss2, b)
    grad_list[i] = grads + grads2

    mu_old = mu
    sigma_old = sigma

    # Apply gradients
    optimizer_actor.apply_gradients(zip(grads, (mu, sigma)))
    if insert_baseline:
        optimizer_critic.apply_gradients(zip([grads2], [b]))

means[-1] = np.array(mu)
stds[-1] = np.array(sigma)
baselines[-1] = np.array(b)
print("means: ", means, '\n')
print("stds: ", stds, '\n')
print("amplitudes: ", amps, '\n')
print("average rewards: ", np.mean(rewards, axis=1))


def plot_examples(colormaps, ax, reward_table):
    """
    Helper function to plot data with associated colormap.
    """

    ax.pcolormesh(reward_table.transpose(), cmap=colormaps, rasterized=True, vmin=-1, vmax=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Episode")
    plt.show()


number_of_steps = 10
x = np.linspace(-2., 2., 200)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
for i in range(0, n_epochs + 1, number_of_steps):
    ax1.plot(x, norm.pdf(x, loc=means[i], scale=np.abs(stds[i])), '.-', label=f'{i}')

ax1.set_xlabel("Action, a")
ax1.set_ylabel("Probability density")
ax2.plot(np.mean(rewards, axis=1), '-.', label='Reward')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Expected reward")
ax2.plot(baselines, '.-', label='baseline')
ax2.legend()
ax1.legend()

cmap = ListedColormap(["blue", "Orange"])
plot_examples(cmap, ax3, rewards)
