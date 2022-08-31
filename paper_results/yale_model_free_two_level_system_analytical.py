"""
Code example reproducing Educational Example described in Appendix A of the paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 05/08/2022
"""

from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

"""This code sets the simplest RL algorithm (Policy Gradient) for solving a quantum control problem. The goal is the 
following: We have access to a quantum computer (here a simulator provided by IBM Q) containing one qubit. The qubit 
originally starts in the state |0>, and we would like to apply a quantum gate (operation) to bring it to the |1> 
state. To do so, we have access to a gate parametrized with an angle, and the RL agent must find the optimal angle 
that maximizes the probability of measuring the qubit in the |1> state. Optimal value for amplitude a (angle/2π) is 
0.5. The RL agent chooses its actions (i.e. picks a random value for a) by drawing a number from a Gaussian 
distribution, of mean mu and standard deviation sigma. The trainable parameters are therefore those two latter 
variables (we expect the mean to be close to 0.5 and the std very low). The reward is a binary number obtained upon 
measurement of the circuit produced (only two possible outcomes can be measured). Note that this problem does contain
only one timestep per episode, meaning that there is no need to introduce any discount factor for future reward (gamma).

"""


def perform_action(amp, shots=1):
    """
    Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
    :param amp: amplitude parameter (given as a ndarray of size batchsize)
    :param shots: number of evaluations to be done on the quantum computer (for simplicity stays to 1)
    :return: Reward table (reward for each run in the batch)
    """
    global qc, qasm

    reward_table = np.zeros(np.shape(amp))
    for j, angle in enumerate(amp):
        qc.rx(2 * np.pi * angle, 0)  # Add parametrized gate for each amplitude in the batch
        qc.measure(0, 0)  # Measure the qubit
        job = qasm.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)  # Returns a dictionary with keys '0' and '1' with number of counts for each key

        #  Calculate reward
        if '1' in counts and '0' in counts:
            reward_table[j] += np.mean(np.array([1] * counts['1'] + [-1] * counts['0']))
        elif '0' in counts:
            reward_table[j] += np.mean([-1] * counts['0'])
        else:
            reward_table[j] += np.mean([1] * counts['1'])
        qc.clear()

    return reward_table  # Shape [batchsize]


# Variables to define environment
qc = QuantumCircuit(1, 1, name="qc")  # Two-level system of interest, 1 qubit
qasm = QasmSimulator()  # Simulation backend (mock quantum computer)
seed = 2024
np.random.seed(seed)

# Hyperparameters for the agent
n_epochs = 100
batchsize = 80
eta = 0.5  # Learning rate for policy gradient
eta_2 = 0.5  # Learning rate for state-value function

# Learnable parameter for the value function (baseline/critic)
insert_baseline = True
if insert_baseline:
    b = np.random.uniform(0, 1)
else:
    b = 0.

# Trainable parameters of the policy (mean and standard deviation of the univariate Gaussian policy)

mu = np.random.uniform(0, 1)
sigma = np.random.uniform(1, 2)

means, std, amps, rewards = np.zeros(n_epochs + 1), np.zeros(n_epochs + 1), \
                            np.zeros(n_epochs), np.zeros([n_epochs, batchsize])
baselines = np.zeros(n_epochs + 1)

for i in tqdm(range(n_epochs)):
    a = np.random.normal(loc=mu, scale=np.abs(sigma), size=batchsize)
    reward = perform_action(a)

    amps[i] = np.mean(a)
    rewards[i] = reward
    means[i] = mu
    std[i] = sigma
    baselines[i] = b

    # REINFORCE Algorithm with Actor-Critic, update the parameters with the derived analytical formulas for gradients
    advantage = reward - b
    mu += eta * np.mean(advantage * (a - mu) / sigma ** 2)
    sigma += eta * np.mean(advantage * ((a - mu) ** 2 / sigma ** 3 - 1 / sigma))
    if insert_baseline:
        b -= eta_2 * np.mean(-2 * advantage)

final_mean = mu
final_std = sigma
means[-1] = final_mean
std[-1] = final_std

print("means: ", means, '\n')
print("stds: ", std, '\n')
print("amplitudes: ", amps, '\n')
print("average rewards: ", np.mean(rewards, axis=1))


def plot_examples(colormaps, reward):
    """
    Helper function to plot data with associated colormap.
    """

    n = len(colormaps)
    fig2, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                             constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(reward.transpose(), cmap=cmap, rasterized=True, vmin=-1, vmax=1)
        fig2.colorbar(psm, ax=ax)
    plt.show()


number_of_steps = 30
x = np.linspace(-5., 5., 200)
fig, (ax1, ax2) = plt.subplots(1, 2)
for i in range(0, n_epochs + 1, number_of_steps):
    ax1.plot(x, norm.pdf(x, loc=means[i], scale=np.abs(std[i])), '.-', label=f'{i}')

ax1.set_xlabel("Action, a")
ax1.set_ylabel("Probability density")
ax2.plot(np.mean(rewards, axis=1), '-.', label='Reward')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Expected reward")
ax1.legend()

cmap = ListedColormap(["red", "green"])
ax2.plot(baselines, '.-', label='baseline')
ax2.legend()
plot_examples([cmap], rewards)
