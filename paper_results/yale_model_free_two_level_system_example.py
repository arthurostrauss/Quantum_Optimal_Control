"""
Code example reproducing Educational Example described in Appendix A of the paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 05/08/2022
"""

from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
import tensorflow as tf
import tensorflow.python.ops.math_ops as math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
This code sets the simplest RL algorithm (Policy Gradient) for solving a quantum control problem.
The goal is the following: We have access to a quantum computer (here a simulator provided by IBMQ) containing one 
qubit. The qubit originally starts in the state |0>, and we would like to apply a quantum gate (operation) to bring it
to the |1> state. To do so, we have access to a gate parametrized with an angle, and the RL agent must find the optimal
angle that maximizes the probability of measuring the qubit in the |1> state. Optimal value for amplitude a (angle/2π) is
0.5. 
The RL agent chooses its actions (that is picks a random value for a) by drawing a number from a Gaussian distribution,
of mean mu and variance sigma_2. The trainable parameters are therefore those two latter variables (we expect the mean 
to be close to 0.5 and the variance very low).
The reward is a binary number obtained upon measurement of the circuit produced (only two possible outcomes can be measured).

"""


def perform_action(a: tf.Tensor, shots=1):
    """
    Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
    :param a: amplitude parameter
    :param shots: number of evaluations to be done on the quantum computer (for simplicity stays to 1)
    :return: Reward table (reward for each run in the batch)
    """
    global qc, qasm
    angles = np.array(a)
    assert len(np.shape(angles)) == 1, f'What happens : {np.shape(angles)}'

    reward_table = np.zeros(np.shape(angles))
    for i, angle in enumerate(angles):
        qc.rx(2 * np.pi * angle, 0)  # Add parametrized gate for each amplitude in the batch
        qc.measure(0, 0)  # Measure the qubit
        job = qasm.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)  # Returns a dictionary with keys '0' and '1' with number of counts for each key

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
qc = QuantumCircuit(1, 1, name="qc")  # Two-level system of interest, 1 qubit
qasm = QasmSimulator()  # Simulation backend (mock quantum computer)

# Hyperparameters for the agent
n_epochs = 100
batchsize = 50
eta = 0.01
# optimizer = Adam(learning_rate=eta)
optimizer = SGD(learning_rate=eta)
mu = tf.Variable(initial_value=0.45, trainable=True, name="mean")
sigma = tf.Variable(initial_value=0.1, trainable=True, name="variance")
losses, means, std, amps, rewards = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(
    n_epochs), np.zeros(n_epochs)
grad_list = [None] * n_epochs


def normal_distrib(a, mu, sigma):
    # return math.divide(math.exp(math.divide(math.pow(a - mu, 2), -2 * math.pow(sigma, 2))),
    #                    math.sqrt(2 * np.pi * math.pow(sigma, 2)))
    return math.exp(-(a - mu) ** 2 / (2 * sigma ** 2)) / math.sqrt(2 * np.pi * sigma ** 2)


for i in range(n_epochs):
    with tf.GradientTape() as tape:
        tape.watch(mu)
        tape.watch(sigma)
        a = tf.random.normal([batchsize], mean=mu, stddev=sigma)
        reward = perform_action(a)

        # Calculate expected return (to be maximized, therefore the minus sign placed in the function below will be
        # useful when applying gradients which minimize the loss),
        # E[R*log(proba(a)] where proba is a gaussian probability density (cf paper of reference, educational example)

        # loss = tf.reduce_mean(-reward * math.log(normal_distrib(a, mu, sigma)))
        loss = tf.reduce_mean(
            -reward * math.log(math.exp(-(a - mu) ** 2 / (2 * sigma ** 2)) / math.sqrt(2 * np.pi * sigma ** 2)))

    amps[i] = np.array(tf.reduce_mean(a))
    losses[i] = np.array(loss)
    rewards[i] = np.array(tf.reduce_mean(reward))
    means[i] = np.array(mu)
    std[i] = np.array(sigma)

    # Compute gradients
    grads = tape.gradient(loss, [mu, sigma])
    grad_list[i] = grads

    # Apply gradients
    optimizer.apply_gradients(zip(grads, (mu, sigma)))
final_mean = np.array(mu)
final_std = np.array(tf.sqrt(sigma))

print("means: ", means, '\n')
print("stds: ", std, '\n')
print("amps: ", amps, '\n')
print("rewards: ", rewards)
x = np.linspace(-1., 1., 300)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, norm.pdf(x, loc=final_mean, scale=final_std))
ax2.plot(range(n_epochs), rewards)
ax1.set_xlabel("Action, a")
ax1.set_ylabel("Probability density")
ax2.plot(range(n_epochs), rewards)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Expected reward")
plt.show()


