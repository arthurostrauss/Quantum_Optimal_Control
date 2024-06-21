"""
Code example reproducing Educational Example described in Appendix A of the paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 05/08/2022
"""

import tensorflow as tf
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.backends.qasm_simulator import QasmSimulator
import qiskit.quantum_info as qi
from tensorflow_probability.python.distributions import Normal
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Union
import csv

"""This code sets the simplest RL algorithm (Policy Gradient) for solving a quantum control problem. The goal is the 
following: We have access to a quantum computer (here a simulator provided by IBM Q) containing one qubit. The qubit 
originally starts in state |0>, and we would like to apply a quantum gate (operation) to bring it to state |1>.
To do so, we have access to a gate parametrized with an angle, and the RL agent must find the optimal angle 
maximizing the probability of measuring the |1> state. Optimal value for amplitude amp (angle/2π) is 
0.5 or -0.5. The RL agent chooses its actions (that is picks a random value for amp) by sampling a number from 
a Gaussian distribution, of mean mu and standard deviation sigma. The trainable parameters are therefore those two 
latter variables (we expect the mean to be close to 0.5 and the variance very low). 
The reward is a binary number obtained upon measurement (only two possible outcomes can be measured). 
"""


def perform_action(
    amp: Union[tf.Tensor, np.array], shots=1, target_state="|1>", epoch=1
):
    """
    Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
    :param amp: amplitude parameter, provided as an array of size batchsize
    :param shots: number of evaluations to be done on the quantum computer (for simplicity stays to 1)
    :param target_state: String indicating which target state is intended (can currently only be "|1>" or "|->")
    :return: Reward table (reward for each run in the batch)
    """
    global qc, qasm, tgt_string  # Use same quantum circuit instance to be reset after each run
    angles, batch = np.array(amp), len(np.array(amp))
    density_matrix = np.zeros([2, 2], dtype="complex128")
    reward_table = np.zeros(batch)

    for j, angle in enumerate(angles):
        if tgt_string == "|1>":
            qc.rx(
                2 * np.pi * angle, 0
            )  # Add parametrized gate for each amplitude in the batch
            q_state = qi.Statevector.from_instruction(qc)
            density_matrix += (
                np.array(q_state.to_operator()) / batch
            )  # Build density matrix as a statistical mixture of
            # states created by the different actions
        elif tgt_string == "|+>":
            qc.ry(
                2 * np.pi * angle, 0
            )  # Add parametrized gate for each amplitude in the batch
            q_state = qi.Statevector.from_instruction(qc)
            density_matrix += (
                np.array(q_state.to_operator()) / batch
            )  # Build density matrix as a statistical mixture
            # of states created by the different actions
            qc.h(0)  # Rotate qubit for measurement in Hadamard basis

        qc.measure(0, 0)  # Measure the qubit

        job = qasm.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(
            qc
        )  # Returns dictionary with keys '0' and '1' with number of counts for each key

        #  Calculate reward (Generalized to include any number of shots for each action)
        if tgt_string == "|1>":
            reward_table[j] += np.mean(
                np.array([1] * counts.get("1", 0) + [-1] * counts.get("0", 0))
            )
        elif tgt_string == "|+>":
            reward_table[j] += np.mean(
                np.array([1] * counts.get("0", 0) + [-1] * counts.get("1", 0))
            )
        qc.clear()  # Reset the Quantum Circuit for next iteration

    return reward_table, qi.DensityMatrix(
        density_matrix
    )  # reward_table is of Shape [batchsize]


# Variables to define environment
qc = QuantumCircuit(1, 1, name="qc")  # Two-level system of interest, 1 qubit
qasm = QasmSimulator(method="statevector")  # Simulation backend (mock quantum computer)

# TODO:: Define a reward function/circuit for each target state in the dictionary
target_states_list = {
    "|1>": qi.DensityMatrix(np.array([[0.0], [1.0]]) @ np.array([[0.0, 1.0]])),
    "|+>": qi.DensityMatrix(0.5 * np.array([[1.0], [1.0]]) @ np.array([[1.0, 1.0]])),
}
tgt_string = "|+>"

# Hyperparameters for the agent
seed = 2364  # Seed for action sampling (ref 2763)

optimizer_string = "Adam"

n_epochs = 60
batch_size = 50
eta = 0.1  # Learning rate for policy update step

critic_loss_coeff = 0.5

use_PPO = True
epsilon = 0.2  # Parameter for ratio clipping value (PPO)
grad_clip = 0.3
sigma_eps = 1e-6

optimizer = None
if optimizer_string == "Adam":
    optimizer = tf.optimizers.Adam(learning_rate=eta)
elif optimizer_string == "SGD":
    optimizer = tf.optimizers.SGD(learning_rate=eta)


def constrain_mean_value(mu_var):
    return tf.clip_by_value(mu_var, -1.0, 1.0)


def constrain_std_value(std_var):
    return tf.clip_by_value(std_var, 1e-3, 3)


# Policy parameters
mu = tf.Variable(
    initial_value=tf.random.normal([], stddev=0.05),
    trainable=True,
    name="µ",
    constraint=constrain_mean_value,
)
sigma = tf.Variable(
    initial_value=1.0, trainable=True, name="sigma", constraint=constrain_std_value
)

# Old parameters are updated with one-step delay, necessary for PPO implementation
mu_old = tf.Variable(initial_value=mu, trainable=False, name="µ_old")
sigma_old = tf.Variable(initial_value=sigma, trainable=False, name="sigma_old")
# Critic parameter (single state-independent baseline b)

b = tf.Variable(initial_value=0.0, name="baseline", trainable=True)

#  Keep track of variables
data = {
    "means": np.zeros(n_epochs),
    "stds": np.zeros(n_epochs),
    "amps": np.zeros([n_epochs, batch_size]),
    "rewards": np.zeros([n_epochs, batch_size]),
    "critic_loss": np.zeros(n_epochs),
    "fidelity": np.zeros(n_epochs),
    "grads": np.zeros((n_epochs, 3)),
    "hyperparams": {
        "learning_rate": eta,
        "seed": seed,
        "clipping_PPO": epsilon,
        "grad_clip_value": grad_clip,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_state": (tgt_string, target_states_list[tgt_string]),
        "PPO?": use_PPO,
        "critic_loss_coeff": critic_loss_coeff,
        "optimizer": optimizer_string,
    },
}
save_data = False  # Decide if data should be saved in a csv file

policy_params = "Policy params:"
for i in tqdm(range(n_epochs)):
    print("EPOCH", i)
    print(f"{policy_params:#<100}")
    print(np.array(mu), "+-", np.array(sigma))
    print("baseline", np.array(b))

    # Sample action from policy (Gaussian distribution with parameters mu and sigma)
    Normal_distrib = Normal(
        loc=mu, scale=sigma, validate_args=True, allow_nan_stats=False
    )
    Normal_distrib_old = Normal(
        loc=mu_old, scale=sigma_old, validate_args=True, allow_nan_stats=False
    )
    a = Normal_distrib.sample(batch_size)
    # Run quantum circuit to retrieve rewards (in this example, only one time step)
    reward, dm_observed = perform_action(a, shots=1, target_state=tgt_string, epoch=i)
    print("Average Return:", np.array(tf.reduce_mean(reward)))

    with tf.GradientTape(persistent=True) as tape:
        """
        Calculate loss function (average return to be maximized, therefore the minus sign placed in front of the loss
        since applying gradients minimize the loss), E[R*log(proba(amp)] where proba is the gaussian
        probability density (cf paper of reference, educational example).
        In case of the PPO, loss function is slightly changed.
        """

        advantage = reward - b
        if use_PPO:
            ratio = Normal_distrib.prob(a) / (Normal_distrib_old.prob(a) + sigma_eps)
            clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
            actor_loss = -tf.reduce_mean(
                tf.minimum(advantage * ratio, advantage * clipped_ratio)
            )

        else:  # REINFORCE algorithm
            log_probs = Normal_distrib.log_prob(a)
            actor_loss = -tf.reduce_mean(advantage * log_probs)

        critic_loss = tf.reduce_mean(advantage**2)

        combined_loss = actor_loss + critic_loss_coeff * critic_loss
    # Compute gradients

    combined_grads = tape.gradient(combined_loss, tape.watched_variables())
    grads = tf.clip_by_value(combined_grads, -grad_clip, grad_clip)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        mu_old.assign(mu)
        sigma_old.assign(sigma)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, tape.watched_variables()))

    data["amps"][i] = np.array(a)
    data["rewards"][i] = reward
    data["means"][i] = np.array(mu)
    data["stds"][i] = np.array(sigma)
    data["critic_loss"][i] = np.array(critic_loss)
    data["fidelity"][i] = qi.state_fidelity(target_states_list[tgt_string], dm_observed)
    data["grads"][i] = grads

# print(data)

if save_data:
    w = csv.writer(open(f"output_seed{seed}_lr{eta}.csv", "w"))

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

    ax.pcolormesh(np.transpose(vals), cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(0, vals.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(0, vals.shape[1], 1), minor=True)
    ax.grid(which="both", color="w", linestyle="-")
    ax.set_aspect("equal")
    ax.set_ylabel("Episode")
    ax.set_xlabel("Epoch")
    plt.show()


x = np.linspace(-1.0, 1.0, 300)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# Plot probability density associated to updated parameters for a few steps
for i in np.linspace(0, n_epochs - 1, 6, dtype=int):
    ax1.plot(
        x,
        norm.pdf(x, loc=data["means"][i], scale=np.abs(data["stds"][i])),
        "-o",
        label=f"{i}",
    )

ax1.set_xlabel("Action, a")
ax1.set_ylabel("Probability density")
ax1.set_ylim(0.0, 20)
#  Plot return as a function of epochs
ax2.plot(np.mean(data["rewards"], axis=1), "-o", label="Return")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Expected return")
# ax2.plot(data["critic_loss"], '-.', label='Critic Loss')
ax2.plot(
    data["fidelity"],
    "-o",
    label=f"State Fidelity (target: {tgt_string})",
    color="green",
)
ax2.legend()
ax1.legend()
plot_examples(ax3, data["rewards"])
