# from qiskit_braket_provider import BraketLocalBackend
import numpy as np
import os
import sys
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.extensions import CXGate, XGate
from qiskit.opflow import Zero, One, Plus, Minus, H, I, X, CX, S, Z
from qiskit_ibm_runtime import Estimator
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)
from helper_functions import generate_model



n_actions = 7

def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Applies a parametrized quantum circuit to a given QuantumCircuit object.

    This function is designed to add a specific set of parametrized gates to a quantum circuit,
    which are intended to be optimized during the reinforcement learning process. The number of
    parameters is determined by the global `n_actions` variable, which should be set before
    this function is called.

    Args:
        qc (QuantumCircuit): The quantum circuit to which the parametrized gates will be added.

    Returns:
        None: The function modifies the QuantumCircuit object in place.
    """

    # TODO: Make this function adaptive to a parameter vector of arbitrary length to generate the circuit from
    # Possible new argument that could be parsed

    global n_actions
    params = ParameterVector('theta', n_actions)
    qc.u(2 * np.pi * params[0], 2 * np.pi * params[1], 2 * np.pi * params[2], 0)
    qc.u(2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], 1)
    qc.rzx(2 * np.pi * params[6], 0, 1)


def get_network():
    """
    Creates and returns the neural network model and initial measurement setup for the RL agent.

    Returns:
        tuple: A tuple containing the network model and the initial measurement.
    """
    n_qubits = 2  
    N_in = n_qubits + 1  
    hidden_units = [20, 20, 30]  
    # Set up Reinforcement Learning Model
    network = generate_model((N_in,), hidden_units, n_actions, actor_critic_together=True)

    init_msmt = np.zeros((1, N_in))

    return network, init_msmt
   
def plot_training_progress(avg_return, fidelities, n_epochs, visualization_steps):
    """
    Plots the training progress of the RL agent and prints the maximum fidelity reached so far.
    """
    clear_output(wait=True)
    _, ax = plt.subplots()
    ax.plot(np.arange(1, n_epochs, 20), avg_return[0:-1:visualization_steps], '-.', label='Average return')
    ax.plot(np.arange(1, n_epochs, 20), fidelities[0:-1:visualization_steps], label='Average Gate Fidelity')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("State Fidelity")
    ax.legend()
    plt.show()
    print("Maximum fidelity reached so far:", np.max(fidelities), "at Epoch", np.argmax(fidelities))

# Training the agent
def train_agent(q_env, training_parameters):
    """
    Runs the training loop for the RL agent in the provided quantum environment.

    Args:
        q_env (QuantumEnvironment): The quantum environment in which to train the agent.
        training_parameters (dict): A dictionary of training parameters such as batch size,
                                    number of epochs, optimizer settings, etc.

    Returns:
        dict: A dictionary containing the training results, including fidelities and the final action vector.
    """
    
    network = training_parameters['network']
    init_msmt = training_parameters['init_msmt']
    
    n_epochs = training_parameters['n_epochs']
    batchsize = training_parameters['batchsize']
    optimizer = training_parameters['optimizer']
    critic_loss_coeff = training_parameters['critic_loss_coeff']
    use_PPO = training_parameters['use_PPO']
    epsilon = training_parameters['epsilon']

    mu_old = tf.Variable(initial_value=network(init_msmt)[0][0], trainable=False)
    sigma_old = tf.Variable(initial_value=network(init_msmt)[1][0], trainable=False)

    avg_return = np.zeros(n_epochs)
    fidelities = np.zeros(n_epochs)

    show_plot = False
    visualization_steps = 20

    for i in tqdm(range(n_epochs)):
        Old_distrib = MultivariateNormalDiag(loc=mu_old, scale_diag=sigma_old, validate_args=True, allow_nan_stats=False)

        with tf.GradientTape(persistent=True) as tape:
            mu, sigma, b = network(init_msmt, training=True)
            mu = tf.squeeze(mu, axis=0)
            sigma = tf.squeeze(sigma, axis=0)
            b = tf.squeeze(b, axis=0)

            Policy_distrib = MultivariateNormalDiag(loc=mu, scale_diag=sigma, validate_args=True, allow_nan_stats=False)

            action_vector = tf.stop_gradient(tf.clip_by_value(Policy_distrib.sample(batchsize), -1., 1.))
            reward = q_env.perform_action(action_vector)
            advantage = reward - b

            if use_PPO:
                ratio = Policy_distrib.prob(action_vector) / (tf.stop_gradient(Old_distrib.prob(action_vector)) + 1e-6)
                actor_loss = - tf.reduce_mean(tf.minimum(advantage * ratio, advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
            else:  # REINFORCE algorithm
                actor_loss = - tf.reduce_mean(advantage * Policy_distrib.log_prob(action_vector))

            critic_loss = tf.reduce_mean(advantage ** 2)
            combined_loss = actor_loss + critic_loss_coeff * critic_loss

        grads = tape.gradient(combined_loss, network.trainable_variables)

        if use_PPO:
            mu_old.assign(mu)
            sigma_old.assign(sigma)

        avg_return[i] = np.mean(q_env.reward_history, axis=1)[i]
        fidelities[i] = q_env.avg_fidelity_history[i]

        if show_plot and i % visualization_steps == 0:
            plot_training_progress(avg_return, fidelities, n_epochs, visualization_steps)

        optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if isinstance(q_env.estimator, Estimator):
        q_env.estimator.session.close()

    return {
        'avg_return': avg_return,
        'fidelities': fidelities,
        # returns the action vector that led to the highest gate fidelity during the training process
        'action_vector': np.mean(q_env.action_history[np.argmax(fidelities)], axis=0),
    }