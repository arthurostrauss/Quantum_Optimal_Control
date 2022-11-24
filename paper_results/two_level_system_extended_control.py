"""
Code for arbitrary state preparation based on scheme described in Appendix D.2b  of paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 11/11/2022
"""

# Qiskit imports for building RL environment (circuit level)
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, IBMQ, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, state_fidelity, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, IBMBackend
from qiskit.primitives import Estimator


# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_probability.python.distributions import MultivariateNormalDiag, Categorical
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_environment import TFEnvironment

from tensorflow.python.keras.callbacks import TensorBoard
# Additional imports
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Union, Dict
import csv
from itertools import product


""" Here we set a RL agent based on PPO and an actor critic network to perform arbitrary state preparation based on
scheme proposed in the appendix of Vladimir Sivak paper. 
In there, we design classes for the environment (Quantum Circuit simulated on IBM Q simulator) and the agent 
"""

# IBMQ.save_account(TOKEN)
IBMQ.load_account()  # Load account from disk
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')


def generate_pauli_ops(n_qubits: int):
    """Return a dictionary containing all Pauli operators for n qubits
    :param n_qubits: number of qubits in considered quantum system

    """
    Pauli_ops = [
        {"name": ''.join(s),
         "matrix": Pauli(''.join(s)).to_matrix()
         }
        for s in product(["I", "X", "Y", "Z"], repeat=n_qubits)
    ]
    return Pauli_ops


def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    :param qc: Quantum Circuit instance to add the gates on
    :return:
    """
    qc.num_qubits
    params = ParameterVector('theta', 1)
    qc.ry(2 * np.pi * params[0], 0)
    qc.cx(0, 1)
    # qc.u(angle[0][0], angle[0][1], angle[0, 2], 0)
    # qc.u(angle[1][0], angle[1][1], angle[1, 2], 1)
    # qc.ecr(0, 1)


class QuantumEnvironment:  # TODO: Build a PyEnvironment out of it
    def __init__(self, n_qubits: int, target_state: Dict[str, Union[str, DensityMatrix]], abstraction_level: str,
                 backend: IBMBackend, sampling_Pauli_space: int = 10, n_shots: int = 1, c_factor: float = 0.5):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param n_qubits: Number of qubits in quantum system
        :param abstraction_level: Circuit or pulse level parametrization of action space
        :param backend: Quantum backend, QASM simulator by default
        :param target_state: control target of interest
        :param sampling_Pauli_space: Number of samples to build fidelity estimator for one action
        :param n_shots: Number of shots to sample for one specific computation (action/Pauli expectation sampling)
        :param c_factor: Scaling factor for reward definition
        """

        self.c_factor = c_factor
        assert abstraction_level == 'circuit' or abstraction_level == 'pulse', 'Abstraction layer parameter can be' \
                                                                               'only pulse or circuit'
        self.abstraction_level = abstraction_level
        if abstraction_level == 'circuit':
            self.q_register = QuantumRegister(n_qubits, name="q")
            self.c_register = ClassicalRegister(n_qubits, name="c")
            self.qc = QuantumCircuit(self.q_register, self.c_register)
            self.backend = backend
        else:
            # TODO: Define pulse level (Schedule most likely, cf Qiskit Pulse doc)
            pass
        self.time_step = 0
        self.Pauli_ops = generate_pauli_ops(n_qubits)
        self.d = 2 ** n_qubits  # Dimension of Hilbert space
        self.density_matrix = np.zeros([self.d, self.d], dtype='complex128')
        self.sampling_Pauli_space = sampling_Pauli_space
        self.n_shots = n_shots
        self.target_state = self.calculate_chi_target_state(target_state)

    def calculate_chi_target_state(self, target_state: Dict):
        """
        Calculate for all P
        :param target_state: Dictionary containing info on target state (name, density matrix)
        :return: target state, initializes self.target_state argument
        """
        # target_state["Chi"] = np.zeros(self.d ** 2, dtype="complex_")
        assert np.imag([np.array(target_state["dm"].to_operator())
                        @ self.Pauli_ops[k]["matrix"] for k in
                        range(self.d ** 2)]).all() == 0.
        target_state["Chi"] = np.array([np.trace(np.array(target_state["dm"].to_operator())
                                                 @ self.Pauli_ops[k]["matrix"]).real / np.sqrt(self.d) for k in
                                        range(
                                            self.d ** 2)])  # Real part is taken to convert it in a good format, but im
        # is 0 systematically as dm is hermitian and Pauli is traceless
        return target_state

    def perform_action(self, actions: Union[tf.Tensor, np.array]):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :return: Reward table (reward for each run in the batch), observations (measurement outcomes),
        obtained density matrix
        """
        global options
        angles, batch_size = np.array(actions), len(np.array(actions))

        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=self.target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, _, pauli_shots = tf.unique_with_counts(k_samples)

        reward_factor = np.round([self.c_factor * self.target_state["Chi"][p] / (self.d * distribution.prob(p))
                                  for p in pauli_index], 5)
        observables = [SparsePauliOp(self.Pauli_ops[p]["name"]) for p in pauli_index]

        # print(type(self.target_state["Chi"]))
        # print("drawn Pauli operators to sample", k_samples)
        # print(pauli_index, pauli_shots)
        # print([distribution.prob(p) for p in pauli_index])
        # print(observables)
        # Perform actions, followed by relevant expectation value sampling for reward calculation

        # Apply parametrized quantum circuit (action)
        apply_parametrized_circuit(self.qc)

        # Keep track of state for benchmarking purposes only
        self.density_matrix = np.zeros([self.d, self.d], dtype='complex128')
        for angle_set in angles:
            qc_2 = self.qc.bind_parameters(angle_set)
            q_state = Statevector.from_instruction(qc_2)
            self.density_matrix += np.array(q_state.to_operator())
        self.density_matrix /= batch_size

        total_shots = self.n_shots * pauli_shots
        job_list = []
        result_list = []
        exp_values = np.zeros((len(pauli_index), batch_size))
        with Session(service=service, backend=self.backend) as session:
            estimator = Estimator(options=options)  # TODO: Figure how to put in the options
            for p in range(len(pauli_index)):
                job = estimator.run(circuits=[self.qc] * batch_size, observables=[observables[p]] * batch_size,
                                    parameter_values=angles,
                                    shots=int(total_shots[p]))
                job_list.append(job)
                result_list.append(job.result())
                exp_values[p] = result_list[p].values

        self.qc.clear()

        reward_table = np.mean(reward_factor[:, np.newaxis] * exp_values, axis=0)
        assert len(reward_table) == batch_size
        return reward_table, DensityMatrix(self.density_matrix)  # Shape [batchsize]


class Agent:
    def __init__(self):
        pass


# Variables to define environment
# If you need to overwrite the account info, please add `overwrite=True`

service = QiskitRuntimeService(channel='ibm_quantum')
seed = 3590  # Seed for action sampling
backend = service.backends(simulator=True)[0]  # Simulation backend (mock quantum computer)
print(backend)
options = {"seed_simulator": 42,
           'resilience_level': 0}
n_qubits = 2

ket0 = np.array([[1.], [0]])
ket1 = np.array([[0.], [1.]])
ket00 = np.kron(ket0, ket0)
ket11 = np.kron(ket1, ket1)
bell_state = (ket00 + ket11) / np.sqrt(2)
bell_dm = bell_state @ bell_state.conj().T
bell_tgt = {"dm": DensityMatrix(bell_dm)}
target_state = bell_tgt

q_env = QuantumEnvironment(2, bell_tgt, "circuit", backend, sampling_Pauli_space=100)
# q_env.perform_action(np.array([[0.25], [0.25]]))

# Hyperparameters for the agent
n_epochs = 50  # Number of epochs
batchsize = 50  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
eta = 0.1  # Learning rate for policy update step
eta_2 = 0.1  # Learning rate for critic (value function) update step

use_PPO = True
epsilon = 0.2  # Parameter for clipping value (PPO)
grad_clip = 0.3

critic_loss_coeff = 0.5


def select_optimizer(optimizer: str = "Adam", concurrent_optimization: bool = True, grad_clip: float = 0.3):
    if concurrent_optimization:
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=eta, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return tf.optimizers.SGD(learning_rate=eta, clipvalue=grad_clip)
    else:
        if optimizer == 'Adam':
            return Adam(learning_rate=eta), Adam(learning_rate=eta_2, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return SGD(learning_rate=eta), SGD(learning_rate=eta_2, clipvalue=grad_clip)


optimizer = select_optimizer()


def constrain_mean_value(mu_var):
    return tf.clip_by_value(mu_var, -1., 1.)


def constrain_std_value(std_var):
    return tf.clip_by_value(std_var, 1e-3, 3)


# Policy parameters
N_in = n_qubits + 1  # One input for each measured qubit state (0 or 1 input for each neuron)
n_actions = 1  # Choose how many control parameters in pulse/circuit parametrization
N_out = 2 * n_actions  # One mean/variance for each action
layers = [3]  # List containing the number of neurons in each hidden layer

input_layer = Input(shape=N_in)
hidden = Sequential([Dense(layer, activation='relu', kernel_initializer=tf.initializers.RandomNormal(stddev=0.01),
                           bias_initializer=tf.initializers.RandomNormal(stddev=0.01))
                     for layer in layers])(input_layer)
actor_output = Dense(N_out, activation=None)(hidden)
critic_output = Dense(1, activation=None)(hidden)
network = Model(inputs=input_layer, outputs=[actor_output, critic_output])
init_msmt = np.zeros([1, N_in])

sigma_eps = 1e-6  # for numerical stability

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
policy_params_str = 'Policy params:'
print('Neural net output', network(init_msmt))
mu_old = tf.Variable(initial_value=network(init_msmt)[0][0][:N_out // 2], trainable=False)
sigma_old = tf.Variable(initial_value=network(init_msmt)[0][0][N_out // 2:], trainable=False)
print("mu_old", mu_old)
print("sigma_old", sigma_old)
for i in tqdm(range(n_epochs)):
    print('Epoch', i)
    print(f"{policy_params_str:#<100}")
    policy_params, b = network(init_msmt)
    print('mu_vec', policy_params[0][:N_out//2])
    print('sigma_vec', policy_params[0][N_out // 2:])
    print('baseline', b)

    Policy_distrib = MultivariateNormalDiag(loc=network(init_msmt)[0][0][:N_out // 2],
                                            scale_diag=network(init_msmt)[0][0][N_out // 2:], validate_args=True,
                                            allow_nan_stats=False)
    Old_distrib = MultivariateNormalDiag(loc=mu_old, scale_diag=sigma_old, validate_args=True,
                                         allow_nan_stats=False)
    e = Policy_distrib.trainable_variables
    print(e)
    action_vector = Policy_distrib.sample(batchsize, seed=seed)
    print('action_vec', action_vector)
    # Run quantum circuit to retrieve rewards (in this example, only one time step)
    reward, dm_observed = q_env.perform_action(action_vector)

    print("reward", reward)
    with tf.GradientTape(persistent=True) as tape:

        """
        Calculate return (to be maximized, therefore the minus sign placed in front of the loss
        since applying gradients minimize the loss), E[R*log(proba(amp)] where proba is the gaussian
        probability density (cf paper of reference, educational example).
        In case of the PPO, loss function is slightly changed.
        """

        advantage = reward - network(init_msmt)[1]
        print("advantage", advantage)
        if use_PPO:
            ratio = Policy_distrib.prob(action_vector) / (Old_distrib.prob(action_vector) + sigma_eps)
            actor_loss = - tf.reduce_mean(tf.minimum(advantage * ratio,
                                                     advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
        else:  # REINFORCE algorithm
            actor_loss = - tf.reduce_mean(advantage * Policy_distrib.log_prob(action_vector))

        critic_loss = tf.reduce_mean(advantage ** 2)
        combined_loss = actor_loss + 0.5 * critic_loss

    # Compute gradients

    grads = tape.gradient(combined_loss, network.trainable_variables)
    # print('grads', combined_grads)
    # grads = tf.clip_by_value(combined_grads, -grad_clip, grad_clip)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        mu_old.assign(Policy_distrib.loc)
        sigma_old.assign(network(init_msmt)[0][0][N_out // 2:])

    data["amps"][i] = np.array(action_vector)
    data["rewards"][i] = reward
    data["means"][i] = np.array(Policy_distrib.loc)
    # data["stds"][i] = np.array(Policy_distrib.scale)
    data["baselines"][i] = np.array(b)
    print('dm', q_env.density_matrix)
    data["fidelity"][i] = state_fidelity(target_state["dm"], dm_observed)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

print(data)


# open file for writing, "w" is writing
# w = csv.writer(open("output.csv", "w"))
#
# # loop over dictionary keys and values
# for key, val in data.items():
#     # write every key and value to file
#     w.writerow([key, val])


#
# """
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# Plotting tools
# """
#
#
# #  Plotting results
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
