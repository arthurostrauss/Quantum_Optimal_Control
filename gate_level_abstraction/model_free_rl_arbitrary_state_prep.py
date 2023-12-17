"""
Code for arbitrary state preparation based on scheme described in Appendix D.2b  of paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using qiskit modules

 Author: Arthur Strauss
 Created on 11/11/2022
"""

import numpy as np
from quantumenvironment import QuantumEnvironment
from helper_functions import select_optimizer, generate_model
from qconfig import QiskitConfig, QEnvConfig
from typing import Optional

# qiskit imports for building RL environment (circuit level)
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Options, RuntimeOptions
from qiskit_ibm_runtime.options import (
    SimulatorOptions,
    TranspilationOptions,
    EnvironmentOptions,
)

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from gymnasium.spaces import Box

# Additional imports
from tqdm import tqdm
import matplotlib.pyplot as plt

""" 
-----------------------------------------------------------------------------------------------------
We set a RL agent based on PPO and an actor critic network to perform arbitrary state preparation based on
scheme proposed in the appendix of Vladimir Sivak paper. 
In there, we design classes for the environment (Quantum Circuit simulated on IBM Q simulator) and the agent 
-----------------------------------------------------------------------------------------------------
"""


def apply_parametrized_circuit(
    qc: QuantumCircuit,
    params: Optional[ParameterVector],
    qr: Optional[QuantumRegister],
    n_actions,
):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit ParameterVector
    :param qc: Quantum Circuit instance to add the gates on
    :param params: Parameters of the custom Gate
    :param qr: Quantum Register formed of target qubits
    :param n_actions: Number of parameters in the parametrized circuit
    :return:
    """
    if params is None:
        params = ParameterVector("theta", n_actions)
    if qr is None:
        qr = qc.qregs[0]
    param_circ = QuantumCircuit(qr)
    param_circ.u(
        2 * np.pi * params[0], 2 * np.pi * params[1], 2 * np.pi * params[2], qr[0]
    )
    param_circ.u(
        2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], qr[1]
    )
    param_circ.rzx(2 * np.pi * params[6], 0, qr[1])
    qc.append(param_circ.to_instruction(), qr)


"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""

n_qubits = 3
sampling_Paulis = 100
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
n_epochs = 800  # Number of epochs : default 1500
batchsize = 200  # Batch size (iterate over a bunch of actions per policy to estimate expected return) default 100
n_actions = 7  # Choose how many control parameters in pulse/circuit parametrization


# qiskit Runtime setup, if selected backend is IBMBackend or IBM cloud simulator
# service = QiskitRuntimeService(channel='ibm_quantum')
seed = 3590  # Seed for action sampling
# print(service.backends())
# backend = service.backends(simulator=True)[0]  # Simulation backend (could be any other backend from the service)

sim_options = SimulatorOptions(
    seed_simulator=seed, noise_model=None, coupling_map=None, basis_gates=None
)
env_options = EnvironmentOptions(
    job_tags=[f"RL_state_prep_epoch{i}" for i in range(n_epochs)]
)
estimator_options = Options(
    simulator=sim_options, environment=env_options, resilience_level=0
)

# Alternatively, simulation can be run locally by using built-in primitives (qiskit_aer, statevector(backend = None), or
# FakeBackend)
backend = None  # Local CPU State vector simulation

# Target state: Bell state
qr = QuantumRegister(n_qubits)
bell_circuit = QuantumCircuit(
    qr
)  # Specify quantum circuit required to prepare the ideal desired quantum state
bell_circuit.h(qr[0])
bell_circuit.cx(qr[0], qr[1])
bell_tgt = {"circuit": bell_circuit, "register": qr}

# In case you want to prepare a state for a specific qubit set in your real backend, you can set the transpiler initial
# layout
transpilation_options = TranspilationOptions(
    skip_transpilation=False, initial_layout={qr[0]: 0, qr[1]: 1}
)
estimator_options.transpilation = transpilation_options

# Alternatively, provide argument density matrix 'dm':
ket0, ket1 = np.array([[1.0], [0]]), np.array([[0.0], [1.0]])
ket00, ket11 = np.kron(ket0, ket0), np.kron(ket1, ket1)
bell_state = Statevector((ket00 + ket11) / np.sqrt(2))
bell_tgt["dm"] = DensityMatrix(bell_state)

Qiskit_setup = QiskitConfig(
    parametrized_circuit=apply_parametrized_circuit,
    backend=backend,
    parametrized_circuit_kwargs={"n_actions": n_actions},
    estimator_options=estimator_options,
)
action_space = Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)
obs_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

training_config = QEnvConfig(
    target=bell_tgt,
    backend_config=Qiskit_setup,
    action_space=action_space,
    observation_space=obs_space,
    batch_size=batchsize,
    sampling_Paulis=sampling_Paulis,
    n_shots=N_shots,
    c_factor=0.125,
    benchmark_cycle=5,
    seed=seed,
    device=None,
)

q_env = QuantumEnvironment(training_config=training_config)


"""
-----------------------------------------------------------------------------------------------------
Hyperparameters for RL agent
-----------------------------------------------------------------------------------------------------
"""
# Hyperparameters for the agent

opti = "Adam"
eta = 0.003  # Learning rate for policy update step
eta_2 = None  # Learning rate for critic (value function) update step

use_PPO = True
epsilon = 0.1  # Parameter for clipping value (PPO)
grad_clip = 0.01
critic_loss_coeff = 0.5
optimizer = select_optimizer(
    lr=eta, optimizer=opti, grad_clip=grad_clip, concurrent_optimization=True, lr2=eta_2
)
sigma_eps = 1e-3  # for numerical stability

"""
-----------------------------------------------------------------------------------------------------
Policy parameters
-----------------------------------------------------------------------------------------------------
"""
# Policy parameters
N_in = obs_space.shape[
    -1
]  # One input for each measured qubit state (0 or 1 input for each neuron)
hidden_units = [82, 82]  # List containing number of units in each hidden layer

network = generate_model((N_in,), hidden_units, n_actions, actor_critic_together=True)
network.summary()
init_msmt = np.zeros(
    (1, N_in)
)  # Here no feedback involved, so measurement sequence is always the same

"""
-----------------------------------------------------------------------------------------------------
Training loop
-----------------------------------------------------------------------------------------------------
"""
mu_old = tf.Variable(initial_value=network(init_msmt)[0][0], trainable=False)
sigma_old = tf.Variable(initial_value=network(init_msmt)[1][0], trainable=False)

policy_params_str = "Policy params:"
print("Neural net output", network(init_msmt), type(network(init_msmt)))
print("mu_old", mu_old)
print("sigma_old", sigma_old)

for i in tqdm(range(n_epochs)):
    Old_distrib = MultivariateNormalDiag(
        loc=mu_old, scale_diag=sigma_old, validate_args=True, allow_nan_stats=False
    )

    with tf.GradientTape(persistent=True) as tape:
        mu, sigma, b = network(init_msmt, training=True)
        mu = tf.squeeze(mu, axis=0)
        sigma = tf.squeeze(sigma, axis=0)
        b = tf.squeeze(b, axis=0)

        Policy_distrib = MultivariateNormalDiag(
            loc=mu, scale_diag=sigma, validate_args=True, allow_nan_stats=False
        )

        action_vector = tf.stop_gradient(
            tf.clip_by_value(Policy_distrib.sample(batchsize, seed=seed), -1.0, 1.0)
        )

        reward = q_env.perform_action(action_vector)
        advantage = reward - b

        if use_PPO:
            ratio = Policy_distrib.prob(action_vector) / (
                tf.stop_gradient(Old_distrib.prob(action_vector)) + 1e-6
            )
            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    advantage * ratio,
                    advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon),
                )
            )
        else:  # REINFORCE algorithm
            actor_loss = -tf.reduce_mean(
                advantage * Policy_distrib.log_prob(action_vector)
            )

        critic_loss = tf.reduce_mean(advantage**2)
        combined_loss = actor_loss + critic_loss_coeff * critic_loss

    grads = tape.gradient(combined_loss, network.trainable_variables)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        mu_old.assign(mu)
        sigma_old.assign(sigma)

    print("\n Epoch", i)
    print(f"{policy_params_str:#<100}")
    print("mu_vec:", np.array(mu))
    print("sigma_vec:", np.array(sigma))
    print("baseline:", np.array(b))
    print("Fidelity:", q_env.state_fidelity_history[i])

    # Apply gradients
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

if isinstance(q_env.estimator, Estimator):
    q_env.estimator.session.close()
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
    ax.set_ylabel("Episode")
    ax.set_xlabel("Epoch")
    fig.colorbar(im, ax=ax, label="Reward")
    plt.show()


figure, (ax1, ax2) = plt.subplots(1, 2)
#  Plot return as a function of epochs
ax1.plot(np.mean(q_env.reward_history, axis=1), "-.", label="Reward")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Expected reward")
# ax1.plot(data["baselines"], '-.', label='baseline')
ax1.plot(q_env.state_fidelity_history, "-o", label="Fidelity")
ax1.legend()
plot_examples(figure, ax2, q_env.reward_history)
