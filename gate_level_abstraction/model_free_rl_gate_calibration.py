"""
Code for arbitrary gate calibration using model-free reinforcement learning

 Author: Arthur Strauss
 Created on 11/11/2022
"""

import numpy as np
from quantumenvironment import QuantumEnvironment
from helper_functions import select_optimizer, generate_model

# Qiskit imports for building RL environment (circuit level)
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.extensions import CXGate, XGate
from qiskit.opflow import H, I, X, S

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag

# Additional imports
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

""" 
-----------------------------------------------------------------------------------------------------
We set a RL agent based on PPO and an actor critic network to perform multi-qubit gate calibration based on
scheme proposed in the appendix of Vladimir Sivak paper (https://doi.org/10.1103/PhysRevX.12.011059) 
Here, we use a class object for the QuantumEnvironment (Quantum Circuit simulated on IBM Q simulator) 
-----------------------------------------------------------------------------------------------------
"""


def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    :param qc: Quantum Circuit instance to add the gates on
    :return:
    """
    # qc.num_qubits
    global n_actions
    params = ParameterVector('theta', n_actions)
    qc.u(2 * np.pi * params[0], 2 * np.pi * params[1], 2 * np.pi * params[2], 0)
    qc.u(2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], 1)
    qc.rzx(2 * np.pi * params[6], 0, 1)


"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""

# service = QiskitRuntimeService(channel='ibm_quantum')
# backend = service.backends(simulator=True)[0]  # Simulation backend (mock quantum computer)

backend = None
seed = 3590  # Seed for action sampling

estimator_options = {"seed_simulator": None, 'resilience_level': 0}
n_qubits = 2
sampling_Paulis = 10
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector

# Target gate: CNOT
circuit_Plus_i = S @ H
circuit_Minus_i = S @ H @ X
cnot_target = {
    "target_type": "gate",
    "gate": CXGate("CNOT"),
    "register": [0, 1],
    "input_states": [{"name": "|00>",  # Drawn from Ref [21] of PhysRevLett.93.080502
                      "circuit": I ^ 2,
                      },
                     {"name": "|01>",
                      "circuit": X ^ I,
                      },
                     {"name": "|10>",
                      "circuit": I ^ X,
                      },
                     {"name": "|11>",
                      "circuit": X ^ X,
                      },
                     {"name": "|+_1>",
                      "circuit": X ^ H,
                      },
                     {"name": "|0_->",
                      "circuit": (H @ X) ^ I,
                      },
                     {"name": "|+_->",
                      "circuit": (H @ X) ^ H,
                      },
                     {"name": "|1_->",
                      "circuit": (H @ X) ^ X,
                      },
                     {"name": "|+_0>",
                      "circuit": I ^ H,
                      },
                     {"name": "|0_->",
                      "circuit": (H @ X) ^ I,
                      },
                     {"name": "|i_0>",
                      "circuit": I ^ circuit_Plus_i,
                      },
                     {"name": "|i_1>",
                      "circuit": X ^ circuit_Plus_i,
                      },
                     {"name": "|0_i>",
                      "circuit": circuit_Plus_i ^ I,
                      },
                     {"name": "|i_i>",
                      "circuit": circuit_Plus_i ^ circuit_Plus_i,
                      },
                     {"name": "|i_->",
                      "circuit": (H @ X) ^ circuit_Plus_i,
                      },
                     {"name": "|+_i->",
                      "circuit": circuit_Minus_i ^ H,
                      },
                     ]

}

Qiskit_setup = {
    "backend": backend,
    "parametrized_circuit": apply_parametrized_circuit,
    "estimator_options": estimator_options
}
target = cnot_target
n_actions = 7  # Choose how many control parameters in pulse/circuit parametrization

q_env = QuantumEnvironment(n_qubits=n_qubits, target=target, abstraction_level="circuit",
                           Qiskit_config=Qiskit_setup,
                           sampling_Pauli_space=sampling_Paulis, n_shots=N_shots, c_factor=0.25)

"""
-----------------------------------------------------------------------------------------------------
Hyperparameters for RL agent
-----------------------------------------------------------------------------------------------------
"""
# Hyperparameters for the agent
n_epochs = 600  # Number of epochs
batchsize = 50  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
opti = "Adam"
eta = 0.0015  # Learning rate for policy update step
eta_2 = None  # Learning rate for critic (value function) update step

use_PPO = True
epsilon = 0.1  # Parameter for clipping value (PPO)
grad_clip = 0.01
critic_loss_coeff = 0.5
optimizer = select_optimizer(lr=eta, optimizer=opti, grad_clip=grad_clip, concurrent_optimization=True, lr2=eta_2)
sigma_eps = 1e-3  # for numerical stability

"""
-----------------------------------------------------------------------------------------------------
Policy parameters
-----------------------------------------------------------------------------------------------------
"""
# Policy parameters
N_in = n_qubits + 1  # One input for each measured qubit state (0 or 1 input for each neuron)
hidden_units = [80]  # List containing number of units in each hidden layer

network = generate_model((N_in,), hidden_units, n_actions, actor_critic_together=True)
network.summary()
init_msmt = np.zeros((1, N_in))  # Here no feedback involved, so measurement sequence is always the same

"""
-----------------------------------------------------------------------------------------------------
Training loop
-----------------------------------------------------------------------------------------------------
"""
mu_old = tf.Variable(initial_value=network(init_msmt)[0][0], trainable=False)
sigma_old = tf.Variable(initial_value=network(init_msmt)[1][0], trainable=False)

policy_params_str = 'Policy params:'
print('Neural net output', network(init_msmt), type(network(init_msmt)))
print("mu_old", mu_old)
print("sigma_old", sigma_old)

for i in tqdm(range(n_epochs)):

    Old_distrib = MultivariateNormalDiag(loc=mu_old, scale_diag=sigma_old,
                                         validate_args=True, allow_nan_stats=False)

    with tf.GradientTape(persistent=True) as tape:

        mu, sigma, b = network(init_msmt, training=True)
        mu = tf.squeeze(mu, axis=0)
        sigma = tf.squeeze(sigma, axis=0)
        b = tf.squeeze(b, axis=0)

        Policy_distrib = MultivariateNormalDiag(loc=mu, scale_diag=sigma,
                                                validate_args=True, allow_nan_stats=False)

        action_vector = tf.stop_gradient(tf.clip_by_value(Policy_distrib.sample(batchsize, seed=seed), -1., 1.))

        reward = q_env.perform_action(action_vector)
        advantage = reward - b

        if use_PPO:
            ratio = Policy_distrib.prob(action_vector) / (tf.stop_gradient(Old_distrib.prob(action_vector)) + 1e-6)
            actor_loss = - tf.reduce_mean(tf.minimum(advantage * ratio,
                                                     advantage * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)))
        else:  # REINFORCE algorithm
            actor_loss = - tf.reduce_mean(advantage * Policy_distrib.log_prob(action_vector))

        critic_loss = tf.reduce_mean(advantage ** 2)
        combined_loss = actor_loss + critic_loss_coeff * critic_loss

    grads = tape.gradient(combined_loss, network.trainable_variables)

    # For PPO, update old parameters to have access to "old" policy
    if use_PPO:
        mu_old.assign(mu)
        sigma_old.assign(sigma)

    print('\n Epoch', i)
    print("Average reward", np.mean(q_env.reward_history[i]))
    print("Average Gate Fidelity:", q_env.avg_fidelity_history[i])

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


figure, (ax1, ax2) = plt.subplots(1, 2)
#  Plot return as a function of epochs
ax1.plot(np.mean(q_env.reward_history, axis=1), '-.', label='Reward')
# ax1.plot(data["baselines"], '-.', label='baseline')
# ax1.plot(q_env.process_fidelity_history, '-o', label='Process Fidelity')
ax1.plot(q_env.avg_fidelity_history, '-.', label='Average Gate Fidelity')
for i in range(len(target["input_states"])):
    ax1.plot(q_env.input_output_state_fidelity_history[i], '.-',
             label=f"State fidelity on {target['gate'].name + target['input_states'][i]['name']}")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Expected reward")
ax1.legend()
# plot_examples(figure, ax2, q_env.reward_history)
# plt.show()
