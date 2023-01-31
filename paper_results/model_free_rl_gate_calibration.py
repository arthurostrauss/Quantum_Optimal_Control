"""
Code for arbitrary state preparation based on scheme described in Appendix D.2b  of paper PhysRevX.12.011059
 (https://doi.org/10.1103/PhysRevX.12.011059) using Qiskit modules

 Author: Arthur Strauss
 Created on 11/11/2022
"""

import numpy as np
from Quantum_Optimal_Control.quantumenvironment import QuantumEnvironment
from Quantum_Optimal_Control.helper_functions import select_optimizer, generate_model

# Qiskit imports for building RL environment (circuit level)
from qiskit import IBMQ
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.extensions import CXGate, XGate
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit.opflow import Zero, One, Plus, Minus, H, I, X, CX, S
from qiskit_ibm_runtime import QiskitRuntimeService

# Tensorflow imports for building RL agent and framework
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag

from tf_agents.specs import array_spec, tensor_spec

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

# IBMQ.save_account(TOKEN)
IBMQ.load_account()  # Load account from disk
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')


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
    qc.rzx(2 * np.pi * params[6], 1, 0)


"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""

service = QiskitRuntimeService(channel='ibm_quantum')
seed = 3590  # Seed for action sampling
backend = service.backends(simulator=True)[0]  # Simulation backend (mock quantum computer)
options = {"seed_simulator": None, 'resilience_level': 0}
n_qubits = 2
sampling_Paulis = 100
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector

# Target gate: CNOT
Plus_i = S @ Plus
Minus_i = S @ Minus
circuit_Plus_i = S @ H
circuit_Minus_i = S @ H @ X
cnot_target = {
    "target_type": "gate",
    "gate": CXGate("CNOT"),
    "input_states": [{"name": "|00>",  # Drawn from Ref [21] of PhysRevLett.93.080502
                      "dm": DensityMatrix(Zero ^ 2),
                      "state_fn": Zero ^ 2,
                      "circuit": I ^ 2,
                      "register": [0, 1]
                      },
                     {"name": "|01>",
                      "dm": DensityMatrix(Zero ^ One),
                      "state_fn": Zero ^ One,
                      "circuit": X ^ I,
                      "register": [0, 1]
                      },
                     {"name": "|10>",
                      "dm": DensityMatrix(One ^ Zero),
                      "state_fn": One ^ Zero,
                      "circuit": I ^ X,
                      "register": [0, 1]
                      },
                     {"name": "|11>",
                      "dm": DensityMatrix(One ^ 2),
                      "state_fn": One ^ 2,
                      "circuit": X ^ X,
                      "register": [0, 1]
                      },
                     {"name": "|+_1>",
                      "dm": DensityMatrix(Plus ^ One),
                      "state_fn": Plus ^ One,
                      "circuit": X ^ H,
                      "register": [0, 1]
                      },
                     {"name": "|0_->",
                      "dm": DensityMatrix(Zero ^ Minus),
                      "state_fn": Zero ^ Minus,
                      "circuit": (H @ X) ^ I,
                      "register": [0, 1]
                      },
                     {"name": "|+_->",
                      "dm": DensityMatrix(Plus ^ Minus),
                      "state_fn": Plus ^ Minus,
                      "circuit": (H @ X) ^ H,
                      "register": [0, 1]
                      },
                     {"name": "|1_->",
                      "dm": DensityMatrix(One ^ Minus),
                      "state_fn": Zero ^ Minus,
                      "circuit": (H @ X) ^ X,
                      "register": [0, 1]
                      },
                     {"name": "|+_0>",
                      "dm": DensityMatrix(Plus ^ Zero),
                      "state_fn": Plus ^ Zero,
                      "circuit": I ^ H,
                      "register": [0, 1]
                      },
                     {"name": "|0_->",
                      "dm": DensityMatrix(Zero ^ Minus),
                      "state_fn": Zero ^ Minus,
                      "circuit": (H @ X) ^ I,
                      "register": [0, 1]
                      },
                     {"name": "|i_0>",
                      "dm": DensityMatrix(Plus_i ^ Zero),
                      "state_fn": Plus_i ^ Zero,
                      "circuit": I ^ circuit_Plus_i,
                      "register": [0, 1]
                      },
                     {"name": "|i_1>",
                      "dm": DensityMatrix(Plus_i ^ One),
                      "state_fn": Plus_i ^ One,
                      "circuit": X ^ circuit_Plus_i,
                      "register": [0, 1]
                      },
                     {"name": "|0_i>",
                      "dm": DensityMatrix(Zero ^ Plus_i),
                      "state_fn": Zero ^ Plus_i,
                      "circuit": circuit_Plus_i ^ I,
                      "register": [0, 1]
                      },
                     {"name": "|i_i>",
                      "dm": DensityMatrix(Plus_i ^ Plus_i),
                      "state_fn": Plus_i ^ Plus_i,
                      "circuit": circuit_Plus_i ^ circuit_Plus_i,
                      "register": [0, 1]
                      },
                     {"name": "|i_->",
                      "dm": DensityMatrix(Plus_i ^ Minus),
                      "state_fn": Plus_i ^ Minus,
                      "circuit": (H @ X) ^ circuit_Plus_i,
                      "register": [0, 1]
                      },
                     {"name": "|+_i->",
                      "dm": DensityMatrix(Plus ^ Minus_i),
                      "state_fn": Plus ^ Minus_i,
                      "circuit": circuit_Minus_i ^ H,
                      "register": [0, 1]
                      },

                     ]
}

# n_qubits = 1
single_qubit_tgt = {
    "target_type": 'gate',
    "gate": XGate("X"),
    "input_states": [
        {"name": '|0>',
         "dm": DensityMatrix(Zero),
         "state_fn": Zero,
         "circuit": I.to_instruction(),
         "register": [0]},

        {"name": '|1>',
         "dm": DensityMatrix(One),
         "state_fn": One,
         "circuit": X.to_instruction(),
         "register": [0]},

        {"name": '|+>',
         "dm": DensityMatrix(Plus),
         "state_fn": Plus,
         "circuit": H.to_instruction(),
         "register": [0]},
        {"name": '|->',
         "dm": DensityMatrix(Minus),
         "state_fn": Minus,
         "circuit": (H @ X).to_instruction(),
         "register": [0]},
    ]

}
Qiskit_setup = {
    "backend": backend,
    "service": service,
    "parametrized_circuit": apply_parametrized_circuit,
    "options": options
}

n_actions = 7  # Choose how many control parameters in pulse/circuit parametrization
time_steps = 1  # Number of time steps within an episode (1 means you do one readout and assign right away the reward)
action_spec = tensor_spec.BoundedTensorSpec(shape=(n_actions,), dtype=tf.float32, minimum=-1., maximum=1.)
observation_spec = array_spec.ArraySpec(shape=(time_steps,), dtype=np.int32)

q_env = QuantumEnvironment(n_qubits=n_qubits, target=cnot_target, abstraction_level="circuit",
                           action_spec=action_spec, observation_spec=observation_spec,
                           Qiskit_config=Qiskit_setup,
                           sampling_Pauli_space=sampling_Paulis, n_shots=N_shots, c_factor=1.)

"""
-----------------------------------------------------------------------------------------------------
Hyperparameters for RL agent
-----------------------------------------------------------------------------------------------------
"""
# Hyperparameters for the agent
n_epochs = 10000  # Number of epochs
batchsize = 200  # Batch size (iterate over a bunch of actions per policy to estimate expected return)
opti = "Adam"
eta = 0.001  # Learning rate for policy update step
eta_2 = None  # Learning rate for critic (value function) update step

use_PPO = True
epsilon = 0.1  # Parameter for clipping value (PPO)
grad_clip = 0.01
critic_loss_coeff = 0.5
optimizer = select_optimizer(lr=eta, optimizer=opti, grad_clip=grad_clip, concurrent_optimization=True, lr2=eta_2)
sigma_eps = 1e-3  # for numerical stability
grad_update_number = 20
# class Agent:
#     def __init__(self, epochs:int, batchsize:int, optimizer, lr: float, lr2: Optional[float], grad_clip:):
#         pass


"""
-----------------------------------------------------------------------------------------------------
Policy parameters
-----------------------------------------------------------------------------------------------------
"""
# Policy parameters
N_in = n_qubits + 1  # One input for each measured qubit state (0 or 1 input for each neuron)
hidden_units = [82, 82]  # List containing number of units in each hidden layer

network = generate_model((N_in,), hidden_units, n_actions, actor_critic_together=True)
network.summary()
init_msmt = np.zeros((1, N_in))  # Here no feedback involved, so measurement sequence is always the same

"""
-----------------------------------------------------------------------------------------------------
Training loop
-----------------------------------------------------------------------------------------------------
"""
# TODO: Use TF-Agents PPO Agent
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

        action_vector = tf.stop_gradient(tf.clip_by_value(Policy_distrib.sample(batchsize), -1., 1.))

        reward = q_env.perform_action_gate_cal(action_vector)
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
    print(f"{policy_params_str:#<100}")
    print('mu_vec:', np.array(mu))
    print('sigma_vec:', np.array(sigma))
    print('baseline:', np.array(b))
    print("Process Fidelity:", q_env.process_fidelity_history[i])
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
    plt.show()


figure, (ax1, ax2) = plt.subplots(1, 2)
#  Plot return as a function of epochs
ax1.plot(np.mean(q_env.reward_history, axis=1), '-.', label='Reward')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Expected reward")
# ax1.plot(data["baselines"], '-.', label='baseline')
ax1.plot(q_env.process_fidelity_history, '-o', label='Process Fidelity')
ax1.plot(q_env.avg_fidelity_history, '-.', label='Average Gate Fidelity')
ax1.legend()
plot_examples(figure, ax2, q_env.reward_history)
