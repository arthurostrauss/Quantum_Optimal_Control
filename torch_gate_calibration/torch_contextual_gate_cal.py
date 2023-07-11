import os

from qiskit.providers.models import QasmBackendConfiguration

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Qiskit imports
from qiskit import pulse, transpile
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister, Gate, \
    ParameterExpression, CircuitInstruction
from qiskit.circuit.random import random_circuit
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.fake_provider import FakeJakartaV2
from qiskit_aer import AerSimulator
from qiskit.circuit.library import XGate, CXGate
from qiskit_experiments.framework import ExperimentData

from qconfig import QiskitConfig
from quantumenvironment import QuantumEnvironment
from helper_functions import select_optimizer, generate_model
from torch_quantum_environment import TorchQuantumEnvironment
from gymnasium.spaces import Box, MultiDiscrete, Space
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal, Normal

import numpy as np
import tqdm
import time
from typing import Union, Optional, List, Sequence


# Save your credentials on disk.
# IBMProvider.save_account(token='<IBM Quantum API key>')


def custom_pulse_schedule(backend: Union[BackendV1, BackendV2], qubit_tgt_register: Union[List[int], QuantumRegister],
                          params: Union[Sequence[ParameterExpression], ParameterVector],
                          default_schedule: Optional[Union[pulse.ScheduleBlock, pulse.Schedule]] = None):
    """
    Define parametrization of the pulse schedule characterizing the target gate
        :param backend: IBM Backend on which schedule shall be added
        :param qubit_tgt_register: Qubit register on which
        :param params: Parameters of the Schedule
        :param default_schedule:  baseline from which one can customize the pulse parameters

        :return: Parametrized Schedule
    """

    if default_schedule is None:  # No baseline pulse, full waveform builder
        pass
    else:

        # Look here for the pulse features to specifically optimize upon, for the x gate here, simply retrieve relevant
        # parameters for the Drag pulse
        pulse_ref = default_schedule.instructions[0][1].pulse

        with pulse.build(backend=backend, name='param_schedule') as parametrized_schedule:

            pulse.play(pulse.Drag(duration=pulse_ref.duration, amp=params[0], sigma=pulse_ref.sigma,
                                  beta=pulse_ref.beta, angle=pulse_ref.angle),
                       channel=pulse.DriveChannel(qubit_tgt_register[0]))

        return parametrized_schedule
        pass


def param_circuit(qc: QuantumCircuit,
                  params: Optional[ParameterVector], q_reg: Optional[QuantumRegister] = None):
    # To build a unique gate identifier, choose the name based on the input circuit name (by default and if used
    # with TFQuantumEnvironment class, this circuit name is of the form c_circ_trunc_{i})
    # custom_gate_name = f"{target['gate'].name}_{qc.name[-1]}"
    # custom_gate = Gate(custom_gate_name, len(target["register"]), params=params.params)
    # custom_sched = custom_pulse_schedule(backend=backend, qubit_tgt_register=physical_qubits, params=params,
    #                                      default_schedule=backend.target.get_calibration(target["gate"].name,
    #                                                                                      tuple(physical_qubits)))
    # qc.add_calibration(custom_gate, physical_qubits, custom_sched)
    # qc.append(custom_gate, physical_qubits)
    # qc.u(np.pi * params[0], np.pi * params[1], np.pi * params[2], 0)
    # qc.u(np.pi * params[3], np.pi * params[4], np.pi * params[5], 1)
    # qc.rzx(np.pi * params[6], 0, 1)
    qc.rx(2*np.pi*params[0], physical_qubits)


"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
sampling_Paulis = 100
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
fake_backend = FakeJakartaV2()

aer_backend = AerSimulator.from_backend(fake_backend, noise_model=None)
backend = aer_backend
target_gate = XGate()
physical_qubits = [0]
n_qubits = len(physical_qubits)
target = {"gate": target_gate, 'register': physical_qubits}
config = QiskitConfig(parametrized_circuit=param_circuit, backend=backend)

q_env = QuantumEnvironment(target, "circuit", config)

# Circuit context
seed = 10
training_steps_per_gate = 1500
# target_circuit = transpile(random_circuit(4, depth=2, max_operands=2, seed=seed), backend)
target_circuit = QuantumCircuit(1)
target_circuit.x(0)
tgt_qubits = [target_circuit.qubits[i] for i in physical_qubits]

tgt_instruction_counts = target_circuit.data.count(CircuitInstruction(target_gate, tgt_qubits))

batchsize = 400  # Batch size (iterate over a bunch of actions per policy to estimate expected return) default 100
n_actions = 1  # Choose how many control parameters in pulse/circuit parametrization
min_bound_actions = 0.
max_bound_actions = 1.
observation_space = Box(low=np.array([0, 0]), high=np.array([4 ** n_qubits, tgt_instruction_counts]), shape=(2,),
                        seed=seed)
action_space = Box(low=min_bound_actions, high=max_bound_actions, shape=(n_actions,), seed=seed)

torch_env = TorchQuantumEnvironment(q_env, target_circuit, action_space, observation_space, batch_size=batchsize,
                                    training_steps_per_gate=training_steps_per_gate, intermediate_rewards=False,
                                    seed=None)


class ActorNetwork(nn.Module):
    def __init__(self, observation_space: Space, hidden_layers: Sequence[int],
                 n_actions: int,
                 hidden_activation_functions: Optional[Sequence[nn.Module]] = None,
                 include_critic=True,
                 chkpt_dir: str = 'tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        # Define a list to hold the layer sizes including input and output sizes
        layer_sizes = [observation_space.shape[0]] + list(hidden_layers)
        if hidden_activation_functions is None:
            hidden_activation_functions = [nn.ReLU() for _ in range(len(layer_sizes))]

        assert len(hidden_activation_functions) == len(layer_sizes)
        # Define a list to hold the layers of the network
        layers = []

        # Iterate over the layer sizes to create the network layers
        for i in range(len(layer_sizes) - 1):
            # Add a linear layer with the current and next layer sizes
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            layers.append(hidden_activation_functions[i])

        # Create the actor network using Sequential container
        self.layers = layers
        self.mean_action = nn.Linear(hidden_layers[-1], n_actions)
        self.std_action = nn.Linear(hidden_layers[-1], n_actions)
        self.std_activation = nn.Sigmoid()

        self.include_critic = include_critic
        self.critic_output = nn.Linear(hidden_layers[-1], 1)

        self.base_network = nn.Sequential(*layers)

        # Initialize the weights of the network
        for layer in self.base_network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.base_network(x)
        mean_action = self.mean_action(x)
        std_action = self.std_action(x)
        std_action = self.std_activation(std_action)
        critic_output = self.critic_output(x)

        if self.critic_output:
            return mean_action, std_action, critic_output
        else:
            return mean_action, std_action

    def get_value(self, x):
        x = self.base_network(x)
        x = self.critic_output(x)
        return x

    def save_checkpoint(self):
        torch.save(self, self.checkpoint_dir)

    def load_checkpoint(self):
        torch.load(self.checkpoint_dir)


class CriticNetwork(nn.Module):
    def __init__(self, observation_space: Space, hidden_layers: Sequence[int],
                 hidden_activation_functions: Optional[Sequence[nn.Module]] = None,
                 chkpt_dir: str = 'tmp/critic_ppo'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        # Define a list to hold the layer sizes including input and output sizes
        layer_sizes = [observation_space.shape[0]] + list(hidden_layers)
        if hidden_activation_functions is None:
            hidden_activation_functions = [nn.ReLU() for _ in range(len(layer_sizes))]

        assert len(hidden_activation_functions) == len(layer_sizes)
        # Define a list to hold the layers of the network
        layers = []

        # Iterate over the layer sizes to create the network layers
        for i in range(len(layer_sizes) - 1):
            # Add a linear layer with the current and next layer sizes
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add a ReLU activation function for all layers except the output layer

            layers.append(hidden_activation_functions[i])

        # Create the actor network using Sequential container
        self.layers = layers
        self.critic_output = nn.Linear(hidden_layers[-1], 1)
        self.layers.append(self.critic_output)
        self.critic_network = nn.Sequential(*self.layers)

        # Initialize the weights of the network
        for layer in self.critic_network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.critic_network(x)

    def save_checkpoint(self):
        torch.save(self, self.checkpoint_dir)

    def load_checkpoint(self):
        torch.load(self.checkpoint_dir)


class Agent(nn.Module):
    def __init__(self, actor_net: ActorNetwork, critic_net: Optional[CriticNetwork]):
        super().__init__()

        self.actor_net = actor_net
        self.critic_net = critic_net

        if self.critic_net is not None:
            assert not self.actor_net.include_critic, "Critic already included in Actor Network"

    def forward(self, x):
        if self.actor_net.include_critic:
            return self.actor_net(x)
        else:
            assert self.critic_net is not None, 'Critic Network not provided and not included in ActorNetwork'
            mean_action, std_action = self.actor_net(x)
            value = self.critic_net(x)
            return mean_action, std_action, value

    def get_value(self, x):
        if self.actor_net.include_critic:
            return self.actor_net.get_value(x)
        else:
            assert self.critic_net is not None, 'Critic Network not provided and not included in ActorNetwork'
            return self.critic_net(x)

    def save_checkpoint(self):
        self.actor_net.save_checkpoint()
        if self.critic_net is not None:
            self.critic_net.save_checkpoint()


actor_net = ActorNetwork(observation_space, [128, 128], n_actions, [nn.ELU(), nn.ELU(), nn.ELU()]).to(device)
critic_net = CriticNetwork(observation_space, [128, 128], [nn.ELU(), nn.ELU(), nn.ELU()]).to(device)
agent = Agent(actor_net, critic_net=None).to(device)
"""
-----------------------------------------------------------------------------------------------------
Hyperparameters for RL agent
-----------------------------------------------------------------------------------------------------
"""
run_name = "test"
writer = SummaryWriter(f"runs/{run_name}")
# writer.add_text(
#     "hyperparameters",
#     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
# )
# Hyperparameters for the agent
n_epochs = 10  # Number of epochs : default 1500
num_updates = 1000
opti = "Adam"
lr_actor = 0.0005  # Learning rate for policy update step
lr_critic = 0.0018  # Learning rate for critic (value function) update step

epsilon = 0.2  # Parameter for clipping value (PPO)
critic_loss_coeff = 0.5
optimizer = optim.Adam(agent.parameters(), lr=lr_actor, eps=1e-5)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr_actor, eps=1e-5)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=lr_critic, eps=1e-5)
minibatch_size = 64
gamma = 1.
gae_lambda = 0.95

# Clipping
clip_vloss = True
grad_clip = 0.005
clip_coef = 0.5
normalize_advantage = False

# other coefficients
ent_coef = 0.
# ALGO Logic: Storage setup
global_step = 0
obs = torch.zeros((tgt_instruction_counts, batchsize) + torch_env.observation_space.shape).to(device)
actions = torch.zeros((tgt_instruction_counts, batchsize) + torch_env.action_space.shape).to(device)
logprobs = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
rewards = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
dones = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
values = torch.zeros((tgt_instruction_counts, batchsize)).to(device)

train_obs = torch.zeros((batchsize,) + torch_env.observation_space.shape, requires_grad=True).to(device)

# Start the Environment
start_time = time.time()


for update in tqdm.tqdm(range(1, num_updates + 1)):
    next_obs, _ = torch_env.reset(seed=seed)
    num_steps = torch_env.episode_length(global_step)
    next_obs = torch.Tensor([next_obs]*batchsize).to(device)
    next_done = torch.zeros(batchsize).to(device)

    # print("episode length:", num_steps)

    for step in range(num_steps):
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            mean_action, std_action, critic_value = agent(next_obs)
            probs = Normal(mean_action, std_action)
            action = torch.clip(probs.sample(), min_bound_actions, max_bound_actions)
            logprob = probs.log_prob(action).sum(1)
            values[step] = critic_value.flatten()

        actions[step] = action
        logprobs[step] = logprob
        # next_obs, reward, terminated, truncated, infos = torch_env.step(action.cpu().numpy())
        next_obs, reward, terminated, truncated, infos = torch_env.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        rewards[step] = torch.tensor(reward).to(device)
        next_obs = torch.Tensor(np.array([next_obs]*batchsize)).to(device)
        next_done = torch.Tensor(np.array([int(done)] * batchsize)).to(device)
        # Only print when at least 1 env is done

        #print(f"global_step={global_step}, episodic_return={np.mean(reward)}")
        writer.add_scalar("charts/episodic_return", np.mean(reward), global_step)
        writer.add_scalar("charts/episodic_length", num_steps, global_step)

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + torch_env.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + torch_env.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batchsize)
    clipfracs = []
    for epoch in range(n_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batchsize, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            new_mean, new_sigma, new_value = agent(b_obs[mb_inds])
            new_dist = Normal(new_mean, new_sigma)
            new_logprob, entropy = new_dist.log_prob(b_actions[mb_inds]).sum(1), new_dist.entropy().sum(1)
            logratio = new_logprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > epsilon).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if normalize_advantage:  # Normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = new_value.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * critic_loss_coeff

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
            optimizer.step()

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    print("Average return:", np.mean(torch_env.reward_history, axis=1)[-1])
    # print(np.mean(torch_env.reward_history, axis =1)[-1])
    print("Circuit fidelity:", torch_env.circuit_fidelity_history[-1])
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/avg_return", np.mean(torch_env.reward_history, axis=1)[-1], global_step)
    # writer.add_scalar("losses/avg_gate_fidelity", torch_env.avg_fidelity_history[-1], global_step)
    writer.add_scalar("losses/circuit_fidelity", torch_env.circuit_fidelity_history[-1], global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)

torch_env.close()
writer.close()
