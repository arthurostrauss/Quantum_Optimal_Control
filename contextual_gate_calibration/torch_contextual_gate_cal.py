import os

# qiskit imports
from qiskit.circuit import (
    ParameterVector,
    QuantumCircuit,
    QuantumRegister,
    CircuitInstruction,
)

# from qiskit.circuit.random import random_circuit
# from qiskit_ibm_provider import IBMProvider
from qiskit.providers.fake_provider import FakeJakartaV2
from qiskit_aer import AerSimulator
from qiskit.circuit.library import CXGate

from agent import ActorNetwork, CriticNetwork, Agent
from qconfig import QiskitConfig, QEnvConfig
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from gymnasium.spaces import Box
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import tqdm
import time
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Save your credentials on disk.
# IBMProvider.save_account(token='<IBM Quantum API key>')


def param_circuit(
    qc: QuantumCircuit,
    params: Optional[ParameterVector] = None,
    q_reg: Optional[QuantumRegister] = None,
):
    my_qc = QuantumCircuit(q_reg)
    my_qc.u(np.pi * params[0], np.pi * params[1], np.pi * params[2], q_reg[0])
    my_qc.u(np.pi * params[3], np.pi * params[4], np.pi * params[5], q_reg[1])
    my_qc.rzx(np.pi * params[6], q_reg[0], q_reg[1])
    qc.append(my_qc.to_instruction(), q_reg)


"""
-----------------------------------------------------------------------------------------------------
Variables to define environment
-----------------------------------------------------------------------------------------------------
"""
device = torch.device("cpu")
sampling_Paulis = 100
N_shots = 1  # Number of shots for sampling the quantum computer for each action vector
fake_backend = FakeJakartaV2()
aer_backend = AerSimulator.from_backend(fake_backend, noise_model=None)

# service = QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q-nus/default/default')
# runtime_backend = service.backend("ibmq_qasm_simulator")
backend = aer_backend
target_gate = CXGate()
physical_qubits = [0, 1]
n_qubits = len(physical_qubits)
target = {"gate": target_gate, "register": physical_qubits}
backend_config = QiskitConfig(parametrized_circuit=param_circuit, backend=backend)

# Circuit context
seed = 10
training_steps_per_gate = 1500
# target_circuit = transpile(random_circuit(4, depth=2, max_operands=2, seed=seed), backend)
target_circuit = QuantumCircuit(2)
target_circuit.cx(0, 1)
tgt_qubits = [target_circuit.qubits[i] for i in physical_qubits]

tgt_instruction_counts = target_circuit.data.count(
    CircuitInstruction(target_gate, tgt_qubits)
)

batchsize = 500  # Batch size (iterate over a bunch of actions per policy to estimate expected return) default 100
n_actions = 7  # Choose how many control parameters in pulse/circuit parametrization
min_bound_actions = 0.0
max_bound_actions = 1.0
observation_space = Box(
    low=np.array([0, 0]),
    high=np.array([4**n_qubits, tgt_instruction_counts]),
    shape=(2,),
    seed=seed,
)
action_space = Box(
    low=min_bound_actions, high=max_bound_actions, shape=(n_actions,), seed=seed
)

config = QEnvConfig(
    target,
    backend_config,
    action_space,
    observation_space,
    batch_size=batchsize,
    sampling_Paulis=sampling_Paulis,
    n_shots=N_shots,
    c_factor=0.125,
    benchmark_cycle=5,
    seed=seed,
    device=device,
)
torch_env = ContextAwareQuantumEnvironment(
    training_config=config,
    circuit_context=target_circuit,
    training_steps_per_gate=training_steps_per_gate,
    intermediate_rewards=False,
)


actor_net = ActorNetwork(
    observation_space, [128, 128], n_actions, [nn.ELU(), nn.ELU(), nn.ELU()]
).to(device)
critic_net = CriticNetwork(
    observation_space, [128, 128], [nn.ELU(), nn.ELU(), nn.ELU()]
).to(device)
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
lr_actor = 0.01  # Learning rate for policy update step
lr_critic = 0.0018  # Learning rate for critic (value function) update step

epsilon = 0.2  # Parameter for clipping value (PPO)
critic_loss_coeff = 0.5
optimizer = optim.Adam(agent.parameters(), lr=lr_actor, eps=1e-5)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr_actor, eps=1e-5)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=lr_critic, eps=1e-5)
minibatch_size = 40
gamma = 1.0
gae_lambda = 0.95

# Clipping
clip_vloss = True
grad_clip = 0.5
clip_coef = 0.5
normalize_advantage = False

# other coefficients
ent_coef = 0.0
# ALGO Logic: Storage setup
global_step = 0
obs = torch.zeros(
    (tgt_instruction_counts, batchsize) + torch_env.observation_space.shape
).to(device)
actions = torch.zeros(
    (tgt_instruction_counts, batchsize) + torch_env.action_space.shape
).to(device)
logprobs = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
rewards = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
dones = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
values = torch.zeros((tgt_instruction_counts, batchsize)).to(device)

train_obs = torch.zeros(
    (batchsize,) + torch_env.observation_space.shape, requires_grad=True
).to(device)
# Start the Environment
start_time = time.time()

try:
    for update in tqdm.tqdm(range(1, num_updates + 1)):
        next_obs, _ = torch_env.reset(seed=seed)
        num_steps = torch_env.episode_length(global_step)
        next_obs = torch.Tensor(np.array([next_obs] * batchsize)).to(device)
        next_done = torch.zeros(batchsize).to(device)

        # print("episode length:", num_steps)

        for step in range(num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                mean_action, std_action, critic_value = agent(next_obs)
                probs = Normal(mean_action, std_action)
                action = torch.clip(
                    probs.sample(), min_bound_actions, max_bound_actions
                )
                logprob = probs.log_prob(action).sum(1)
                values[step] = critic_value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            # next_obs, reward, terminated, truncated, infos = torch_env.step(action.cpu().numpy())
            next_obs, reward, terminated, truncated, infos = torch_env.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(np.array([next_obs] * batchsize)).to(device)
            next_done = torch.Tensor(np.array([int(done)] * batchsize)).to(device)
            # Only print when at least 1 env is done

            # print(f"global_step={global_step}, episodic_return={np.mean(reward)}")
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
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
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
                new_logprob, entropy = new_dist.log_prob(b_actions[mb_inds]).sum(
                    1
                ), new_dist.entropy().sum(1)
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                print("new_logprob", new_logprob)
                print("b_logprobs[mb_inds]", b_logprobs[mb_inds])
                print("logratio", logratio)
                print("ratio", ratio)
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > epsilon).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if normalize_advantage:  # Normalize advantage
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

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
        print("mean", mean_action[0])
        print("sigma", std_action[0])
        print("Average return:", np.mean(torch_env.reward_history, axis=1)[-1])
        # print(np.mean(torch_env.reward_history, axis =1)[-1])
        print("Circuit fidelity:", torch_env.circuit_fidelity_history[-1])
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar(
            "losses/avg_return",
            np.mean(torch_env.reward_history, axis=1)[-1],
            global_step,
        )
        # writer.add_scalar("losses/avg_gate_fidelity", torch_env.avg_fidelity_history[-1], global_step)
        writer.add_scalar(
            "losses/circuit_fidelity",
            torch_env.circuit_fidelity_history[-1],
            global_step,
        )
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

except Exception as e:
    writer.close()
    raise e

torch_env.close()
writer.close()
