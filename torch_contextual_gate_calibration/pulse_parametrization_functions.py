# %%
import os
import sys
import numpy as np
import tqdm
from typing import Optional, Union, List

module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)

from basis_gate_library import FixedFrequencyTransmon
from helper_functions import remove_unused_wires, get_control_channel_map, get_solver_and_freq_from_backend
from quantumenvironment import QuantumEnvironment
from torch_quantum_environment import TorchQuantumEnvironment

import jax
jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
# import Array and set default backend
from qiskit_dynamics.array import Array
Array.set_default_backend('jax')

from qiskit import pulse, transpile, schedule
from qiskit_dynamics.backend.dynamics_backend import _get_backend_channel_freqs, DynamicsBackend
from qiskit_dynamics import Solver
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector, Gate
from qiskit.circuit.library.standard_gates import ECRGate, SXGate, XGate
from qiskit.providers import Backend, BackendV1, BackendV2
from qiskit.transpiler import CouplingMap
from qiskit_experiments.calibration_management import Calibrations
from qiskit.pulse.library import Gaussian
from qiskit.providers.fake_provider import FakeValencia, FakeJakarta, FakeJakartaV2, FakeHanoi, FakeCairo, FakeCambridge
from qiskit.visualization import plot_coupling_map, plot_circuit_layout, gate_map, plot_gate_map
from qiskit.visualization.pulse_v2 import IQXStandard
from qiskit_ibm_runtime.options import Options, ExecutionOptions, EnvironmentOptions
from qconfig import QiskitConfig

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from agent import ActorNetwork, CriticNetwork, Agent

# Circuit context and Reinforcement Learning libraries
import gymnasium as gym
from gymnasium.spaces import Box, Space

from IPython.display import clear_output

from torch.distributions import Normal


fake_backend = FakeJakarta()
fake_backend_v2 = FakeJakartaV2()

## Currently, including duration as a parameter does not work due to ``qiskit.pulse.exceptions.UnassignedDurationError: 'All instruction durations should be assigned before creating `Schedule`.Please check `.parameters` to find unassigned parameter objects.'``
n_actions = 4
params = ParameterVector('theta', n_actions)



def get_sx_params(backend, physical_qubits):
    """
    Retrieve parameters for an SX gate from a quantum backend.

    This function retrieves the default parameters for an SX gate, including amplitude, beta, sigma, and duration,
    from a given quantum backend.

    Parameters:
    - backend (Backend): The quantum backend from which to retrieve the parameters.
    - physical_qubits (Union[int, tuple, list]): The physical qubits on which the SX gate is applied.
    
    Returns:
    - default_params (dict): A dictionary containing default parameters for the SX gate.
    - basis_gate_instructions (InstructionSchedule): The instruction schedule for the SX gate.
    - instructions_array (numpy.ndarray): An array of instructions for the SX gate.
    """


    if isinstance(backend, BackendV1):
        instruction_schedule_map = backend.defaults().instruction_schedule_map
    else:
        instruction_schedule_map = backend.target.instruction_schedule_map()

    basis_gate_instructions = instruction_schedule_map.get('sx', qubits=physical_qubits)
    
    instructions_array = np.array(basis_gate_instructions.instructions)[:,1]

    sx_pulse = basis_gate_instructions.instructions[0][1].pulse 

    default_params = {
        ("amp", physical_qubits, "sx"): sx_pulse.amp,
        # ("angle", physical_qubits, "sx"): sx_pulse.angle,
        ("β", physical_qubits, 'sx'): sx_pulse.beta,
        ("σ", physical_qubits, "sx"): sx_pulse.sigma,
        ("duration", physical_qubits, "sx"): sx_pulse.duration
    }

    return default_params, basis_gate_instructions, instructions_array

def custom_sx_schedule(backend: Backend, physical_qubits=Union[int, tuple, list], params: ParameterVector=None):
    """
    Generate a custom parameterized schedule for an SX gate.

    This function generates a custom parameterized schedule for an SX gate on specified physical qubits
    with the given parameters.

    Parameters:
    - backend (Backend): The quantum backend used to obtain default SX gate parameters.
    - physical_qubits (Union[int, tuple, list]): The physical qubits on which the SX gate is applied.
    - params (ParameterVector, optional): The parameter vector specifying the custom parameters for the SX gate.

    Returns:
    - parametrized_schedule (Schedule): A parameterized schedule for the SX gate with custom parameters.
    """
    
    # pulse_features = ["amp", "angle", "β", "σ"]
    pulse_features = ['amp', 'β', 'σ'] #, 'duration']
 
    global n_actions
    assert n_actions == len(params), f"Number of actions ({n_actions}) does not match length of ParameterVector {params.name} ({len(params)})" 

    if isinstance(physical_qubits, int):
        physical_qubits = tuple(physical_qubits)

    new_params, _, _ = get_sx_params(backend=fake_backend, physical_qubits=physical_qubits)

    for i, feature in enumerate(pulse_features):
        new_params[(feature, physical_qubits, "sx")] += params[i]

    cals = Calibrations.from_backend(backend, [FixedFrequencyTransmon(["x", "sx"])],
                                        add_parameter_defaults=True)
    
    parametrized_schedule = cals.get_schedule("sx", physical_qubits, assign_params=new_params)

    return parametrized_schedule

def add_parametrized_circuit(qc: QuantumCircuit, params: Optional[ParameterVector]=None, tgt_register: Optional[QuantumRegister]=None):
    """
    Add a parametrized gate to a QuantumCircuit.

    This function adds a parametrized gate to a given QuantumCircuit with optional custom parameters and target register.

    Parameters:
    - qc (QuantumCircuit): The QuantumCircuit to which the parametrized gate will be added.
    - params (ParameterVector, optional): The parameter vector specifying the custom parameters for the gate.
    - tgt_register (QuantumRegister, optional): The target quantum register for applying the parametrized gate.
    """
    
    global n_actions, fake_backend, target

    gate, physical_qubits = target["gate"], target["register"]
    physical_qubits = tuple(physical_qubits)

    if params is None:
        params = ParameterVector('theta', n_actions)
    if tgt_register is None:
        tgt_register = qc.qregs[0]

    parametrized_gate = Gate(name='sx_cal', num_qubits=1, params=params)
    parametrized_schedule = custom_sx_schedule(backend=fake_backend, physical_qubits=physical_qubits, params=params)
    qc.add_calibration(parametrized_gate, physical_qubits, parametrized_schedule)
    qc.append(parametrized_gate, tgt_register)


def get_target_gate(gate: Gate, register: list[int]):
    # X gate as the target gate
    sx_gate = {"gate": gate, 
               "register": register}
    target = sx_gate
    print('Target: ', target)

    return target

target = get_target_gate(gate=XGate(), register=[0])

# %%
def get_circuit_context(num_total_qubits: int):
    # Quantum Circuit context
    target_circuit = QuantumCircuit(num_total_qubits)
    target_circuit.x(0)
    target_circuit.h(0)
    target_circuit.y(0)
    return target_circuit

# %%
def transpile_circuit(target_circuit, fake_backend):
    # Transpile the (context) quantum circuit to the provided (Fake-) Backend
    transpiled_circ = transpile(target_circuit, fake_backend, 
                                initial_layout=[0],
                                basis_gates=fake_backend.configuration().basis_gates, 
                                # scheduling_method='asap',
                                optimization_level=1)
    return remove_unused_wires(transpiled_circ)

# %%
def get_estimator_options(physical_qubits):
    sampling_Paulis = 50
    N_shots = 200
    abstraction_level = 'pulse'

    # control_channel_map = get_control_channel_map(fake_backend, physical_qubits)
    dt = fake_backend.configuration().dt

    dynamics_options = {
                        'seed_simulator': None, #"configuration": fake_backend.configuration(),
                        # 'control_channel_map': control_channel_map, 
                        "solver_options": {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax":dt}
                        }
    qubit_properties = fake_backend_v2.qubit_properties(physical_qubits)

    # Extract channel frequencies and Solver instance from backend to provide a pulse level simulation enabling
    # fidelity benchmarking
    channel_freq, solver = get_solver_and_freq_from_backend(
        backend=fake_backend,
        subsystem_list=physical_qubits,
        rotating_frame="auto",
        evaluation_mode="dense",
        rwa_cutoff_freq=None,
        static_dissipators=None,
        dissipator_channels=None,
        dissipator_operators=None
    )
    calibration_files=None

    estimator_options = Options(resilience_level=0, optimization_level=0, 
                                execution= ExecutionOptions(shots=N_shots*sampling_Paulis))
    
    return qubit_properties, dynamics_options, estimator_options, channel_freq, solver, sampling_Paulis, abstraction_level, N_shots


def get_own_solver(qubit_properties):

    qubit_properties, _, _, _, _ = get_estimator_options()

    dim = 3

    v = [prop.frequency for prop in qubit_properties]
    v0 = 4.86e9
    anharm0 = -0.32e9
    r0 = 0.22e9

    v1 = 4.97e9
    anharm1 = -0.32e9
    r1 = 0.26e9

    J = 0.002e9

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))

    ident = np.eye(dim, dtype=complex)
    full_ident = np.eye(dim**2, dtype=complex)

    N0 = np.kron(ident, N)
    N1 = np.kron(N, ident)

    a0 = np.kron(ident, a)
    a1 = np.kron(a, ident)

    a0dag = np.kron(ident, adag)
    a1dag = np.kron(adag, ident)


    static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
    static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)

    static_ham_full = static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))

    drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
    drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)

    # build solver
    dt = 1/4.5e9

    solver = Solver(
        static_hamiltonian=static_ham_full,
        hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1, drive_op1, drive_op0],
        rotating_frame=static_ham_full,
        hamiltonian_channels=["d0", "d1", "u0", "u1", "u2", "u3"],
        channel_carrier_freqs={"d0": v0, "d1": v1, "u0": v1, "u1": v0, "u2":v0, "u3": v1},
        dt=dt,
        evaluation_mode="sparse"
    )
    # Consistent solver option to use throughout notebook
    solver_options = {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8}

    custom_backend2 = DynamicsBackend(
        solver=solver,
        subsystem_dims=[dim, dim], # for computing measurement data
        solver_options=solver_options, # to be used every time run is called
    )

    return custom_backend2


def get_db_qiskitconfig(target, physical_qubits, qubit_properties, estimator_options, channel_freq, solver, sampling_Paulis, abstraction_level, N_shots, dynamics_options):
    # subsystem_list takes the qubits indices that are the qubits with the parametrized gate AND its nearest neighbours
    dynamics_backend = DynamicsBackend.from_backend(fake_backend, subsystem_list=physical_qubits, **dynamics_options)
    dynamics_backend.target.qubit_properties = qubit_properties

    # Wrap all info in one QiskitConfig
    Qiskit_setup = QiskitConfig(parametrized_circuit=add_parametrized_circuit, backend=dynamics_backend,
                                estimator_options=estimator_options, channel_freq=channel_freq,
                                solver=solver)

    q_env = QuantumEnvironment(target=target, abstraction_level=abstraction_level,
                           Qiskit_config=Qiskit_setup,
                           sampling_Pauli_space=sampling_Paulis, n_shots=N_shots, c_factor=0.5)
    
    return dynamics_backend, Qiskit_setup, q_env


# %%
def get_torch_env(q_env, target_circuit):
    seed = 10
    training_steps_per_gate = 2000
    benchmark_cycle = 100
    # tgt_instruction_counts = target_circuit.data.count(CircuitInstruction(target_gate, tgt_qubits))
    tgt_instruction_counts = 2  # Number of times target Instruction is applied in Circuit
    batchsize = 200  # Batch size (iterate over a bunch of actions per policy to estimate expected return) default 100
    n_actions = 4 # Choose how many control parameters in pulse/circuit parametrization
    min_bound_actions =  - 0.1
    max_bound_actions =  0.1
    scale_factor = 0.1
    observation_space = Box(low=np.array([0, 0]), high=np.array([1, tgt_instruction_counts]), shape=(2,),
                            seed=seed)
    action_space = Box(low=min_bound_actions, high=max_bound_actions, shape=(n_actions,), seed=seed)

    torch_env = TorchQuantumEnvironment(q_env, target_circuit,
                                        action_space,
                                        observation_space, batch_size=batchsize,
                                        training_steps_per_gate=training_steps_per_gate,
                                        benchmark_cycle = benchmark_cycle,
                                        intermediate_rewards=False,
                                        seed=None,)
    return torch_env, observation_space, action_space, tgt_instruction_counts, batchsize, min_bound_actions, max_bound_actions, scale_factor, seed

# %% [markdown]
def get_network(device, observation_space):
    # Definition of the Agent
    hidden_units = [64, 64]
    activation_functions = [nn.Tanh(), nn.Tanh(), nn.Tanh()]
    include_critic = False
    chkpt_dir = 'tmp/ppo'
    chkpt_dir_critic = 'tmp/critic_ppo'

    actor_net = ActorNetwork(observation_space, hidden_units, n_actions, activation_functions, include_critic, chkpt_dir).to(device)
    critic_net = CriticNetwork(observation_space, hidden_units, activation_functions, chkpt_dir_critic).to(device)
    agent = Agent(actor_net, critic_net=critic_net).to(device)

    return actor_net, critic_net, agent

# %%
def clear_history(torch_env, tgt_instruction_counts, batchsize, device):
    global_step = 0
    torch_env.clear_history()
    obs = torch.zeros((tgt_instruction_counts, batchsize) + torch_env.observation_space.shape).to(device)
    actions = torch.zeros((tgt_instruction_counts, batchsize) + torch_env.action_space.shape).to(device)
    logprobs = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
    rewards = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
    dones = torch.zeros((tgt_instruction_counts, batchsize)).to(device)
    values = torch.zeros((tgt_instruction_counts, batchsize)).to(device)

    train_obs = torch.zeros((batchsize,) + torch_env.observation_space.shape, requires_grad=True).to(device)
    visualization_steps = 10

    return global_step, obs, actions, logprobs, rewards, dones, values, train_obs, visualization_steps

# %%
def train_agent(torch_env, global_step, num_updates, seed, device, batchsize, obs, agent, scale_factor, min_bound_actions, max_bound_actions, logprobs, actions, rewards, dones, values, n_epochs, optimizer, minibatch_size, gamma, gae_lambda, critic_loss_coeff, epsilon, clip_vloss, grad_clip, clip_coef, normalize_advantage, ent_coef, writer, visualization_steps):
    for update in tqdm.tqdm(range(1, num_updates + 1)):
        next_obs, _ = torch_env.reset(seed=seed)
        num_steps = torch_env.episode_length(global_step)
        print('num_steps', num_steps)
        
        next_obs = torch.Tensor(np.array([next_obs] * batchsize)).to(device)
        next_done = torch.zeros(batchsize).to(device)

        for step in range(num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                mean_action, std_action, critic_value = agent(next_obs)
                mean_action*=scale_factor
                probs = Normal(mean_action, std_action)
                action = torch.clip(probs.sample(), torch.Tensor(np.array(min_bound_actions)), torch.Tensor(np.array(max_bound_actions)))
                logprob = probs.log_prob(action).sum(1)
                values[step] = critic_value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            # next_obs, reward, terminated, truncated, infos = torch_env.step(action.cpu().numpy())
            next_obs, reward, terminated, truncated, infos = torch_env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(np.array([next_obs] * batchsize)).to(device)
            next_done = torch.Tensor(np.array([int(done)] * batchsize)).to(device)
            # Only print when at least 1 env is done

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
                logratio = new_logprob - b_logprobs[mb_inds] + torch.log(torch.Tensor([1e-6]))
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
        print("mean", mean_action[0])
        print("sigma", std_action[0])
        print("Average return:", np.mean(torch_env.reward_history, axis=1)[-1])
        print(torch_env._get_info())
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/avg_return", np.mean(torch_env.reward_history, axis=1)[-1], global_step)
        #for i in range(num_steps):
        #   writer.add_scalar(f"losses/avg_gate_{i}_fidelity", torch_env.avg_fidelity_history[-1][i], global_step)
        #writer.add_scalar("losses/circuit_fidelity", torch_env.circuit_fidelity_history[-1], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if global_step%visualization_steps == 0:
            clear_output(wait=True) # for animation

    torch_env.close()
    writer.close()

    return {
        'avg_return': np.mean(torch_env.reward_history, axis=1), # shape: (num_updates,)
        'mean_action': mean_action[0],
        'sigma_action': std_action[0],
    }