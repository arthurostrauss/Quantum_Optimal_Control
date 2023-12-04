"""
This script contains a set of functions used in the HPO script hpo_pulse_01.py.
NO interaction / input from the user is required.

Author: Lukas Voss
Created on 16/11/2023
"""

# %%
import os
import sys
import numpy as np
import tqdm
from functools import partial
from typing import Optional, Union

module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)

from basis_gate_library import FixedFrequencyTransmon
from helper_functions import remove_unused_wires, get_solver_and_freq_from_backend
from quantumenvironment import QuantumEnvironment
module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control/torch_contextual_gate_calibration'))
if module_path not in sys.path:
    sys.path.append(module_path)
from torch_quantum_environment import TorchQuantumEnvironment

import jax
jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
# import Array and set default backend
from qiskit_dynamics.array import Array
Array.set_default_backend('jax')

from qiskit import transpile
from qiskit_dynamics.backend.dynamics_backend import DynamicsBackend
from qiskit_dynamics import Solver
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector, Gate
from qiskit.circuit.library.standard_gates import XGate, SXGate, YGate, ZGate, HGate, CXGate, SGate, ECRGate
from qiskit.providers import Backend, BackendV1
from qiskit_experiments.calibration_management import Calibrations
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2, FakeMelbourne, FakeMelbourneV2, FakeRome, FakeRomeV2, FakeSydney, FakeSydneyV2, FakeValencia, FakeValenciaV2, FakeVigo, FakeVigoV2, FakeJakarta, FakeJakartaV2
# from qiskit.visualization import plot_coupling_map, plot_circuit_layout, gate_map, plot_gate_map
from qiskit_ibm_runtime.options import Options, ExecutionOptions
from qconfig import QiskitConfig

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# Circuit context and Reinforcement Learning libraries
from agent import ActorNetwork, CriticNetwork, Agent
from gymnasium.spaces import Box

from IPython.display import clear_output

import logging
# Create a custom logger with the level WARNING because INFO would trigger too many log message by qiskit itself
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s", # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def map_json_inputs(config):
    """
    Map input parameters from a json file to the corresponding Python objects.
    """

    quantum_gates_mapping = {
        'XGate': XGate(),
        'YGate': YGate(),
        'ZGate': ZGate(),
        'HGate': HGate(),
        'CXGate': CXGate(),
        'SGate': SGate(),
    }
    try:
        config['target_gate'] = quantum_gates_mapping[config['target_gate']]
    except KeyError:
        logging.warning(f"Target gate {config['target_gate']} not found in the quantum gates mapping. Please check the spelling. Example: 'XGate'")
        raise KeyError(f"Target gate {config['target_gate']} not found in the quantum gates mapping. Please check the spelling. Example: 'XGate'")



    fake_backends_mapping = {
        'FakeMelbourne': FakeMelbourne(),
        'FakeMelbourneV2': FakeMelbourneV2(),
        'FakeRome': FakeRome(),
        'FakeRomeV2': FakeRomeV2(),
        'FakeSydney': FakeSydney(),
        'FakeSydneyV2': FakeSydneyV2(),
        'FakeValencia': FakeValencia(),
        'FakeValenciaV2': FakeValenciaV2(),
        'FakeVigo': FakeVigo(),
        'FakeVigoV2': FakeVigoV2(),
        'FakeJakarta': FakeJakarta(),
        'FakeJakartaV2': FakeJakartaV2(),
    }
    try:
        config['backend'] = fake_backends_mapping[config['backend']]
    except KeyError:
        logging.warning(f"Fake backend {config['backend']} not found in the fake backends mapping. Please check the spelling. Example: 'FakeMelbourne' and 'FakeMelbourneV2'")
        raise KeyError(f"Fake backend {config['backend']} not found in the fake backends mapping. Please check the spelling. Example: 'FakeMelbourne' and 'FakeMelbourneV2'")

    torch_devices_mapping = {
        'cpu': torch.device('cpu'),
        'cuda': torch.device('cuda:0'),  # Assuming you have a CUDA-enabled GPU
    }
    try:
        config['device'] = torch_devices_mapping[config['device']]
    except KeyError:
        logging.warning(f"Torch device {config['device']} not found in the torch devices mapping. Please check the spelling. Choose either: 'cpu' and 'cuda'")
        raise KeyError(f"Torch device {config['device']} not found in the torch devices mapping. Please check the spelling. Choose either: 'cpu' and 'cuda'")

    return config


def get_gate_params(backend, physical_qubits, gate_str):
    """
    Retrieve parameters for an SX gate from a quantum backend.

    This function retrieves the default parameters for an SX gate, including amplitude, beta, sigma, and duration,
    from a given quantum backend.

    Args:
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

    basis_gate_instructions = instruction_schedule_map.get(gate_str, qubits=physical_qubits)
    
    instructions_array = np.array(basis_gate_instructions.instructions)[:, 1]

    gate_pulse = basis_gate_instructions.instructions[0][1].pulse 

    default_params = {
        ("amp", physical_qubits, gate_str): gate_pulse.amp,
        # ("angle", physical_qubits, gate_str): gate_pulse.angle,
        ("β", physical_qubits, gate_str): gate_pulse.beta,
        ("σ", physical_qubits, gate_str): gate_pulse.sigma,
        ("duration", physical_qubits, gate_str): gate_pulse.duration
    }

    return default_params, basis_gate_instructions, instructions_array

def custom_gate_schedule(backend: Backend, physical_qubits=Union[int, tuple, list], params: ParameterVector=None, n_actions: int=4, gate_str: str=None):
    """
    Generate a custom parameterized schedule for an SX gate.

    This function generates a custom parameterized schedule for an SX gate on specified physical qubits
    with the given parameters.

    Args:
        - backend (Backend): The quantum backend used to obtain default SX gate parameters.
        - physical_qubits (Union[int, tuple, list]): The physical qubits on which the SX gate is applied.
        - params (ParameterVector, optional): The parameter vector specifying the custom parameters for the SX gate.

    Returns:
        - parametrized_schedule (Schedule): A parameterized schedule for the SX gate with custom parameters.
    """
    
    # pulse_features = ["amp", "angle", "β", "σ"]
    pulse_features = ['amp', 'β', 'σ'] #, 'duration']
 
    assert n_actions == len(params), f"Number of actions ({n_actions}) does not match length of ParameterVector {params.name} ({len(params)})" 

    if isinstance(physical_qubits, int):
        physical_qubits = tuple(physical_qubits)

    new_params, _, _ = get_gate_params(backend=backend, physical_qubits=physical_qubits, gate_str=gate_str)

    for i, feature in enumerate(pulse_features):
        new_params[(feature, physical_qubits, gate_str)] += params[i]

    cals = Calibrations.from_backend(backend, [FixedFrequencyTransmon(["x", "sx"])],
                                        add_parameter_defaults=True)
    
    parametrized_schedule = cals.get_schedule(gate_str, physical_qubits, assign_params=new_params)

    return parametrized_schedule

def add_parametrized_circuit(qc: QuantumCircuit, params: Optional[ParameterVector]=None, tgt_register: Optional[QuantumRegister]=None, backend: BackendV1 = FakeJakarta(), n_actions: int=4, target: dict=None, gate_str: str=None):
    """
    Add a parametrized gate to a QuantumCircuit.

    This function adds a parametrized gate to a given QuantumCircuit with optional custom parameters and target register.

    Args:
        - qc (QuantumCircuit): The QuantumCircuit to which the parametrized gate will be added.
        - params (ParameterVector, optional): The parameter vector specifying the custom parameters for the gate.
        - tgt_register (QuantumRegister, optional): The target quantum register for applying the parametrized gate.
    """
    
    physical_qubits = tuple(target["register"])

    if params is None:
        params = ParameterVector('theta', n_actions)
    if tgt_register is None:
        tgt_register = qc.qregs[0]

    gate_name = gate_str + '_cal'
    parametrized_gate = Gate(name=gate_name, num_qubits=1, params=params)
    parametrized_schedule = custom_gate_schedule(backend=backend, physical_qubits=physical_qubits, params=params, gate_str=gate_str)
    qc.add_calibration(parametrized_gate, physical_qubits, parametrized_schedule)
    qc.append(parametrized_gate, tgt_register)


def get_target_gate(gate: Gate, register: Union[tuple[int], list[int]]):
    """
    Constructs and returns a dictionary representing a target quantum gate with its associated register.

    The function takes a quantum gate object and a either a tuple or a list of integers representing qubit indices specifying the target register (so the register the target gate is applied to),
    and combines them into a dictionary. This dictionary can then be used to apply the gate to the specified qubits
    in a quantum circuit.

    Args:
        - gate (Gate): A quantum gate object from qiskit
        - register (list[int]): A list of integers representing the qubit indices the gate will be applied to.

    Returns:
        - dict: A dictionary with two keys: 'gate', holding the quantum gate object, and 'register', holding
              the list of qubit indices.
    """
    target = {
        "gate": gate, 
        "register": register,
    }
    return target

# %%
def transpile_circuit(target_circuit: QuantumCircuit, backend: Backend):
    """
    This function takes a quantum circuit and transpiles it for a given backend (which can be a real
    or a fake quantum computing backend). The transpilation process may include optimizing the circuit,
    applying an initial layout, and conforming to the basis gates of the backend. The function also
    removes any unused wires from the transpiled circuit.

    Args:
        - target_circuit (QuantumCircuit): The quantum circuit to be transpiled.
        - backend (Backend): The backend (real or simulated) for which the circuit is being transpiled.

    Returns:
        - QuantumCircuit: The transpiled quantum circuit, optimized and compatible with the specified backend.
    """
    transpiled_circ = transpile(target_circuit, backend, 
                                initial_layout=[0],
                                basis_gates=backend.configuration().basis_gates, 
                                # scheduling_method='asap',
                                optimization_level=1)
    return remove_unused_wires(transpiled_circ)

# %%
def get_estimator_options(sampling_Paulis, N_shots, physical_qubits, backend):
    
    # abstraction_level = 'pulse'
    dt = backend.configuration().dt

    dynamics_options = {
                        'seed_simulator': None,
                        "solver_options": {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax":dt}
                        }

    # Extract channel frequencies and Solver instance from backend to provide a pulse level simulation enabling
    # fidelity benchmarking
    channel_freq, solver = get_solver_and_freq_from_backend(
        backend=backend,
        subsystem_list=physical_qubits,
        rotating_frame="auto",
        evaluation_mode="dense",
        rwa_cutoff_freq=None,
        static_dissipators=None,
        dissipator_channels=None,
        dissipator_operators=None
    )

    estimator_options = Options(resilience_level=0, optimization_level=0, 
                                execution= ExecutionOptions(shots=N_shots*sampling_Paulis))
    
    return dynamics_options, estimator_options, channel_freq, solver


def get_own_solver():
    """
    This function constructs a Hamiltonian representing two coupled qubits and their interaction dynamics.
    It initializes the solver with this Hamiltonian and various operational parameters like channel frequencies
    and time step (dt). The solver is configured for sparse matrix evaluation using JAX.

    Args:
        - qubit_properties (list): A list containing the properties of qubits, such as frequency.

    Returns:
        - DynamicsBackend: An object representing the custom quantum dynamics backend, configured with
                           the static Hamiltonian, drive operations, rotating frame, and other solver options.
    """

    dim = 3
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

    custom_backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=[dim, dim], # for computing measurement data
        solver_options=solver_options, # to be used every time run is called
    )
    return custom_backend


def get_db_qiskitconfig(backend: Backend, target: dict, physical_qubits: tuple, gate_str: str, estimator_options, channel_freq, solver, sampling_Paulis, abstraction_level, N_shots, dynamics_options):
    """
    Configures and returns a quantum environment setup for Qiskit simulations.

    This function sets up a quantum environment using a given backend and target circuit configuration.
    It includes the creation of a DynamicsBackend, configuring a Qiskit environment, and setting up
    various simulation parameters like the number of shots, channel frequencies, and quantum environment options.

    Args:
        - backend (Backend): The simulated backend for the quantum environment.
        - target (dict): A dictionary specifying the target quantum gate and the register it acts on.
        - physical_qubits (list[int]): List of qubits indices to be included in the subsystem.
        - qubit_properties (list): Properties of the qubits used in the simulation.
        - estimator_options (dict): Options for the quantum state estimator.
        - channel_freq (dict): Dictionary mapping channels to their frequencies.
        - solver (Solver): The solver for the quantum dynamics.
        - sampling_Paulis (list): List of Pauli operators for sampling.
        - abstraction_level (str): The level of abstraction for the quantum simulation.
        - N_shots (int): Number of shots (repetitions) for each measurement.
        - dynamics_options (dict): Configuration options for the dynamics backend.

    Returns:
        - Dynamics Backend
        - Qiskit configuration setup
        - Quantum environment object

    Note:
    This function assumes the availability of certain global variables and specific library imports.
    """
    dynamics_backend = DynamicsBackend.from_backend(backend, subsystem_list=physical_qubits, **dynamics_options)
    # dynamics_backend.target.qubit_properties = qubit_properties

    # Create a partial function with target passed
    parametrized_circuit_with_target = partial(add_parametrized_circuit, target=target, gate_str=gate_str)

    Qiskit_setup = QiskitConfig(parametrized_circuit=parametrized_circuit_with_target, backend=dynamics_backend,
                                estimator_options=estimator_options, channel_freq=channel_freq,
                                solver=solver)

    q_env = QuantumEnvironment(target=target, abstraction_level=abstraction_level,
                           Qiskit_config=Qiskit_setup,
                           sampling_Pauli_space=sampling_Paulis, n_shots=N_shots, c_factor=0.5)
    
    return dynamics_backend, Qiskit_setup, q_env


# %%
def get_torch_env(q_env: QuantumEnvironment, target_circuit: QuantumCircuit, n_actions: int):
    """
    Initializes and returns a Torch-based quantum environment for RL application

    This function sets up a quantum environment for use with PyTorch, including defining action and observation spaces.
    It configures the environment for training quantum circuits with RL, specifying parameters
    like the number of actions, batch size, and training steps per gate.

    Args:
        - q_env (QuantumEnvironment): The quantum environment setup, typically from Qiskit.
        - target_circuit (QuantumCircuit): The quantum circuit to be used in the environment.
        - n_actions (int): The number of actions available in the RL model.

    Returns:
        - Tuple containing the TorchQuantumEnvironment object and its configuration details like
               observation space, action space, and other parameters relevant to the RL setup.
    """
    seed = 10
    training_steps_per_gate = 2000
    benchmark_cycle = 100
    tgt_instruction_counts = 2  # Number of times target Instruction is applied in Circuit
    batchsize = 200  # Batch size (iterate over a bunch of actions per policy to estimate expected return) default 100
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

# %%
def get_network(device: torch.device, observation_space: Box, n_actions: int):
    """
    Initializes and returns the neural network models (actor and critic) and an agent for RL

    This function creates an actor network and a critic network using the specified observation space and
    number of actions. These networks are used in policy-based reinforcement learning methods. The function also
    initializes an agent with these networks.

    Args:
        - device (torch.device): The device (e.g., CPU or GPU) on which the networks will be loaded.
        - observation_space (space): The observation space of the environment.
        - n_actions (int): The number of actions available in the reinforcement learning model.

    Returns:
        - A tuple containing the actor network, critic network, and the agent. These are used for
               training and decision-making in reinforcement learning scenarios.
    """
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
def clear_history(torch_env: TorchQuantumEnvironment, tgt_instruction_counts: int, batchsize: int, device: torch.device):
    """
    Clears the history of the TorchQuantumEnvironment and initializes training variables for reinforcement learning (RL)

    This function resets the environment's history and initializes various tensors (like observations, actions,
    log probabilities, rewards, dones, and values) used in the training loop of a RL algorithm.
    These tensors are initialized to zeros and are configured to the specified batch size and instruction counts.

    Args:
        - torch_env (TorchQuantumEnvironment): The Torch-based quantum environment used in RL training.
        - tgt_instruction_counts (int): The number of instruction counts used in the environment.
        - batchsize (int): The batch size used for training.
        - device (torch.device): The device (e.g., CPU or GPU) where the tensors will be stored.

    Returns:
        - A tuple containing initialized global step count, tensors for observations, actions, log probabilities,
          rewards, dones, values, train observations, and the number of visualization steps.
    """
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
def train_agent(torch_env: TorchQuantumEnvironment, 
                global_step: int, 
                num_updates: int, 
                seed: int, 
                device: torch.device, 
                batchsize: int, 
                obs: torch.tensor, 
                agent: Agent, 
                scale_factor: float, 
                min_bound_actions: float, 
                max_bound_actions: float, 
                logprobs: torch.tensor, 
                actions: torch.tensor, 
                rewards: torch.tensor, 
                dones: torch.tensor, 
                values: torch.tensor, 
                n_epochs: int, 
                optimizer: torch.optim.Adam, 
                minibatch_size: int, 
                gamma: float, 
                gae_lambda, 
                critic_loss_coeff: float, 
                epsilon: float, 
                clip_vloss: bool, 
                grad_clip: float, 
                clip_coef: float, 
                normalize_advantage: bool, 
                ent_coef: float, 
                writer: SummaryWriter, 
                visualization_steps: int):
    """
    Trains an agent in a reinforcement learning environment using the Proximal Policy Optimization (PPO) algorithm.

    This function performs training iterations on an agent within a specified quantum environment using PPO. It involves
    collecting data by interacting with the environment, computing advantages, and optimizing both the policy (actor) and
    value (critic) networks. Additionally, it logs various training metrics for analysis.

    Args:
        - torch_env (TorchQuantumEnvironment): The Torch-based quantum environment used for training.
        - global_step (int): Global step count, used for tracking the training progress.
        - num_updates (int): Number of training iterations to perform.
        - seed (int): Seed for environment reset for consistency.
        - device (torch.device): The device on which the tensors are stored and computations are performed.
        - batchsize (int), obs, agent, scale_factor, min_bound_actions, max_bound_actions, logprobs,    actions, rewards, dones, values: Training-related parameters and tensors.
        - n_epochs (int): Number of epochs for training.
        - optimizer (torch.optim.Optimizer): The optimizer used for training.
        - minibatch_size (int): Size of the minibatch for training.
        - gamma (float): Discount factor for rewards.
        - gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).
        - critic_loss_coeff (float): Coefficient for critic loss in the total loss calculation.
        - epsilon (float): Clipping parameter for PPO.
        - clip_vloss (bool): Whether to clip the value loss.
        - grad_clip (float): Gradient clipping value.
        - clip_coef (float): Coefficient for value loss clipping.
        - normalize_advantage (bool): Flag to normalize advantage values.
        - ent_coef (float): Coefficient for entropy in the total loss calculation.
        - writer (torch.utils.tensorboard.SummaryWriter): Tensorboard writer for logging training metrics.
        - visualization_steps (int): Number of steps for visualization.

    Returns:
        - dict: A dictionary containing the average return, mean action, and sigma (standard deviation) of the action.
    """
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