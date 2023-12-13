from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union, Sequence, Any
from agent import Agent, ActorNetwork, CriticNetwork

import torch
import torch.optim as optim
import torch.nn as nn
from gymnasium import Space

from qiskit_ibm_runtime import QiskitRuntimeService, Options
from qiskit.providers import Backend
from qiskit.circuit import (
    QuantumCircuit,
    Parameter,
    ParameterVector,
    QuantumRegister,
    Gate,
)
from qiskit_dynamics import Solver
from qualang_tools.config.configuration import QMConfiguration


class BackendConfig(ABC):
    """
    Abstract base class for backend configurations.
    """

    def __init__(self, config_type: str = ""):
        self._config_type = config_type

    @property
    def config_type(self) -> str:
        return self._config_type


@dataclass
class QiskitConfig(BackendConfig):
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QuantumCircuit instance
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        additional_args: Additional arguments to feed the parametrized_circuit function
        estimator_options: Options to feed the Estimator primitive
        solver: Relevant only if dealing with pulse simulation (typically with DynamicsBackend), gives away solver used
        to run simulations for computing exact fidelity benchmark
        channel_freq: Relevant only if dealing with pulse simulation, Dictionary containing information mapping
        the channels and the qubit frequencies
        calibration_files: Feature not available yet, load existing gate calibrations from csv files for DynamicsBackend
        baseline gate calibrations for running algorithm

    """

    parametrized_circuit: Callable[
        [
            QuantumCircuit,
            Optional[Sequence[Parameter] | ParameterVector],
            Optional[QuantumRegister],
            Any,
        ],
        None,
    ]
    backend: Optional[Backend] = None
    parametrized_circuit_kwargs: Optional[Dict] = field(default_factory=dict)
    estimator_options: Optional[Options] = None
    solver: Optional[Solver] = None
    channel_freq: Optional[Dict] = field(default_factory=dict)
    calibration_files: Optional[List[str]] = None
    do_calibrations: bool = True

    def __post_init__(self):
        super().__init__(config_type="Qiskit")


@dataclass
class QuaConfig(BackendConfig):
    """
    QUA Configuration
    """

    parametrized_macro: Callable
    hardware_config: QMConfiguration

    def __post_init__(self):
        super().__init__(config_type="Qua")


@dataclass
class QEnvConfig:
    """
    Quantum Environment configuration. This is used to define all hyperparameters characterizing the Quantum Environment.
    Those include a description of the backend, the action and observation spaces, the batch size (number of actions per
    policy evaluation), the number of Pauli observables to sample for the fidelity estimation scheme,
    the number of shots per Pauli for the fidelity estimation, the renormalization factor, and the device on which the simulation is run.

    Args:
        target (Dict): Target state or target gate to prepare
        backend_config (BackendConfig): Backend configuration
        action_space (Space): Action space
        observation_space (Space): Observation space
        batch_size (int, optional): Batch size (iterate over a bunch of actions per policy to estimate expected return). Defaults to 50.
        sampling_Paulis (int, optional): Number of Paulis to sample for the fidelity estimation scheme. Defaults to 100.
        n_shots (int, optional): Number of shots per Pauli for the fidelity estimation. Defaults to 1.
        c_factor (float, optional): Renormalization factor. Defaults to 0.5.
        benchmark_cycle (int, optional): Number of epochs between two fidelity benchmarking. Defaults to 5.
        seed (int, optional): Seed for Observable sampling. Defaults to 1234.
        device (Optional[torch.device], optional): Device on which the simulation is run. Defaults to None.
    """

    target: Dict[str, List | Gate | QuantumRegister | QuantumCircuit]
    backend_config: BackendConfig
    action_space: Space
    observation_space: Space
    batch_size: int = 50
    sampling_Paulis: int = 100
    n_shots: int = 1
    c_factor: float = 0.5
    benchmark_cycle: int = 1
    seed: int = 1234
    device: Optional[torch.device] = None


def get_module_from_str(module_str):
    module_dict = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
        "none": nn.ReLU,
        "softmax": nn.Softmax,
        "log_softmax": nn.LogSoftmax,
        "gelu": nn.GELU,
    }
    return module_dict[module_str]


def get_optimizer_from_str(optim_str):
    optim_dict = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "rmsprop": optim.RMSprop,
        "rprop": optim.Rprop,
        "sgd": optim.SGD,
    }
    return optim_dict[optim_str]


@dataclass
class AgentConfig:
    """
    Agent configuration. This is used to define all hyperparameters characterizing the Agent.
    """

    optim: str = "adam"
    num_updates: int = 100
    n_epochs: int = 10
    mini_batch_size: int = 64
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    grad_clip: float = 0.5
    clip_value_loss: bool = True
    clip_ratio: float = 0.2
    n_units: list = field(default_factory=lambda: [64, 64])
    activations: list = field(default_factory=lambda: ["tanh", "tanh"])
    observation_space: Space = None
    action_space: Space = None
    include_critic: bool = True
    chkpt_dir: str = "tmp/ppo"

    def __post_init__(self):
        for activation in self.activations:
            activation = get_module_from_str(activation)()
        self.actor_network = ActorNetwork(
            self.observation_space,
            self.n_units,
            self.action_space.shape[-1],
            self.activations,
            self.include_critic,
            self.chkpt_dir,
        )
        self.critic_network = CriticNetwork(
            self.observation_space, self.n_units, self.activations
        )
        self.agent = Agent(
            self.actor_network,
            self.critic_network,
            self.lr_actor,
            self.lr_critic,
            self.gamma,
            self.gae_lambda,
            self.ent_coef,
            self.vf_coef,
            self.grad_clip,
            self.clip_value_loss,
            self.clip_ratio,
            self.mini_batch_size,
            self.chkpt_dir,
        )
