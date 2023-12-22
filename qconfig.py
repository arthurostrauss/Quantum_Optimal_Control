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
            ParameterVector,
            QuantumRegister,
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
    training_with_cal: bool = True
    device: Optional[torch.device] = None