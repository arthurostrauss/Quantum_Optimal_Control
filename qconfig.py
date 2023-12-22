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
    parametrized_macro: Callable = None
    hardware_config: QMConfiguration = None
