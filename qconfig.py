from dataclasses import dataclass
from typing import Callable, Dict, Optional, List, Union, Sequence

from qiskit_ibm_runtime import QiskitRuntimeService, Options
from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit, Gate, ParameterExpression, ParameterVector, QuantumRegister
from qiskit_dynamics import Solver
from qualang_tools.config.configuration import QMConfiguration

import torch

@dataclass
class QiskitConfig:
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QuantumCircuit instance
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        estimator_options: Options to feed the Estimator primitive
        solver: Relevant only if dealing with pulse simulation (typically with DynamicsBackend), gives away solver used
        to run simulations for computing exact fidelity benchmark
        channel_freq: Relevant only if dealing with pulse simulation, Dictionary containing information mapping
        the channels and the qubit frequencies
        calibration_files: Feature not available yet, load existing gate calibrations from csv files for DynamicsBackend
        baseline gate calibrations for running algorithm

    """
    parametrized_circuit: Callable[[QuantumCircuit, Optional[Union[Sequence[ParameterExpression], ParameterVector]],
                                    Optional[Union[List, QuantumRegister]]], None] = None
    backend: Optional[Backend] = None
    estimator_options: Optional[Options] = None
    solver: Optional[Solver] = None
    channel_freq: Optional[Dict] = None
    do_calibrations: Optional[bool] = True
    calibration_files: Optional[List[str]] = None


@dataclass
class QuaConfig:
    """
    QUA Configuration
    """
    parametrized_macro: Callable = None
    hardware_config: QMConfiguration = None


@dataclass
class SimulationConfig:
    """
    Simulation Configuration
    
    Args:
        abstraction_level (str): The abstraction level of the simulation. Can be either ``'ideal'`` or ``'pulse'``.
        target_gate (Gate): The target gate to calibrate.
        register (List[int]): The register on which the target gate is applied.
        custum_backend (Backend): The custom backend to use for the simulation.
        fake_backend (Backend): The fake backend to use for the simulation.
        fake_backend_v2 (Backend): The fake backend to use for the simulation.
        n_actions (int): The number of actions the agent can take.
        sampling_Paulis (int): The number of Paulis to sample for the estimation of the fidelity.
        n_shots (int): The number of shots to use for the estimation of the fidelity.
        device (torch.device): The device to use for the simulation.
    """
    
    abstraction_level: str = None
    target_gate: Gate = None
    register: List[int] = None
    custum_backend: Optional[Backend] = None
    fake_backend: Optional[Backend] = None
    fake_backend_v2: Optional[Backend] = None
    n_actions: int = None
    sampling_Paulis: int = None
    n_shots: int = None
    device: torch.device = None