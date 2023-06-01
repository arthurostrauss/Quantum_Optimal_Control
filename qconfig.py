from dataclasses import dataclass
from typing import Callable, Dict, Optional, List

from qiskit_ibm_runtime import QiskitRuntimeService, Options
from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit
from qiskit_dynamics import Solver


@dataclass
class QiskitConfig:
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QuantumCircuit instance
        backend: Quantum backend, if None is provided, then statevector simulation is used
        estimator_options: Options to feed the Estimator primitive
        solver: Relevant only if dealing with pulse simulation (typically with DynamicsBackend), gives away solver used
        to run simulations for computing exact fidelity benchmark
        channel_freq: Relevant only if dealing with pulse simulation, Dictionary containing information mapping
        the channels and the qubit frequencies
        calibration_files: Feature not available yet, load existing gate calibrations from csv files for DynamicsBackend
        baseline gate calibrations for running algorithm

    """
    parametrized_circuit: Callable[[QuantumCircuit], None] = None
    backend: Optional[Backend] = None
    estimator_options: Optional[Options] = None
    solver: Optional[Solver] = None
    channel_freq: Optional[Dict] = None
    calibration_files: Optional[List[str]] = None



