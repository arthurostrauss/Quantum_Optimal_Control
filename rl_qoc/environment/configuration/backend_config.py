from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union, Callable, Any, Optional, Dict, Iterable, List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Parameter

from qiskit.primitives import EstimatorPubLike, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import BackendV2
from qiskit.transpiler import PassManager, InstructionDurations

PubLike = Union[
    EstimatorPubLike,
    SamplerPubLike,
]
Pub = Union[EstimatorPub, SamplerPub]


@dataclass
class BackendConfig(ABC):
    """
    Abstract base class for backend configurations.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager: Pass manager to transpile the circuit
        instruction_durations: Dictionary containing the durations of the instructions in the circuit
        primitive_options: Options to feed the primitives (estimator or sampler). If None, the default options of each primitive are used.

    """

    parametrized_circuit: Optional[
        Callable[
            [
                QuantumCircuit,
                ParameterVector | List[Parameter],
                QuantumRegister,
                Any,
            ],
            None,
        ]
    ] = None
    backend: Optional[Any] = None
    parametrized_circuit_kwargs: Dict = field(default_factory=dict)
    skip_transpilation: bool = False
    pass_manager: Optional[Any] = None
    instruction_durations: Optional[InstructionDurations] = None
    primitive_options: Optional[dict] = None

    @property
    @abstractmethod
    def config_type(self):
        return "backend"

    def as_dict(self):
        return {
            "parametrized_circuit": self.parametrized_circuit,
            "backend": self.backend,
            "parametrized_circuit_kwargs": self.parametrized_circuit_kwargs,
            "pass_manager": self.pass_manager,
            "instruction_durations": self.instruction_durations,
            "primitive_options": self.primitive_options,
        }


@dataclass
class QiskitConfig(BackendConfig):
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager
        instruction_durations: Dictionary containing the durations of the instructions in the circuit
    """

    backend: Optional[BackendV2] = None
    pass_manager: Optional[PassManager] = None

    @property
    def config_type(self):
        return "qiskit"


@dataclass
class DynamicsConfig(QiskitConfig):
    """
    Qiskit Dynamics configuration elements.

    Args:
        calibration_files: load existing gate calibrations from json file for DynamicsBackend
        do_calibrations: whether to do gate calibrations for the backend

    """

    calibration_files: Optional[str] = None
    do_calibrations: bool = True

    @property
    def config_type(self):
        return "dynamics"

    def as_dict(self):
        return super().as_dict() | {
            "calibration_files": self.calibration_files,
            "do_calibrations": self.do_calibrations,
        }
