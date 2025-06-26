from dataclasses import dataclass

from qm import CompilerOptionArguments

from ..environment.configuration.backend_config import BackendConfig
from qiskit_qm_provider import QMBackend
from typing import Literal, Union, Optional
from qiskit_qm_provider import InputType

Input_Type = Union[Literal["INPUT_STREAM", "IO1", "IO2", "DGX"]]


@dataclass
class QMConfig(BackendConfig):
    """
    QUA Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
    """

    backend: QMBackend = None
    input_type: InputType = "INPUT_STREAM"
    verbosity: int = 1
    num_updates: int = 1000
    compiler_options: Optional[CompilerOptionArguments] = None
    opnic_dev_path: str = "/home/dpoulos/opnic-dev"
    timeout: int = 60

    @property
    def config_type(self):
        return "qm"

    def __post_init__(self):
        self.input_type = (
            InputType(self.input_type) if isinstance(self.input_type, str) else self.input_type
        )

    def as_dict(self):
        return {
            "input_type": str(self.input_type),
            "verbosity": self.verbosity,
            "num_updates": self.num_updates,
            "opnic_dev_path": self.opnic_dev_path,
        }
