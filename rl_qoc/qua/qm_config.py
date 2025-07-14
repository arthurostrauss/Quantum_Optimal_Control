from dataclasses import dataclass, field

from qm import CompilerOptionArguments

from ..environment.configuration.backend_config import BackendConfig
from qiskit_qm_provider import QMBackend
from typing import Literal, Union, Optional, Dict, Any
from qiskit_qm_provider import InputType

Input_Type = Union[Literal["INPUT_STREAM", "IO1", "IO2", "DGX"], InputType]


@dataclass
class QMConfig(BackendConfig):
    """
    QUA Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        input_type: Type of input for the QUA program
        verbosity: Verbosity level for the QUA program
        num_updates: Number of updates for the QUA program
        compiler_options: Compiler options for the QUA program
        opnic_dev_path: Path to the Opnic development directory
        timeout: Timeout for the QUA program execution
        test_mode: Whether to run in test mode (saves everything in streams)
    """

    backend: Optional[QMBackend] = None
    input_type: Input_Type = InputType.INPUT_STREAM
    verbosity: int = 1
    num_updates: int = 1000
    wrapper_data: Dict[str, Any] = field(default_factory=dict)
    compiler_options: Optional[CompilerOptionArguments] = None
    opnic_dev_path: str = "/home/dpoulos/opnic-dev"
    timeout: int = 60
    test_mode: bool = False

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
