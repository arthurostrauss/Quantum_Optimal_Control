from ctypes import CDLL
from dataclasses import dataclass
from ..environment.configuration.backend_config import BackendConfig
from .qm_backend import QMBackend
from typing import Any, Callable, Literal, Optional


@dataclass
class QMConfig(BackendConfig):
    """
    QUA Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        hardware_config: Hardware configuration
        channel_mapping: Dictionary mapping channels to quantum elements
    """

    backend: QMBackend = None
    hardware_config: Any = None
    apply_macro: Callable = None
    reset_type: Literal["active", "thermalize"] = "active"
    input_type: Literal["input_stream", "IO1", "IO2", "dgx"] = "input_stream"
    qubit_pair: Any = None

    @property
    def config_type(self):
        return "qua"


@dataclass
class DGXConfig(QMConfig):
    """
    DGX Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        hardware_config: Hardware configuration
    """

    dgx_lib: Optional[CDLL] = None
    dgx_stream = None

    def __post_init__(self):
        if self.input_type != "dgx":
            raise ValueError("DGXConfig must have input_type as 'dgx'")
        if self.dgx_lib is None:
            raise ValueError("DGXConfig must have dgx_lib defined")
        if self.dgx_stream is None:
            raise ValueError("DGXConfig must have dgx_stream defined")

    @property
    def config_type(self):
        return "qua"
