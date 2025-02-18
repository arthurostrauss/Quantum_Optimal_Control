from dataclasses import dataclass
from ..environment.configuration.backend_config import BackendConfig
from .qm_backend import QMBackend
from typing import Any, Callable, Literal


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

    backend: QMBackend = None
    hardware_config: Any = None
    apply_macro: Callable = None
    reset_type: Literal["active", "thermalize"] = "active"
    input_type: Literal["input_stream", "IO1", "IO2", "dgx"] = "dgx"
    qubit_pair: Any = None

    @property
    def config_type(self):
        return "qua"
