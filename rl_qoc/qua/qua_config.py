from dataclasses import dataclass
from ..environment.configuration.backend_config import BackendConfig
from .qua_backend import QMBackend
from typing import Any, Callable, Literal


@dataclass
class QMBackendConfig(BackendConfig):
    """
    QUA Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        hardware_config: Hardware configuration
    """

    backend: QMBackend = None
    hardware_config: Any = None
    apply_macro: Callable = None
    input_type: Literal["input_stream", "IO1", "IO2", "dgx"] = "input_stream"

    @property
    def config_type(self):
        return "qua"
