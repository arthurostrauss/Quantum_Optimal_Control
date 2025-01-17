from dataclasses import dataclass
from ..environment.configuration.backend_config import BackendConfig
from .qua_backend import QMBackend
from typing import Any


@dataclass
class QuaConfig(BackendConfig):
    """
    QUA Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        hardware_config: Hardware configuration
        channel_mapping: Dictionary mapping channels to quantum elements
    """

    backend: QMBackend
    hardware_config: Any = None

    @property
    def config_type(self):
        return "qua"
