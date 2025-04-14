import ctypes
from _ctypes import Structure, POINTER
from ctypes import CDLL
from dataclasses import dataclass

from ..environment.configuration.backend_config import BackendConfig
from .qm_backend import QMBackend
from typing import Any, Callable, Literal, Union
from .parameter_table.input_type import InputType

Input_Type = Union[Literal["INPUT_STREAM", "IO1", "IO2", "DGX"]]


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
    input_type: InputType = "INPUT_STREAM"
    num_updates: int = 500

    @property
    def config_type(self):
        return "qua"

    def __post_init__(self):
        self.input_type = InputType(self.input_type) if isinstance(self.input_type, str) else self.input_type


@dataclass
class DGXConfig(QMConfig):
    """
    DGX Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        hardware_config: Hardware configuration
    """
    opnic_dev_path: str = "/home/dpoulos/aps_demo"
    verbosity: int = 1
    MAX_VARIABLE_TRANSFERS: int = 100
    STREAM_TYPE_CPU: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.input_type != InputType.DGX:
            raise ValueError("DGXConfig must have input_type as 'dgx'")

    @property
    def config_type(self):
        return "dgx"
