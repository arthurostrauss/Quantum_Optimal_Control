import ctypes
from _ctypes import Structure, POINTER
from ctypes import CDLL
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

    opnic_so_lib_path: str = "/home/dpoulos/libopnic_sdk.so"
    verbosity: int = 1
    MAX_VARIABLE_TRANSFERS: int = 800
    STREAM_TYPE_CPU: int = 1

    def __post_init__(self):
        if self.input_type != "dgx":
            raise ValueError("DGXConfig must have input_type as 'dgx'")

        self._dgx_lib = CDLL(self.opnic_so_lib_path)
        self._dgx_lib.DgxStream_new.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._dgx_lib.DgxStream_new.restype = ctypes.c_void_p
        self._dgx_lib.DgxStream_pop.argtypes = [ctypes.c_void_p]
        self._dgx_lib.DgxStream_push.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._dgx_lib.DgxStream_delete.argtypes = [ctypes.c_void_p]
        self._lib_fully_initialized = False

    @property
    def dgx_stream(self):
        return self._dgx_stream

    @dgx_stream.setter
    def dgx_stream(self, value):
        self._dgx_stream = value

    @property
    def dgx_lib(self):
        return self._dgx_lib

    @property
    def config_type(self):
        return "dgx"

    def shape_dgx_stream(self, packet: Structure):
        self._dgx_lib.DgxStream_pop.restype = POINTER(packet)
        self._dgx_stream = self._dgx_lib.DgxStream_new(
            self.STREAM_TYPE_CPU, ctypes.sizeof(packet), 1
        )
        self._lib_fully_initialized = True

    @property
    def lib_fully_initialized(self):
        return self._lib_fully_initialized
