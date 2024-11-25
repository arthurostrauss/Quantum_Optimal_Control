from dataclasses import dataclass, field
from ..environment.qconfig import BackendConfig
from typing import Any, List, Tuple, Callable


@dataclass
class QiboConfig(BackendConfig):
    """
    Qibo configuration elements.

    Args:
        backend: Qibo backend
        physical_qubits: Qubit pair to be used for the Qibo platform
        platform: Qibo platform to be used
        coupling_map: Coupling map for the Qibo platform
        n_qubits: Number of qubits for the Qibo platform
        options: Options to feed the Qibo platform
    """

    backend: Any
    physical_qubits: Tuple[int | str, int | str] = (0, 1)
    platform: str = "qibolab"
    coupling_map: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 1)])
    n_qubits: int = 2
    gate_rule: Tuple[str, Callable] | str = "cz"

    @property
    def config_type(self):
        return "qibo"
