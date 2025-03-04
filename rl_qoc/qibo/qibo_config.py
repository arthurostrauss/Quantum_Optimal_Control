from __future__ import annotations
from dataclasses import dataclass, field
from .. import BackendConfig
from ..environment.configuration.backend_config import PubLike, Pub
from typing import Any, List, Tuple, Callable, Iterable, Optional, Union
from ..environment.backend_info import BackendInfo
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike


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

    def process_pubs(self, pubs: Iterable[Pub | PubLike]) -> List[Pub]:
        """
        Process the pub to the correct type for the backend
        """
        new_pubs = []
        for pub in pubs:
            if isinstance(pub, (EstimatorPubLike, EstimatorPub)):
                new_pubs.append(EstimatorPub.coerce(pub))
            elif isinstance(pub, (SamplerPubLike, SamplerPub)):
                new_pubs.append(SamplerPub.coerce(pub))
            else:
                raise ValueError(f"Pub type {type(pub)} not recognized")
        return new_pubs


class QiboBackendInfo(BackendInfo):
    """
    Class to store information on Qibo backend (can also generate some dummy information for the case of no backend)
    """

    def __init__(
        self,
        n_qubits: int = 0,
        coupling_map: Optional[List[Tuple[int, int]]] = None,
        pass_manager=None,
    ):
        """
        Initialize the backend information
        :param n_qubits: Number of qubits for the quantum environment
        :param coupling_map: Coupling map for the quantum environment

        """
        super().__init__(n_qubits, pass_manager)
        self._coupling_map = coupling_map

    @property
    def basis_gates(self):
        return ["rx", "cz", "measure", "rz", "x", "sx"]

    @property
    def dt(self):
        return 1e-9

    @property
    def instruction_durations(self):
        return None

    @property
    def coupling_map(self):
        return (
            CouplingMap(self._coupling_map)
            if self._coupling_map is not None
            else CouplingMap.from_full(self._n_qubits)
        )

    def custom_transpile(
        self, qc_input: Union[QuantumCircuit, List[QuantumCircuit]], *args, **kwargs
    ):
        return (
            qc_input.decompose()
            if isinstance(qc_input, QuantumCircuit)
            else [circ.decompose() for circ in qc_input]
        )

    def asdict(self):
        return {
            "n_qubits": self._n_qubits,
            "coupling_map": self._coupling_map,
            "pass_manager": self._pass_manager,
        }
