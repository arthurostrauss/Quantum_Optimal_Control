from __future__ import annotations
from dataclasses import dataclass, field
from ..environment.qconfig import BackendConfig, PubLike, Pub
from typing import Any, List, Tuple, Callable, Iterable
from  ..environment.calibration_pubs import (CalibrationEstimatorPub, CalibrationEstimatorPubLike,
                                             CalibrationSamplerPub, CalibrationSamplerPubLike)
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
    
    def process_pubs(self, pubs: Iterable[Pub|PubLike]) -> List[Pub]:
        """
        Process the pub to the correct type for the backend
        """
        new_pubs = []
        for pub in pubs:
            if isinstance(pub, (CalibrationEstimatorPubLike, CalibrationEstimatorPub)):
                new_pubs.extend(CalibrationEstimatorPub.coerce(pub).to_pub_list())
            elif isinstance(pub, (CalibrationSamplerPubLike, CalibrationSamplerPub)):
                new_pubs.extend(CalibrationSamplerPub.coerce(pub).to_pub_list())
            elif isinstance(pub, (EstimatorPubLike, EstimatorPub)):
                new_pubs.append(EstimatorPub.coerce(pub))
            elif isinstance(pub, (SamplerPubLike, SamplerPub)):
                new_pubs.append(SamplerPub.coerce(pub))
            else:
                raise ValueError(f"Pub type {type(pub)} not recognized")
        return new_pubs
