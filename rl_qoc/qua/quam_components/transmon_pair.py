from typing import Dict, Any, Optional, Union
from dataclasses import field

from quam.core import QuamComponent, quam_dataclass
from quam.components import QubitPair
from .transmon import Transmon
from .tunable_coupler import TunableCoupler


__all__ = ["TransmonPair"]


@quam_dataclass
class TransmonPair(QubitPair):
    id: Union[int, str]
    qubit_control: Transmon = None
    qubit_target: Transmon = None
    coupler: Optional[TunableCoupler] = None

    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"
