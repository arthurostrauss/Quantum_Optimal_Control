from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.quantum_info import Statevector

from ..reward_data import RewardData, RewardDataList
import numpy as np


@dataclass
class XEBRewardData(RewardData):
    pub: SamplerPub | SamplerPubLike
    state: Statevector
    causal_cone_qubit_indices: List[int]

    def __post_init__(self):
        self.pub = SamplerPub.coerce(self.pub)

    @property
    def shots(self) -> int:
        return self.pub.shots

    @property
    def total_shots(self) -> int:
        return self.pub.shots * self.pub.parameter_values.shape[0]

    @property
    def probabilities(self) -> np.array:
        return self.state.probabilities(decimals=6)


@dataclass
class XEBRewardDataList(RewardDataList):
    reward_data: List[XEBRewardData]

    def __post_init__(self):
        causal_cone_indices = self.reward_data[0].causal_cone_qubit_indices
        for reward_data in self.reward_data:
            if reward_data.causal_cone_qubit_indices != causal_cone_indices:
                raise ValueError("Causal cone qubit indices must be the same")

    @property
    def pubs(self) -> List[SamplerPub]:
        return [reward_data.pub for reward_data in self.reward_data]

    @property
    def total_shots(self) -> int:
        return sum([reward_data.total_shots for reward_data in self.reward_data])

    @property
    def shots(self) -> List[int]:
        return [reward_data.shots for reward_data in self.reward_data]

    @property
    def causal_cone_qubit_indices(self) -> List[int]:
        return
