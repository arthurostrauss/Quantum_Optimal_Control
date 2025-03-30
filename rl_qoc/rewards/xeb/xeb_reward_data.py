from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike

from ..reward_data import RewardData, RewardDataList


@dataclass
class XEBRewardData(RewardData):
    pub: SamplerPub | SamplerPubLike

    @property
    def shots(self) -> int:
        return self.pub.shots

    @property
    def total_shots(self) -> int:
        return self.pub.shots * self.pub.parameter_values.shape[0]


@dataclass
class XEBRewardDataList(RewardDataList):
    reward_data: List[XEBRewardData]

    @property
    def pubs(self) -> List[SamplerPub]:
        return [reward_data.pub for reward_data in self.reward_data]

    @property
    def total_shots(self) -> int:
        return sum([reward_data.total_shots for reward_data in self.reward_data])

    @property
    def shots(self) -> List[int]:
        return [reward_data.shots for reward_data in self.reward_data]
