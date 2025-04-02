from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..reward_data import RewardData, RewardDataList
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import GateTarget, StateTarget
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.circuit import QuantumCircuit


@dataclass
class FidelityRewardData(RewardData):
    pub: SamplerPub | SamplerPubLike

    @property
    def total_shots(self) -> int:
        return 0

    @property
    def shots(self) -> int:
        return 0


@dataclass
class FidelityRewardDataList(RewardDataList):
    reward_data: List[FidelityRewardData]
    env_config: QEnvConfig
    target: GateTarget | StateTarget

    @property
    def pubs(self) -> List[SamplerPub]:
        return [reward_data.pub for reward_data in self.reward_data]

    @property
    def total_shots(self) -> int:
        return 0

    @property
    def shots(self) -> List[int]:
        return [0] * len(self.reward_data)
