from rl_qoc.rewards.reward_data import RewardData, RewardDataList
from dataclasses import dataclass
from typing import List
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike


@dataclass
class PiPulseRewardData(RewardData):
    pub: SamplerPub | SamplerPubLike

    def __post_init__(self):
        # Check if number of qubits is consistent between input circuit and observables
        self.pub = SamplerPub.coerce(self.pub)


@dataclass
class PiPulseRewardDataList(RewardDataList):
    reward_data: List[PiPulseRewardData]
