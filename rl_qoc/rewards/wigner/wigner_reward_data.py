from typing import List
from dataclasses import dataclass
from ..reward_data import RewardData, RewardDataList
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.quantum_info import Statevector


@dataclass
class WignerRewardData(RewardData):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the Wigner reward.
    """

    pub: SamplerPub | SamplerPubLike
    state: Statevector
    causal_cone_qubits_indices: List[int]

@dataclass
class WignerRewardDataList(RewardDataList):
    """
    List of WignerRewardData objects.
    """

    data: List[WignerRewardData]