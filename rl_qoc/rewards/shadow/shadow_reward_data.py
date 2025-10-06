from __future__ import annotations

from ..reward_data import RewardData, RewardDataList
from ...environment.target import StateTarget, GateTarget
from dataclasses import dataclass
from typing import Optional, Tuple, List
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
import numpy as np

@dataclass
class ShadowRewardData(RewardData):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the Shadow reward.

    Args:
        pub: SamplerPub
        u_out: unitary output of the circuit
        u_in: unitary input of the circuit

    """
    #add data for unitary list
    pub: SamplerPub
    unitary: List[int]                            
    u_in: Optional[List[int]]=None
    b_in: Optional[List[int]]=None
    

    def __post_init__(self):
        self.pub = SamplerPub.coerce(self.pub)

@dataclass
class ShadowRewardDataList(RewardDataList):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the Shadow reward.
    """

    reward_data: List[ShadowRewardData]
    target: StateTarget | GateTarget
    
    @property
    def unitaries(self) -> List[List[int]]:
        """
        Return the list of unitaries.
        """        
        return [data.unitary for data in self.reward_data]

    @property
    def shadow_size(self) -> int:
        return sum(int(pub.shots) for pub in self.pubs)
    

    @property
    def unitaries_in(self) -> List[Optional[List[int]]]:
        """
        Return the list of unitaries.
        """        
        return [data.u_in for data in self.reward_data if data.u_in is not None]
    
    @property
    def bitstrings(self) -> List[Optional[List[int]]]:
        """
        Return the list of bitstrings.
        """        
        return [data.b_in for data in self.reward_data if data.b_in is not None]
        """
        Return the list of unitaries.
        """        
        return [data.unitary for data in self.reward_data]