from __future__ import annotations

from ..reward_data import RewardData, RewardDataList
from dataclasses import dataclass
from typing import Optional, Tuple, List
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
import numpy as np

@dataclass
class ShadowRewardData(RewardData):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the Shadow reward.
    """
    #add data for unitary list
    pub: SamplerPub | SamplerPubLike
    unitary: List[int]                             #one reward = one shadow generation = one list of list of random variables
    

    def __post_init__(self):
        # Check if number of qubits is consistent between input circuit and observables
        
        self.pub = SamplerPub.coerce(self.pub)

    @property
    def shots(self) -> int:
        """
        Return the number of shots
        """
        return self.pub.shots



@dataclass
class ShadowRewardDataList(RewardDataList):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the Shadow reward.
    """

    reward_data: List[ShadowRewardData]

    def __post_init__(self):
        pass
        

    @property
    def pubs(self) -> List[SamplerPub]:
        """
        Return the list of SamplerPubs.
        """
        return [SamplerPub.coerce(reward_data.pub) for reward_data in self.reward_data]
    
    @property
    def unitaries(self) -> List[List[int]]:
        """
        Return the list of unitaries.
        """        
        return [data.unitary for data in self.reward_data]

    @property
    def shadow_size(self):
        return np.sum([pub.shots for pub in self.pubs])
    
    @property
    def causal_cone_qubits_indices(self) -> List[int]:
        """
        Return the causal cone qubits indices.
        """
        return self.reward_data[0].causal_cone_qubits_indices

    @property
    def causal_cone_size(self) -> int:
        """
        Return the size of the causal cone.
        """
        return self.reward_data[0].causal_cone_size

    @property
    def shots(self) -> List[int]:
        """
        Return the number of shots.
        """
        return [reward_data.shots for reward_data in self.reward_data]

    @property
    def total_shots(self) -> int:
        """
        Return the total number of shots.
        """
        return sum(
            reward_data.shots * reward_data.pub.parameter_values.shape[0]
            for reward_data in self.reward_data
        )

    @property
    def n_reps(self) -> int:
        """
        Return the number of repetitions.
        """
        return self.reward_data[0].n_reps

    @property
    def input_indices(self) -> List[Tuple[int]]:
        """
        Return the input indices.
        """
        return [reward_data.input_indices for reward_data in self.reward_data]

    @property
    def inverse_circuits(self) -> List[QuantumCircuit]:
        """
        Return the inverse circuits.
        """
        return [reward_data.inverse_circuit for reward_data in self.reward_data]

    @property
    def input_circuits(self) -> List[QuantumCircuit]:
        """
        Return the input circuits.
        """
        return [reward_data.input_circuit for reward_data in self.reward_data]
