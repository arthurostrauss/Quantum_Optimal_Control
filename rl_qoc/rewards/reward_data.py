from __future__ import annotations

from dataclasses import dataclass

from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.circuit import QuantumCircuit
from typing import List, Union
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike

from ..helpers import precision_to_shots

Pub = Union[SamplerPub, EstimatorPub]
PubLike = Union[SamplerPubLike, EstimatorPubLike]
EstimatorPubs = List[EstimatorPub]
SamplerPubs = List[SamplerPub]
Pubs = List[Pub]


@dataclass
class RewardData:
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the reward.
    """

    pub: Pub | PubLike

    @property
    def parameter_values(self) -> BindingsArray:
        """
        Return the parameter values.
        """
        return self.pub.parameter_values

    @property
    def circuit(self) -> QuantumCircuit:
        return self.pub.circuit

    @property
    def total_shots(self) -> int:
        """
        Return the total number of shots.
        """
        return self.pub.shots * self.pub.parameter_values.shape[0]

    @property
    def shots(self) -> int:
        """
        Return the number of shots.
        """
        if isinstance(self.pub, SamplerPub):
            return self.pub.shots
        elif isinstance(self.pub, EstimatorPub):
            return precision_to_shots(self.pub.precision)
        return 0


@dataclass
class RewardDataList:
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the reward.
    """

    reward_data: List[RewardData]

    @property
    def pubs(self) -> List[Pub]:
        """
        Return the list of EstimatorPubs.
        """
        return [reward_data.pub for reward_data in self.reward_data]

    @property
    def total_shots(self) -> int:
        """
        Return the total number of shots across all reward data.
        """
        return sum(reward_data.total_shots for reward_data in self.reward_data)

    def __len__(self):
        """
        Return the number of reward data.
        """
        return len(self.reward_data)

    def __getitem__(self, index):
        """
        Return the reward data at the given index.
        """
        return self.reward_data[index]

    def __iter__(self):
        """
        Return an iterator over the reward data.
        """
        return iter(self.reward_data)

    def __repr__(self):
        """
        Return a string representation of the reward data.
        """
        return f"RewardDataList({self.reward_data})"

    def __str__(self):
        """
        Return a string representation of the reward data.
        """
        return f"RewardDataList({self.reward_data})"
