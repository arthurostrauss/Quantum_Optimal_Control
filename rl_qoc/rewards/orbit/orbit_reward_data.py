from dataclasses import dataclass
from ..base_reward import RewardData, RewardDataList
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from typing import List


@dataclass
class ORBITRewardData(RewardData):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the ORBIT reward.
    """

    pub: SamplerPub | SamplerPubLike
    causal_cone_qubits_indices: List[int]
    inverse_circuit: QuantumCircuit
    n_reps: int

    def __post_init__(self):
        # Check if number of qubits is consistent between input circuit and observables
        self.pub = SamplerPub.coerce(self.pub)

    @property
    def shots(self) -> int:
        """
        Return the number of shots.
        """
        return self.pub.shots

    @property
    def total_shots(self) -> int:
        """
        Return the total number of shots.
        """
        return self.pub.shots * self.pub.parameter_values.shape[0]


@dataclass
class ORBITRewardDataList(RewardDataList):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the ORBIT reward.
    """

    reward_data: List[ORBITRewardData]

    def __post_init__(self):
        # Check if all reward data have the same number of qubits
        num_qubits = self.reward_data[0].pub.circuit.num_qubits
        causal_cone_qubits_indices = self.reward_data[0].causal_cone_qubits_indices

        for reward_data in self.reward_data:
            if reward_data.pub.circuit.num_qubits != num_qubits:
                raise ValueError(
                    f"Number of qubits in input circuit ({num_qubits}) does not match number of qubits in reward data ({reward_data.pub.num_qubits})"
                )
            if reward_data.causal_cone_qubits_indices != causal_cone_qubits_indices:
                raise ValueError(
                    f"Causal cone qubits indices in input circuit ({causal_cone_qubits_indices}) does not match number of qubits in reward data ({reward_data.causal_cone_qubits_indices})"
                )

    @property
    def pubs(self) -> List[SamplerPub]:
        """
        Return the list of SamplerPubs.
        """
        return [SamplerPub.coerce(reward_data.pub) for reward_data in self.reward_data]

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
        return sum(reward_data.total_shots for reward_data in self.reward_data)

    @property
    def causal_cone_qubits_indices(self) -> List[int]:
        """
        Return the causal cone qubits indices.
        """
        return self.reward_data[0].causal_cone_qubits_indices
