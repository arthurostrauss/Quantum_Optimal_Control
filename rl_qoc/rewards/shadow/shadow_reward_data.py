from __future__ import annotations

from ..reward_data import RewardData, RewardDataList
from dataclasses import dataclass
from typing import Optional, Tuple, List
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike


@dataclass
class ShadowRewardData(RewardData):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the Shadow reward.
    """
    #add data for unitary list
    pub: SamplerPub | SamplerPubLike

    unitary: List[List[int]]                                #one reward = one shadow generation = one list of list of random variables
    

    def __post_init__(self):
        # Check if number of qubits is consistent between input circuit and observables
        num_qubits = self.input_circuit.num_qubits
        if self.input_indices is not None and len(self.input_indices) != num_qubits:
            raise ValueError(
                f"Number of qubits in input circuit ({num_qubits}) does not match number of qubits in input indices ({len(self.input_indices)})"
            )
        if (
            self.inverse_circuit is not None
            and self.inverse_circuit.num_qubits != num_qubits
        ):
            raise ValueError(
                f"Number of qubits in input circuit ({num_qubits}) does not match number of qubits in inverse circuit ({self.inverse_circuit.num_qubits})"
            )
        self.pub = SamplerPub.coerce(self.pub)

    @property
    def shots(self) -> int:
        """
        Return the number of shots.S
        """
        return self.pub.shots

    @property
    def causal_cone_size(self) -> int:
        """
        Return the size of the causal cone.
        """
        if self.causal_cone_qubits_indices is not None:
            return len(self.causal_cone_qubits_indices)
        else:
            return 0


@dataclass
class ShadowRewardDataList(RewardDataList):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the Shadow reward.
    """

    reward_data: List[ShadowRewardData]

    def __post_init__(self):
        # Check if all reward data have the same number of qubits
        num_qubits = self.reward_data[0].input_circuit.num_qubits
        n_reps = self.reward_data[0].n_reps
        for reward_data in self.reward_data:
            if reward_data.input_circuit.num_qubits != num_qubits:
                raise ValueError(
                    f"Number of qubits in input circuit ({num_qubits}) does not match number of qubits in reward data ({reward_data.input_circuit.num_qubits})"
                )
            if reward_data.n_reps != n_reps:
                raise ValueError(
                    f"Number of repetitions in input circuit ({n_reps}) does not match number of repetitions in reward data ({reward_data.n_reps})"
                )
        causal_cone_qubits_indices = self.reward_data[0].causal_cone_qubits_indices
        for reward_data in self.reward_data:
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
