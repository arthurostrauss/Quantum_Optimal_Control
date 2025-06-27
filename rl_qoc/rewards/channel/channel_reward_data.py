from __future__ import annotations

from typing import List, Tuple, Optional
from dataclasses import dataclass
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from ..reward_data import RewardData, RewardDataList
from ...helpers import precision_to_shots


@dataclass
class ChannelRewardData(RewardData):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the channel reward.
    """

    pub: EstimatorPub | EstimatorPubLike
    input_circuit: QuantumCircuit
    observables: SparsePauliOp | List[SparsePauliOp]
    n_reps: int
    causal_cone_qubits_indices: List[int]
    input_pauli: Optional[Pauli|PauliList] = None
    input_indices: Optional[Tuple[int]] = None
    observables_indices: Optional[List[Tuple[int]]] = None

    def __post_init__(self):
        # Check if number of qubits is consistent between input circuit and observables
        self.pub = EstimatorPub.coerce(self.pub)

    @property
    def num_qubits(self) -> int:
        """
        Return the number of qubits in the input circuit.
        """
        return len(self.causal_cone_qubits_indices)

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """
        Return the Hamiltonian to be estimated.
        """
        return (
            SparsePauliOp.sum(self.observables).simplify()
            if isinstance(self.observables, List)
            else self.observables.simplify()
        )

    @property
    def total_shots(self) -> int:
        """
        Return the total number of shots.
        """
        return (
            self.shots
            * len(self.hamiltonian.group_commuting(True))
            * self.pub.parameter_values.shape[0]
        )

    @property
    def fiducials_indices(self) -> Optional[Tuple[Tuple[int], List[Tuple[int]]]]:
        """
        Return the fiducial indices.
        """
        if self.input_indices is not None:
            return self.input_indices, self.observables_indices
        else:
            return None

    @property
    def fiducials(self) -> Tuple[Pauli, SparsePauliOp]:
        """
        Return the fiducials.
        """
        return self.input_pauli, self.hamiltonian

    @property
    def full_fiducials(self) -> Tuple[QuantumCircuit, SparsePauliOp]:
        """
        Return the full fiducials, expressed as the sampled Pauli eigenstate
        and the Hamiltonian.
        """
        return self.input_circuit, self.hamiltonian

    @property
    def precision(self) -> float:
        """
        Return the precision of the estimator.
        """
        return self.pub.precision

    @property
    def pauli_eigenstate(self) -> str:
        """
        Get a string representation of the pauli eigenstate.
        """
        state = "|"
        if self.input_pauli is not None and self.input_indices is not None:
            input_indices = [self.input_indices[i] for i in self.causal_cone_qubits_indices]
            if isinstance(self.input_pauli, Pauli):
                input_pauli = self.input_pauli
            else:
                import numpy as np
                input_pauli = Pauli((np.logical_or.reduce(self.input_pauli.z), np.logical_or.reduce(self.input_pauli.x)))
            for i, term in enumerate(reversed(input_pauli.to_label())):
                if term == "Z" or term == "I":
                    if input_indices[i] % 2 == 0:
                        state += "0"
                    else:
                        state += "1"
                elif term == "X":
                    if input_indices[i] % 2 == 0:
                        state += "+"
                    else:
                        state += "-"
                else:
                    if input_indices[i] % 2 == 0:
                        state += "+i"
                    else:
                        state += "-i"
        state += ">"
        return state

    @property
    def pauli_rep(self) -> Pauli:
        """
        Return the Pauli representative of the commuting group formed by the input PauliList.
        """
        if isinstance(self.input_pauli, PauliList):
            import numpy as np
            return Pauli((np.logical_or.reduce(self.input_pauli.z), np.logical_or.reduce(self.input_pauli.x)))
        elif isinstance(self.input_pauli, Pauli):
            return self.input_pauli
        else:
            raise TypeError("input_pauli not set or not of type Pauli or PauliList")



@dataclass
class ChannelRewardDataList(RewardDataList):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the channel reward.
    """

    reward_data: List[ChannelRewardData]
    pauli_sampling: int
    id_coeff: float

    def __post_init__(self):
        # Check if all reward data have the same number of qubits
        num_qubits = self.reward_data[0].num_qubits
        n_reps = self.reward_data[0].n_reps
        for reward_data in self.reward_data:
            if reward_data.num_qubits != num_qubits:
                raise ValueError(
                    f"Number of qubits in input circuit ({num_qubits}) does not match number of qubits in reward data ({reward_data.input_circuit.num_qubits})"
                )
            if reward_data.n_reps != n_reps:
                raise ValueError(
                    f"Number of repetitions in input circuit ({n_reps}) does not match number of repetitions in reward data ({reward_data.n_reps})"
                )

    @property
    def num_qubits(self) -> int:
        """
        Return the number of qubits in the input circuit.
        """
        return self.reward_data[0].num_qubits

    @property
    def pubs(self) -> List[EstimatorPub]:
        """
        Return the list of EstimatorPubs.
        """
        return [EstimatorPub.coerce(reward_data.pub) for reward_data in self.reward_data]

    @property
    def shots(self) -> List[int]:
        """
        Return the number of shots.
        """
        return [reward_data.shots for reward_data in self.reward_data]

    @property
    def n_reps(self) -> int:
        """
        Return the number of repetitions.
        """
        return self.reward_data[0].n_reps

    @property
    def total_shots(self) -> int:
        """
        Return the total number of shots.
        """
        return sum(reward_data.total_shots for reward_data in self.reward_data)

    @property
    def hamiltonian(self) -> List[SparsePauliOp]:
        """
        Return the Hamiltonian to be estimated.
        """
        ham = [SparsePauliOp("I" * self.num_qubits, coeffs=self.id_coeff)]
        for reward_data in self.reward_data:
            if isinstance(reward_data.observables, List):
                ham += reward_data.observables
            else:
                ham.append(reward_data.observables)
        return ham

    @property
    def observables(self) -> List[SparsePauliOp]:
        """
        Return the observables.
        """
        return [reward_data.observables for reward_data in self.reward_data]

    @property
    def input_paulis(self) -> List[Pauli]:
        """
        Return the input Pauli operators.
        """
        return [reward_data.input_pauli for reward_data in self.reward_data]

    @property
    def input_circuits(self) -> List[QuantumCircuit]:
        """
        Return the input circuits.
        """
        return [reward_data.input_circuit for reward_data in self.reward_data]

    @property
    def pauli_eigenstates(self) -> List[str]:
        """
        Return the Pauli eigenstates.
        """
        return [reward_data.pauli_eigenstate for reward_data in self.reward_data]

    @property
    def input_indices(self) -> List[Tuple[int]]:
        """
        Return the input indices.
        """
        return [reward_data.input_indices for reward_data in self.reward_data]

    @property
    def observables_indices(self) -> List[List[Tuple[int]]]:
        """
        Return the observables indices.
        """
        return [reward_data.observables_indices for reward_data in self.reward_data]

    @property
    def fiducials_indices(self) -> List[Tuple[Tuple[int], List[Tuple[int]]]]:
        """
        Return the fiducial indices.
        """
        return [reward_data.fiducials_indices for reward_data in self.reward_data]

    @property
    def fiducials(self) -> List[Tuple[Pauli, SparsePauliOp]]:
        """
        Return the fiducials.
        """
        return [reward_data.fiducials for reward_data in self.reward_data]

    @property
    def full_fiducials(self) -> List[Tuple[QuantumCircuit, SparsePauliOp]]:
        """
        Return the full fiducials, expressed as the sampled Pauli eigenstate
        and the Hamiltonian.
        """
        return [reward_data.full_fiducials for reward_data in self.reward_data]

    def __repr__(self):
        """
        Return a string representation of the reward data.
        """
        return f"ChannelRewardDataList({self.reward_data})"

    def __str__(self):
        """
        Return a string representation of the reward data.
        """
        return f"ChannelRewardDataList({self.reward_data})"
