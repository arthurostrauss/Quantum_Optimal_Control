from dataclasses import dataclass
from ..reward_data import RewardData, RewardDataList
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from typing import Optional, List, Tuple


@dataclass
class StateRewardData(RewardData):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the state reward.
    """

    pub: EstimatorPub | EstimatorPubLike
    pauli_sampling: int
    id_coeff: float
    input_circuit: QuantumCircuit
    observables: SparsePauliOp | List[SparsePauliOp]
    shots: int
    n_reps: int
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
        return self.input_circuit.num_qubits

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """
        Return the Hamiltonian to be estimated.
        """
        num_qubits = self.input_circuit.num_qubits
        if isinstance(self.observables, SparsePauliOp):
            ham = (
                SparsePauliOp("I" * num_qubits, coeffs=self.id_coeff) + self.observables
            )
        else:
            ham = SparsePauliOp("I" * num_qubits, coeffs=self.id_coeff)
            ham += sum(self.observables)
        return ham.simplify()

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
    def precision(self) -> float:
        """
        Return the precision of the estimator.
        """
        return self.pub.precision

    @property
    def fiducials(self) -> Tuple[QuantumCircuit, SparsePauliOp]:
        """
        Return the fiducials.
        """
        return self.input_circuit, self.hamiltonian


@dataclass
class StateRewardDataList(RewardDataList):
    """
    Dataclass enabling the storage and tracking of all the items that will be used to compute the state reward.
    """

    reward_data: List[StateRewardData]

    @property
    def shots(self) -> int:
        """
        Return the number of shots.
        """
        return self.reward_data[0].shots

    @property
    def n_reps(self) -> int:
        """
        Return the number of repetitions.
        """
        return self.reward_data[0].n_reps

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """
        Return the Hamiltonian to be estimated.
        """
        return sum(
            [reward_data.hamiltonian for reward_data in self.reward_data]
        ).simplify()

    @property
    def pauli_sampling(self) -> int:
        """
        Return the number of Pauli samples.
        """
        return self.reward_data[0].pauli_sampling

    @property
    def id_coeff(self) -> float:
        """
        Return the identity coefficient.
        """
        return self.reward_data[0].id_coeff

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
        return [
            EstimatorPub.coerce(reward_data.pub) for reward_data in self.reward_data
        ]

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
    def fiducials(self) -> List[Tuple[QuantumCircuit, SparsePauliOp]]:
        """
        Return the fiducials.
        """
        return [reward_data.fiducials for reward_data in self.reward_data]

    def __repr__(self):
        """
        Return a string representation of the reward data.
        """
        return f"StateRewardDataList({self.reward_data})"

    def __str__(self):
        """
        Return a string representation of the reward data.
        """
        return f"StateRewardDataList({self.reward_data})"
