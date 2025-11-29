from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from ..reward_data import RewardData, RewardDataList
import numpy as np


@dataclass
class MultiTargetStateRewardData(RewardData):
    """
    Dataclass for storing MultiTarget state reward data.
    Stores per-target observables and ID coefficients while sharing common circuit references.
    """

    pub: EstimatorPub | EstimatorPubLike
    # Common circuit with joint input state preparations
    input_circuit: QuantumCircuit
    n_reps: int
    
    # Per-target data
    target_indices: List[int]  # List of target indices
    target_id_coeffs: List[float]  # ID coefficient for each target
    target_pauli_samplings: List[int]  # Pauli sampling count for each target
    target_observables: List[SparsePauliOp]  # Observables for each target (extended to full circuit)
    target_qubits: List[List[int]]  # Qubit indices for each target
    target_input_indices: List[Tuple[int, ...]]  # Input state indices for each target
    target_observable_indices: List[List[Tuple[int]]]  # Observable indices for each target
    
    # Combined observables (sum of all target observables)
    combined_observables: SparsePauliOp = field(init=False)
    
    def __post_init__(self):
        """Initialize combined observables and validate data."""
        self.pub = EstimatorPub.coerce(self.pub)
        
        # Validate that all lists have the same length
        n_targets = len(self.target_indices)
        if not all(
            len(lst) == n_targets
            for lst in [
                self.target_id_coeffs,
                self.target_pauli_samplings,
                self.target_observables,
                self.target_qubits,
                self.target_input_indices,
                self.target_observable_indices,
            ]
        ):
            raise ValueError("All per-target lists must have the same length")
        
        # Compute combined observables
        if self.target_observables:
            self.combined_observables = sum(self.target_observables).simplify()
        else:
            raise ValueError("At least one target observable must be provided")
    
    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the input circuit."""
        return self.input_circuit.num_qubits
    
    @property
    def num_targets(self) -> int:
        """Return the number of targets."""
        return len(self.target_indices)
    
    @property
    def total_id_coeff(self) -> float:
        """Return the sum of all target ID coefficients."""
        return sum(self.target_id_coeffs)
    
    @property
    def total_pauli_sampling(self) -> int:
        """Return the sum of all target Pauli samplings."""
        return sum(self.target_pauli_samplings)
    
    @property
    def hamiltonian(self) -> SparsePauliOp:
        """
        Return the combined Hamiltonian (sum of all target observables plus identity terms).
        """
        num_qubits = self.input_circuit.num_qubits
        ham = SparsePauliOp("I" * num_qubits, coeffs=self.total_id_coeff)
        ham += self.combined_observables
        return ham.simplify()
    
    @property
    def precision(self) -> float:
        """Return the precision of the estimator."""
        return self.pub.precision
    
    @property
    def total_shots(self) -> int:
        """Return the total number of shots."""
        return (
            self.shots
            * len(self.hamiltonian.group_commuting(True))
            * self.pub.parameter_values.shape[0]
        )
    
    def get_target_hamiltonian(self, target_idx: int) -> SparsePauliOp:
        """
        Get the Hamiltonian for a specific target.
        
        Args:
            target_idx: Index of the target
            
        Returns:
            SparsePauliOp representing the Hamiltonian for this target
        """
        if target_idx not in self.target_indices:
            raise ValueError(f"Target index {target_idx} not found")
        
        idx = self.target_indices.index(target_idx)
        num_qubits = self.input_circuit.num_qubits
        ham = SparsePauliOp("I" * num_qubits, coeffs=self.target_id_coeffs[idx])
        ham += self.target_observables[idx]
        return ham.simplify()


@dataclass
class MultiTargetStateRewardDataList(RewardDataList):
    """
    Dataclass for storing a list of MultiTarget state reward data.
    """

    reward_data: List[MultiTargetStateRewardData]
    
    @property
    def num_targets(self) -> int:
        """Return the number of targets (should be consistent across all reward data)."""
        if not self.reward_data:
            return 0
        return self.reward_data[0].num_targets
    
    @property
    def target_indices(self) -> List[int]:
        """Return the list of target indices."""
        if not self.reward_data:
            return []
        return self.reward_data[0].target_indices
    
    @property
    def shots(self) -> List[int]:
        """Return the number of shots for each reward data."""
        return [data.shots for data in self.reward_data]
    
    @property
    def n_reps(self) -> int:
        """Return the number of repetitions."""
        if not self.reward_data:
            return 1
        return self.reward_data[0].n_reps
    
    @property
    def total_id_coeff(self) -> float:
        """Return the total ID coefficient (sum across all targets)."""
        if not self.reward_data:
            return 0.0
        return self.reward_data[0].total_id_coeff
    
    @property
    def total_pauli_sampling(self) -> int:
        """Return the total Pauli sampling (sum across all targets)."""
        if not self.reward_data:
            return 0
        return self.reward_data[0].total_pauli_sampling
    
    @property
    def hamiltonian(self) -> SparsePauliOp:
        """Return the combined Hamiltonian."""
        if not self.reward_data:
            raise ValueError("No reward data available")
        return self.reward_data[0].hamiltonian
    
    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        if not self.reward_data:
            return 0
        return self.reward_data[0].num_qubits
    
    @property
    def pubs(self) -> List[EstimatorPub]:
        """Return the list of EstimatorPubs."""
        return [EstimatorPub.coerce(data.pub) for data in self.reward_data]
    
    @property
    def total_shots(self) -> int:
        """Return the total number of shots."""
        return sum(data.total_shots for data in self.reward_data)
    
    def get_target_hamiltonians(self, target_idx: int) -> List[SparsePauliOp]:
        """
        Get the Hamiltonians for a specific target across all reward data.
        
        Args:
            target_idx: Index of the target
            
        Returns:
            List of SparsePauliOp representing the Hamiltonians for this target
        """
        return [data.get_target_hamiltonian(target_idx) for data in self.reward_data]
