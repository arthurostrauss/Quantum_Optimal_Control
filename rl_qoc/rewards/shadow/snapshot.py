from typing import List
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli


@dataclass
class Snapshot:
    b: List[int]
    pauli_int: List[int]

    def __post_init__(self):
        if len(self.b) != len(self.pauli):
            raise ValueError("b and pauli must have the same length")

    @property
    def num_qubits(self):
        return len(self.b)
    
    @property
    def bitstring(self):
        """
        Return the bitstring corresponding to |b> in equation S13
        """
        return "".join(map(str, self.b))
    
    @property
    def pauli_string(self):
        """
        Returns Pauli operator as a string for the sampled basis
        """
        return "".join(map(str, self.pauli_int))[::-1]
    
    @property
    def pauli(self):
        return Pauli(self.pauli_string)
    
    @property
    def circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        for i, pauli in enumerate(self.pauli_int):
            if pauli == 0:  # X Pauli: Hadamard basis
                qc.h(i)
            elif pauli == 1:  # Y Pauli: Sdg-H basis
                qc.sdg(i)
                qc.h(i)
            
        return qc
    
    @property
    def U(self):
        return Operator(self.circuit)
                

