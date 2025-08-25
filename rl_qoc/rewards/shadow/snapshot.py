from typing import List
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli


@dataclass
class Snapshot:
    b: List[int]    # takes input data made of one snapshot - one bitstring and one pauli string
    pauli_int: List[int]    # note that we are already assuming the pauli_int is reversed, i.e. in little endian form

    """
    for convenience, listed below all the properties in this class
    num_qubits: number of qubits in bitstring/shadow
    pauli_string: converts list of Paulis into a string of letters eg [1,2,3,0] -> "XYZI"
    pauli: returns Pauli object of pauli_string which can be used directly into qiskit simulation - this may not be needed
    circuit: returns circuit that has undergone evolution as per pauli_int
    unitary: returns actual unitary that acts on the bitstring b in operator form

    
    """

    def __post_init__(self):
        if len(self.b) != len(self.pauli_int):
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
        mapping = {0: 'X', 1: 'Y', 2: 'Z', 3: 'I'}
        return ''.join(mapping[i] for i in self.pauli_int)
   
    
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
    def unitary(self):
        return Operator(self.circuit)
                

@dataclass
class SnapshotList:
    pass
            