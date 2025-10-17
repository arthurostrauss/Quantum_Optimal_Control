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

    def circuit_single(self, pauli):
        qc = QuantumCircuit(1)

        if pauli == 0:  # X Pauli: Hadamard basis
            qc.h(0)
        elif pauli == 1:  # Y Pauli: Sdg-H basis
            qc.sdg(0)
            qc.h(0)
        return qc
    
    @property
    def unitary_single(self):
        return [Operator(self.circuit_single(pauli)) for pauli in self.pauli_int]


@dataclass
class SnapshotList:
    snapshots: List[Snapshot]
    """
    Builds a list of snapshots from many unitary operators and their corresponding bitstrings.
    Creates object that shows the nth qubit bit and pauli.
    """

    def __post_init__(self):
        self.snapshots = []
        self.bitstring_no = bitstring_no
        self.unitary_no = unitary_no

    def add_snapshot(self, snapshot: Snapshot):
        self.snapshots.append(snapshot)

    def get_snapshot(self, index: int) -> Snapshot:
        return self.snapshots[index]
    @property
    def num_snapshots(self) -> int:
        return len(self.snapshots)

    def bit_n(self, bitstring_no):
        return [snapshot.b[bitstring_no] for snapshot in self.snapshots]

    def unitary_n(self, unitary_no):
        return [snapshot.unitary for snapshot in self.snapshots]