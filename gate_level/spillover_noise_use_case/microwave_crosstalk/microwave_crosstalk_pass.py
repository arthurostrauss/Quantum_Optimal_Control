from typing import Callable, List, Optional
from qiskit.transpiler import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Gate, Parameter, Qubit
from qiskit.circuit.library import (
    get_standard_gate_name_mapping as gate_map,
    UnitaryGate,
    RXGate,
    RYGate,
    RZGate,
)
from qiskit_aer.noise import coherent_unitary_error
from qiskit.quantum_info import Operator
import numpy as np

GateParamDependence = Callable[[Gate, List[Qubit]], List[Parameter | float]]

class MicrowaveCrosstalkPass(TransformationPass):
    """
    A pass to transform the circuit to make it compatible with a Qiskit Aer built microwave crosstalk model.
    """
    def __init__(self, crosstalk_rate_matrix: tuple[tuple[float]], gate_param_dependence: Optional[GateParamDependence] = None):
        """
        Initialize the pass with the crosstalk rate matrix and the gate parameter dependence.
        Parameters:
        - crosstalk_rate_matrix: 2D or 3D array representing the crosstalk rate between qubits, has to be
            passed as a tuple because of hashing in the Pass. The first two dimensions are the qubits, the third dimension is the list of crosstalk rates associated with coherent errors on all X, Y and Z axes.
            If the matrix is 2D, it is assumed that only X-axis crosstalk is present.
            The crosstalk matrix item [i, j] represents the crosstalk rate from qubit i to qubit j.
            The crosstalk matrix item [i, j, k] represents the crosstalk rate from qubit i to qubit j for the k-th coherent error axis (X, Y, Z).
        - gate_param_dependence: Function that takes a gate and returns a list of parameters or floats that depend on the gate parameter (could be used to represent complex angle dependencies of the crosstalk)
        """
        super().__init__()
        self.crosstalk_rate_matrix = np.array(crosstalk_rate_matrix)
        if self.crosstalk_rate_matrix.ndim == 2:
            zeros = np.zeros(self.crosstalk_rate_matrix.shape + (3,))
            zeros[...,0] = self.crosstalk_rate_matrix
            self.crosstalk_rate_matrix = zeros
            print("Crosstalk rate matrix is 2D, adding a new dimension")
            print(self.crosstalk_rate_matrix.shape)
        assert self.crosstalk_rate_matrix.shape[2] == 3, "Crosstalk rate matrix length on the third dimension must be 3"
        # Check that the crosstalk rate matrix is a square matrix on the first two dimensions
        assert self.crosstalk_rate_matrix.shape[0] == self.crosstalk_rate_matrix.shape[1], "Crosstalk rate matrix must be a square matrix on the first two dimensions"
        if gate_param_dependence is None:
            gate_param_dependence = lambda gate: [gate.params[0]] * 3
        self.gate_param_dependence = gate_param_dependence

    def run(self, dag: DAGCircuit):
        new_dag = DAGCircuit()
        new_dag.add_qubits(dag.qubits)
        new_dag.add_clbits(dag.clbits)
        num_qubits = dag.num_qubits()
        if num_qubits != self.crosstalk_rate_matrix.shape[0]:
            raise ValueError(f"Number of qubits in the circuit ({dag.num_qubits}) does not match the number of qubits in the crosstalk rate matrix ({self.crosstalk_rate_matrix.shape[0]})")
        gate_map_ = gate_map()
        for layer in dag.layers():
            for node in layer["graph"].op_nodes():
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            for node in layer["graph"].op_nodes():
                if node.name in ["rx", "ry", "rz"]:
                    qubit = node.qargs[0]
                    qubit_index = dag.find_bit(qubit).index
                    crosstalk_rates = self.crosstalk_rate_matrix[qubit_index, :, :]
                    gate_param_dependence = self.gate_param_dependence(node.op)
                    assert len(gate_param_dependence) == 3, "Gate parameter dependence must return a list of 3 elements"
                    
                    for j in range(num_qubits):
                        target_qubit_index = j
                        target_qubit = dag.qubits[target_qubit_index]
                        noisy_ops = [type(gate_map_[gate])(gate_param_dependence[k] * crosstalk_rates[j, k], label=f"{gate}_{qubit_index}_{target_qubit_index}")
                        if crosstalk_rates[j, k] != 0.0 else None for k, gate in enumerate(["rx", "ry", "rz"])] 
                        for noisy_op in noisy_ops:
                            if noisy_op is not None:
                                new_dag.apply_operation_back(noisy_op, [target_qubit])
        return new_dag
                        
