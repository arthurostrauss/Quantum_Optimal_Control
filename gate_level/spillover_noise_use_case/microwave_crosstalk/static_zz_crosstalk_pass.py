from qiskit.circuit.library import RZZGate
from qiskit.transpiler import TransformationPass, InstructionDurations
from qiskit.dagcircuit import DAGCircuit


class StaticZZCrosstalkPass(TransformationPass):
    """
    A pass to transform the circuit to make it compatible with a static ZZ crosstalk model.
    """
    def __init__(self, crosstalk_rate_matrix: dict[tuple[int, int], float], instruction_durations: InstructionDurations):
        """
        Initialize the pass with the coupling map and the crosstalk rate.
        """
        super().__init__()
        self.crosstalk_rate_matrix = crosstalk_rate_matrix
        self.instruction_durations = instruction_durations
    def run(self, dag: DAGCircuit):
        new_dag = DAGCircuit()
        new_dag.add_qubits(dag.qubits)
        new_dag.add_clbits(dag.clbits)

        for layer in dag.layers():
            for node in layer["graph"].op_nodes():
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

            durations = []
            layer_nodes = layer["graph"].op_nodes()
            for node in layer_nodes:
                qubits = node.qargs
                indices = [dag.find_bit(qubit).index for qubit in qubits]
                duration = self.instruction_durations.get(node.op.name, indices)
                durations.append(duration)
            layer_duration = max(durations)
            layer_duration_dt = layer_duration * self.instruction_durations.dt
            
            if layer_duration > 0:
                for (q0, q1), rate in self.crosstalk_rate_matrix.items():
                    if rate != 0.0:
                        duration = layer_duration_dt
                        qubits = [dag.qubits[q0], dag.qubits[q1]]
                        new_dag.apply_operation_back(RZZGate(duration*rate), qubits)
                    
        return new_dag
            
                
                    
        
        