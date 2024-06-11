from typing import Dict, Sequence
from qiskit.circuit import (
    QuantumCircuit,
    Instruction,
    Qubit,
)
from qiskit.quantum_info import Operator


def get_baseline_fid_from_phi_gamma(param_tuple):
    # prevent key errors with rounding
    param_tuple = (param_tuple[0], round(param_tuple[1], 2))

    if any([param_tuple[0] == 0, param_tuple[1] == 0]):
        return 1.0

    baseline_gate_fidelities = {
        (0.7853981633974483, 0.01): 0.9999845788223948,
        (0.7853981633974483, 0.02): 0.9999383162408302,
        (0.7853981633974483, 0.03): 0.9998612151090003,
        (0.7853981633974483, 0.04): 0.9997532801828659,
        (0.7853981633974483, 0.05): 0.9996145181203613,
        (0.7853981633974483, 0.06): 0.999444937480985,
        (0.7853981633974483, 0.07): 0.9992445487252688,
        (0.7853981633974483, 0.08): 0.9990133642141359,
        (0.7853981633974483, 0.09): 0.9987513982081349,
        (0.7853981633974483, 0.1): 0.9984586668665639,
        (0.7853981633974483, 0.11): 0.9981351882464706,
        (0.7853981633974483, 0.12): 0.9977809823015402,
        (0.7853981633974483, 0.13): 0.9973960708808629,
        (0.7853981633974483, 0.14): 0.9969804777275899,
        (0.7853981633974483, 0.15): 0.9965342284774632,
        (1.5707963267948966, 0.01): 0.9999383162408302,
        (1.5707963267948966, 0.02): 0.9997532801828658,
        (1.5707963267948966, 0.03): 0.9994449374809848,
        (1.5707963267948966, 0.04): 0.9990133642141359,
        (1.5707963267948966, 0.05): 0.9984586668665638,
        (1.5707963267948966, 0.06): 0.9977809823015402,
        (1.5707963267948966, 0.07): 0.9969804777275897,
        (1.5707963267948966, 0.08): 0.9960573506572388,
        (1.5707963267948966, 0.09): 0.9950118288582785,
        (1.5707963267948966, 0.1): 0.9938441702975689,
        (1.5707963267948966, 0.11): 0.9925546630773869,
        (1.5707963267948966, 0.12): 0.9911436253643442,
        (1.5707963267948966, 0.13): 0.9896114053108829,
        (1.5707963267948966, 0.14): 0.9879583809693737,
        (1.5707963267948966, 0.15): 0.9861849601988382,
        (2.356194490192345, 0.01): 0.9998612151090003,
        (2.356194490192345, 0.02): 0.9994449374809851,
        (2.356194490192345, 0.03): 0.998751398208135,
        (2.356194490192345, 0.04): 0.9977809823015402,
        (2.356194490192345, 0.05): 0.9965342284774632,
        (2.356194490192345, 0.06): 0.9950118288582788,
        (2.356194490192345, 0.07): 0.9932146285882479,
        (2.356194490192345, 0.08): 0.9911436253643444,
        (2.356194490192345, 0.09): 0.9887999688823954,
        (2.356194490192345, 0.1): 0.9861849601988384,
        (2.356194490192345, 0.11): 0.9833000510084537,
        (2.356194490192345, 0.12): 0.9801468428384714,
        (2.356194490192345, 0.13): 0.9767270861595005,
        (2.356194490192345, 0.14): 0.9730426794137726,
        (2.356194490192345, 0.15): 0.9690956679612422,
        (3.141592653589793, 0.01): 0.9997532801828659,
        (3.141592653589793, 0.02): 0.9990133642141359,
        (3.141592653589793, 0.03): 0.99778098230154,
        (3.141592653589793, 0.04): 0.9960573506572388,
        (3.141592653589793, 0.05): 0.9938441702975689,
        (3.141592653589793, 0.06): 0.9911436253643443,
        (3.141592653589793, 0.07): 0.9879583809693738,
        (3.141592653589793, 0.08): 0.9842915805643158,
        (3.141592653589793, 0.09): 0.9801468428384714,
        (3.141592653589793, 0.1): 0.9755282581475768,
        (3.141592653589793, 0.11): 0.970440384477113,
        (3.141592653589793, 0.12): 0.9648882429441258,
        (3.141592653589793, 0.13): 0.9588773128419905,
        (3.141592653589793, 0.14): 0.9524135262330098,
        (3.141592653589793, 0.15): 0.9455032620941839,
    }
    return baseline_gate_fidelities[param_tuple]


def create_circuit_from_own_unitaries(circuit_context: QuantumCircuit, **kwargs):
    """
    Generates a new quantum circuit, replacing all gates except the specified target gate with custom unitary gates.

    This function iterates through each gate in the provided `circuit_context`, and for each gate that does not match
    the specified target operation, it extracts its matrix representation and adds a corresponding unitary gate to a new
    quantum circuit. The target operation, if present, is added unchanged to the new circuit. Each non-target gate in the
    new circuit is labeled uniquely to indicate its type, the qubits it acts on, and its occurrence count.

    Parameters:
    - circuit_context (QuantumCircuit): The original quantum circuit from which the gate operations are extracted.
    - **kwargs: Arbitrary keyword arguments. Expected to contain 'target_gate_info' specifying the target operation to leave
      unchanged. This should be a dictionary with 'register' (a list of qubit indices the gate acts on) and 'gate' (the gate
      instruction).

    Returns:
    - QuantumCircuit: A new quantum circuit where non-target gates are replaced by unitary gates and the target gate is
      left as in the original circuit. Each unitary gate is labeled to reflect its origin.
    """
    new_circuit = QuantumCircuit(circuit_context.num_qubits)
    gate_count = {}
    target_gate_instruction = kwargs.get("gate")
    target_gate_qubits = kwargs.get("register", [])

    for inst, qargs, _ in circuit_context.data:
        gate_name = inst.name
        qubits_indices = [circuit_context.find_bit(q).index for q in qargs]

        # Check if the current gate matches the target operation
        if (
            gate_name == target_gate_instruction.name
            and qubits_indices == target_gate_qubits
        ):
            # Add the target gate unchanged to the new circuit
            print("Target gate found")
            new_circuit.append(instruction=inst, qargs=qargs)
        else:
            # Process non-target gates
            gate_label, qubit_indices = process_gate(
                inst, qargs, gate_count, circuit_context
            )
            matrix_representation = Operator(inst).data
            new_circuit.unitary(matrix_representation, qubit_indices, label=gate_label)

    return new_circuit


def process_gate(
    inst: Instruction, qargs: Sequence[Qubit], gate_count: Dict, circuit_context: QuantumCircuit
):
    """
    Processes a gate to generate a unique label and determine qubit indices for the new unitary gate.

    Parameters:
    - inst: The gate instruction from the original circuit.
    - qargs: Qubit arguments for the gate.
    - gate_count: A dictionary tracking the occurrence count of each gate label.
    - circuit_context: The original quantum circuit context.

    Returns:
    - gate_label (str): A unique label for the gate.
    - qubit_indices (tuple or int): The qubit indices for the new unitary gate.
    """
    gate_name = inst.name
    qubits_indices = [
        circuit_context.find_bit(q).index for q in qargs
    ]  # Define qubits_indices here

    if len(qargs) == 2:  # Special handling for two-qubit gates like CNOT
        control_index, target_index = qubits_indices
        gate_label = f"{gate_name}_c{control_index}_t{target_index}"
        qubit_indices = (control_index, target_index)
    else:  # Handling for single-qubit gates and other types of gates
        gate_label = f"{gate_name}_q{'_'.join(map(str, qubits_indices))}"
        qubit_indices = (
            qubits_indices[0] if len(qubits_indices) == 1 else tuple(qubits_indices)
        )

    # Increment gate instance count and update label
    gate_count[gate_label] = gate_count.get(gate_label, 0) + 1
    gate_label += f"_o{gate_count[gate_label]}"  # o for occurrences

    return gate_label, qubit_indices
