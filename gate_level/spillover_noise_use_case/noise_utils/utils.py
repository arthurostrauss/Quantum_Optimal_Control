from typing import Dict, Sequence
from qiskit.circuit import (
    QuantumCircuit,
    Instruction,
    Qubit,
)
from qiskit.quantum_info import Operator


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
    inst: Instruction,
    qargs: Sequence[Qubit],
    gate_count: Dict,
    circuit_context: QuantumCircuit,
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
