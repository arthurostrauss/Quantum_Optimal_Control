"""
This file contains the functions to generate a generic spillover noise model for a given quantum circuit.
The example circuit is drawn from a typical layer of single and two qubit gates in variational circuit.
"""

from __future__ import annotations

from typing import Optional, Literal, List, Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter, ParameterVector
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, TransformationPass, PassManager
import qiskit_aer.noise as noise
from qiskit.quantum_info import Operator
from qiskit.circuit.library import (
    get_standard_gate_name_mapping as gate_map,
    UnitaryGate,
)
from qiskit_aer import AerSimulator
from qiskit_aer.backends.backend_utils import BASIS_GATES
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration


def numpy_to_hashable(matrix):
    return tuple(map(tuple, matrix)) if matrix is not None else None


def validate_args(
    num_qubits: int,
    rotation_axes: List[Literal["rx", "ry", "rz"]],
    rotation_angles: List[float] | np.ndarray,
    coupling_map: Optional[CouplingMap] = None,
):
    """
    Validate the arguments passed to the function
    If coupling_map is not provided, a linear coupling map is generated

    Parameters:
    - num_qubits: Number of qubits in the circuit
    - rotation_axes: List of strings specifying the rotation axis for each qubit
    - rotation_angles: List of floats specifying the rotation angle for each qubit
    - coupling_map: Qiskit CouplingMap object that represents the qubit connectivity (default: Line connectivity)
    """

    qubits = list(range(num_qubits))
    num_qubits = len(qubits)
    if not isinstance(rotation_axes, list):
        raise ValueError("rotation_axis must be a list of strings")
    if not all(axis in ["rx", "ry", "rz"] for axis in rotation_axes):
        raise ValueError("rotation_axis must be a list of strings containing only 'rx', 'ry', 'rz'")
    if len(rotation_axes) != num_qubits:
        raise ValueError("rotation_axis must have the same length as num_qubits")
    if (
        not isinstance(rotation_angles, list)
        and not isinstance(rotation_angles, ParameterVector)
        or not all([isinstance(angle, (float, Parameter)) for angle in rotation_angles])
    ):
        try:
            rotation_angles = rotation_angles.tolist()
        except AttributeError:
            raise ValueError("rotation_angles must be a list of floats")
    if len(rotation_angles) != num_qubits:
        raise ValueError("rotation_angles must have the same length as num_qubits")
    if coupling_map is None:
        coupling_map = CouplingMap.from_line(num_qubits, False)
    else:
        if not isinstance(coupling_map, CouplingMap):
            raise ValueError("coupling_map must be a CouplingMap object")
        if coupling_map.size() < num_qubits:
            raise ValueError("coupling_map must have at least num_qubits qubits")

    return qubits, rotation_axes, rotation_angles, coupling_map


from itertools import combinations
from qiskit.transpiler import CouplingMap


def get_parallel_gate_combinations(coupling_map: CouplingMap, direction="forward"):
    """
    Returns combinations of qubit pairs for which two-qubit gates can be applied in parallel,
    ensuring that the maximum number of qubits are entangled in each configuration.

    Parameters:
    - coupling_map: Qiskit CouplingMap object representing the qubit connectivity.
    - direction: 'forward' or 'reverse' to indicate which direction of qubit pairs should be selected.

    Returns:
    - List of valid combinations where the maximum number of qubits are entangled in parallel.
    """
    # Get all possible two-qubit gate pairs
    qubit_pairs = coupling_map.get_edges()

    # Filter pairs based on direction
    filtered_pairs = set()
    for q1, q2 in qubit_pairs:
        if direction == "forward":
            if (q2, q1) not in filtered_pairs:
                filtered_pairs.add((q1, q2))
        elif direction == "reverse":
            if (q1, q2) not in filtered_pairs:
                filtered_pairs.add((q2, q1))

    qubit_pairs = list(filtered_pairs)

    # Collect all valid combinations of parallel gates
    valid_combinations = []

    # Iterate over all possible combinations of pairs
    for r in range(1, len(qubit_pairs) + 1):
        for combo in combinations(qubit_pairs, r):
            # Check if all pairs in the combination can be applied in parallel
            used_qubits = set()
            valid = True
            for pair in combo:
                if pair[0] in used_qubits or pair[1] in used_qubits:
                    valid = False
                    break
                used_qubits.update(pair)

            if valid:
                valid_combinations.append(set(combo))

    # Filter out sub-combinations that are part of larger combinations
    filtered_combinations = []
    for combo in valid_combinations:
        is_subset = any(combo < other for other in valid_combinations)
        if not is_subset:
            filtered_combinations.append(combo)

    return [list(combo) for combo in filtered_combinations]


def circuit_context(
    num_qubits: int,
    rotation_axes: List[Literal["rx", "ry", "rz"]],
    rotation_angles: List[float | Parameter] | ParameterVector,
    coupling_map: Optional[CouplingMap] = None,
    n_layers: int = 1,
):
    """
    Generate a circuit containing a layer of single qubit rotations specified by the user
    and a layer of parallel CNOT gates based on the given coupling map

    Parameters:
    - num_qubits: Number of qubits in the circuit
    - rotation_axes: List of strings specifying the rotation axis for each qubit
    - rotation_angles: List of floats specifying the rotation angle for each qubit
    - coupling_map: Qiskit CouplingMap object that represents the qubit connectivity (default: Line connectivity)

    """

    qubits, rotation_axes, rotation_angles, coupling_map = validate_args(
        num_qubits, rotation_axes, rotation_angles, coupling_map
    )
    qc = QuantumCircuit(num_qubits)
    g_library = gate_map()
    available_combinations = get_parallel_gate_combinations(coupling_map)
    # Sort out the combinations by applying the longest ones first
    available_combinations.sort(key=lambda x: len(x), reverse=True)
    combination_count = 0
    for _ in range(n_layers):
        for qubit, axis, angle in zip(qubits, rotation_axes, rotation_angles):
            rotation_gate = type(g_library[axis])(angle)
            qc.append(rotation_gate, [qubit])

        for edge in available_combinations[combination_count]:
            qc.cx(*edge)
        combination_count = (combination_count + 1) % len(available_combinations)

    return qc


class SpilloverNoiseAerPass(TransformationPass):
    """
    A pass to transform the circuit to make it compatible with a Qiskit Aer built spillover noise model
    This pass essentially looks at every local rotation gate such as rx, ry, rz, and transforms it in a generic n qubit
    operation on which arbitrary spillover noise from any qubit can be applied. To make this transformation more efficient,
    the user can pass in advance the spillover rate matrix which represents the spillover rate between qubits. If one
    contribution of spillover noise is to be applied to a qubit, the corresponding element in the matrix should be non-zero.
    If it is zero, then no spillover noise is applied, and we reduce the number of qubits involved in the generic operation.
    """

    def __init__(self, spillover_rate_matrix: Optional[tuple[tuple[float]]] = None):
        """
        Initialize the pass with the spillover rate matrix if provided

        Parameters:
        - spillover_rate_matrix: 2D array representing the spillover rate between qubits, has to be
            passed as a tuple because of hashing in the Pass (default: None)
        """
        super().__init__()
        self._spillover_rate_matrix = spillover_rate_matrix

    @property
    def spillover_rate_matrix(self):
        return np.array(self._spillover_rate_matrix)

    def run(self, dag: DAGCircuit):
        """
        Run the pass on the input DAG circuit

        Parameters:
        - dag: Input DAG circuit

        Returns:
        - Transformed DAG circuit compatible with a Qiskit Aer built spillover noise model
        """
        new_dag = dag.copy_empty_like()
        for layer in dag.layers():
            for node in layer["graph"].op_nodes():
                if node.name in ["rx", "ry", "rz"]:
                    qubit = node.qargs[0]
                    qubit_index = dag.find_bit(qubit).index
                    angle = node.op.params[0]
                    rotation_gate = type(node.op)(angle)

                    if (
                        self.spillover_rate_matrix is not None
                    ):  # Create a custom multi-qubit operation for qubits connected by spillover noise
                        # Find all qubits that have a non-zero spillover rate with the qubit under consideration
                        involved_qubits_indices = [qubit_index] + np.where(
                            self.spillover_rate_matrix[qubit_index] != 0.0
                        )[0].tolist()

                        involved_qubits = [dag.qubits[q] for q in involved_qubits_indices]
                        gate_op = Operator.from_label(
                            "I" * len(involved_qubits)
                        )  # Identity operator
                        gate_op = gate_op.compose(
                            Operator(rotation_gate),
                            qargs=[
                                0
                            ],  # Always 0 as first qubit of involved qubits list is the one undergoing the rotation
                        )  # Apply the rotation gate to the qubit under consideration

                    else:  # Transform the rotation gate to a generic n-qubit operation
                        involved_qubits = dag.qubits
                        gate_op = Operator.from_label("I" * dag.num_qubits())
                        gate_op = gate_op.compose(Operator(rotation_gate), qargs=[qubit_index])

                    gate_label = f"{node.name}({angle:.2f}, {qubit_index})"
                    new_dag.apply_operation_back(
                        UnitaryGate(gate_op, label=gate_label), qargs=involved_qubits
                    )
                else:
                    new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
        return new_dag


def circuit_context2(
    num_qubits: int,
    rotation_axes: List[Literal["rx", "ry", "rz"]],
    rotation_angles: List[float],
    coupling_map: Optional[CouplingMap] = None,
):
    """
    Generate a circuit containing a layer of single qubit rotations specified by the user
    and a layer of parallel CNOT gates based on the given coupling map

    Parameters:
    - num_qubits: Number of qubits in the circuit
    - rotation_axes: List of strings specifying the rotation axis for each qubit
    - rotation_angles: List of floats specifying the rotation angle for each qubit
    - coupling_map: Qiskit CouplingMap object that represents the qubit connectivity (default: Line connectivity)

    """

    qubits, rotation_axes, rotation_angles, coupling_map = validate_args(
        num_qubits, rotation_axes, rotation_angles, coupling_map
    )
    qc = QuantumCircuit(num_qubits)
    g_library = gate_map()
    for qubit, axis, angle in zip(qubits, rotation_axes, rotation_angles):
        rotation_gate = type(g_library[axis])(angle)
        angle_in_pi = angle / np.pi
        gate_op = Operator.from_label("I" * num_qubits)
        gate_op = gate_op.compose(Operator(rotation_gate), qargs=[qubit])
        gate_label = f"{axis}({angle_in_pi:.2f}π, {qubit})"
        qc.unitary(gate_op, qubits, label=gate_label)

    for edge in get_parallel_gate_combinations(coupling_map)[0]:
        qc.cx(*edge)

    return qc


def create_spillover_noise_model_from_circuit(
    qc: QuantumCircuit, rotation_angles: Dict[int, List[float]], spillover_rate_matrix
):
    """
    Create a spillover noise model based on the given quantum circuit and spillover rate matrix.
    Matrix should be provided such that the element (i, j) represents the spillover rate from qubit i to qubit j.
    The noise model is constructed by applying spillover noise to each gate in the circuit based on the spillover rate matrix
    and the specified rotation angles for each qubit.

    Parameters:
    - qc: QuantumCircuit object containing the circuit to generate the noise model for
    - rotation_angles: List of floats specifying the rotation angle for each qubit
    - spillover_rate_matrix: 2D numpy array representing the spillover rate between qubits

    Returns:
    - NoiseModel object containing the spillover noise model for the given circuit
    """
    noise_model = noise.NoiseModel(
        ["unitary", "rzx", "cx", "u", "h", "x", "s", "z", "rx", "ry", "rz", "sdg"]
    )
    num_qubits = qc.num_qubits
    custom_instructions_counter = {i: 0 for i in range(num_qubits)}
    assert (
        num_qubits == spillover_rate_matrix.shape[0] == spillover_rate_matrix.shape[1]
    ), "Spillover rate matrix must be a square matrix with the same size as the number of qubits"
    assert num_qubits == len(
        rotation_angles
    ), "The number of rotation angles must be equal to the number of qubits in the circuit"
    # Loop through all pairs based on gamma_matrix and check for custom gates
    for instruction in qc.data:
        # Extract gate type, target qubits, and rotation angle (phi)
        gate: Gate = instruction.operation
        if gate.label is not None:  # Check if the gate is a custom gate
            main_qubit_index = int(
                gate.label[-2]
            )  # Extract the main qubit from the gate label (e.g., 'ry(0.25π, 0)')
            gate_name = gate.label[:2]  # Extract gate name from the label (e.g., 'ry')
            gate_type = type(gate_map()[gate_name])  # Get the gate type from the gate name
            instruction_index = custom_instructions_counter[main_qubit_index]
            phi = rotation_angles[main_qubit_index][
                instruction_index
            ]  # Rotation angle of original gate

            custom_instructions_counter[main_qubit_index] += 1
            # Construct the noise operator based  on the gate type
            noise_ops = []
            noise_qubits = [main_qubit_index]
            for j in range(num_qubits):
                if spillover_rate_matrix[main_qubit_index, j] != 0.0:
                    noise_rotation_op = Operator(
                        gate_type(spillover_rate_matrix[main_qubit_index, j] * phi)
                    )
                    noise_ops.append(noise_rotation_op)
                    if j != main_qubit_index:
                        noise_qubits.append(j)
                elif (
                    j == main_qubit_index
                ):  # No noise for the main qubit but has to be included in the noise operator
                    # This is because the noise operator has to be bound to the instruction containing the main qubit and
                    # noisy qubits
                    noise_ops.append(Operator.from_label("I"))
                else:
                    noise_ops.append(None)

            # Construct the noise operator associated to main qubit (which spillovers the rest)
            noise_op = noise_ops[main_qubit_index]
            for j, op in enumerate(noise_ops):
                if op is not None and j != main_qubit_index:
                    noise_op = noise_op.expand(op)

            noise_model.add_quantum_error(
                noise.coherent_unitary_error(noise_op),
                [gate],
                noise_qubits,
            )

    return noise_model


def noisy_backend(
    circuit_context: QuantumCircuit,
    spillover_rate_matrix: np.ndarray,
    coupling_map: Optional[CouplingMap] = None,
    seed_simulator: Optional[int] = None,
):
    """
    Generate a noisy backend object with spillover noise based on the given circuit and spillover rate matrix

    Parameters:

    - circuit_context: QuantumCircuit object containing the circuit to generate the noise model for
        (should be generated using circuit_context function)
    - spillover_rate_matrix: 2D numpy array representing the spillover rate between qubits
    - coupling_map: Qiskit CouplingMap object that represents the qubit connectivity (default: Line connectivity)

    Returns:
    - AerSimulator object with the spillover noise model applied and the specified coupling map

    """
    rotation_angles = {i: [] for i in range(circuit_context.num_qubits)}
    for instruction in circuit_context.data:
        if instruction.operation.name in ["rx", "ry", "rz"]:
            qubit_index = circuit_context.find_bit(instruction.qubits[0]).index
            rotation_angles[qubit_index].append(instruction.operation.params[0])

    pm = PassManager([SpilloverNoiseAerPass(numpy_to_hashable(spillover_rate_matrix))])
    qc = pm.run(circuit_context)
    noise_model = create_spillover_noise_model_from_circuit(
        qc, rotation_angles, spillover_rate_matrix
    )
    if coupling_map is None:
        coupling_map = CouplingMap.from_line(circuit_context.num_qubits, False)
    config = AerBackendConfiguration(
        "custom_spillover_impact_simulator",
        "2",
        circuit_context.num_qubits,
        BASIS_GATES["automatic"],
        [],
        int(1e7),
        list(coupling_map.get_edges()),
        description="Custom simulator with spillover noise model",
        custom_instructions=AerSimulator._CUSTOM_INSTR["automatic"],
        simulator=True,
    )

    backend = AerSimulator(
        noise_model=noise_model, configuration=config, seed_simulator=seed_simulator
    )

    return backend
