"""
This file contains the functions to generate a generic spillover noise model for a given quantum circuit.
The example circuit is drawn from a typical layer of single and two qubit gates in variational circuit.
"""

from itertools import combinations
from typing import Optional, Literal, List

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
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration


def numpy_to_hashable(matrix):
    return tuple(map(tuple, matrix)) if matrix is not None else None


def validate_args(
    num_qubits: int,
    rotation_axes: List[Literal["rx", "ry", "rz"]],
    rotation_angles: List[float],
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
        raise ValueError(
            "rotation_axis must be a list of strings containing only 'rx', 'ry', 'rz'"
        )
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
        coupling_map = CouplingMap.from_line(num_qubits, True)
    else:
        if not isinstance(coupling_map, CouplingMap):
            raise ValueError("coupling_map must be a CouplingMap object")
        if coupling_map.size() < num_qubits:
            raise ValueError("coupling_map must have at least num_qubits qubits")

    return qubits, rotation_axes, rotation_angles, coupling_map


def get_parallel_gate_combinations(coupling_map: CouplingMap, direction="forward"):
    """
    Returns all possible combinations of qubit pairs for which a two-qubit gate can be applied in parallel,
    respecting the specified direction constraint.

    Parameters:
    - coupling_map: Qiskit CouplingMap object that represents the qubit connectivity.
    - direction: 'forward' or 'reverse' to indicate which direction of qubit pairs should be selected.

    Returns:
    - List of combinations where the maximum number of two-qubit gates can be applied in parallel.
    """
    # Get all possible two-qubit gate pairs
    qubit_pairs = coupling_map.get_edges()

    # Create a set to store unique pairs in the specified direction
    filtered_pairs = set()
    for q1, q2 in qubit_pairs:
        if direction == "forward":
            # Add the pair if it's in the forward direction
            if (q2, q1) not in filtered_pairs:
                filtered_pairs.add((q1, q2))
        elif direction == "reverse":
            # Add the reversed pair if the original forward pair exists
            if (q1, q2) not in filtered_pairs:
                filtered_pairs.add((q2, q1))

    # Convert the set back to a list for further processing
    qubit_pairs = list(filtered_pairs)

    max_parallel_combinations = []
    max_num_parallel_gates = 0

    # Check all possible combinations of the qubit pairs
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
                if len(combo) > max_num_parallel_gates:
                    max_num_parallel_gates = len(combo)
                    max_parallel_combinations = [combo]
                elif len(combo) == max_num_parallel_gates:
                    max_parallel_combinations.append(combo)

    return max_parallel_combinations


def circuit_context(
    num_qubits: int,
    rotation_axes: List[Literal["rx", "ry", "rz"]],
    rotation_angles: List[float | Parameter] | ParameterVector,
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
        qc.append(rotation_gate, [qubit])

    for edge in get_parallel_gate_combinations(coupling_map)[0]:
        qc.cx(*edge)

    return qc


class LocalSpilloverNoiseAerPass(TransformationPass):
    """
    A pass to transform the circuit to make it compatible with a Qiskit Aer built spillover noise model
    This pass essentially looks at every local rotation gate such as rx, ry, rz, and transforms it in a generic n qubit
    operation on which arbitrary spillover noise from any qubit can be applied. To make this transformation more efficient,
    the user can pass in advance the spillover rate matrix which represents the spillover rate between qubits. If one
    contribution of spillover noise is to be applied to a qubit, the corresponding element in the matrix should be non-zero.
    If it is zero, then no spillover noise is applied, and we reduce the number of qubits involved in the generic operation.
    """

    def __init__(
        self,
        spillover_rate_matrix: Optional[tuple[tuple[float]]] = None,
        target_subsystem: tuple = None,
    ):
        """
        Initialize the pass with the spillover rate matrix if provided

        Parameters:
        - spillover_rate_matrix: 2D array representing the spillover rate between qubits, has to be
            passed as a tuple because of hashing in the Pass (default: None)
        """
        super().__init__()

        assert all(
            [isinstance(q, int) for q in target_subsystem]
        ), "target_subsystem must be a tuple of integers"
        assert all(
            [q < len(spillover_rate_matrix) for q in target_subsystem]
        ), "target_subsystem must be within the total number of qubits"
        self._spillover_rate_matrix = spillover_rate_matrix
        self._target_subsystem = target_subsystem

    @property
    def spillover_rate_matrix(self):
        return np.array(self._spillover_rate_matrix)

    @property
    def target_subsystem(self):
        return self._target_subsystem

    def run(self, dag: DAGCircuit):
        """
        Run the pass on the input DAG circuit

        Parameters:
        - dag: Input DAG circuit

        Returns:
        - Transformed DAG circuit compatible with a Qiskit Aer built spillover noise model
        """
        new_dag = DAGCircuit()
        new_dag.add_qubits([dag.qubits[q] for q in self.target_subsystem])
        for node in dag.op_nodes():
            if node.name in ["rx", "ry", "rz"]:
                if node.qargs[0] not in new_dag.qubits:
                    continue
                else:
                    qubit = node.qargs[0]
                    qubit_index = dag.find_bit(qubit).index
                    angle = node.op.params[0]
                    rotation_gate = type(node.op)(angle)
                    if self.spillover_rate_matrix is not None:
                        involved_qubits_indices = [qubit_index]
                        for other_subsystem_qubit in new_dag.qubits:
                            if other_subsystem_qubit != qubit:
                                other_index = dag.find_bit(other_subsystem_qubit).index
                                if (
                                    self.spillover_rate_matrix[qubit_index, other_index]
                                    != 0.0
                                ):
                                    involved_qubits_indices.append(other_index)

                        involved_qubits = [
                            dag.qubits[i] for i in involved_qubits_indices
                        ]

                        gate_op = Operator.from_label("I" * len(involved_qubits))
                        gate_op = gate_op.compose(
                            Operator(rotation_gate),
                            qargs=[involved_qubits_indices.index(qubit_index)],
                        )

                        gate_label = f"{node.name}({angle:.2f}, {qubit_index})"
                        new_dag.apply_operation_back(
                            UnitaryGate(gate_op, label=gate_label),
                            qargs=involved_qubits,
                        )
            elif all([q in new_dag.qubits for q in node.qargs]):
                new_dag.apply_operation_back(node.op, qargs=node.qargs)
        return new_dag


def create_spillover_noise_model_from_circuit(
    qc: QuantumCircuit,
    rotation_angles: List[float],
    rotation_axes: List[str],
    spillover_rate_matrix,
    target_subsystem: tuple,
):
    """
    Create a spillover noise model based on the given quantum circuit and spillover rate matrix.
    Matrix should be provided such that the element (i, j) represents the spillover rate from qubit i to qubit j.
    The noise model is constructed by applying spillover noise to each gate in the circuit based on the spillover rate matrix
    and the specified rotation angles for each qubit.

    Parameters:
    - qc: QuantumCircuit object containing the circuit to generate the noise model for
    - rotation_angles: List of floats specifying the rotation angle for each qubit
    - rotation_axes: List of strings specifying the rotation axis for each qubit
    - spillover_rate_matrix: 2D numpy array representing the spillover rate between qubits
    - target_subsystem: Tuple containing the qubit indices to apply the spillover noise on

    Returns:
    - NoiseModel object containing the spillover noise model for the given circuit
    """
    noise_model = noise.NoiseModel(
        ["unitary", "rzx", "cx", "u", "h", "x", "s", "z", "rx", "ry", "rz"]
    )
    total_num_qubits = len(rotation_angles)
    assert all(
        [isinstance(q, int) for q in target_subsystem]
    ), "target_subsystem must be a tuple of integers"
    assert all(
        [q < total_num_qubits for q in target_subsystem]
    ), "target_subsystem must be within the total number of qubits"
    assert len(rotation_angles) == len(
        rotation_axes
    ), "rotation_angles and rotation_axes must have the same length"
    assert spillover_rate_matrix.shape == (
        total_num_qubits,
        total_num_qubits,
    ), "spillover_rate_matrix must be a square matrix"

    qubit_index_mapping = {q: i for i, q in enumerate(target_subsystem)}
    noisy_operations = {
        qubit_index_mapping[q]: {
            "main_op": None,
            "noisy_op": Operator.from_label("I"),
            "qargs": [qubit_index_mapping[q]],
        }
        for q in target_subsystem
    }

    for instruction in qc.data:
        if instruction.operation.label is not None:
            n_qargs = len(instruction.qubits)
            qubit_index_in_subcircuit = qc.find_bit(
                instruction.qubits[0]
            ).index  # Main qubit undergoing rotation
            noisy_operations[qubit_index_in_subcircuit][
                "main_op"
            ] = instruction.operation
            noisy_operations[qubit_index_in_subcircuit]["noisy_op"] = (
                Operator.from_label("I" * n_qargs)
            )
            noisy_operations[qubit_index_in_subcircuit]["qargs"] = [
                qc.find_bit(q).index for q in instruction.qubits
            ]

    for q, angle in enumerate(rotation_angles):
        for target_q in target_subsystem:
            gamma = spillover_rate_matrix[q, target_q]
            if gamma != 0.0 and angle != 0.0:
                noisy_unitary = Operator(
                    type(gate_map()[rotation_axes[q]])(gamma * angle)
                )
                noisy_op = noisy_operations[qubit_index_mapping[target_q]]["noisy_op"]

                noisy_operations[qubit_index_mapping[target_q]]["noisy_op"] = (
                    noisy_op.compose(
                        noisy_unitary,
                        qargs=(
                            [qubit_index_mapping[target_q]]
                            if noisy_op.num_qubits > 1
                            else None
                        ),
                    )
                )

    for q in target_subsystem:
        q_ = qubit_index_mapping[q]
        noise_model.add_quantum_error(
            noise.coherent_unitary_error(noisy_operations[q_]["noisy_op"]),
            noisy_operations[q_]["main_op"],
            noisy_operations[q_]["qargs"],
        )

    return noise_model


def noisy_backend(
    circuit_context: QuantumCircuit,
    spillover_rate_matrix: np.ndarray,
    target_subsystem: tuple,
    coupling_map: Optional[CouplingMap] = None,
):
    """
    Generate a noisy backend object with spillover noise based on the given circuit and spillover rate matrix

    Parameters:
    - circuit_context: QuantumCircuit object containing the circuit to generate the noise model for
    - spillover_rate_matrix: 2D numpy array representing the spillover rate between qubits
    - target_subsystem: Tuple containing the qubit indices to apply the spillover noise on

    Returns:
    - AerSimulator object with the spillover noise model applied and the specified coupling map

    """
    assert all(
        [isinstance(q, int) for q in target_subsystem]
    ), "target_subsystem must be a tuple of integers"
    assert all(
        [q < len(spillover_rate_matrix) for q in target_subsystem]
    ), "target_subsystem must be within the total number of qubits"

    rotation_angles = []
    rotation_axes = []
    for instruction in circuit_context.data:
        if instruction.operation.name in ["rx", "ry", "rz"]:
            rotation_angles.append(instruction.operation.params[0])
            rotation_axes.append(instruction.operation.name)

    pm = PassManager(
        [
            LocalSpilloverNoiseAerPass(
                numpy_to_hashable(spillover_rate_matrix),
                target_subsystem=target_subsystem,
            )
        ]
    )
    qc = pm.run(circuit_context)
    noise_model = create_spillover_noise_model_from_circuit(
        qc, rotation_angles, rotation_axes, spillover_rate_matrix, target_subsystem
    )
    backend_configuration = AerSimulator._DEFAULT_CONFIGURATION
    backend_configuration["coupling_map"] = list(coupling_map.get_edges())
    backend_configuration = AerBackendConfiguration(**backend_configuration)
    backend = AerSimulator(noise_model=noise_model, configuration=backend_configuration)

    return backend
