from __future__ import annotations

import keyword
import re

import numpy as np
from qiskit.circuit import (
    QuantumCircuit,
    Qubit,
    Delay,
    Gate,
    ParameterVector,
    Parameter,
    CircuitInstruction,
    QuantumRegister,
)
from qiskit.circuit.library import get_standard_gate_name_mapping as gate_map
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import DensityMatrix, Statevector, Operator
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.providers import BackendV2, Backend
from qiskit_experiments.framework import BaseAnalysis, BatchExperiment
from qiskit_experiments.library import ProcessTomography, StateTomography

from typing import Tuple, Optional, Sequence, List, Union, Dict

from .transpiler_passes import CustomGateReplacementPass

QuantumTarget = Union[Statevector, Operator, DensityMatrix]


def shots_to_precision(n_shots: int) -> float:
    """
    Convert number of shots to precision on expectation value
    """
    return 1 / np.sqrt(n_shots)


def precision_to_shots(precision: float) -> int:
    """
    Convert precision on expectation value to number of shots
    """
    return int(np.ceil(1 / precision**2))


def handle_n_reps(qc: QuantumCircuit, n_reps: int = 1, backend=None, control_flow=True):
    """
    Handle n_reps for a Quantum Circuit

    Args:
        qc: Quantum Circuit
        n_reps: Number of repetitions
        backend: Backend instance
        control_flow: Control flow flag (uses for_loop if True and backend supports it)
    """
    # Repeat the circuit n_reps times and prepend the input state preparation
    if n_reps == 1:
        return qc.copy()
    if (
        isinstance(backend, BackendV2)
        and "for_loop" in backend.operation_names
        and control_flow
    ):
        prep_circuit = qc.copy_empty_like()

        with prep_circuit.for_loop(range(n_reps)) as i:
            prep_circuit.compose(qc, inplace=True)
    else:
        prep_circuit = qc.repeat(n_reps).decompose()
    return prep_circuit


def causal_cone_circuit(
    circuit: QuantumCircuit,
    qubits: Sequence[int | Qubit] | QuantumRegister,
) -> Tuple[QuantumCircuit, List[Qubit]]:
    """
    Get the causal cone circuit of the specified qubits as well as the qubits involved in the causal cone

    Args:
        circuit: Quantum Circuit
        qubits: Qubits of interest
    """

    dag = circuit_to_dag(circuit)
    if isinstance(qubits, List) and all(isinstance(q, int) for q in qubits):
        qubits = [dag.qubits[q] for q in qubits]
    involved_qubits = [dag.quantum_causal_cone(q) for q in qubits]
    involved_qubits = list(set([q for sublist in involved_qubits for q in sublist]))
    filtered_dag = dag.copy_empty_like()
    for node in dag.topological_op_nodes():
        if all(q in involved_qubits for q in node.qargs):
            filtered_dag.apply_operation_back(node.op, node.qargs)

    filtered_dag.remove_qubits(
        *[q for q in filtered_dag.qubits if q not in involved_qubits]
    )
    return dag_to_circuit(filtered_dag, False), involved_qubits


def to_python_identifier(s):
    # Prepend underscore if the string starts with a digit
    if s[0].isdigit():
        s = "_" + s

    # Replace non-alphanumeric characters with underscore
    s = re.sub("\W|^(?=\d)", "_", s)

    # Append underscore if the string is a Python keyword
    if keyword.iskeyword(s):
        s += "_"

    return s


def count_gates(qc: QuantumCircuit):
    """
    Count number of gates in a Quantum Circuit
    """
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            if not isinstance(gate.operation, Delay):
                gate_count[qubit] += 1
    return gate_count


def remove_unused_wires(qc: QuantumCircuit):
    """
    Remove unused wires from a Quantum Circuit
    """
    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if count == 0:
            for instr in qc.data:
                if qubit in instr.qubits:
                    qc.data.remove(instr)
            qc.qubits.remove(qubit)
    return qc


def get_instruction_timings(circuit: QuantumCircuit):
    # Initialize the timings for each qubit
    qubit_timings = {i: 0 for i in range(circuit.num_qubits)}

    # Initialize the list of start times
    start_times = []

    # Loop over each instruction in the circuit
    for instruction in circuit.data:
        qubits = instruction.qubits
        qubit_indices = [circuit.find_bit(qubit).index for qubit in qubits]
        # Find the maximum time among the qubits involved in the instruction
        start_time = max(qubit_timings[i] for i in qubit_indices)

        # Add the start time to the list of start times
        start_times.append(start_time)

        # Update the time for each qubit involved in the instruction
        for i in qubit_indices:
            qubit_timings[i] = start_time + 1

    return start_times


def density_matrix_to_statevector(density_matrix: DensityMatrix):
    """
    Convert a density matrix to a statevector (if the density matrix represents a pure state)

    Args:
        density_matrix: DensityMatrix object representing the pure state

    Returns:
        Statevector: Statevector object representing the pure state

    Raises:
        ValueError: If the density matrix does not represent a pure state
    """
    # Check if the state is pure by examining if Tr(rho^2) is 1
    if np.isclose(density_matrix.purity(), 1):
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix.data)

        # Find the eigenvector corresponding to the eigenvalue 1 (pure state)
        # The statevector is the eigenvector corresponding to the maximum eigenvalue
        max_eigenvalue_index = np.argmax(eigenvalues)
        statevector = eigenvectors[:, max_eigenvalue_index]

        # Return the Statevector object
        return Statevector(statevector)
    else:
        raise ValueError("The density matrix does not represent a pure state.")


def fidelity_from_tomography(
    qc_input: List[QuantumCircuit] | QuantumCircuit,
    backend: Optional[Backend],
    physical_qubits: Optional[Sequence[int]],
    target: Optional[QuantumTarget | List[QuantumTarget]] = None,
    analysis: Union[BaseAnalysis, None, str] = "default",
    **run_options,
):
    """
    Extract average state or gate fidelity from batch of Quantum Circuit for target state or gate

    Args:
        qc_input: Quantum Circuit input to benchmark (Note that we handle removing final measurements if any)
        backend: Backend instance
        physical_qubits: Physical qubits on which state or process tomography is to be performed
        analysis: Analysis instance
        target: Target state or gate for fidelity calculation (must be either Operator or QuantumState)
    Returns:
        avg_fidelity: Average state or gate fidelity (over the batch of Quantum Circuits)
    """
    if isinstance(qc_input, QuantumCircuit):
        qc_input = [qc_input.remove_final_measurements(False)]
    else:
        qc_input = [qc.remove_final_measurements(False) for qc in qc_input]
    if isinstance(target, QuantumTarget):
        target = [target] * len(qc_input)
    elif target is not None:
        if len(target) != len(qc_input):
            raise ValueError(
                "Number of target states/gates does not match the number of input circuits"
            )
    else:
        target = [Statevector(qc) for qc in qc_input]

    exps = []
    fids = []
    for qc, tgt in zip(qc_input, target):
        if isinstance(tgt, Operator):
            exps.append(
                ProcessTomography(
                    qc, physical_qubits=physical_qubits, analysis=analysis, target=tgt
                )
            )
            fids.append("process_fidelity")
        elif isinstance(tgt, QuantumState):
            exps.append(
                StateTomography(
                    qc, physical_qubits=physical_qubits, analysis=analysis, target=tgt
                )
            )
            fids.append("state_fidelity")
        else:
            raise TypeError("Target must be either Operator or QuantumState")
    batch_exp = BatchExperiment(exps, backend=backend, flatten_results=False)

    exp_data = batch_exp.run(backend_run=True, **run_options).block_for_results()
    results = []
    for fid, tgt, child_data in zip(fids, target, exp_data.child_data()):
        result = child_data.analysis_results(fid).value
        if fid == "process_fidelity" and tgt.is_unitary():
            # Convert to average gate fidelity metric
            dim, _ = tgt.dim
            result = dim * result / (dim + 1)
        results.append(result)

    return results if len(results) > 1 else results[0]


def get_gate(gate: Gate | str) -> Gate:
    """
    Get gate from gate_map
    """
    if isinstance(gate, str):
        if gate.lower() == "cnot":
            gate = "cx"
        elif gate.lower() == "cphase":
            gate = "cz"
        elif gate.lower() == "x/2":
            gate = "sx"
        try:
            gate = gate_map()[gate.lower()]
        except KeyError:
            raise ValueError("Invalid target gate name")
    return gate


def substitute_target_gate(
    circuit: QuantumCircuit,
    target_gate: Gate | str,
    custom_gate: Gate | str,
    qubits: Optional[Sequence[int]] = None,
    parameters: ParameterVector | List[Parameter] | List[float] = None,
):
    """
    Substitute target gate in Quantum Circuit with a parametrized version of the gate.
    The parametrized_circuit function signature should match the expected one for a QiskitConfig instance.

    Args:
        circuit: Quantum Circuit instance
        target_gate: Target gate to be substituted
        custom_gate: Custom gate to be substituted with
        qubits: Physical qubits on which the gate is to be applied (if None, all qubits of input circuit are considered)
        parameters: Parameters for the custom gate
    """

    if isinstance(custom_gate, str):
        try:
            custom_gate2 = gate_map()[custom_gate]
        except KeyError:
            raise ValueError(f"Custom gate {custom_gate} not found in gate map")
        if custom_gate2.params and parameters is not None:
            assert len(custom_gate2.params) == len(parameters), (
                f"Number of parameters ({len(parameters)}) does not match number of parameters "
                f"required by the custom gate ({len(custom_gate2.params)})"
            )

    pass_manager = PassManager(
        [
            CustomGateReplacementPass(
                (target_gate, qubits), custom_gate, parameters=parameters
            )
        ]
    )

    return pass_manager.run(circuit)


def retrieve_neighbor_qubits(coupling_map: CouplingMap, target_qubits: List):
    """
    Retrieve neighbor qubits of target qubits

    Args:
        coupling_map: Coupling map
        target_qubits: Target qubits

    Returns:
        neighbor_qubits: List of neighbor qubits indices for specified target qubits
    """
    neighbors = set()
    for target_qubit in target_qubits:
        neighbors.update(coupling_map.neighbors(target_qubit))
    return list(neighbors - set(target_qubits))


def isolate_qubit_instructions(circuit: QuantumCircuit, physical_qubits: list[int]):
    """
    Extracts instructions from a circuit that act on the specified physical qubits
    :param circuit: QuantumCircuit to extract instructions from
    :param physical_qubits: List of physical qubits to extract instructions for
    :return: List of instructions that act on the specified physical qubits
    """
    qubits_to_index = {qubit: circuit.find_bit(qubit).index for qubit in circuit.qubits}
    instructions = filter(
        lambda instr: all(
            qubits_to_index[qubit] in physical_qubits for qubit in instr.qubits
        ),
        circuit.data,
    )
    return list(instructions)


def retrieve_tgt_instruction_count(qc: QuantumCircuit, target: Dict):
    """
    Retrieve count of target instruction in Quantum Circuit

    Args:
        qc: Quantum Circuit (ideally already transpiled)
        target: Target in form of {"gate": "X", "physical_qubits": [0, 1]}
    """
    tgt_instruction = CircuitInstruction(
        target["gate"], [qc.qubits[i] for i in target["physical_qubits"]]
    )
    return qc.data.count(tgt_instruction)
