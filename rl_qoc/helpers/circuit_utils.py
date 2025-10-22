from __future__ import annotations

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
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    Operator,
    SparsePauliOp,
    Pauli,
    PauliList,
)
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.transpiler import PassManager, CouplingMap, InstructionProperties
from qiskit.providers import BackendV2

from qiskit_experiments.framework import BaseAnalysis, BatchExperiment
from qiskit_experiments.library import ProcessTomography, StateTomography
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
    Pauli6PreparationBasis,
)

from .transpiler_passes import CustomGateReplacementPass
from typing import Tuple, Optional, Sequence, List, Union, Dict, Callable

QuantumTarget = Union[Statevector, Operator, DensityMatrix]


def shots_to_precision(n_shots: int) -> float:
    """
    Converts the number of shots to a precision value.

    Args:
        n_shots: The number of shots.

    Returns:
        The precision value.
    """
    return 1 / np.sqrt(n_shots)


def precision_to_shots(precision: float) -> int:
    """
    Converts a precision value to the number of shots.

    Args:
        precision: The precision value.

    Returns:
        The number of shots.
    """
    return int(np.ceil(1 / precision**2))


def handle_n_reps(qc: QuantumCircuit, n_reps: int = 1, backend=None, control_flow=True):
    """
    Repeats a quantum circuit a given number of times.

    Args:
        qc: The quantum circuit to repeat.
        n_reps: The number of repetitions.
        backend: The backend to use for the repetition.
        control_flow: Whether to use control flow for the repetition.

    Returns:
        The repeated quantum circuit.
    """
    if n_reps == 1:
        return qc.copy()
    if isinstance(backend, BackendV2) and "for_loop" in backend.operation_names and control_flow:
        prep_circuit = qc.copy_empty_like()

        with prep_circuit.for_loop(range(n_reps)) as i:
            prep_circuit.compose(qc, inplace=True)
    else:
        prep_circuit = qc.repeat(n_reps).decompose()
    return prep_circuit


def add_custom_gate(
    qc: QuantumCircuit,
    gate: Gate | QuantumCircuit | str,
    qubits: QuantumRegister | Sequence[Qubit | int],
    parameters: Optional[ParameterVector | List[Parameter] | List[float]] = None,
    physical_qubits: Optional[Sequence[int]] = None,
    backend: Optional[BackendV2] = None,
    instruction_properties: Optional[InstructionProperties] = None,
    inplace: bool = True,
) -> QuantumCircuit:
    """
    Adds a custom gate to a quantum circuit.

    Args:
        qc: The quantum circuit to add the gate to.
        gate: The gate to add.
        qubits: The qubits to apply the gate to.
        parameters: The parameters for the gate.
        physical_qubits: The physical qubits to apply the gate to.
        backend: The backend to use for the gate.
        instruction_properties: The instruction properties for the gate.
        inplace: Whether to add the gate in place.

    Returns:
        The quantum circuit with the added gate.
    """
    if isinstance(gate, str):
        gate = Gate(
            gate.lower(),
            num_qubits=len(qubits),
            params=parameters.params if isinstance(parameters, ParameterVector) else parameters,
        )
    elif isinstance(gate, QuantumCircuit):
        if gate.num_qubits != len(qubits):
            raise ValueError(
                f"QuantumCircuit gate must have {len(qubits)} qubits, but has {gate.num_qubits}."
            )
        gate = gate.to_gate()
    validate_qubits(qc, qubits)
    if inplace:
        qc.append(gate, qubits)
    else:
        qc_new = qc.copy()
        qc_new.append(gate, qubits)

    if instruction_properties:
        validate_qm_instruction_properties(instruction_properties, parameters)
    if backend is not None and physical_qubits is not None:
        if gate.name not in backend.target.operation_names:
            backend.target.add_instruction(gate, {tuple(physical_qubits): instruction_properties})
        else:
            backend.target.update_instruction_properties(
                gate.name, tuple(physical_qubits), instruction_properties
            )
    return qc_new if not inplace else qc


def validate_qm_instruction_properties(
    instruction_properties: InstructionProperties,
    parameters: ParameterVector | List[Parameter] | List[float],
):
    """
    Validates the QUA pulse macro for a custom gate.

    Args:
        instruction_properties: The instruction properties for the gate.
        parameters: The parameters for the gate.
    """
    if hasattr(instruction_properties, "qua_pulse_macro"):
        qua_pulse_macro = instruction_properties.qua_pulse_macro
        if qua_pulse_macro:
            if not callable(qua_pulse_macro):
                raise TypeError(
                    "`qua_pulse_macro` must be a callable function implementing the QUA macro."
                )
            if len(signature(qua_pulse_macro).parameters) != len(parameters):
                raise ValueError(
                    "Mismatch between expected and provided parameters for the QUA macro."
                )


def validate_qubits(qc: QuantumCircuit, qubits: QuantumRegister | Sequence[Qubit | int]):
    """
    Validates that the given qubits are valid for the given quantum circuit.

    Args:
        qc: The quantum circuit.
        qubits: The qubits to validate.
    """
    if isinstance(qubits, QuantumRegister):
        if not all(q in qc.qubits for q in qubits):
            raise ValueError(
                "All qubits in the QuantumRegister must be part of the QuantumCircuit."
            )
    elif isinstance(qubits, Sequence) and all(isinstance(q, Qubit) for q in qubits):
        if not all(q in qc.qubits for q in qubits):
            raise ValueError("All qubits in the sequence must be part of the QuantumCircuit.")
    elif isinstance(qubits, Sequence) and all(isinstance(q, int) for q in qubits):
        if any(q < 0 or q >= qc.num_qubits for q in qubits):
            raise ValueError(
                "All qubit indices in the sequence must be within the range of the QuantumCircuit."
            )


def causal_cone_circuit(
    circuit: QuantumCircuit,
    qubits: Sequence[int | Qubit] | QuantumRegister,
) -> Tuple[QuantumCircuit, List[Qubit]]:
    """
    Returns the causal cone circuit for the given qubits.

    Args:
        circuit: The quantum circuit.
        qubits: The qubits to get the causal cone for.

    Returns:
        The causal cone circuit and the qubits involved in the causal cone.
    """

    dag = circuit_to_dag(circuit)
    if isinstance(qubits, Sequence) and all(isinstance(q, int) for q in qubits):
        qubits = [dag.qubits[q] for q in qubits]
    involved_qubits = [dag.quantum_causal_cone(q) for q in qubits]
    involved_qubits = list(set([q for sublist in involved_qubits for q in sublist]))
    filtered_dag = dag.copy_empty_like()
    for node in dag.topological_op_nodes():
        if all(q in involved_qubits for q in node.qargs):
            filtered_dag.apply_operation_back(node.op, node.qargs)

    filtered_dag.remove_qubits(*[q for q in filtered_dag.qubits if q not in involved_qubits])
    return dag_to_circuit(filtered_dag, False), involved_qubits


def count_gates(qc: QuantumCircuit):
    """
    Counts the number of gates in a quantum circuit.

    Args:
        qc: The quantum circuit.

    Returns:
        A dictionary mapping each qubit to the number of gates acting on it.
    """
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            if not isinstance(gate.operation, Delay):
                gate_count[qubit] += 1
    return gate_count


def remove_unused_wires(qc: QuantumCircuit):
    """
    Removes unused wires from a quantum circuit.

    Args:
        qc: The quantum circuit.

    Returns:
        The quantum circuit with unused wires removed.
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
    """
    Gets the timings of the instructions in a quantum circuit.

    Args:
        circuit: The quantum circuit.

    Returns:
        A list of the start times of the instructions.
    """
    qubit_timings = {i: 0 for i in range(circuit.num_qubits)}

    start_times = []

    for instruction in circuit.data:
        qubits = instruction.qubits
        qubit_indices = [circuit.find_bit(qubit).index for qubit in qubits]
        start_time = max(qubit_timings[i] for i in qubit_indices)

        start_times.append(start_time)

        for i in qubit_indices:
            qubit_timings[i] = start_time + 1

    return start_times


def density_matrix_to_statevector(density_matrix: DensityMatrix):
    """
    Converts a density matrix to a statevector.

    Args:
        density_matrix: The density matrix to convert.

    Returns:
        The converted statevector.
    """
    if np.isclose(density_matrix.purity(), 1):
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix.data)

        max_eigenvalue_index = np.argmax(eigenvalues)
        statevector = eigenvectors[:, max_eigenvalue_index]

        return Statevector(statevector)
    else:
        raise ValueError("The density matrix does not represent a pure state.")


def fidelity_from_tomography(
    qc_input: List[QuantumCircuit] | QuantumCircuit,
    backend: Optional[BackendV2],
    physical_qubits: Optional[Sequence[int]],
    target: Optional[QuantumTarget | List[QuantumTarget]] = None,
    analysis: Union[BaseAnalysis, None, str] = "default",
    **run_options,
):
    """
    Calculates the fidelity of a quantum circuit using tomography.

    Args:
        qc_input: The quantum circuit(s) to calculate the fidelity for.
        backend: The backend to use for the tomography.
        physical_qubits: The physical qubits to perform the tomography on.
        target: The target state or gate.
        analysis: The analysis method to use.
        **run_options: Additional run options.

    Returns:
        The fidelity of the quantum circuit(s).
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
                StateTomography(qc, physical_qubits=physical_qubits, analysis=analysis, target=tgt)
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
            dim, _ = tgt.dim
            result = dim * result / (dim + 1)
        results.append(result)

    return results if len(results) > 1 else results[0]


def get_gate(gate: Gate | str) -> Gate:
    """
    Gets a gate from the standard gate map.

    Args:
        gate: The name of the gate or a Gate object.

    Returns:
        The Gate object.
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
    Substitutes a target gate in a quantum circuit with a custom gate.

    Args:
        circuit: The quantum circuit.
        target_gate: The target gate to substitute.
        custom_gate: The custom gate to substitute with.
        qubits: The qubits to apply the gate to.
        parameters: The parameters for the custom gate.

    Returns:
        The quantum circuit with the substituted gate.
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
        [CustomGateReplacementPass((target_gate, qubits), custom_gate, parameters=parameters)]
    )

    return pass_manager.run(circuit)


def retrieve_neighbor_qubits(coupling_map: CouplingMap, target_qubits: List):
    """
    Retrieves the neighbor qubits of a set of target qubits.

    Args:
        coupling_map: The coupling map of the backend.
        target_qubits: The target qubits.

    Returns:
        A list of the neighbor qubits.
    """
    neighbors = set()
    for target_qubit in target_qubits:
        neighbors.update(coupling_map.neighbors(target_qubit))
    return list(neighbors - set(target_qubits))


def isolate_qubit_instructions(circuit: QuantumCircuit, physical_qubits: list[int]):
    """
    Isolates the instructions in a quantum circuit that act on a given set of physical qubits.

    Args:
        circuit: The quantum circuit.
        physical_qubits: The physical qubits to isolate the instructions for.

    Returns:
        A list of the isolated instructions.
    """
    qubits_to_index = {qubit: circuit.find_bit(qubit).index for qubit in circuit.qubits}
    instructions = filter(
        lambda instr: all(qubits_to_index[qubit] in physical_qubits for qubit in instr.qubits),
        circuit.data,
    )
    return list(instructions)


def retrieve_tgt_instruction_count(qc: QuantumCircuit, target: Dict):
    """
    Retrieves the number of target instructions in a quantum circuit.

    Args:
        qc: The quantum circuit.
        target: The target instruction.

    Returns:
        The number of target instructions.
    """
    tgt_instruction = CircuitInstruction(
        target["gate"], [qc.qubits[i] for i in target["physical_qubits"]]
    )
    return qc.data.count(tgt_instruction)


def get_single_qubit_input_states(input_state_choice) -> List[QuantumCircuit]:
    """
    Gets a list of single-qubit input states.

    Args:
        input_state_choice: The choice of input states.

    Returns:
        A list of single-qubit input states.
    """
    if input_state_choice == "pauli4":
        input_states = [PauliPreparationBasis().circuit([i]) for i in range(4)]
    elif input_state_choice == "pauli6":
        input_states = [Pauli6PreparationBasis().circuit([i]) for i in range(6)]
    elif input_state_choice == "2-design":
        states = get_2design_input_states(2)
        input_circuits = [QuantumCircuit(1) for _ in states]
        for input_circ, state in zip(input_circuits, states):
            input_circ.prepare_state(state)
        input_states = input_circuits
    else:
        raise ValueError("Invalid input state choice")
    return input_states


def get_input_states_cardinality_per_qubit(input_state_choice: str) -> int:
    """
    Gets the cardinality of the input states per qubit.

    Args:
        input_state_choice: The choice of input states.

    Returns:
        The cardinality of the input states per qubit.
    """
    if input_state_choice == "pauli4":
        return 4
    elif input_state_choice == "pauli6":
        return 6
    elif input_state_choice == "2-design":
        return 4
    else:
        raise ValueError("Invalid input state choice")


def get_2design_input_states(d: int = 4) -> List[Statevector]:
    """
    Gets a list of 2-design input states.

    Args:
        d: The dimension of the Hilbert space.

    Returns:
        A list of 2-design input states.
    """
    golden_ratio = (np.sqrt(5) - 1) / 2
    omega = np.exp(2 * np.pi * 1j / d)

    e0 = np.array([1, 0, 0, 0], dtype=complex)
    e1 = np.array([0, 1, 0, 0], dtype=complex)
    e2 = np.array([0, 0, 1, 0], dtype=complex)
    e3 = np.array([0, 0, 0, 1], dtype=complex)

    Z = np.diag([omega**r for r in range(d)])

    X = np.zeros((d, d), dtype=complex)
    for r in range(d - 1):
        X[r, r + 1] = 1
    X[d - 1, 0] = 1

    coefficients = (
        1
        / (2 * np.sqrt(3 + golden_ratio))
        * np.array(
            [
                1 + np.exp(-1j * np.pi / 4),
                np.exp(1j * np.pi / 4) + 1j * golden_ratio ** (-3 / 2),
                1 - np.exp(-1j * np.pi / 4),
                np.exp(1j * np.pi / 4) - 1j * golden_ratio ** (-3 / 2),
            ]
        )
    )

    fiducial_state = (
        coefficients[0] * e0 + coefficients[1] * e1 + coefficients[2] * e2 + coefficients[3] * e3
    ).reshape(d, 1)
    states = []
    for k in range(0, d):
        for l in range(0, d):
            state = np.linalg.matrix_power(X, k) @ np.linalg.matrix_power(Z, l) @ fiducial_state
            states.append(Statevector(state))

    return states


def observables_to_indices(
    observables: List[SparsePauliOp, Pauli, str] | SparsePauliOp | PauliList | Pauli | str,
):
    """
    Converts a list of observables to a list of indices.

    Args:
        observables: The observables to convert.

    Returns:
        A list of indices.
    """
    if isinstance(observables, (str, Pauli)):
        observables = PauliList(Pauli(observables) if isinstance(observables, str) else observables)
    elif isinstance(observables, List) and all(
        isinstance(obs, (str, Pauli)) for obs in observables
    ):
        observables = PauliList(
            [Pauli(obs) if isinstance(obs, str) else obs for obs in observables]
        )
    observable_indices = []
    observables_grouping = (
        observables.group_commuting(qubit_wise=True)
        if isinstance(observables, (SparsePauliOp, PauliList))
        else observables
    )
    for obs_group in observables_grouping:
        current_indices = []
        paulis = obs_group.paulis if isinstance(obs_group, SparsePauliOp) else obs_group
        reference_pauli = Pauli((np.logical_or.reduce(paulis.z), np.logical_or.reduce(paulis.x)))
        for pauli_term in reversed(
            reference_pauli.to_label()
        ):
            if pauli_term == "I" or pauli_term == "Z":
                current_indices.append(0)
            elif pauli_term == "X":
                current_indices.append(1)
            elif pauli_term == "Y":
                current_indices.append(2)
        observable_indices.append(tuple(current_indices))
    return observable_indices


def pauli_input_to_indices(prep: Pauli | str, inputs: List[int]):
    """
    Converts a Pauli input to a list of indices.

    Args:
        prep: The Pauli input.
        inputs: The list of inputs.

    Returns:
        A list of indices.
    """
    prep = prep if isinstance(prep, Pauli) else Pauli(prep)
    prep_indices = []
    assert all(input < 2 for input in inputs), "Only single qubit inputs are supported"
    for i, pauli_op in enumerate(reversed(prep.to_label())):
        if pauli_op == "X":
            prep_indices.append(2 + inputs[i])
        elif pauli_op == "Y":
            prep_indices.append(4 + inputs[i])
        else:
            prep_indices.append(inputs[i])
    return prep_indices


def extend_input_state_prep(
    input_circuit: QuantumCircuit, qc: QuantumCircuit, gate_target, indices
) -> Tuple[QuantumCircuit, Tuple[int, ...]]:
    """
    Extends the input state preparation to all qubits in a quantum circuit.

    Args:
        input_circuit: The input state preparation circuit.
        qc: The quantum circuit.
        gate_target: The target gate.
        indices: The indices of the qubits to apply the input state preparation to.

    Returns:
        The extended quantum circuit and the new indices.
    """
    if (
        qc.num_qubits > gate_target.causal_cone_size
    ):
        other_qubits_indices = set(range(qc.num_qubits)) - set(
            gate_target.causal_cone_qubits_indices
        )
        new_indices: List[int] = []
        initial_indices = indices.copy()
        other_qubits = [qc.qubits[i] for i in other_qubits_indices]
        random_input_indices = np.random.randint(0, 6, len(other_qubits)).tolist()
        random_input_context = Pauli6PreparationBasis().circuit(random_input_indices)
        for i in range(qc.num_qubits):
            if i in other_qubits_indices:
                new_indices.append(random_input_indices.pop(0))
            else:
                new_indices.append(initial_indices.pop(0))

        new_circuit = input_circuit.compose(random_input_context, other_qubits, front=True)
        return new_circuit, tuple(new_indices)
    return input_circuit, tuple(indices)


def extend_observables(
    observables: SparsePauliOp, qc: QuantumCircuit, target_qubit_indices: List[int]
) -> SparsePauliOp:
    """
    Extends the observables to all qubits in a quantum circuit.

    Args:
        observables: The observables to extend.
        qc: The quantum circuit.
        target_qubit_indices: The indices of the target qubits.

    Returns:
        The extended observables.
    """

    size = len(target_qubit_indices)
    if qc.num_qubits > size:
        other_qubits_indices = set(range(qc.num_qubits)) - set(target_qubit_indices)
        observables = observables.apply_layout(None, qc.num_qubits).apply_layout(
            target_qubit_indices + list(other_qubits_indices)
        )

    return observables
