from __future__ import annotations

import keyword
import re
from inspect import signature
from typing import Tuple, Optional, Sequence, List, Union, Dict, Callable

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

# Qiskit Experiments imports
from qiskit_experiments.framework import BaseAnalysis, BatchExperiment
from qiskit_experiments.library import ProcessTomography, StateTomography
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
    Pauli6PreparationBasis,
)

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


def handle_n_reps(
    qc: QuantumCircuit,
    n_reps: int = 1,
    backend: Optional[BackendV2] = None,
    control_flow: bool = True,
):
    """
    Returns a Quantum Circuit with qc repeated n_reps times
    Depending on the backend and the control_flow flag,
    the circuit is repeated using the built-in for_loop method if available

    Args:
        qc: Quantum Circuit
        n_reps: Number of repetitions
        backend: Backend instance
        control_flow: Control flow flag (uses for_loop if True and if backend supports it)
    """
    # Repeat the circuit n_reps times and prepend the input state preparation
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
) -> QuantumCircuit:
    """
    Add a custom gate to the Quantum Circuit and register it in the backend target for the specified physical qubits
    (if provided).
    Args:
        qc: Quantum Circuit to which the gate is to be added
        gate: Gate to be added (can be a Gate, QuantumCircuit, or string name of the gate)
        qubits: Virtual qubits on which the gate is to be applied (within qc.qubits)
        parameters: Parameters of the gate (if applicable, e.g., for parametrized gates)
        physical_qubits: Physical qubits on which the gate is to be applied (if applicable)
        backend: Backend instance to register the gate in the target
        instruction_properties: Instruction properties for the gate (if applicable)
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
    qc.append(gate, qubits)

    if instruction_properties:
        validate_qm_instruction_properties(instruction_properties, parameters)
    if backend is not None and physical_qubits is not None:
        if gate.name not in backend.target.operation_names:
            backend.target.add_instruction(gate, {tuple(physical_qubits): instruction_properties})
        else:
            backend.target.update_instruction_properties(
                gate.name, tuple(physical_qubits), instruction_properties
            )
    return qc


def validate_qm_instruction_properties(
    instruction_properties: InstructionProperties,
    parameters: ParameterVector | List[Parameter] | List[float],
):
    # Validate the qua_pulse_macro if provided
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
    Validate that the specified qubits are part of the QuantumCircuit.
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
    Get the causal cone circuit of the specified qubits as well as the qubits involved in the causal cone

    Args:
        circuit: Quantum Circuit
        qubits: Qubits of interest
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
    backend: Optional[BackendV2],
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
        [CustomGateReplacementPass((target_gate, qubits), custom_gate, parameters=parameters)]
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
        lambda instr: all(qubits_to_index[qubit] in physical_qubits for qubit in instr.qubits),
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


def get_single_qubit_input_states(input_state_choice) -> List[QuantumCircuit]:
    """
    Get single qubit input states for a given choice of input states
    (pauli4, pauli6, 2-design)
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
    Get the cardinality of the input states for a given choice of input states
    (pauli4, pauli6, 2-design)
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
    Function that return the 2-design input states (used for CAFE reward scheme)
    Follows this Reference: https://arxiv.org/pdf/1008.1138 (see equations 2 and 13)
    """
    # Define constants
    golden_ratio = (np.sqrt(5) - 1) / 2
    omega = np.exp(2 * np.pi * 1j / d)

    # Define computational basis states
    e0 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    e1 = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
    e2 = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
    e3 = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩

    # Create the Z matrix (diagonal matrix with powers of omega)
    Z = np.diag([omega**r for r in range(d)])

    # Create the X matrix (shift matrix)
    X = np.zeros((d, d), dtype=complex)
    for r in range(d - 1):
        X[r, r + 1] = 1
    X[d - 1, 0] = 1  # Wrap-around to satisfy the condition for |e_0>

    # Define the fiducial state from Eq. (13)
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
    # Prepare all 16 states
    states = []
    for k in range(0, d):
        for l in range(0, d):
            # state = apply_hw_group(p1, p2, coefficients)
            state = np.linalg.matrix_power(X, k) @ np.linalg.matrix_power(Z, l) @ fiducial_state
            states.append(Statevector(state))

    return states


def observables_from_array(pauli_array: np.ndarray, coeff_array: np.ndarray) -> list:
    """
    Convert a numpy array to a list of SparsePauliOp objects.
    Args:
        pauli_array: Last dimension of the array is the Pauli string per qubit
        coeff_array: Coefficients for each Pauli string

    Returns:
        list: List of SparsePauliOp objects

    """
    pauli_list = []
    if pauli_array.shape[:-1] != coeff_array.shape:
        raise ValueError("Shapes of pauli_array and coeff_array do not match")

    # Mapping for Pauli operators
    pauli_map = {0: "I", 1: "X", 2: "Y", 3: "Z"}

    # Flatten all dimensions except the last one
    flat_paulis = pauli_array.reshape(-1, pauli_array.shape[-1])
    flat_coeffs = coeff_array.flatten()

    for pauli_string, coeff in zip(flat_paulis, flat_coeffs):
        # Convert numbers to Pauli string
        pauli_str = "".join(pauli_map[int(p)] for p in pauli_string)
        pauli_list.append(SparsePauliOp.from_list([(pauli_str, coeff)]))

    # Manually reshape using the original shape
    def nested_reshape(flat_list, shape):
        """Recursively reshape a flat list into nested lists."""
        if not shape:
            return flat_list.pop(0)
        size = shape[0]
        return [nested_reshape(flat_list, shape[1:]) for _ in range(size)]

    return nested_reshape(pauli_list.copy(), list(coeff_array.shape))


def observables_to_indices(
    observables: List[SparsePauliOp, Pauli, str] | SparsePauliOp | PauliList | Pauli | str,
):
    """
    Get single qubit indices of Pauli observables for the reward computation.

    Args:
        observables: Pauli observables to sample
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
    for obs_group in observables_grouping:  # Get indices of Pauli observables
        current_indices = []
        paulis = obs_group.paulis if isinstance(obs_group, SparsePauliOp) else obs_group
        reference_pauli = Pauli((np.logical_or.reduce(paulis.z), np.logical_or.reduce(paulis.x)))
        for pauli_term in reversed(
            reference_pauli.to_label()
        ):  # Get individual qubit indices for each Pauli term
            if pauli_term == "I" or pauli_term == "Z":
                current_indices.append(0)
            elif pauli_term == "X":
                current_indices.append(1)
            elif pauli_term == "Y":
                current_indices.append(2)
        observable_indices.append(tuple(current_indices))
    return observable_indices


def pauli_input_to_indices(prep: Pauli | str, inputs: List[int] | Tuple[int]):
    """
    Convert the input state to single qubit state indices for the reward computation

    Args:
        prep: Pauli input state
        inputs: List of binary numbers indicating which eigenstate is selected (0 -> +1 eigenstate, 1 -> -1 eigenstate)
    """
    prep = prep if isinstance(prep, Pauli) else Pauli(prep)
    prep_indices = []
    assert all(input < 2 for input in inputs), "Only single qubit inputs are supported"
    for i, pauli_op in enumerate(reversed(prep.to_label())):
        # Build input state in Pauli6 basis from Pauli prep: look at each qubit individually
        if pauli_op == "X":
            prep_indices.append(2 + inputs[i])
        elif pauli_op == "Y":
            prep_indices.append(4 + inputs[i])
        else:  # pauli_op == "I" or pauli_op == "Z"
            prep_indices.append(inputs[i])
    return prep_indices


def extend_input_state_prep(
    input_circuit: QuantumCircuit, qc: QuantumCircuit, gate_target, indices
) -> Tuple[QuantumCircuit, Tuple[int]]:
    """
    Extend the input state preparation to all qubits in the quantum circuit if necessary

    Args:
        input_circuit: Input state preparation circuit
        qc: Quantum circuit to be executed on quantum system
        gate_target: Target gate to prepare (possibly within a wider circuit context)
    """
    if (
        qc.num_qubits > gate_target.causal_cone_size
    ):  # Add random input state on all qubits (not part of reward calculation)
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

        new_circuit: QuantumCircuit = input_circuit.compose(
            random_input_context, other_qubits, front=True
        )
        return new_circuit, tuple(new_indices)
    return input_circuit, tuple(indices)


def extend_observables(
    observables: SparsePauliOp, qc: QuantumCircuit, target_qubit_indices: List[int]
) -> SparsePauliOp:
    """
    Extend the observables to all qubits in the quantum circuit if necessary

    Args:
        observables: Pauli observables to sample
        qc: Quantum circuit to be executed on quantum system
        target_qubit_indices: Target qubit indices for the observables

    Returns:
        Extended Pauli observables
    """

    size = len(target_qubit_indices)
    if qc.num_qubits > size:
        other_qubits_indices = set(range(qc.num_qubits)) - set(target_qubit_indices)
        observables = observables.apply_layout(None, qc.num_qubits).apply_layout(
            target_qubit_indices + list(other_qubits_indices)
        )

    return observables


def pauli_weight(pauli_obj: Union[Pauli, SparsePauliOp, PauliList]) -> Union[int, List[int]]:
    """
    Return the weight(s) of a Pauli object:
    - For Pauli: returns a single int (number of non-identity terms).
    - For SparsePauliOp: returns a list of ints (one per Pauli term).
    - For PauliList: returns a list of ints (one per Pauli term).
    """

    if isinstance(pauli_obj, Pauli):
        return pauli_obj.x | pauli_obj.z
    elif isinstance(pauli_obj, SparsePauliOp):
        return np.sum(pauli_obj.paulis.x | pauli_obj.paulis.z, axis=1).tolist()
    elif isinstance(pauli_obj, PauliList):
        return np.sum(pauli_obj.x | pauli_obj.z, axis=1).tolist()
    else:
        raise TypeError("Input must be a Pauli or SparsePauliOp.")


def are_qubit_wise_commuting(p1: Pauli, p2: Pauli) -> bool:
    """Check qubit-wise commutation: commute on each qubit independently."""
    for c1, c2 in zip(p1.to_label(), p2.to_label()):
        if c1 != "I" and c2 != "I" and c1 != c2:
            return False
    return True


def group_input_paulis_by_qwc(input_paulis: PauliList, counts) -> List[PauliList]:
    """Group input Pauli operators by qubit-wise commutation (QWC), sorted by descending importance.

    Each group is returned as a PauliList.
    """
    # Extract input Pauli operators

    # Sort input Paulis by descending count importance
    sorted_indices = sorted(range(len(counts)), key=lambda i: -counts[i])
    sorted_paulis = [input_paulis[i] for i in sorted_indices]

    grouped = []
    while sorted_paulis:
        ref = sorted_paulis.pop(0)
        group = [ref]
        to_remove = []
        for i, other in enumerate(sorted_paulis):
            if are_qubit_wise_commuting(ref, other):
                group.append(other)
                to_remove.append(i)
        for i in reversed(to_remove):
            sorted_paulis.pop(i)
        grouped.append(PauliList(group))  # Convert to PauliList

    return grouped
