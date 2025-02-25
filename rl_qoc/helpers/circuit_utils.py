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
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    Operator,
    SparsePauliOp,
    Pauli,
    pauli_basis,
)
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.providers import BackendV2, Backend
from qiskit_experiments.framework import BaseAnalysis, BatchExperiment
from qiskit_experiments.library import ProcessTomography, StateTomography

from typing import Tuple, Optional, Sequence, List, Union, Dict

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


def handle_n_reps(qc: QuantumCircuit, n_reps: int = 1, backend=None, control_flow=True):
    """
    Returns a Quantum Circuit with qc repeated n_reps times
    Depending on the backend and the control_flow flag,
    the circuit is repeated using the built-in for_loop method if available

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
        coefficients[0] * e0
        + coefficients[1] * e1
        + coefficients[2] * e2
        + coefficients[3] * e3
    ).reshape(d, 1)
    # Prepare all 16 states
    states = []
    for k in range(0, d):
        for l in range(0, d):
            # state = apply_hw_group(p1, p2, coefficients)
            state = (
                np.linalg.matrix_power(X, k)
                @ np.linalg.matrix_power(Z, l)
                @ fiducial_state
            )
            states.append(Statevector(state))

    return states


def observables_to_indices(observables: List[SparsePauliOp] | SparsePauliOp):
    """
    Get single qubit indices of Pauli observables for the reward computation.

    Args:
        observables: Pauli observables to sample
    """
    observable_indices = []
    observables_grouping = (
        observables.group_commuting(qubit_wise=True)
        if isinstance(observables, SparsePauliOp)
        else observables
    )
    for obs_group in observables_grouping:  # Get indices of Pauli observables
        current_indices = []
        paulis = obs_group.paulis  # Get Pauli List out of the SparsePauliOp
        reference_pauli = Pauli(
            (np.logical_or.reduce(paulis.z), np.logical_or.reduce(paulis.x))
        )
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


def pauli_input_to_indices(prep: Pauli | str, inputs):
    """
    Convert the input state to single qubit state indices for the reward computation

    Args:
        prep: Pauli input state
        inputs: List of qubit indices
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


def retrieve_observables(
    target_state,
    dfe_tuple: Optional[Tuple[float, float]] = None,
    sampling_paulis: int = 100,
    c_factor: float = 1.0,
    observables_rng: Optional[np.random.Generator] = None,
) -> Tuple[SparsePauliOp, List[int]]:
    """
    Retrieve observables to sample for the DFE protocol (PhysRevLett.106.230501) for given target state

    :param target_state: Target state to prepare
    :param dfe_tuple: Optional Tuple (Ɛ, δ) from DFE paper
    :param sampling_paulis: Number of Pauli observables to sample
    :param c_factor: Constant factor for reward calculation
    :return: Observables to sample, number of shots for each observable
    """
    # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
    probabilities = target_state.Chi**2
    full_basis = pauli_basis(target_state.n_qubits)
    if not np.isclose(np.sum(probabilities), 1, atol=1e-5):
        print("probabilities sum um to", np.sum(probabilities))
        print("probabilities normalized")
        probabilities = probabilities / np.sum(probabilities)

    sample_size = (
        sampling_paulis
        if dfe_tuple is None
        else int(np.ceil(1 / (dfe_tuple[0] ** 2 * dfe_tuple[1])))
    )
    if observables_rng is not None:
        k_samples = observables_rng.choice(
            len(probabilities), size=sample_size, p=probabilities
        )
    else:
        k_samples = np.random.choice(
            len(probabilities), size=sample_size, p=probabilities
        )

    pauli_indices, pauli_shots = np.unique(k_samples, return_counts=True)
    reward_factor = c_factor / (
        np.sqrt(target_state.dm.dim) * target_state.Chi[pauli_indices]
    )

    if dfe_tuple is not None:
        pauli_shots = np.ceil(
            2
            * np.log(2 / dfe_tuple[1])
            / (
                target_state.dm.dim
                * sample_size
                * dfe_tuple[0] ** 2
                * target_state.Chi[pauli_indices] ** 2
            )
        )
    # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
    observables = SparsePauliOp(full_basis[pauli_indices], reward_factor, copy=False)

    shots_per_basis = []
    # Group observables by qubit-wise commuting groups to reduce the number of PUBs
    for i, commuting_group in enumerate(
        observables.paulis.group_qubit_wise_commuting()
    ):
        max_pauli_shots = 0
        for pauli in commuting_group:
            pauli_index = list(full_basis).index(pauli)
            ref_index = list(pauli_indices).index(pauli_index)
            max_pauli_shots = max(max_pauli_shots, pauli_shots[ref_index])
        shots_per_basis.append(max_pauli_shots)

    return observables, shots_per_basis


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
        new_indices = []
        other_qubits = [qc.qubits[i] for i in other_qubits_indices]
        random_input_indices = np.random.randint(0, 6, len(other_qubits)).tolist()
        random_input_context = Pauli6PreparationBasis().circuit(random_input_indices)
        for i in range(qc.num_qubits):
            if i in other_qubits_indices:
                new_indices.append(random_input_indices.pop(0))
            else:
                new_indices.append(indices[i])

        return input_circuit.compose(
            random_input_context, other_qubits, front=True
        ), tuple(new_indices)
    return input_circuit, tuple(indices)


def extend_observables(
    observables: SparsePauliOp, qc: QuantumCircuit, gate_target
) -> SparsePauliOp:
    """
    Extend the observables to all qubits in the quantum circuit if necessary

    Args:
        observables: Pauli observables to sample
        qc: Quantum circuit to be executed on quantum system
        gate_target: Target gate to prepare (possibly within a wider circuit context)

    Returns:
        Extended Pauli observables
    """

    if qc.num_qubits > gate_target.causal_cone_size:
        other_qubits_indices = set(range(qc.num_qubits)) - set(
            gate_target.causal_cone_qubits_indices
        )
        observables = observables.apply_layout(None, qc.num_qubits).apply_layout(
            gate_target.causal_cone_qubits_indices + list(other_qubits_indices)
        )

    return observables
