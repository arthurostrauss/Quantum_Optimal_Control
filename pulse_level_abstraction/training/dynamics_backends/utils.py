from typing import List, Tuple, Dict, Optional
import numpy as np
import copy

from qiskit.quantum_info import Operator


def create_quantum_operators(
    dims: List[int],
) -> Tuple[List[Operator], List[Operator], List[Operator], Operator, Operator]:
    """
    Creates quantum operators \(a\), \(a^\dagger\), \(N\), and the identity matrix for each subsystem
    specified by its dimension in `dims`.
    """

    a = [Operator(np.diag(np.sqrt(np.arange(1, dim)), 1)) for dim in dims]
    adag = [Operator(np.diag(np.sqrt(np.arange(1, dim)), -1)) for dim in dims]
    N = [Operator(np.diag(np.arange(dim))) for dim in dims]
    ident = [Operator(np.eye(dim, dtype=complex)) for dim in dims]
    return a, adag, N, ident


def expand_operators(
    operators: List[Operator], dims: List[int], ident: List[Operator]
) -> List[Operator]:
    """Expands quantum operators across multiple qubits based on dimensions."""
    expanded_ops = operators.copy()
    n_qubits = len(dims)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if j > i:
                expanded_ops[i] = expanded_ops[i].expand(ident[j])
            elif j < i:
                expanded_ops[i] = expanded_ops[i].tensor(ident[j])
    return expanded_ops


def get_full_identity(dims: List[int], ident: List[Operator]) -> Operator:
    """Generates the full identity operator for the given dimensions of subsystems."""
    full_ident = ident[0]
    for i in range(1, len(dims)):
        full_ident = full_ident.tensor(ident[i])
    return full_ident


def construct_static_hamiltonian(
    dims: List[int],
    freqs: List[float],
    anharmonicities: List[float],
    N_ops: List[Operator],
    full_ident: Operator,
) -> Operator:
    """Constructs the static part of the Hamiltonian."""
    static_ham = Operator(
        np.zeros((np.prod(dims), np.prod(dims)), dtype=complex),
        input_dims=tuple(dims),
        output_dims=tuple(dims),
    )
    n_qubits = len(dims)
    for i in range(n_qubits):
        static_ham += 2 * np.pi * freqs[i] * N_ops[i] + np.pi * anharmonicities[
            i
        ] * N_ops[i] @ (N_ops[i] - full_ident)
    return static_ham


def get_couplings(
    couplings: Dict[Tuple[int, int], float],
    static_ham: Operator,
    a_ops: List[Operator],
    adag_ops: List[Operator],
    channels: Dict[str, float],
    freqs: List[float],
    ecr_ops: List[Operator],
    drive_ops: List[Operator],
    num_controls: int,
):
    """
    Processes coupling information to update the static Hamiltonian, control channels, and drive operators with
    cross-resonance terms.
    """
    control_channel_map = {}
    keys = list(couplings.keys())
    for i, j in keys:
        couplings[(j, i)] = couplings[(i, j)]
    for (i, j), coupling in couplings.items():
        static_ham += (
            2 * np.pi * coupling * (a_ops[i] + adag_ops[i]) @ (a_ops[j] + adag_ops[j])
        )
        channels[f"u{num_controls}"] = freqs[j]
        control_channel_map[(i, j)] = num_controls
        num_controls += 1
        ecr_ops.append(drive_ops[i])

    return static_ham, channels, ecr_ops, num_controls, control_channel_map


def noise_coupling_sanity_check(qbit, errors, drive_ops_errorfree, drive_ops_error):
    # Sum the error contributions from all neighbors for the current qubit
    total_noise_contribution = sum(errors[qbit]) if qbit in errors else 0

    # Calculate the expected total drive operation with noise for the current qubit
    expected_total_with_noise = drive_ops_errorfree[qbit] + total_noise_contribution

    # Check if the expected total matches the actual total drive operation with noise
    if not np.allclose(expected_total_with_noise, drive_ops_error[qbit]):
        raise ValueError(f"Error in adding noise to drive operators for qubit {qbit}")


def get_pulse_spillover_noise(
    pulse_spillover_rates: Dict[Tuple[int, int], float],
    drive_ops_errorfree: List[float],
    n_qubits: int,
    rabi_freqs: List[float],
    a_ops: List[Operator],
    adag_ops: List[Operator],
    drive_ops: List[Operator],
) -> List[Operator]:
    keys = list(pulse_spillover_rates.keys())
    """
    This function describes the noise effect of pulse-spillover for qubit-pairs specified in noise_couplings. This noise only occurs when a pulse (intended for qubit i) leaks
    into another qubit j. So, it only affects the drive operators of a qubit.
    """
    neighbours = get_pulse_spillover_noise_neighbours(pulse_spillover_rates)
    for i, j in keys:
        if (j, i) not in pulse_spillover_rates:
            pulse_spillover_rates[(j, i)] = pulse_spillover_rates[
                (i, j)
            ]  # Make the noise coupling symmetric if not specified otherwise

    # Add spill-over error terms to the drive operators
    errors = {qbit: [] for qbit in range(n_qubits)}
    for qbit in range(n_qubits):
        for neighbour in neighbours[qbit]:
            if (qbit, neighbour) in pulse_spillover_rates:
                noise_strength = pulse_spillover_rates[(qbit, neighbour)]
                noise_contribution = noise_strength * (
                    2
                    * np.pi
                    * rabi_freqs[qbit]
                    * (a_ops[neighbour] + adag_ops[neighbour])
                )
                errors[qbit].append(noise_contribution)
                drive_ops[qbit] += noise_contribution

    drive_ops_error = copy.deepcopy(drive_ops)
    # Check if the noise has been added correctly to the drive operators
    for qbit in range(n_qubits):
        noise_coupling_sanity_check(qbit, errors, drive_ops_errorfree, drive_ops_error)

    return drive_ops


def get_pulse_spillover_noise_neighbours(noise_couplings: Dict = None) -> Dict:
    """
    This function processes a dictionary of noise couplings between qubits to determine the
    neighbor relationships among them. Each key in the noise_couplings dictionary is a tuple
    representing a pair of qubits, and its corresponding value indicates the coupling strength
    between these qubits. Only positive coupling strengths are considered as valid connections.

    Parameters:
    - noise_couplings (Dict, optional): A dictionary where each key is a tuple of two integers
      representing qubit indices, and each value is a float representing the coupling strength
      between these qubits. Default is None.

    Returns:
    - dict: A dictionary where each key is an integer representing a qubit index, and each value
      is a list of integers representing the indices of its neighboring qubits based on the noise
      couplings. The neighbors are determined by positive coupling strengths, and each neighbor
      relationship is bidirectional unless specified otherwise.
    """
    if noise_couplings is None:
        raise ValueError("Noise couplings dictionary not provided")
    # Initialize neighbor lists for each qubit
    # Assuming the maximum qubit index from the adjacency dictionary
    max_qubit_index = max(max(pair) for pair in noise_couplings.keys())
    neighbors = {i: [] for i in range(max_qubit_index + 1)}

    # Populate the neighbor lists based on the adjacency dictionary
    for (qubit1, qubit2), coupling_strength in noise_couplings.items():
        if (
            coupling_strength > 0
        ):  # Assuming a positive coupling strength indicates a connection
            if qubit1 != qubit2:
                # Check if qubit2 is not already a neighbor of qubit1 before adding
                if qubit2 not in neighbors[qubit1]:
                    neighbors[qubit1].append(qubit2)
                # Check if qubit1 is not already a neighbor of qubit2 before adding
                if qubit1 not in neighbors[qubit2]:
                    neighbors[qubit2].append(qubit1)

    return neighbors
