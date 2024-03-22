from typing import List, Tuple, Dict, Optional
import copy
from qiskit_dynamics.array import Array
from qiskit.quantum_info import Operator
import jax
from qiskit_dynamics import DynamicsBackend, Solver
from custom_jax_sim import JaxSolver
import numpy as np
from helper_functions import get_noise_coupling_neighbours

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
# import Array and set default backend
Array.set_default_backend("jax")




def custom_backend(
    dims: List[int],
    freqs: List[float],
    anharmonicities: List[float],
    rabi_freqs: List[float],
    couplings: Optional[Dict[Tuple[int, int], float]] = None,
    noise_couplings: Optional[Dict[Tuple[int, int], float]] = None,
    solver_options: Optional[Dict] = None,
):
    """
    Custom backend for the dynamics simulation.

    Args:
        dims: The dimensions of the subsystems.
        freqs: The frequencies of the subsystems.
        anharmonicities: The anharmonicities of the subsystems.
        couplings: The coupling constants between the subsystems.
        noise_couplings: The noise coupling constants between (neighbouring) qubits.
        rabi_freqs: The Rabi frequencies of the subsystems.

    """
    assert (
        len(dims) == len(freqs) == len(anharmonicities) == len(rabi_freqs)
    ), "The number of subsystems, frequencies, and anharmonicities must be equal."
    n_qubits = len(dims)
    a = [Operator(np.diag(np.sqrt(np.arange(1, dim)), 1)) for dim in dims]
    adag = [Operator(np.diag(np.sqrt(np.arange(1, dim)), -1)) for dim in dims]
    N = [Operator(np.diag(np.arange(dim))) for dim in dims]
    ident = [Operator(np.eye(dim, dtype=complex)) for dim in dims]

    full_ident = ident[0]
    for i in range(1, n_qubits):
        full_ident = full_ident.tensor(ident[i])

    N_ops = N
    a_ops = a
    adag_ops = adag
    for i in range(n_qubits):
        for j in range(n_qubits):
            if j > i:
                N_ops[i] = N_ops[i].expand(ident[j])
                a_ops[i] = a_ops[i].expand(ident[j])
                adag_ops[i] = adag_ops[i].expand(ident[j])
            elif j < i:
                N_ops[i] = N_ops[i].tensor(ident[j])
                a_ops[i] = a_ops[i].tensor(ident[j])
                adag_ops[i] = adag_ops[i].tensor(ident[j])

    static_ham = Operator(
        np.zeros((np.prod(dims), np.prod(dims)), dtype=complex),
        input_dims=tuple(dims),
        output_dims=tuple(dims),
    )

    for i in range(n_qubits):
        static_ham += 2 * np.pi * freqs[i] * N_ops[i] + np.pi * anharmonicities[
            i
        ] * N_ops[i] @ (N_ops[i] - full_ident)
    drive_ops = [
        2 * np.pi * rabi_freqs[i] * (a_ops[i] + adag_ops[i]) for i in range(n_qubits)
    ]
    channels = {f"d{i}": freqs[i] for i in range(n_qubits)}
    ecr_ops = []
    num_controls = 0
    if couplings is not None:
        keys = list(couplings.keys())
        for i, j in keys:
            couplings[(j, i)] = couplings[(i, j)]
        for (i, j), coupling in couplings.items():
            static_ham += (
                2
                * np.pi
                * coupling
                * (a_ops[i] + adag_ops[i])
                @ (a_ops[j] + adag_ops[j])
            )
            channels[f"u{num_controls}"] = freqs[j]
            num_controls += 1
            ecr_ops.append(drive_ops[i])

    # ZZ-Crosstalk Noise Couplings
    drive_ops_errorfree = copy.deepcopy(drive_ops)
    if noise_couplings is not None:
        keys = list(noise_couplings.keys())
        # Get the neighbours of each qubit based on the noise couplings dictionary provided
        neighbours = get_noise_coupling_neighbours(noise_couplings)
        for i, j in keys:
            if (j, i) not in noise_couplings:
                noise_couplings[(j, i)] = noise_couplings[(i, j)] # Make the noise coupling symmetric if not specified otherwise
        
        # Add spill-over error terms to the drive operators
        errors = {qbit: [] for qbit in range(n_qubits)}
        for qbit in range(n_qubits):
            for neighbour in neighbours[qbit]:
                if (qbit, neighbour) in noise_couplings:
                    noise_strength = noise_couplings[(qbit, neighbour)]
                    noise_contribution = noise_strength * (2 * np.pi * rabi_freqs[qbit] * (a_ops[neighbour] + adag_ops[neighbour]))
                    errors[qbit].append(noise_contribution)
                    drive_ops[qbit] += noise_contribution

        drive_ops_error = copy.deepcopy(drive_ops)

        # Check if the noise has been added correctly to the drive operators
        for qbit in range(n_qubits):
            # Sum the error contributions from all neighbors for the current qubit
            total_noise_contribution = sum(errors[qbit]) if qbit in errors else 0

            # Calculate the expected total drive operation with noise for the current qubit
            expected_total_with_noise = drive_ops_errorfree[qbit] + total_noise_contribution

            # Check if the expected total matches the actual total drive operation with noise
            if not np.allclose(expected_total_with_noise, drive_ops_error[qbit]):
                raise ValueError(f'Error in adding noise to drive operators for qubit {qbit}')

    dt = 2.2222e-10

    jax_solver = JaxSolver(
        static_hamiltonian=static_ham,
        hamiltonian_operators=drive_ops + ecr_ops,
        rotating_frame=static_ham,
        hamiltonian_channels=list(channels.keys()),
        channel_carrier_freqs=channels,
        dt=dt,
        evaluation_mode="dense",
    )

    solver = Solver(
        static_hamiltonian=static_ham,
        hamiltonian_operators=drive_ops + ecr_ops,
        rotating_frame=static_ham,
        hamiltonian_channels=list(channels.keys()),
        channel_carrier_freqs=channels,
        dt=dt,
        evaluation_mode="dense",
    )
    if solver_options is None:
        solver_options = {
            "method": "jax_odeint",
            "atol": 1e-5,
            "rtol": 1e-7,
            "hmax": dt,
        }

    jax_backend = DynamicsBackend(
        solver=jax_solver,
        subsystem_dims=dims,  # for computing measurement data
        solver_options=solver_options,  # to be used every time run is called
    )

    dynamics_backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=dims,  # for computing measurement data
        solver_options=solver_options,  # to be used every time run is called
    )
    return jax_backend, dynamics_backend
