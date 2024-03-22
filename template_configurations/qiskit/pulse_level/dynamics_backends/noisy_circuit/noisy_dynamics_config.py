from typing import List, Tuple, Dict, Optional
import copy
from qiskit_dynamics.array import Array
from qiskit.quantum_info import Operator
import jax
from qiskit_dynamics import DynamicsBackend, Solver
from custom_jax_sim import JaxSolver
import numpy as np
from helper_functions import (
    create_quantum_operators,
    expand_operators,
    get_full_identity,
    construct_static_hamiltonian,
    get_couplings,
    get_noise_couplings,
)

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
    Custom noisy backend for the dynamics simulation.
    We allow for ZZ-crosstalk noise couplings between neighbouring qubits modeling hardware noise in transmon architectures.
    This noise occurs when pulses are applied to one qubit and affect the neighbouring qubits as well due to wires being close to each other.
    See https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.21.024016

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
    # Create operators
    a, adag, N, ident = create_quantum_operators(dims)

    # Expand operators across qudits
    N_ops = expand_operators(N, dims, ident)
    a_ops = expand_operators(a, dims, ident)
    adag_ops = expand_operators(adag, dims, ident)

    full_ident = get_full_identity(dims, ident)

    # Construct the static part of the Hamiltonian
    static_ham = construct_static_hamiltonian(
        dims, freqs, anharmonicities, N_ops, full_ident
    )

    drive_ops = [
        2 * np.pi * rabi_freqs[i] * (a_ops[i] + adag_ops[i]) for i in range(n_qubits)
    ]
    channels = {f"d{i}": freqs[i] for i in range(n_qubits)}
    ecr_ops = []
    num_controls = 0
    if couplings is not None:
        static_ham, channels, ecr_ops, num_controls = get_couplings(
            couplings,
            static_ham,
            a_ops,
            adag_ops,
            channels,
            freqs,
            ecr_ops,
            drive_ops,
            num_controls,
        )

    # ZZ-Crosstalk Noise Couplings
    drive_ops_errorfree = copy.deepcopy(drive_ops)
    if noise_couplings is not None:
        drive_ops = get_noise_couplings(
            noise_couplings,
            drive_ops_errorfree,
            n_qubits,
            rabi_freqs,
            a_ops,
            adag_ops,
            drive_ops,
        )

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
