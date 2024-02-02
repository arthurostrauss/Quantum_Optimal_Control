from typing import List, Tuple, Dict

from qiskit_dynamics.array import Array
from qiskit.quantum_info import Operator
import jax
from qiskit_dynamics import DynamicsBackend, Solver
from custom_jax_sim import JaxSolver
import numpy as np

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
# import Array and set default backend
Array.set_default_backend("jax")


def custom_backend(
    dims: List[int],
    freqs: List[float],
    anharmonicities: List[float],
    couplings: Dict[Tuple[float], float],
):
    """
    Custom backend for the dynamics simulation.

    Args:
        dims: The dimensions of the subsystems.
        freqs: The frequencies of the subsystems.
        anharmonicities: The anharmonicities of the subsystems.
        couplings: The coupling constants between the subsystems.

    """
    a = [np.diag(np.sqrt(np.arange(1, dim)), 1) for dim in dims]
    adag = np.array([np.diag(np.sqrt(np.arange(1, dim)), -1) for dim in dims])
    N = np.array([np.diag(np.arange(dim)) for dim in dims])

    ident = np.array([np.eye(dim, dtype=complex) for dim in dims])
    full_ident = np.eye(dims[0] ** 2, dtype=complex)

    N0 = np.kron(ident, N)
    N1 = np.kron(N, ident)

    a0 = np.kron(ident, a)
    a1 = np.kron(a, ident)

    a0dag = np.kron(ident, adag)
    a1dag = np.kron(adag, ident)

    static_ham0 = 2 * np.pi * freqs[0] * N0 + np.pi * anharmonicities[0] * N0 * (
        N0 - full_ident
    )
    static_ham1 = 2 * np.pi * freqs[1] * N1 + np.pi * anharmonicities[1] * N1 * (
        N1 - full_ident
    )

    static_ham_full = (
        static_ham0
        + static_ham1
        + 2 * np.pi * couplings[0] * ((a0 + a0dag) @ (a1 + a1dag))
    )

    # build solver
    dt = 2.2222e-10

    jax_solver = JaxSolver(
        static_hamiltonian=static_ham_full,
        hamiltonian_operators=[
            2 * np.pi * couplings[1] * (a0 + a0dag),
            2 * np.pi * couplings[2] * (a1 + a1dag),
            2 * np.pi * couplings[3] * (a0 + a0dag),
            2 * np.pi * couplings[4] * (a1 + a1dag),
        ],
        rotating_frame=static_ham_full,
        hamiltonian_channels=["d0", "d1", "u0", "u1"],
        channel_carrier_freqs={
            "d0": freqs[0],
            "d1": freqs[1],
            "u0": freqs[1],
            "u1": freqs[0],
        },
        dt=dt,
        evaluation_mode="dense",
    )
    # Consistent solver option to use throughout notebook
    solver_options = {"method": "jax_odeint", "atol": 1e-5, "rtol": 1e-7, "hmax": dt}

    solver = Solver(
        static_hamiltonian=static_ham_full,
        hamiltonian_operators=[
            2 * np.pi * couplings[1] * (a0 + a0dag),
            2 * np.pi * couplings[2] * (a1 + a1dag),
            2 * np.pi * couplings[3] * (a0 + a0dag),
            2 * np.pi * couplings[4] * (a1 + a1dag),
        ],
        rotating_frame=static_ham_full,
        hamiltonian_channels=["d0", "d1", "u0", "u1"],
        channel_carrier_freqs={
            "d0": freqs[0],
            "d1": freqs[1],
            "u0": freqs[1],
            "u1": freqs[0],
        },
        dt=dt,
        evaluation_mode="dense",
    )

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


dim = 3
v0 = 4.86e9
anharm0 = -0.32e9
r0 = 0.22e9

v1 = 4.97e9
anharm1 = -0.32e9
r1 = 0.26e9

J = 0.002e9

a = np.diag(np.sqrt(np.arange(1, dim)), 1)
adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
N = np.diag(np.arange(dim))

ident = np.eye(dim, dtype=complex)
full_ident = np.eye(dim**2, dtype=complex)

N0 = np.kron(ident, N)
N1 = np.kron(N, ident)

a0 = np.kron(ident, a)
a1 = np.kron(a, ident)

a0dag = np.kron(ident, adag)
a1dag = np.kron(adag, ident)

static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)

static_ham_full = (
    static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))
)

drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)

# build solver
dt = 2.2222e-10

jax_solver = JaxSolver(
    static_hamiltonian=static_ham_full,
    hamiltonian_operators=[
        drive_op0,
        drive_op1,
        drive_op0,
        drive_op1,
    ],
    rotating_frame=static_ham_full,
    hamiltonian_channels=["d0", "d1", "u0", "u1"],
    channel_carrier_freqs={
        "d0": v0,
        "d1": v1,
        "u0": v1,
        "u1": v0,
    },
    dt=dt,
    evaluation_mode="dense",
)
# Consistent solver option to use throughout notebook
solver_options = {"method": "jax_odeint", "atol": 1e-5, "rtol": 1e-7, "hmax": dt}

solver = Solver(
    static_hamiltonian=static_ham_full,
    hamiltonian_operators=[
        drive_op0,
        drive_op1,
        drive_op0,
        drive_op1,
    ],
    rotating_frame=static_ham_full,
    hamiltonian_channels=["d0", "d1", "u0", "u1"],
    channel_carrier_freqs={
        "d0": v0,
        "d1": v1,
        "u0": v1,
        "u1": v0,
    },
    dt=dt,
    evaluation_mode="dense",
)

jax_backend = DynamicsBackend(
    solver=jax_solver,
    # target = fake_backend_v2.target,
    subsystem_dims=[dim, dim],  # for computing measurement data
    solver_options=solver_options,  # to be used every time run is called
)

dynamics_backend = DynamicsBackend(
    solver=solver,
    # target = fake_backend_v2.target,
    subsystem_dims=[dim, dim],  # for computing measurement data
    solver_options=solver_options,  # to be used every time run is called
)
