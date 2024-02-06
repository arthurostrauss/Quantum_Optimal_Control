from typing import List, Tuple, Dict, Optional

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
    rabi_freqs: List[float],
    couplings: Optional[Dict[Tuple[int, int], float]] = None,
):
    """
    Custom backend for the dynamics simulation.

    Args:
        dims: The dimensions of the subsystems.
        freqs: The frequencies of the subsystems.
        anharmonicities: The anharmonicities of the subsystems.
        couplings: The coupling constants between the subsystems.
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

    # Consistent solver option to use throughout notebook
    solver_options = {"method": "jax_odeint", "atol": 1e-5, "rtol": 1e-7, "hmax": dt}

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
