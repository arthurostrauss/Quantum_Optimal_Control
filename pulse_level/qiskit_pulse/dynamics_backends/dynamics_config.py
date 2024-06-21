from typing import List, Tuple, Dict, Optional
from qiskit.quantum_info import Operator
import jax
from qiskit_dynamics import DynamicsBackend, Solver
from rl_qoc.custom_jax_sim import JaxSolver
import numpy as np
from pulse_level.qiskit_pulse.dynamics_backends import (
    create_quantum_operators,
    expand_operators,
    get_full_identity,
    construct_static_hamiltonian,
    get_couplings,
)

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")


def custom_backend(
    dims: List[int],
    freqs: List[float],
    anharmonicities: List[float],
    rabi_freqs: List[float],
    couplings: Optional[Dict[Tuple[int, int], float]] = None,
    solver_options: Optional[Dict] = None,
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
    control_channel_map = None
    if couplings is not None:
        static_ham, channels, ecr_ops, num_controls, control_channel_map = (
            get_couplings(
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
        )

    dt = 2.2222e-10

    jax_solver = JaxSolver(
        static_hamiltonian=static_ham,
        hamiltonian_operators=drive_ops + ecr_ops,
        rotating_frame=static_ham,
        hamiltonian_channels=list(channels.keys()),
        channel_carrier_freqs=channels,
        dt=dt,
        array_library="jax",
    )

    solver = Solver(
        static_hamiltonian=static_ham,
        hamiltonian_operators=drive_ops + ecr_ops,
        rotating_frame=static_ham,
        hamiltonian_channels=list(channels.keys()),
        channel_carrier_freqs=channels,
        dt=dt,
        array_library="jax",
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
        control_channel_map=control_channel_map,
    )

    dynamics_backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=dims,  # for computing measurement data
        solver_options=solver_options,  # to be used every time run is called
        control_channel_map=control_channel_map,
    )
    return jax_backend, dynamics_backend


def single_qubit_backend(w, r, dt, solver_options=None):
    """
    Custom single qubit backend for the dynamics simulation.
    """
    X = Operator.from_label("X")
    Z = Operator.from_label("Z")

    drift = 2 * np.pi * w * Z / 2
    operators = [2 * np.pi * r * X / 2]

    # construct the solver
    solver = Solver(
        static_hamiltonian=drift,
        hamiltonian_operators=operators,
        rotating_frame=drift,
        rwa_cutoff_freq=2 * 5.0,
        hamiltonian_channels=["d0"],
        channel_carrier_freqs={"d0": w},
        dt=dt,
        array_library="jax",
    )

    jax_solver = JaxSolver(
        static_hamiltonian=drift,
        hamiltonian_operators=operators,
        rotating_frame=drift,
        rwa_cutoff_freq=2 * 5.0,
        hamiltonian_channels=["d0"],
        channel_carrier_freqs={"d0": w},
        dt=dt,
        array_library="jax",
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
        subsystem_dims=[2],
        solver_options=solver_options,
    )

    dynamics_backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=[2],
        solver_options=solver_options,
    )

    return jax_backend, dynamics_backend


def surface_code_plaquette():
    """
    Custom backend for the dynamics simulation of a surface code plaquette.
    """
    # Define the parameters
    return custom_backend(
        [2] * 5,
        freqs=[4.8e9, 4.9e9, 5.0e9, 5.1e9, 5.2e9],
        anharmonicities=[-0.33e9] * 5,
        rabi_freqs=[0.1e6] * 5,
        couplings={(0, 1): 0.1e6, (0, 2): 0.1e6, (0, 3): 0.1e6, (0, 4): 0.1e6},
    )
