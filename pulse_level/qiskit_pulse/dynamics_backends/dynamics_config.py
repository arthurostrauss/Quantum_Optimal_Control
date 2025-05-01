from qiskit.quantum_info import Statevector
import jax
from qiskit_dynamics import DynamicsBackend, Solver
from .utils import *

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")


def fixed_frequency_transmon_backend(
    dims: List[int],
    freqs: List[float],
    anharmonicities: List[float],
    rabi_freqs: List[float],
    couplings: Optional[Dict[Tuple[int, int], float]] = None,
    dt: float = 2.222e-10,
    solver_options: Optional[Dict] = None,
    t1s: Optional[List[float]] = None,
    t2s: Optional[List[float]] = None,
    **backend_options,
) -> DynamicsBackend:
    """
    Custom Transmon backend for the dynamics simulation.

    Args:
        dims: The dimensions of the subsystems.
        freqs: The frequencies of the subsystems.
        anharmonicities: The anharmonicities of the subsystems.
        rabi_freqs: The Rabi frequencies of the subsystems.
        couplings: The coupling constants between the subsystems.
        dt: Time step for Backend (also fixes step of Solver simulation by default)
        solver_options: The options for the solver.
        t1s: The T1 times of the subsystems.
        t2s: The T2 times of the subsystems.

    """
    assert (
        len(dims) == len(freqs) == len(anharmonicities) == len(rabi_freqs)
    ), "The number of subsystems, frequencies, and anharmonicities must be equal."
    n_systems = len(dims)
    # Create operators
    a, adag, N = create_quantum_operators(dims)

    # Expand operators across qudits
    N_ops = expand_operators(N, dims)
    a_ops = expand_operators(a, dims)
    adag_ops = expand_operators(adag, dims)

    # Construct the static part of the Hamiltonian
    static_ham = construct_static_hamiltonian(dims, freqs, anharmonicities, N_ops)

    drive_ops = [2 * np.pi * rabi_freqs[i] * (a_ops[i] + adag_ops[i]) for i in range(n_systems)]
    drive_channels = {f"d{i}": freqs[i] for i in range(n_systems)}
    ecr_ops = []
    control_channel_map = None
    control_channels = {}
    if couplings is not None:
        coupling_ham, control_channels, ecr_ops, control_channel_map = get_couplings(
            couplings,
            a_ops,
            adag_ops,
            freqs,
            drive_ops,
        )
        static_ham += coupling_ham

    channels = {**drive_channels, **control_channels}
    dissipator_channels, static_dissipators = get_t1_t2_dissipators(t1s, t2s, a_ops, adag_ops)
    solver = Solver(
        static_hamiltonian=static_ham,
        hamiltonian_operators=drive_ops + ecr_ops,
        dissipator_operators=static_dissipators if static_dissipators else None,
        rotating_frame=static_ham,
        hamiltonian_channels=list(channels.keys()),
        dissipator_channels=dissipator_channels if dissipator_channels else None,
        channel_carrier_freqs=channels,
        dt=dt,
        array_library="jax",
    )
    if solver_options is None:
        solver_options = {
            "method": "jax_odeint",
            "atol": 1e-6,
            "rtol": 1e-8,
            "hmax": dt,
        }

    dynamics_backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=dims,  # for computing measurement data
        solver_options=solver_options,  # to be used every time run is called
        control_channel_map=control_channel_map,
        initial_state=Statevector.from_int(0, dims),
        **backend_options,
    )
    return dynamics_backend


def single_qubit_backend(w, r, dt, solver_options=None, **backend_options):
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

    if solver_options is None:
        solver_options = {
            "method": "jax_odeint",
            "atol": 1e-5,
            "rtol": 1e-7,
            "hmax": dt,
        }

    dynamics_backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=[2],
        solver_options=solver_options,
        **backend_options,
    )

    return dynamics_backend


def surface_code_plaquette(**backend_options):
    """
    Custom backend for the dynamics simulation of a surface code plaquette.
    """
    # Define the parameters
    return fixed_frequency_transmon_backend(
        [2] * 5,
        freqs=[4.8e9, 4.9e9, 5.0e9, 5.1e9, 5.2e9],
        anharmonicities=[-0.33e9] * 5,
        rabi_freqs=[0.1e6] * 5,
        couplings={(0, 1): 0.1e6, (0, 2): 0.1e6, (0, 3): 0.1e6, (0, 4): 0.1e6},
        **backend_options,
    )
