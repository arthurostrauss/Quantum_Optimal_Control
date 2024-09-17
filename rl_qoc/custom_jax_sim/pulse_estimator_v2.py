from typing import List, Iterable, Callable, Optional, Any

import jax.debug
import numpy as np
from qiskit import schedule, QiskitError
from qiskit.circuit import ControlFlowOp
from qiskit.primitives import (
    PubResult,
    PrimitiveResult,
    DataBin,
    BaseEstimatorV2,
    PrimitiveJob,
)
from qiskit.primitives.backend_estimator_v2 import Options
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, Pauli
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit_dynamics import DynamicsBackend, Solver, solve_lmde
from qiskit_dynamics import DYNAMICS_NUMPY as unp

from qiskit_dynamics.solvers.solver_classes import (
    format_final_states,
    validate_and_format_initial_state,
)
from scipy.integrate._ivp.ivp import OdeResult
from jax import vmap, jit
from qiskit.pulse import Schedule, ScheduleBlock, SymbolicPulse


def simulate_pulse_level(
    pub: EstimatorPub | SamplerPub | SamplerPubLike,
    backend: DynamicsBackend,
    y0: Optional[Any] = None,
) -> list[OdeResult]:
    """
    Simulate the quantum state at the pulse level with DynamicsBackend.

    Args:
        pub: The pub to simulate. Only has to contain the circuit and parameter values.
        backend: The backend to use for the simulation.
        y0: Initial state. If None, the backend's initial state is used.
    """
    if not isinstance(pub, EstimatorPub):
        pub = SamplerPub.coerce(pub)

    # Extract parameters
    dt = backend.dt
    solver: Solver = backend.options.solver
    model = solver.model
    pub_parameters = pub.parameter_values
    if not pub_parameters.shape:
        pub_parameters = pub_parameters.reshape(-1)
    parameter_values = pub_parameters.data
    # Build pulse schedule generator function (with or without parameter values)
    sched_generator = lambda: schedule(pub.circuit, backend)

    # Compute t_span given the circuit duration
    sched_duration = schedule(pub.circuit, backend).duration
    t_span = [0, sched_duration * dt]  # Time span for simulation

    # Define jittable simulation function
    if parameter_values:

        def jittable_sim(t_span, y0, y0_input, y0_cls, param_dict):
            return pulse_level_core_sim(
                sched_generator, backend, t_span, y0, y0_input, y0_cls, param_dict
            )

        param_vmap = vmap(jittable_sim, in_axes=(None, None, None, None, 0))
        jit_sim = jit(param_vmap, static_argnums=(3,))
    else:

        def jittable_sim(t_span, y0, y0_input, y0_cls):
            return pulse_level_core_sim(
                sched_generator, backend, t_span, y0, y0_input, y0_cls
            )

        jit_sim = jit(jittable_sim, static_argnums=(3,))

    all_results = []
    if y0 is None:
        y0 = backend.options.initial_state
        if isinstance(y0, str) and y0 == "ground_state":
            y0 = Statevector(
                backend._dressed_states[:, 0], dims=backend.options.subsystem_dims
            )
    y0, y0_input, y0_cls, state_type_wrapper = validate_and_format_initial_state(
        y0, model
    )

    # jax.config.update('jax_disable_jit', True)
    if parameter_values:
        batch_results_t, batch_results_y = jit_sim(
            unp.asarray(t_span),
            unp.asarray(y0),
            parameter_values,
            unp.asarray(y0),
            y0_cls,
        )
    else:
        batch_results_t, batch_results_y = jit_sim(
            unp.asarray(t_span), unp.asarray(y0), unp.asarray(y0_input), y0_cls
        )
        batch_results_t = [batch_results_t]
        batch_results_y = [batch_results_y]

    for results_t, results_y in zip(batch_results_t, batch_results_y):
        results = OdeResult(t=results_t, y=results_y)
        if y0_cls is not None:
            results.y = [y0_cls(np.array(yi)) for yi in results.y]
        all_results.append(results)
    return all_results


def pulse_level_core_sim(
    sched_generator: Callable[[], Schedule],
    backend: DynamicsBackend,
    t_span: list[float],
    y0,
    y0_input,
    y0_cls,
    param_dict: Optional[dict] = None,
):
    """
    Core simulation function for pulse level simulation (called within a JIT compiled function).

    Args:
        sched_generator: Schedule generator function
        backend: DynamicsBackend object
        t_span: Time span for simulation
        y0: Initial state
        y0_input: Initial state input type
        y0_cls: Initial state class
        param_dict: Parameter dictionary for parameterized schedules
    """

    solver: Solver = backend.options.solver
    model = solver.model
    model_sigs = model.signals
    sched = sched_generator()
    if sched.is_parameterized():
        sched.assign_parameters(param_dict)
    signals = solver._schedule_to_signals(sched)
    solver._set_new_signals(signals)
    results = solve_lmde(model, t_span, y0, **backend.options.solver_options)
    results.y = format_final_states(results.y, model, y0_input, y0_cls)
    model.signals = model_sigs
    return results.t, results.y


class PulseEstimatorV2(BaseEstimatorV2):
    """
    Evaluates expectation value using Pauli rotation gates.

    This :class:`~.PulseEstimatorV2` class is a specific implementation of the
    :class:`~.BackendEstimatorV2` interface that is used to wrap a :class:`~.DynamicsBackend`
    object in the :class:`~.BackendEstimatorV2` API. This version slightly changes from the original BackendEstimatorV2
    in the main function, which essentially loads parameters values as well as other useful parameters that can be used
    for running the RL training (typically, observables information and the initial state that should be priorly set in
    the estimator options before calling estimator.run()) into the backend and solver options.

    """

    def __init__(self, *, backend: DynamicsBackend, options: dict | None = None):
        """
        Args:
            backend: The backend to use for the estimation.
            options: The options for the estimator.
        """
        if not isinstance(backend, DynamicsBackend):
            raise TypeError("PulseEstimatorV2 can only be used with a DynamicsBackend.")

        ScheduleBlock.disable_parameter_validation = True
        SymbolicPulse.disable_validation = True
        Schedule.disable_parameter_validation = True
        self._backend = backend
        self._options = Options(**options) if options else Options()

        basis = PassManagerConfig.from_backend(backend).basis_gates
        opt1q = Optimize1qGatesDecomposition(basis=basis, target=backend.target)
        self._passmanager = PassManager([opt1q])

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    @property
    def backend(self) -> DynamicsBackend:
        """Returns the backend which this sampler object based on."""
        return self._backend

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        """Estimate expectation values for each provided pub (Primitive Unified Bloc).

        Args:
            pubs: An iterable of pub-like objects, such as tuples ``(circuit, observables)``
                  or ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                       run Estimator Pub that does not specify its own precision. If None
                       the estimator's default precision value will be used.

        Returns:
            A job object that contains results.
        """
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult(
            [self._run_pub(pub) for pub in pubs], metadata={"version": 2}
        )

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        """
        Run the estimator on the given pubs.

        Args:
            pubs: The pubs to compute.

        Returns:
            The results of the computation.
        """

        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # calculate broadcasting of parameters and observables
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(
            param_shape
        )
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        subsystem_dims: List[int] = list(
            filter(lambda x: x > 1, self._backend.options.subsystem_dims)
        )
        sim_results = simulate_pulse_level(pub, self._backend)
        states = [result.y[-1] for result in sim_results]
        if not all(isinstance(state, (Statevector, DensityMatrix)) for state in states):
            raise TypeError("States must be either Statevector or DensityMatrix object")
        # Project obtained states to computational multi-qubit space
        qubitized_results = [projected_state(state, subsystem_dims) for state in states]

        # calculate expectation values (evs) and standard errors (stds)
        flat_indices = list(param_indices.ravel())
        evs = np.zeros_like(bc_param_ind, dtype=float)
        stds = np.full(bc_param_ind.shape, precision)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            flat_index = flat_indices.index(param_index)
            for pauli, coeff in bc_obs[index].items():
                expval = qubitized_results[flat_index].expectation_value(Pauli(pauli))
                evs[index] += expval * coeff
        if precision > 0:
            rng = np.random.default_rng(self.options.seed_simulator)
            if not np.all(np.isreal(evs)):
                raise ValueError(
                    "Given operator is not Hermitian and noise cannot be added."
                )
            evs = rng.normal(evs, precision, evs.shape)
        return PubResult(
            DataBin(evs=evs, stds=stds, shape=evs.shape),
            metadata={
                "target_precision": precision,
                "circuit_metadata": pub.circuit.metadata,
                "simulated_statevectors": states,
                "simulated_qubitized_statevectors": qubitized_results,
            },
        )

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        """
        Validate the pubs to ensure they are compatible with the estimator.

        Args:
            pubs: The pubs to validate.
        """

        for i, pub in enumerate(pubs):
            if pub.precision <= 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than or equal to 0 ({pub.precision}). ",
                    "But precision should be larger than 0.",
                )
            if pub.circuit.cregs:
                raise QiskitError(
                    "PulseEstimatorV2 does not support classical registers."
                )
            if pub.circuit.num_vars > 0:
                raise QiskitError("PulseEstimatorV2 does not support dynamic circuits.")
            for op in pub.circuit.data:
                if op.operation.name == "measure":
                    raise QiskitError("PulseEstimatorV2 does not support measurements.")
                if isinstance(op.operation, ControlFlowOp):
                    raise QiskitError(
                        "PulseEstimatorV2 does not support control flow operations."
                    )


def build_qubit_space_projector(initial_subsystem_dims: list) -> Operator:
    """
    Build projector on qubit space from initial subsystem dimensions
    The returned operator is a non-square matrix mapping the qudit space to the qubit space.
    It can be applied to convert multi-qudit states/unitaries to multi-qubit states/unitaries.

    Args:
        initial_subsystem_dims: Initial subsystem dimensions

    Returns: Projector on qubit space as a Qiskit Operator object
    """
    total_dim = np.prod(initial_subsystem_dims)
    output_dims = (2,) * len(initial_subsystem_dims)
    total_qubit_dim = np.prod(output_dims)
    projector = Operator(
        np.zeros((total_qubit_dim, total_dim), dtype=np.complex128),
        input_dims=tuple(initial_subsystem_dims),
        output_dims=output_dims,
    )  # Projector initialized in the qudit space
    for i in range(
        total_dim
    ):  # Loop over all computational basis states in the qudit space
        s = Statevector.from_int(i, initial_subsystem_dims)  # Computational qudit state
        for key in s.to_dict().keys():  # Loop over all computational basis states
            if all(
                c in "01" for c in key
            ):  # Check if basis state is in the qubit space
                s_qubit = Statevector.from_label(key)  # Computational qubit state
                projector += Operator(
                    s_qubit.data.reshape(total_qubit_dim, 1)
                    @ s.data.reshape(total_dim, 1).conj().T,
                    input_dims=tuple(initial_subsystem_dims),
                    output_dims=output_dims,
                )  # Add |s_qubit><s_qudit| to projector
                break
            else:
                continue
    return projector


def projected_state(
    state: np.ndarray | Statevector | DensityMatrix,
    subsystem_dims: List[int],
    normalize: bool = True,
) -> Statevector | DensityMatrix:
    """
    Project statevector on qubit space

    Args:
        state: State, given as numpy array or QuantumState object
        subsystem_dims: Subsystem dimensions
        normalize: Normalize statevector
    """
    if not isinstance(state, (np.ndarray, QuantumState)):
        raise TypeError("State must be either numpy array or QuantumState object")
    proj = build_qubit_space_projector(
        subsystem_dims
    )  # Projector on qubit space (in qudit space)
    if isinstance(state, np.ndarray):
        state_type = DensityMatrix if state.ndim == 2 else Statevector
        output_state: Statevector | DensityMatrix = state_type(state)
    else:
        output_state: Statevector | DensityMatrix = state
    qubitized_state = output_state.evolve(proj)

    if (
        normalize
    ) and qubitized_state.trace() != 0:  # Normalize the projected state (which is for now unnormalized due to selection of components)
        qubitized_state = (
            qubitized_state / qubitized_state.trace()
            if isinstance(qubitized_state, DensityMatrix)
            else qubitized_state / np.linalg.norm(qubitized_state.data)
        )

    return qubitized_state


def qubit_projection(
    unitary: np.ndarray | Operator, subsystem_dims: List[int]
) -> Operator:
    """
    Project unitary on qubit space

    Args:
        unitary: Unitary, given as numpy array or Operator object
        subsystem_dims: Subsystem dimensions

    Returns: unitary projected on qubit space as a Qiskit Operator object
    """

    proj = build_qubit_space_projector(
        subsystem_dims
    )  # Projector on qubit space (in qudit space)
    unitary_op = (
        Operator(
            unitary, input_dims=tuple(subsystem_dims), output_dims=tuple(subsystem_dims)
        )
        if isinstance(unitary, np.ndarray)
        else unitary
    )  # Unitary operator (in qudit space)

    qubitized_op = (
        proj @ unitary_op @ proj.adjoint()
    )  # Projected unitary (in qubit space)
    # (Note that is actually not unitary at this point, it's a Channel on the multi-qubit system)
    return qubitized_op
