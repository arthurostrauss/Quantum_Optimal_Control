from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_dynamics import Solver, RotatingFrame, solve_lmde
from qiskit_dynamics.solvers.solver_classes import (
    format_final_states,
    validate_and_format_initial_state,
)
from typing import Optional, List, Union, Callable, Tuple

from qiskit import QuantumCircuit

from qiskit.quantum_info import Operator
from qiskit.pulse import Schedule, SymbolicPulse
from qiskit_dynamics.array import Array
from qiskit_dynamics.array import wrap
from jax import vmap, jit, numpy as jnp
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

jit_wrap = wrap(jit, decorator=True)


# qd_vmap = wrap(vmap, decorator=True)


def PauliToQuditOperator(qubit_ops: List[Operator], subsystem_dims: List[int]):
    """
    This function operates very similarly to SparsePauliOp from Qiskit, except this can produce
    arbitrary dimension qudit operators that are the equivalent to the Qubit Operators desired.

    This functionality is useful for qudit simulations of standard qubit workflows like state preparation
    and choosing measurement observables, without losing any information from the simulation.

    All operators produced remain as unitaries.
    """
    qudit_op_list = []
    for op, dim in zip(qubit_ops, subsystem_dims):
        qud_op = np.identity(dim, dtype=np.complex64)
        qud_op[:2, :2] = op.to_matrix()
        qudit_op_list.append(qud_op)
    complete_op = qudit_op_list[0]
    for i in range(1, len(qudit_op_list)):
        complete_op = np.kron(complete_op, qudit_op_list[i])
    return Operator(
        complete_op, input_dims=tuple(subsystem_dims), output_dims=tuple(subsystem_dims)
    )


class JaxSolver(Solver):
    """This custom Solver behaves exactly like the original Solver object except one difference, the user
    provides the function that can be jitted for faster simulations (should be provided to the class
    non-jit compiled)"""

    def __init__(
            self,
            static_hamiltonian: Optional[Array] = None,
            hamiltonian_operators: Optional[Array] = None,
            static_dissipators: Optional[Array] = None,
            dissipator_operators: Optional[Array] = None,
            hamiltonian_channels: Optional[List[str]] = None,
            dissipator_channels: Optional[List[str]] = None,
            channel_carrier_freqs: Optional[dict] = None,
            dt: Optional[float] = None,
            rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
            in_frame_basis: bool = False,
            evaluation_mode: str = "dense",
            rwa_cutoff_freq: Optional[float] = None,
            rwa_carrier_freqs: Optional[Union[Array, Tuple[Array, Array]]] = None,
            validate: bool = True,
            schedule_func: Optional[Callable[[], Schedule]] = None,
    ):
        """Initialize solver with model information.

        Args:
            static_hamiltonian: Constant Hamiltonian term. If a ``rotating_frame``
                                is specified, the ``frame_operator`` will be subtracted from
                                the static_hamiltonian.
            hamiltonian_operators: Hamiltonian operators.
            static_dissipators: Constant dissipation operators.
            dissipator_operators: Dissipation operators with time-dependent coefficients.
            hamiltonian_channels: List of channel names in pulse schedules corresponding to
                                  Hamiltonian operators.
            dissipator_channels: List of channel names in pulse schedules corresponding to
                                 dissipator operators.
            channel_carrier_freqs: Dictionary mapping channel names to floats which represent
                                   the carrier frequency of the pulse channel with the
                                   corresponding name.
            dt: Sample rate for simulating pulse schedules.
            rotating_frame: Rotating frame to transform the model into. Rotating frames which
                            are diagonal can be supplied as a 1d array of the diagonal elements,
                            to explicitly indicate that they are diagonal.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                            frame operator is diagonalized. See class documentation for a more
                            detailed explanation on how this argument affects object behaviour.
            evaluation_mode: Method for model evaluation. See documentation for
                             ``HamiltonianModel.evaluation_mode`` or
                             ``LindbladModel.evaluation_mode``.
                             (if dissipators in model) for valid modes.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency. If ``None``, no
                             approximation is made.
            rwa_carrier_freqs: Carrier frequencies to use for rotating wave approximation.
                               If no time dependent coefficients in model leave as ``None``,
                               if no time-dependent dissipators specify as a list of frequencies
                               for each Hamiltonian operator, and if time-dependent dissipators
                               present specify as a tuple of lists of frequencies, one for
                               Hamiltonian operators and one for dissipators.
            validate: Whether or not to validate Hamiltonian operators as being Hermitian.
            jittable_func: Callable or list of Callables taking as inputs arrays such that parametrized pulse simulation can be done in an
                           optimized manner

        Raises:
            QiskitError: If arguments concerning pulse-schedule interpretation are insufficiently
            specified.
        """
        super().__init__(
            static_hamiltonian,
            hamiltonian_operators,
            static_dissipators,
            dissipator_operators,
            hamiltonian_channels,
            dissipator_channels,
            channel_carrier_freqs,
            dt,
            rotating_frame,
            in_frame_basis,
            evaluation_mode,
            rwa_cutoff_freq,
            rwa_carrier_freqs,
            validate,
        )
        self._schedule_func = schedule_func
        self._param_values = None
        self._param_names = None
        self._subsystem_dims = None
        self._t_span = None
        self._batched_sims = []
        self._kwargs = None
        SymbolicPulse.disable_validation = True

    @property
    def circuit_macro(self):
        return self._schedule_func

    @circuit_macro.setter
    def circuit_macro(self, func):
        """
        This setter should be done each time one wants to switch the target circuit truncation
        """
        self._schedule_func = func

    @property
    def batched_sims(self):
        return self._batched_sims

    def unitary_solve(self, param_values):
        """
        This method is used to solve the unitary evolution of the system and get the total unitary
        (not just the final state)
        It assumes that a DynamicsEstimator job has been run previously
        """
        if self.circuit_macro is None:
            raise ValueError(
                "No circuit macro has been provided, please provide a circuit macro"
            )

        def sim_function(t_span, params):
            parametrized_schedule = self.circuit_macro()
            model_sigs = self.model.signals

            parametrized_schedule.assign_parameters(
                {
                    param_obj: param
                    for (param_obj, param) in zip(self._param_names, params)
                }
            )
            signals = self._schedule_to_signals(parametrized_schedule)
            self._set_new_signals(signals)
            results = solve_lmde(
                generator=self.model,
                t_span=t_span,
                y0=jnp.eye(
                    np.prod(self._subsystem_dims), np.prod(self._subsystem_dims)
                ),
                **self._kwargs,
            )
            self.model.signals = model_sigs  # reset signals to original
            return Array(results.t).data, Array(results.y).data

        jit_func = jit(vmap(sim_function, in_axes=(None, 0)))
        batch_results_t, batch_results_y = jit_func(
            Array(self._t_span).data, Array(param_values).data
        )
        return batch_results_y

    def _solve_schedule_list_jax(
            self,
            t_span_list: List[Array],
            y0_list: List[Union[Array, QuantumState, BaseOperator]],
            schedule_list: List[Schedule],
            convert_results: bool = True,
            **kwargs,
    ) -> List[OdeResult]:
        if (
                "parameter_dicts" not in kwargs
                or "parameter_values" not in kwargs
                or "observables" not in kwargs
        ):
            # If the user is not using the estimator, then we can just use the original solver method
            return super()._solve_schedule_list_jax(
                t_span_list, y0_list, schedule_list, convert_results, **kwargs
            )
        else:
            # Otherwise, we need to load the parameters and observables from the solver options
            param_dicts = kwargs["parameter_dicts"]
            param_names = self._param_names = param_dicts[0].keys()
            param_values = self._param_values = kwargs["parameter_values"]
            subsystem_dims = self._subsystem_dims = kwargs["subsystem_dims"]
            if self.circuit_macro is None:
                raise ValueError(
                    "No circuit macro has been provided, please provide a circuit macro"
                )

            # print(kwargs["observables"])
            observables_circuits: List[QuantumCircuit] = [
                circ.remove_final_measurements(inplace=False)
                for circ in kwargs["observables"]
            ]
            # print(observables_circuits)
            pauli_rotations = [
                [Operator.from_label("I") for _ in range(circ.num_qubits)]
                for circ in observables_circuits
            ]
            for i, circuit in enumerate(observables_circuits):
                qubit_counter = 0
                qubit_list = []
                for circuit_instruction in circuit.data:
                    assert (
                            len(circuit_instruction.qubits) == 1
                    ), "Operation non local, need local rotations"
                    if circuit_instruction.qubits[0] not in qubit_list:
                        qubit_list.append(circuit_instruction.qubits[0])
                        qubit_counter += 1

                    pauli_rotations[i][qubit_counter - 1] = pauli_rotations[i][
                        qubit_counter - 1
                        ].compose(Operator(circuit_instruction.operation))

            observables = [
                PauliToQuditOperator(pauli_rotations[i], subsystem_dims)
                for i in range(len(pauli_rotations))
            ]

            for key in [
                "parameter_dicts",
                "parameter_values",
                "observables",
                "subsystem_dims",
            ]:
                kwargs.pop(key)
            self._kwargs = kwargs

            def sim_function(t_span, y0, params, y0_input, y0_cls):
                parametrized_schedule = self.circuit_macro()
                model_sigs = self.model.signals

                parametrized_schedule.assign_parameters(
                    {
                        param_obj: param
                        for (param_obj, param) in zip(param_names, params)
                    }
                )
                signals = self._schedule_to_signals(parametrized_schedule)
                self._set_new_signals(signals)
                results = solve_lmde(
                    generator=self.model, t_span=t_span, y0=y0, **kwargs
                )
                results.y = format_final_states(results.y, self.model, y0_input, y0_cls)
                self.model.signals = model_sigs

                return Array(results.t).data, Array(results.y).data

            jit_func = jit(
                vmap(sim_function, in_axes=(None, None, 0, None, None)),
                static_argnums=(4,),
            )
            all_results = []

            # setup initial state
            (
                y0,
                y0_input,
                y0_cls,
                state_type_wrapper,
            ) = validate_and_format_initial_state(y0_list[0], self.model)
            t_span = self._t_span = t_span_list[0]

            batch_results_t, batch_results_y = jit_func(
                Array(t_span).data,
                Array(y0).data,
                Array(param_values).data,
                Array(y0_input).data,
                y0_cls,
            )

            self._batched_sims.append(batch_results_y)

            for results_t, results_y in zip(batch_results_t, batch_results_y):
                for observable in observables:
                    results = OdeResult(
                        t=results_t, y=Array(results_y, backend="jax", dtype=complex)
                    )
                    if y0_cls is not None and convert_results:
                        results.y = [state_type_wrapper(yi) for yi in results.y]
                        # Rotate final state with Pauli basis rotations to sample all corresponding Pauli observables
                        results.y = [yi.evolve(observable) for yi in results.y]
                    all_results.append(results)

            return all_results
