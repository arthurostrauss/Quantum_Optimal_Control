from __future__ import annotations

import time

from qiskit.circuit import Gate, QuantumCircuit
from qiskit import QiskitError
from qiskit.quantum_info import Operator, SuperOp, DensityMatrix, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_dynamics import Solver, RotatingFrame, solve_lmde
from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.solvers.solver_classes import (
    is_lindblad_model_vectorized,
    is_lindblad_model_not_vectorized,
    format_final_states,
    initial_state_converter,
    validate_and_format_initial_state,
)
from typing import Optional, List, Union, Callable, Tuple, Type, Any

from qiskit.pulse import Schedule, SymbolicPulse, ScheduleBlock
from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS as numpy_alias
from qiskit_dynamics.arraylias.alias import _isArrayLike
from qiskit_dynamics import ArrayLike

from jax import vmap, jit, numpy as jnp
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult


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
        if dim > 1:
            qud_op = np.identity(dim, dtype=np.complex64)
            qud_op[:2, :2] = op.to_matrix()
            qudit_op_list.append(qud_op)
    complete_op = Operator(qudit_op_list[0])
    for i in range(1, len(qudit_op_list)):
        complete_op = complete_op.tensor(Operator(qudit_op_list[i]))
    assert complete_op.is_unitary(), "The operator is not unitary"
    assert (
        complete_op.input_dims()
        == complete_op.output_dims()
        == tuple(filter(lambda x: x > 1, subsystem_dims))
    ), "The operator is not the right dimension"
    return complete_op


class JaxSolver(Solver):
    """This custom Solver behaves exactly like the original Solver object except one difference, the user
    provides the function that can be jitted for faster simulations (should be provided to the class
    non-jit compiled)"""

    def __init__(
        self,
        static_hamiltonian: Optional[ArrayLike] = None,
        hamiltonian_operators: Optional[ArrayLike] = None,
        static_dissipators: Optional[ArrayLike] = None,
        dissipator_operators: Optional[ArrayLike] = None,
        hamiltonian_channels: Optional[List[str]] = None,
        dissipator_channels: Optional[List[str]] = None,
        channel_carrier_freqs: Optional[dict] = None,
        dt: Optional[float] = None,
        rotating_frame: Optional[Union[ArrayLike, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        array_library: Optional[str] = None,
        vectorized: Optional[bool] = None,
        rwa_cutoff_freq: Optional[float] = None,
        rwa_carrier_freqs: Optional[
            Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
        ] = None,
        validate: bool = True,
        schedule_func: Optional[Callable[[], Schedule]] = None,
    ):
        """Initialize solver with model information.

        Args:
            static_hamiltonian: Constant Hamiltonian term. If a ``rotating_frame``
                is specified, the ``frame_operator`` will be subtracted from the static_hamiltonian.
            hamiltonian_operators: Hamiltonian operators.
            static_dissipators: Constant dissipation operators.
            dissipator_operators: Dissipation operators with time-dependent coefficients.
            hamiltonian_channels: List of channel names in pulse schedules corresponding to
                Hamiltonian operators.
            dissipator_channels: List of channel names in pulse schedules corresponding to
                dissipator operators.
            channel_carrier_freqs: Dictionary mapping channel names to floats which represent the
                carrier frequency of the pulse channel with the corresponding name.
            dt: Sample rate for simulating pulse schedules.
            rotating_frame: Rotating frame to transform the model into. Rotating frames which are
                diagonal can be supplied as a 1d array of the diagonal elements, to explicitly
                indicate that they are diagonal.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                frame operator is diagonalized. See class documentation for a more detailed
                explanation on how this argument affects object behaviour.
            array_library: Array library to use for storing operators of underlying model. See the
                :ref:`model evaluation section of the Models API documentation <model evaluation>`
                for a more detailed description of this argument.
            vectorized: If including dissipator terms, whether or not to construct the
                :class:`.LindbladModel` in vectorized form. See the
                :ref:`model evaluation section of the Models API documentation <model evaluation>`
                for a more detailed description of this argument.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency. If ``None``, no
                approximation is made.
            rwa_carrier_freqs: Carrier frequencies to use for rotating wave approximation.
                If no time dependent coefficients in model leave as ``None``, if no time-dependent
                dissipators specify as a list of frequencies for each Hamiltonian operator, and if
                time-dependent dissipators present specify as a tuple of lists of frequencies, one
                for Hamiltonian operators and one for dissipators.
            validate: Whether or not to validate Hamiltonian operators as being Hermitian.
            schedule_func: Callable or list of Callables taking as inputs arrays such that parametrized pulse simulation can be done in an
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
            array_library,
            vectorized,
            rwa_cutoff_freq,
            rwa_carrier_freqs,
            validate,
        )
        self._y0 = None
        self.stored_results = []
        self.observables = []
        self._schedule_func = schedule_func
        self._jit_func = None
        self._unitary_jit_func = None
        self._param_values = None
        self._param_names = None
        self._subsystem_dims = None
        self._t_span = None
        self._batched_sims = None
        self._kwargs = None
        self.circuit_macro_counter = 0
        SymbolicPulse.disable_validation = True

    @property
    def circuit_macro(self):
        return self._schedule_func

    def get_signals(self, params):
        """
        This method generates the call to the circuit macro and sets the signals of the model.
        It also returns the signals of the model before the new signals are set for easy resetting.

        Args:
            params: The parameters to be assigned to the parametrized schedule
        """
        parametrized_schedule = self.circuit_macro()
        model_sigs = self.model.signals
        if parametrized_schedule.is_parameterized():
            parametrized_schedule.assign_parameters(
                {
                    param_obj: param
                    for (param_obj, param) in zip(self._param_names, params)
                }
            )
        signals = self._schedule_to_signals(parametrized_schedule)
        self._set_new_signals(signals)
        return model_sigs

    @circuit_macro.setter
    def circuit_macro(self, func):
        """
        This setter should be done each time one wants to switch the target circuit truncation
        """
        self._schedule_func = func

        def run_sim_function(t_span, y0, params, y0_input, y0_cls):
            model_sigs = self.get_signals(params)
            results = solve_lmde(
                generator=self.model, t_span=t_span, y0=y0, **self._kwargs
            )
            results.y = format_final_states(results.y, self.model, y0_input, y0_cls)
            self.model.signals = model_sigs

            return results.t, results.y

        def unitary_sim_function(t_span, params):
            model_sigs = self.get_signals(params)
            results = solve_lmde(
                generator=self.model,
                t_span=t_span,
                y0=jnp.eye(np.prod(self._subsystem_dims)),
                **self._kwargs,
            )

            results.y.at[-1].set(
                self.model.rotating_frame.state_out_of_frame(t_span[-1], results.y[-1])
            )
            self.model.signals = model_sigs

            return results.t, results.y

        self._jit_func = jit(
            vmap(run_sim_function, in_axes=(None, None, 0, None, None)),
            static_argnums=(4,),
        )
        self._unitary_jit_func = jit(vmap(unitary_sim_function, in_axes=(None, 0)))

    @property
    def batched_sims(self):
        return self._batched_sims

    def unitary_solve(self, param_values: Optional[ArrayLike] = None):
        """
        This method is used to solve the unitary evolution of the system and get the total unitary
        (not just the final state)
        It assumes that a DynamicsEstimator job has been run previously
        """
        if self.circuit_macro is None:
            raise ValueError(
                "No circuit macro has been provided, please provide a circuit macro"
            )
        assert isinstance(
            self.model, HamiltonianModel
        ), "Model must be a HamiltonianModel"
        batch_results_t, batch_results_y = self._unitary_jit_func(
            self._t_span, param_values
        )

        return np.array(batch_results_y)

    def _solve_schedule_list_jax(
        self,
        t_span_list: List[ArrayLike],
        y0_list: List[Union[ArrayLike, QuantumState, BaseOperator]],
        schedule_list: List[Schedule],
        convert_results: bool = True,
        **kwargs,
    ) -> List[OdeResult]:
        """
        This method is overriding the one from the original Solver and is used to solve the
        dynamics of the system using the jax backend for the specific use case of the DynamicsEstimator.
        It assumes that a DynamicsEstimator job has been run previously and that the user has provided
        the parameter values and the observables to be measured as solver options
        """
        if (
            "parameter_dicts" not in kwargs
            or "parameter_values" not in kwargs
            or "observables" not in kwargs
            or "subsystem_dims" not in kwargs
        ):
            # If the user is not using the estimator, then we can just use the original solver method
            return super()._solve_schedule_list_jax(
                t_span_list, y0_list, schedule_list, convert_results, **kwargs
            )
        else:
            estimator_usage = False  # Flag to check if the estimator is being used
            if "observables" in kwargs and "subsystem_dims" in kwargs:
                estimator_usage = True
                observables_circuits: List[QuantumCircuit] = [
                    circ.remove_final_measurements(inplace=False)
                    for circ in kwargs["observables"]
                ]

                pauli_rotations = [
                    [Operator.from_label("I") for _ in range(circ.num_qubits)]
                    for circ in observables_circuits
                ]
                for i, circuit in enumerate(observables_circuits):
                    qubit_counter, qubit_list = 0, []

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
                subsystem_dims = self._subsystem_dims = kwargs["subsystem_dims"]
                observables = [
                    PauliToQuditOperator(pauli_rotations[i], subsystem_dims)
                    for i in range(len(pauli_rotations))
                ]
                self.observables = observables
                kwargs.pop("observables")
                kwargs.pop("subsystem_dims")

            self._param_names = kwargs["parameter_dicts"][0].keys()
            param_values = self._param_values = kwargs["parameter_values"]

            if self.circuit_macro is None:
                raise ValueError(
                    "No circuit macro has been provided, please provide a circuit macro"
                )

            for key in [
                "parameter_dicts",
                "parameter_values",
            ]:
                kwargs.pop(key)
            self._kwargs = kwargs

            all_results = []

            # setup initial state
            (
                y0,
                y0_input,
                y0_cls,
                state_type_wrapper,
            ) = validate_and_format_initial_state(y0_list[0], self.model)
            t_span = self._t_span = t_span_list[0]
            start_time = time.time()

            batch_results_t, batch_results_y = self._jit_func(
                unp.asarray(t_span),
                unp.asarray(y0),
                unp.asarray(param_values),
                unp.asarray(y0_input),
                y0_cls,
            )
            print("Time to run simulation: ", time.time() - start_time)
            self._batched_sims = batch_results_y
            if estimator_usage:
                for results_t, results_y in zip(batch_results_t, batch_results_y):
                    for observable in observables:
                        results = OdeResult(t=results_t, y=results_y)
                        if y0_cls is not None and convert_results:
                            results.y = [
                                y0_cls(np.array(yi), dims=subsystem_dims)
                                for yi in results.y
                            ]

                            # Rotate final state with Pauli basis rotations to sample all corresponding Pauli observables
                            results.y[-1] = results.y[-1].evolve(observable)
                            self.stored_results.append(results.y[-1])
                        all_results.append(results)
            else:
                for results_t, results_y in zip(batch_results_t, batch_results_y):
                    results = OdeResult(t=results_t, y=results_y)
                    if y0_cls is not None and convert_results:
                        results.y = [y0_cls(np.array(yi)) for yi in results.y]
                    all_results.append(results)

            return all_results

    def solve_param_schedule(
        self,
        parameter_values: ArrayLike,
        t_span: Optional[ArrayLike] = None,
        y0: Optional[ArrayLike | QuantumState | BaseOperator] = None,
        schedule: Optional[Schedule | ScheduleBlock] = None,
        convert_results: bool = True,
        **kwargs,
    ):
        """
        Solve a parametrized schedule for a given set of parameters.

        Args:
            parameter_values: The parameter values to use for the simulation.
            t_span: The time span to simulate over (if not specified, standard [0, sched.duration] is used
            y0: The initial state to simulate from.
            schedule: The schedule to simulate.
            convert_results: Whether to convert the results to the correct state type.
            kwargs: Additional solver options.
        """
        if schedule is not None:
            self.circuit_macro = lambda: schedule
        if t_span is not None:
            self._t_span = t_span
        if y0 is not None:
            self._y0 = y0
        if self._y0 is None:
            self._y0 = Statevector.from_int(0, self.model.dim)
        if self._t_span is None:
            self._t_span = [0, schedule.duration]
        (
            y0,
            y0_input,
            y0_cls,
            state_type_wrapper,
        ) = validate_and_format_initial_state(y0, self.model)
        self._param_values = parameter_values

        return self._jit_func(t_span, y0, parameter_values, y0_input, y0_cls)
