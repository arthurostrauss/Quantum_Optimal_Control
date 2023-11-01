import copy
import uuid
import datetime

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_dynamics import DynamicsBackend, Solver, Signal, RotatingFrame
from qiskit_dynamics.solvers.solver_classes import format_final_states, validate_and_format_initial_state
from typing import Optional, List, Union, Callable, Tuple

from qiskit_dynamics.array import wrap
from qiskit import pulse

from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.models.hamiltonian_model import is_hermitian
from qiskit_dynamics.type_utils import to_numeric_matrix_type
from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.quantum_info import Statevector
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit_dynamics.array import Array
import jax
import jax.numpy as jnp
from jax import block_until_ready, vmap
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

jit = wrap(jax.jit, decorator=True)
qd_vmap = wrap(vmap, decorator=True)

Runnable = Union[QuantumCircuit, Schedule, ScheduleBlock]


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


    @property
    def circuit_macro(self):
        return self._schedule_func

    @circuit_macro.setter
    def set_macro(self, func):
        """
        This setter should be done each time one wants to switch the target circuit truncation
        """
        self._schedule_func = func

    def _solve_schedule_list_jax(
        self,
        t_span_list: List[Array],
        y0_list: List[Union[Array, QuantumState, BaseOperator]],
        schedule_list: List[Schedule],
        convert_results: bool = True,
        **kwargs,
    ) -> List[OdeResult]:

        param_dicts = kwargs["parameter_dicts"]
        relevant_params = param_dicts[0].keys()
        observables_circuits = kwargs["observables"]
        param_values = kwargs["parameter_values"]
        for key in ["parameter_dicts", "parameter_values", "parameter_values"]:
            kwargs.pop(key)

        def sim_function(params, t_span, y0_input, y0_cls):
            parametrized_schedule = self.circuit_macro()

            parametrized_schedule.assign_parameters({param_obj: param
                                                     for (param_obj, param) in zip(relevant_params, params)})
            signals = self._schedule_converter.get_signals(parametrized_schedule)
            
            # Perhaps replace below by solve_lmde
            results = self.solve(t_span, y0_input, signals, **kwargs)
            results.y = format_final_states(results.y, self.model, y0_input, y0_cls)

            return Array(results.t).data, Array(results.y).data

