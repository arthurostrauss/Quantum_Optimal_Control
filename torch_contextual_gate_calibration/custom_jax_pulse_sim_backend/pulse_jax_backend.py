import copy
import uuid
import datetime

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_dynamics import DynamicsBackend, Solver, Signal
from typing import Optional, List, Union, Callable, Tuple
from qiskit_dynamics.backend.dynamics_job import DynamicsJob
from qiskit_dynamics.array import Array
from helper_functions import get_schedule_dict
from qiskit_dynamics.array import wrap
from qiskit import pulse

from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.backend.dynamics_backend import (
    _to_schedule_list,
    _validate_run_input,
    _get_acquire_instruction_timings,
    default_experiment_result_function,
)
from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.quantum_info import Statevector
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.transpiler import Target
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

    def __init__(self, solver: Solver, jittable_function: Optional[Callable] = None, rwa_cutoff_freq: Optional[float] = None,
                 rwa_carrier_freqs: Optional[Union[Array, Tuple[Array, Array]]] = None,
                 validate: bool = True):
        model = solver.model
        self.jit_function = jit(jittable_function)
        if isinstance(model, HamiltonianModel):
            super().__init__(model.static_operator, model.operators, static_dissipators=None, dissipator_operators=None,
                             hamiltonian_channels=solver._hamiltonian_channels, dissipator_channels=None,
                             channel_carrier_freqs=solver._channel_carrier_freqs, dt=solver._dt,
                             rotating_frame=model.rotating_frame, in_frame_basis=model.in_frame_basis,
                             evaluation_mode=model.evaluation_mode, rwa_cutoff_freq=rwa_cutoff_freq,
                             rwa_carrier_freqs=rwa_carrier_freqs, validate=validate)
        else:
            super().__init__(model.static_hamiltonian, model.hamiltonian_operators, model.static_dissipators,
                             model.dissipator_operators, solver._hamiltonian_channels, solver._dissipator_channels,
                             channel_carrier_freqs=solver._channel_carrier_freqs, dt=solver._dt,
                             rotating_frame=model.rotating_frame, in_frame_basis=model.in_frame_basis,
                             evaluation_mode=model.evaluation_mode, rwa_cutoff_freq=rwa_cutoff_freq,
                             rwa_carrier_freqs=rwa_carrier_freqs, validate=validate)

    def _solve_schedule_list_jax(
        self,
        t_span_list: List[Array],
        y0_list: List[Union[Array, QuantumState, BaseOperator]],
        schedule_list: List[Schedule],
        convert_results: bool = True,
        **kwargs,
    ) -> List[OdeResult]:
        # determine fixed array shape for containing all samples
        max_duration = 0
        schedule_dicts = []
        for idx, sched in enumerate(schedule_list):
            max_duration = max(sched.duration, max_duration)
            # TODO: Append necessary delays to have same duration for each schedule

        for sched in schedule_list:
            if sched.duration < max_duration:
                sched
        all_samples_shape = (len(self._all_channels), max_duration)

        # run simulations
        all_results = []
        schedule_dicts = [get_schedule_dict(sched) for sched in schedule_list]
