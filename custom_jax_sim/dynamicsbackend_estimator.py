from __future__ import annotations

from itertools import accumulate
from typing import Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BackendEstimator, EstimatorResult
from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_dynamics.backend import DynamicsBackend
from qiskit.transpiler import PassManager


class DynamicsBackendEstimator(BackendEstimator):
    """Evaluates expectation value using Pauli rotation gates.

    This :class:`~.DynamicsBackendEstimator` class is a specific implementation of the
    :class:`~.BaseEstimator` interface that is used to wrap a :class:`~.DynamicsBackend`
    object in the :class:`~.BaseEstimator` API. This version slightly changes from the original BackendEstimator in
    the main function, which essentially loads parameters values as well as other useful parameters that can be used
    for running the RL training (typically, observables information and the initial state that should be priorly set in
    the estimator options before calling estimator.run()) into the backend and solver options.

    """

    def __init__(
        self,
        backend: DynamicsBackend,
        options: dict | None = None,
        abelian_grouping: bool = True,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
    ):
        if not isinstance(backend, DynamicsBackend):
            raise TypeError(
                "DynamicsBackendEstimator can only be used with a DynamicsBackend."
            )
        super().__init__(
            backend, options, abelian_grouping, bound_pass_manager, skip_transpilation
        )

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        # Transpile
        self._grouping = list(zip(circuits, observables))
        transpiled_circuits = self.transpiled_circuits
        num_observables = [len(m) for (_, m) in self.preprocessed_circuits]
        accum = [0] + list(accumulate(num_observables))

        # Bind parameters
        parameter_dicts = [
            dict(zip(self._parameters[i], value))
            for i, value in zip(circuits, parameter_values)
        ]

        self.backend.options.solver_options[
            "parameter_dicts"
        ] = parameter_dicts  # To be given as PyTree
        self.backend.options.solver_options[
            "subsystem_dims"
        ] = self.backend.options.subsystem_dims
        self.backend.options.solver_options[
            "parameter_values"
        ] = parameter_values  # To be given as PyTree alternatively
        self.backend.options.solver_options["observables"] = transpile(
            self.preprocessed_circuits[0][1], self.backend
        )
        self.backend.set_options(**run_options)
        run_options = {}
        bound_circuits = [
            transpiled_circuits[circuit_index]
            if len(p) == 0
            else transpiled_circuits[circuit_index].bind_parameters(p)
            for i, (p, n) in enumerate(zip(parameter_dicts, num_observables))
            for circuit_index in range(accum[i], accum[i] + n)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)
        new_bound_circuits = []
        for _ in range(len(parameter_values)):
            for circ in bound_circuits:
                new_bound_circuits.append(circ.copy())
        accum = [0] + list(accumulate([num_observables[0]] * len(parameter_values)))
        # Run
        result, metadata = _run_circuits(
            new_bound_circuits, self._backend, **run_options
        )
        for option in [
            "parameter_dicts",
            "subsystem_dims",
            "parameter_values",
            "observables",
        ]:
            self.backend.options.solver_options.pop(option)
        self.backend.set_options(initial_state="ground_state")
        return self._postprocessing(result, accum, metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[BaseOperator, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ):
        custom_circuit_list = (circuits[0],)
        custom_observation_list = (observables[0],)
        return super()._run(
            custom_circuit_list,
            custom_observation_list,
            parameter_values,
            **run_options,
        )
