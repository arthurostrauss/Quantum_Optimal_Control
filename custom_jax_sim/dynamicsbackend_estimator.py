from __future__ import annotations

from itertools import accumulate
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimator, EstimatorResult
from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.result import Result
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
            "parameter_values"
        ] = parameter_values  # To be given as PyTree alternatively
        self.backend.options.solver_options["observables"] = self.preprocessed_circuits[0][1]
        bound_circuits = [
            transpiled_circuits[circuit_index]
            if len(p) == 0
            else transpiled_circuits[circuit_index].bind_parameters(p)
            for i, (p, n) in enumerate(zip(parameter_dicts, num_observables))
            for circuit_index in range(accum[i], accum[i] + n)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)
        # Run
        result, metadata = _run_circuits(bound_circuits, self._backend, **run_options)

        return self._postprocessing(result, accum, metadata)
