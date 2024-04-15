from __future__ import annotations

from itertools import accumulate
from typing import Sequence

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.qiskit.primitives import BackendEstimatorV2
from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qua_backend import QuaBackend
from qiskit.transpiler import PassManager


class QuaEstimator(BackendEstimatorV2):
    """Evaluates expectation value using Pauli rotation gates.

    This :class:`~.QuaEstimator` class is a specific implementation of the
    :class:`~.BaseEstimator` interface that is used to wrap a :class:`~.QuaBackend`
    object in the :class:`~.BaseEstimator` API. This version slightly changes from the original BackendEstimator in
    the main function, which essentially loads parameters values as well as other useful parameters that can be used
    for running the RL training (typically, observables information and the initial state that should be priorly set in
    the estimator options before calling estimator.run()) into the backend and solver options.

    """

    def __init__(
        self,
        *,
        backend: QuaBackend,
        options: dict | None = None,
    ):
        """
        Args:
            backend: The backend to run the primitive on.
            options: The options to control the default precision (``default_precision``),
                the operator grouping (``abelian_grouping``), and
                the random seed for the simulator (``seed_simulator``).
        """
        if not isinstance(backend, QuaBackend):
            raise TypeError("QuaEstimator can only be used with a QuaBackend.")
        super().__init__(backend=backend, options=options)
        for g in ["h", "sdg"]:
            self.backend.options.solver.run_options[g + "_cal"] = [
                lambda: self.backend.target.get_calibration(g, (qubit,))
                for qubit in range(self.backend.num_qubits)
            ]
        self.backend.options.solver.run_options["subsystem_dims"] = (
            self.backend.options.subsystem_dims
        )
        self._mean_action = None
        self._std_action = None

    @property
    def mean_action(self):
        return self._mean_action

    @property
    def std_action(self):
        return self._std_action

    @mean_action.setter
    def mean_action(self, value):
        self._mean_action = value

    @std_action.setter
    def std_action(self, value):
        self._std_action = value
