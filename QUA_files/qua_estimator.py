from __future__ import annotations

from collections import defaultdict
from itertools import accumulate
from typing import Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BackendEstimatorV2, PrimitiveResult, PubResult
from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qua_backend import QMBackend
from qualang_tools.video_mode import ParameterTable
from qiskit.transpiler import PassManager
from qiskit.pulse.library import SymbolicPulse
from typing import List
from qm.qua import *
from qm import QuantumMachinesManager, QuantumMachine, QmJob, Program

pauli_mapping = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def pauli_string_to_numbers(pauli_string: str):
    """
    This function converts a Pauli string to a list of numbers that represent the Pauli operators
    """
    return [pauli_mapping[pauli] for pauli in reversed(pauli_string)]


def numbers_to_pauli_string(pauli_numbers: List[int]):
    """
    This function converts a list of numbers that represent the Pauli operators to a Pauli string
    """
    return "".join(
        [key for key, value in pauli_mapping.items() if value in pauli_numbers]
    )


def pad_pauli_numbers(pauli_numbers: List[int], n_qubits: int):
    """
    This function pads a list of numbers that represent the Pauli operators to the desired number of qubits
    """
    return pauli_numbers + [0] * (n_qubits - len(pauli_numbers))


class QMEstimator(BackendEstimatorV2):
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
        backend: QMBackend,
        options: dict | None = None,
    ):
        """
        Args:
            backend: The backend to run the primitive on.
            options: The options to control the default precision (``default_precision``),
                the operator grouping (``abelian_grouping``), and
                the random seed for the simulator (``seed_simulator``).
        """
        SymbolicPulse.disable_validation = True
        if not isinstance(backend, QMBackend):
            raise TypeError("QMEstimator can only be used with a QMBackend.")
        super().__init__(backend=backend, options=options)

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        """
        Run the primitive on the backend.
        Args:
            pubs: The list of PUBs to run.

        Returns:
            The result of the run.
        """
        pub_dict = defaultdict(list)
        self.backend.run(self.estimator_qua_program())

    def estimator_qua_program(
        self,
        qc: QuantumCircuit,
        observables: ObservablesArray,
        parameter_values: BindingsArray,
    ):
        """
        Generate the QUA program for the estimator primitive
        """
        n_qubits = qc.num_qubits
        param_table = ParameterTable(
            {parameter_name: 0.0 for parameter_name in parameter_values.data.keys()}
        )

        with program() as estimator:
            n_shots = declare(int)
            observables_indices = declare_input_stream(
                int, name="observables_indices", size=n_qubits
            )
            param_vars = param_table.declare_variables()

            self.backend.schedule_to_qua_program(qc, param_vars)

            for q in range(n_qubits):
                with switch_(observables_indices[q]):
                    with case_(1):
                        self.backend.schedule_to_qua_program()
