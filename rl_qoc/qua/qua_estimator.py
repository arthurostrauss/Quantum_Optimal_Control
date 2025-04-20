from __future__ import annotations

import math
from collections import defaultdict
from itertools import accumulate
from typing import Sequence, Iterable
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import (
    BackendEstimatorV2,
    PrimitiveResult,
    PubResult,
    BaseEstimatorV2,
    EstimatorPubLike,
    BasePrimitiveJob,
)
from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from qm_backend import QMBackend
from qualang_tools.video_mode import ParameterTable
from qiskit.transpiler import PassManager, PassManagerConfig
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


@dataclass
class Options:
    """Options for :class:`~.BackendEstimatorV2`."""

    default_precision: float = 0.015625
    """The default precision to use if none are specified in :meth:`~run`.
    Default: 0.015625 (1 / sqrt(4096)).
    """

    abelian_grouping: bool = True
    """Whether the observables should be grouped into sets of qubit-wise commuting observables.
    Default: True.
    """


class QMEstimator(BaseEstimatorV2):
    """Evaluates expectation value using Pauli rotation gates.

    This :class:`~.QMEstimator` class is a specific implementation of the
    :class:`~.BaseEstimator` interface that is used to wrap a :class:`~.QMBackend`
    object in the :class:`~.BaseEstimator` API. This version slightly changes from the original BackendEstimatorV2,
    as it leverages the QOP input streams mechanism to load both the observables and the parameter values, making the
    whole PUB evaluation process more efficient. Circuit parameters are typically updated within the QUA program,
    and the observables translate into qubit-wise switch-case statements that select the corresponding Pauli rotation
    to measure in the appropriate basis.
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
        if not isinstance(backend, QMBackend):
            raise TypeError("QMEstimator can only be used with a QMBackend.")
        self._backend = backend
        self._options = Options(**options) if options else Options()

        basis = PassManagerConfig.from_backend(backend).basis_gates
        opt1q = Optimize1qGatesDecomposition(basis, target=backend.target)
        self._passmanager = PassManager([opt1q])

    @property
    def options(self) -> Options:
        """Return the options for the estimator."""
        return self._options

    @property
    def backend(self) -> QMBackend:
        """Return the backend for the estimator."""
        return self._backend

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> BasePrimitiveJob[PrimitiveResult[PubResult]]:
        """
        Run the primitive on the backend.
        """

        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)

        programs = [self.estimator_qua_program(pub) for pub in coerced_pubs]

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        """
        Validate the PUBs to run, and add necessary metadata to run it on QM backend
        (namely, a ParameterTable and a set of updated pulse operations on relevant elements
        if custom calibrations are attached to the circuit).
        """
        for i, pub in enumerate(pubs):
            if pub.precision <= 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than or equal to 0 ({pub.precision}). ",
                    "But precision should be larger than 0.",
                )
            self.backend.update_calibrations(pub.circuit)

    def estimator_qua_program(
        self,
        pub: EstimatorPub,
    ):
        """
        Generate the QUA program for the estimator primitive
        """
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision
        shots = int(math.ceil(1.0 / precision**2))

        parameter_table = circuit.metadata.get("parameter_table", None)

        with program() as estimator_prog:
            pass
