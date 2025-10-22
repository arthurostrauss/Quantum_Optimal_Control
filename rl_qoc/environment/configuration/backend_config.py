from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union, Callable, Any, Optional, Dict, List

from qiskit import QiskitError, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers import BackendV2 as QiskitBackend
from qiskit.transpiler import (
    PassManager,
    InstructionDurations,
    Target,
    Layout,
    CouplingMap,
    generate_preset_pass_manager,
)
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES


@dataclass
class BackendConfig(ABC):
    """
    An abstract base class for backend configurations.

    This class provides a common interface for backend configurations, including methods for transpiling circuits
    and accessing backend properties.
    """

    parametrized_circuit: Optional[
        Callable[
            [
                QuantumCircuit,
                ParameterVector | List[Parameter],
                QuantumRegister,
                Any,
            ],
            None,
        ]
    ] = None
    backend: Optional[Any] = None
    parametrized_circuit_kwargs: Dict = field(default_factory=dict)
    skip_transpilation: bool = False
    pass_manager: Optional[Any] = None
    custom_instruction_durations: Optional[InstructionDurations] = None
    primitive_options: Optional[dict] = None

    @property
    @abstractmethod
    def config_type(self):
        """The type of the backend configuration."""
        pass

    def custom_transpile(self, qc_input, *args, **kwargs) -> QuantumCircuit | List[QuantumCircuit]:
        """
        Transpiles the given quantum circuit(s).

        Args:
            qc_input: The quantum circuit(s) to transpile.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The transpiled quantum circuit(s).
        """
        pass

    def as_dict(self):
        """
        Returns a dictionary representation of the backend configuration.

        Returns:
            A dictionary representation of the backend configuration.
        """
        return {
            "parametrized_circuit": self.parametrized_circuit,
            "backend": self.backend,
            "parametrized_circuit_kwargs": self.parametrized_circuit_kwargs,
            "pass_manager": self.pass_manager,
            "instruction_durations": self.custom_instruction_durations,
            "primitive_options": self.primitive_options,
        }

    @property
    def instruction_durations(self):
        """The instruction durations of the backend."""
        if (
            isinstance(self.backend, QiskitBackend)
            and self.backend.instruction_durations.duration_by_name_qubits
        ):
            return self.backend.instruction_durations
        return self.custom_instruction_durations


@dataclass
class QiskitConfig(BackendConfig):
    """
    A backend configuration for Qiskit backends.
    """

    backend: Optional[QiskitBackend] = None
    pass_manager: Optional[PassManager] = None

    @property
    def config_type(self):
        """The type of the backend configuration."""
        return "qiskit"

    def custom_transpile(
        self,
        qc_input: Union[QuantumCircuit, List[QuantumCircuit]],
        initial_layout: Optional[Layout] = None,
        scheduling: bool = True,
        optimization_level: int = 0,
        remove_final_measurements: bool = True,
    ):
        """
        Transpiles the given quantum circuit(s) using a Qiskit backend.

        Args:
            qc_input: The quantum circuit(s) to transpile.
            initial_layout: The initial layout for the transpilation.
            scheduling: Whether to perform scheduling.
            optimization_level: The optimization level for the transpilation.
            remove_final_measurements: Whether to remove final measurements.

        Returns:
            The transpiled quantum circuit(s).
        """
        if self.backend is None and self.instruction_durations is None and scheduling:
            raise QiskitError("Backend or instruction durations should be provided for scheduling")
        if remove_final_measurements:
            if isinstance(qc_input, QuantumCircuit):
                circuit = qc_input.remove_final_measurements(inplace=False)
            else:
                circuit = [circ.remove_final_measurements(inplace=False) for circ in qc_input]
        else:
            circuit = qc_input
        if isinstance(self.backend, QiskitBackend):
            num_qubits = self.backend.num_qubits
            coupling_map = self.backend.coupling_map
        elif isinstance(qc_input, QuantumCircuit):
            num_qubits = qc_input.num_qubits
            coupling_map = CouplingMap.from_full(num_qubits)
        else:
            num_qubits = max(qc.num_qubits for qc in qc_input)
            coupling_map = CouplingMap.from_full(num_qubits)
        if not self.skip_transpilation:
            if self.pass_manager is None:
                if self.backend is None:
                    target = Target.from_configuration(
                        self.basis_gates,
                        num_qubits,
                        coupling_map if coupling_map.size() != 0 else None,
                        instruction_durations=self.instruction_durations,
                        dt=self.dt,
                    )
                else:
                    target = self.backend.target
                self.pass_manager = generate_preset_pass_manager(
                    optimization_level=optimization_level,
                    backend=self.backend,
                    target=target if self.backend is None else None,
                    initial_layout=initial_layout,
                    scheduling_method=(
                        "asap" if self.instruction_durations is not None and scheduling else None
                    ),
                    translation_method="translator",
                )
            if self.backend is not None:
                circuit = self.pass_manager.run(circuit)
            else:
                if isinstance(circuit, QuantumCircuit):
                    circuit = circuit.decompose()
                else:
                    circuit = [circ.decompose() for circ in circuit]
        return circuit

    @property
    def basis_gates(self):
        """The basis gates of the backend."""
        return (
            self.backend.operation_names
            if isinstance(self.backend, QiskitBackend)
            else ["u", "rzx", "cx", "rz", "h", "s", "sdg", "x", "z", "measure", "reset"]
        )

    @property
    def dt(self):
        """The time unit of the backend."""
        return self.backend.dt if isinstance(self.backend, QiskitBackend) else 1e-9

    @property
    def instruction_durations(self):
        """The instruction durations of the backend."""
        return (
            self.backend.instruction_durations
            if isinstance(self.backend, QiskitBackend)
            and self.backend.instruction_durations.duration_by_name_qubits
            else self.custom_instruction_durations
        )

    @property
    def control_flow(self) -> bool:
        """Whether the backend supports control flow."""
        if isinstance(self.backend, QiskitBackend):
            if any(
                operation_name in CONTROL_FLOW_OP_NAMES
                for operation_name in self.backend.operation_names
            ):
                return True
        return False


@dataclass
class DynamicsConfig(QiskitConfig):
    """
    A backend configuration for Qiskit Dynamics backends.
    """

    calibration_files: Optional[str] = None
    do_calibrations: bool = True

    @property
    def config_type(self):
        """The type of the backend configuration."""
        return "dynamics"

    def as_dict(self):
        """
        Returns a dictionary representation of the backend configuration.

        Returns:
            A dictionary representation of the backend configuration.
        """
        return super().as_dict() | {
            "calibration_files": self.calibration_files,
            "do_calibrations": self.do_calibrations,
        }
