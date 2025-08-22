from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union, Callable, Any, Optional, Dict, Iterable, List

from qiskit import QiskitError, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers import BackendV2 as QiskitBackend
from qiskit.transpiler import(PassManager,
                              InstructionDurations,
                              Target, Layout, CouplingMap,
                              generate_preset_pass_manager)
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES


@dataclass
class BackendConfig(ABC):
    """
    Abstract base class for backend configurations.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager: Pass manager to transpile the circuit
        instruction_durations: Dictionary containing the durations of the instructions in the circuit
        primitive_options: Options to feed the primitives (estimator or sampler). If None, the default options of each primitive are used.

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
        pass

    def custom_transpile(self, qc_input, *args, **kwargs) -> QuantumCircuit | List[QuantumCircuit]:
        pass

    def as_dict(self):
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
        return self.custom_instruction_durations

@dataclass
class QiskitConfig(BackendConfig):
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager
        instruction_durations: Dictionary containing the durations of the instructions in the circuit
    """

    backend: Optional[QiskitBackend] = None
    pass_manager: Optional[PassManager] = None

    @property
    def config_type(self):
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
        Custom transpile function to transpile the quantum circuit
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
        """
        Retrieve the basis gates of the backend (default is ['x', 'sx', 'cx', 'rz', 'measure', 'reset'])
        """
        return (
            self.backend.operation_names
            if isinstance(self.backend, QiskitBackend)
            else ["u", "rzx", "cx", "rz", "h", "s", "sdg", "x", "z", "measure", "reset"]
        )

    @property
    def dt(self):
        """
        Retrieve the time unit of the backend (default is 1e-9)
        """
        return self.backend.dt if isinstance(self.backend, QiskitBackend) else 1e-9

    @property
    def instruction_durations(self):
        """
        Retrieve the instruction durations of the backend (default is None)
        """
        return (
            self.backend.instruction_durations
            if isinstance(self.backend, QiskitBackend)
            and self.backend.instruction_durations.duration_by_name_qubits
            else self.custom_instruction_durations
        )

    @property
    def control_flow(self) -> bool:
        """
        Assess if the backend supports real-time control flow
        """
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
    Qiskit Dynamics configuration elements.

    Args:
        calibration_files: load existing gate calibrations from json file for DynamicsBackend
        do_calibrations: whether to do gate calibrations for the backend

    """

    calibration_files: Optional[str] = None
    do_calibrations: bool = True

    @property
    def config_type(self):
        return "dynamics"

    def as_dict(self):
        return super().as_dict() | {
            "calibration_files": self.calibration_files,
            "do_calibrations": self.do_calibrations,
        }
