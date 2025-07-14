from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.transpiler import (
    PassManager,
    InstructionDurations,
    Layout,
    CouplingMap,
    generate_preset_pass_manager,
    Target,
)
from qiskit.providers import BackendV2 as QiskitBackend


class BackendInfo(ABC):

    def __init__(self, n_qubits: int = 0, pass_manager=None, skip_transpilation: bool = False):
        """
        Initialize the backend information

        :param n_qubits: Number of qubits for the quantum environment
        :param pass_manager: Pass manager for the quantum environment (default is None)
        """
        self._n_qubits = n_qubits
        self._pass_manager = pass_manager
        self._skip_transpilation = skip_transpilation
        self._backend = None

    @abstractmethod
    def custom_transpile(self, qc_input, *args, **kwargs) -> QuantumCircuit | List[QuantumCircuit]:
        pass

    @property
    @abstractmethod
    def coupling_map(self) -> CouplingMap | List[Tuple[int, int]]:
        pass

    @property
    @abstractmethod
    def basis_gates(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        pass

    @property
    @abstractmethod
    def instruction_durations(self):
        pass

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    @num_qubits.setter
    def num_qubits(self, n_qubits: int):
        assert n_qubits > 0, "Number of qubits should be positive"
        self._n_qubits = n_qubits

    @property
    def pass_manager(self) -> Optional[PassManager]:
        return self._pass_manager

    @property
    def skip_transpilation(self):
        return self._skip_transpilation

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend: QiskitBackend):
        self._backend = backend


class QiskitBackendInfo(BackendInfo):
    """
    Class to store information on Qiskit backend (can also generate some dummy information for the case of no backend)
    """

    def __init__(
        self,
        backend: Optional[QiskitBackend] = None,
        custom_instruction_durations: Optional[InstructionDurations] = None,
        pass_manager: Optional[PassManager] = None,
        skip_transpilation: bool = False,
        n_qubits: int = 0,
    ):
        """
        Initialize the backend information
        :param backend: Backend object for the quantum environment
        :param custom_instruction_durations: Custom instruction durations for the backend
        :param pass_manager: Custom pass manager for the backend (applied on top and prior to Qiskit preset pass manager)
        :param n_qubits: Number of qubits for the quantum environment
        """
        super().__init__(
            backend.num_qubits if isinstance(backend, QiskitBackend) else n_qubits,
            pass_manager,
            skip_transpilation,
        )
        # if isinstance(backend, BackendV2) and backend.coupling_map is None:
        #     raise QiskitError("Coupling map should be provided for the backend")
        self._backend = backend
        self._instruction_durations = custom_instruction_durations

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

        if not self.skip_transpilation:
            if self._pass_manager is None:
                if self.backend is None:
                    target = Target.from_configuration(
                        self.basis_gates,
                        self.num_qubits,
                        self.coupling_map if self.coupling_map.size() != 0 else None,
                        instruction_durations=self.instruction_durations,
                        dt=self.dt,
                    )
                else:
                    target = self.backend.target
                self._pass_manager = generate_preset_pass_manager(
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
    def coupling_map(self):
        """
        Retrieve the coupling map of the backend (default is fully connected if backend is None)
        """
        return (
            self.backend.coupling_map
            if isinstance(self.backend, QiskitBackend) and self.backend.coupling_map is not None
            else CouplingMap.from_full(self._n_qubits)
        )

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
            else self._instruction_durations
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
