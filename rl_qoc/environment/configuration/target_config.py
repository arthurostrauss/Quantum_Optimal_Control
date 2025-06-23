from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Optional, Literal
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.circuit import Gate, QuantumRegister, Qubit
from qiskit.exceptions import QiskitError
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.transpiler import Layout

from ...helpers.circuit_utils import density_matrix_to_statevector, get_gate


@dataclass
class TargetConfig:
    """
    Configuration for the target state or gate to prepare

    Args:
        gate: Target gate to prepare
        physical_qubits: Physical qubits on which the target gate is applied
    """

    physical_qubits: Sequence[int]
    tgt_register: QuantumRegister | List[Qubit] = field(init=False)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key {key} not found in " f" Target configuration")

    def get(self, attribute_name, default_val=None):
        try:
            return getattr(self, attribute_name)
        except AttributeError:
            return default_val

    def __post_init__(self):
        self.tgt_register = QuantumRegister(len(self.physical_qubits), "tgt")
        if not isinstance(self.physical_qubits, Sequence):
            raise QiskitError("Physical qubits must be a sequence of integers")


@dataclass
class GateTargetConfig(TargetConfig):
    """
    Configuration for the target gate to prepare

    Args:
        physical_qubits: Physical qubits on which the target gate is applied
        gate: Target gate to prepare
        circuit_context: Quantum circuit context in which the target gate is applied
        virtual_target_qubits: Virtual qubits on which the target gate is applied (relevant if circuit context is used
            and is larger than physical qubits)
        layout: Layout for transpiling the circuit to the backend (needs to be provided if circuit context is larger
            than physical qubits as it is used to map the virtual target qubits to physical qubits)
    """

    gate: Gate | str
    circuit_context: Optional[QuantumCircuit | List[QuantumCircuit]] = None
    virtual_target_qubits: Optional[List[int]] = None
    layout: Optional[Layout | List[Layout]] = None
    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"

    def __post_init__(self):
        self.gate = get_gate(self.gate)
        if self.circuit_context is not None:
            if isinstance(self.circuit_context, QuantumCircuit):
                self.circuit_context = [self.circuit_context]
            if isinstance(self.circuit_context, list):
                if any(
                    circ.num_qubits < len(self.physical_qubits) for circ in self.circuit_context
                ):
                    raise ValueError(
                        "Circuit context must have at least as many qubits as physical qubits"
                    )
                if self.virtual_target_qubits is None:
                    self.virtual_target_qubits = list(range(len(self.physical_qubits)))
                self.tgt_register = [
                    self.circuit_context[0].qubits[q] for q in self.virtual_target_qubits
                ]
                if self.layout is None:
                    if any(
                        circ.num_qubits > len(self.physical_qubits) for circ in self.circuit_context
                    ):
                        raise ValueError(
                            "If circuit context is larger than physical qubits, "
                            "circuit_context_layout must be provided"
                        )
                    self.layout = [
                        Layout(
                            {
                                self.tgt_register[i]: self.physical_qubits[i]
                                for i in range(len(self.physical_qubits))
                            }
                        )
                        for _ in self.circuit_context
                    ]
        else:
            self.tgt_register = QuantumRegister(len(self.physical_qubits), "tgt")
            qc = QuantumCircuit(self.tgt_register)
            qc.append(self.gate, self.tgt_register)
            self.circuit_context = [qc]
            self.virtual_target_qubits = list(range(len(self.physical_qubits)))
            if self.layout is None:
                self.layout = [
                    Layout(
                        {
                            self.tgt_register[i]: self.physical_qubits[i]
                            for i in range(len(self.physical_qubits))
                        }
                    )
                ]

    def as_dict(self):
        return {
            "gate": self.gate.name,
            "physical_qubits": self.physical_qubits,
            "virtual_target_qubits": self.virtual_target_qubits,
        }


@dataclass
class StateTargetConfig(TargetConfig):
    """
    Configuration for the target state to prepare

    Args:
        state: Target state to prepare
        physical_qubits: Physical qubits on which the target state is prepared
    """

    state: QuantumCircuit | DensityMatrix | Statevector | str

    def __post_init__(self):
        if isinstance(self.state, str):
            try:
                self.state = Statevector.from_label(self.state)
            except QiskitError as e:
                raise QiskitError(f"State {self.state} not recognized") from e
        elif isinstance(self.state, QuantumCircuit):
            self.state = Statevector(self.state)
        elif isinstance(self.state, DensityMatrix):
            self.state = density_matrix_to_statevector(self.state)

    def as_dict(self):
        return {
            "state": self.state.data,
            "physical_qubits": self.physical_qubits,
        }
