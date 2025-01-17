from __future__ import annotations

from dataclasses import dataclass
from typing import List
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.circuit import Gate
from qiskit.exceptions import QiskitError
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping


@dataclass
class TargetConfig:
    """
    Configuration for the target state or gate to prepare

    Args:
        gate: Target gate to prepare
        physical_qubits: Physical qubits on which the target gate is applied
    """

    physical_qubits: List[int]

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


@dataclass
class GateTargetConfig(TargetConfig):
    """
    Configuration for the target gate to prepare

    Args:
        gate: Target gate to prepare
        physical_qubits: Physical qubits on which the target gate is applied
    """

    gate: Gate | str

    def __post_init__(self):
        if isinstance(self.gate, str):
            if self.gate.lower() == "cnot":
                self.gate = "cx"
            elif self.gate.lower() == "cphase":
                self.gate = "cz"
            elif self.gate.lower() == "x/2":
                self.gate = "sx"
            try:
                self.gate = get_standard_gate_name_mapping()[self.gate.lower()]
            except KeyError as e:
                raise ValueError(f"Gate {self.gate} not recognized") from e

    def as_dict(self):
        return {
            "gate": self.gate.name,
            "physical_qubits": self.physical_qubits,
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

    def as_dict(self):
        return {
            "state": self.state.data,
            "physical_qubits": self.physical_qubits,
        }
