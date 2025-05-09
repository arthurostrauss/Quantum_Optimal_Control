from __future__ import annotations

from dataclasses import dataclass
from typing import List
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.circuit import Gate
from qiskit.exceptions import QiskitError
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

from ...helpers.circuit_utils import density_matrix_to_statevector, get_gate


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
        self.gate = get_gate(self.gate)

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
        elif isinstance(self.state, QuantumCircuit):
            self.state = Statevector(self.state)
        elif isinstance(self.state, DensityMatrix):
            self.state = density_matrix_to_statevector(self.state)

    def as_dict(self):
        return {
            "state": self.state.data,
            "physical_qubits": self.physical_qubits,
        }
