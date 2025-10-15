from dataclasses import dataclass, field
from typing import Literal, List, Union, Optional, Dict, Callable, Any, TYPE_CHECKING
from ...quam_config.components import Transmon, TransmonPair, Quam
from ..two_qubit_xeb import QUAGate, QUAGateSet
from qiskit.circuit import QuantumCircuit
import numpy as np

@dataclass
class ShadowConfig:
    """
    Configuration for the classical shadow experiment.

    Args:
        
    """
    shadow_size: int
    shots_per_snapshot: int
    input_state_prep_macro: Callable[[Any], None]
    input_state_circuit: Callable[[Any], QuantumCircuit]
    measurement_basis: Union[str, Dict[int, QUAGate]]
    qubits: List[Transmon]
    input_state_prep_macro_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_angles: int = 100
    readout_qubits: Optional[List[Transmon]] = None
    readout_pulse_name: str = "readout"
    reset_method: Literal["active", "cooldown"] = "cooldown"
    reset_kwargs: Optional[Dict[str, Union[float, str, int]]] = field(
        default_factory=lambda: {
            "cooldown_time": 20,
            "max_tries": None,
            "pi_pulse": None,
        }
    )
    gate_indices: List[List[int]]| np.ndarray | None = None
    save_dir: str = ""
    should_save_data: bool = True
    data_folder_name: Optional[str] = None
    generate_new_data: bool = True
    seed: int = 1234
    
    def __post_init__(self):
        self.n_qubits = len(self.qubits)
        self.dim = 2**self.n_qubits
        self.measurement_basis = QUAGateSet(self.measurement_basis)
        if self.gate_indices is not None:
            if isinstance(self.gate_indices, list):
                self.gate_indices = np.array(self.gate_indices)
            self.gate_indices = self.gate_indices.astype(int)
            if self.gate_indices.ndim != 2:
                raise ValueError("gate_indices must be a 2D array")
            if self.gate_indices.shape[1] != self.n_qubits:
                raise ValueError("gate_indices must have the same number of columns as the number of qubits")
            if self.gate_indices.shape[0] != self.shadow_size:
                raise ValueError("gate_indices must have the same number of rows as the shadow size")
            # Check if there is no index that is negative or higher than length of dictionary of macros
            if any(index < 0 or index >= len(self.measurement_basis) for index in self.gate_indices.flatten()):
                raise ValueError("gate_indices must contain only indices that are within the range of the measurement basis")

        
    def as_dict(self):
        """
        Return the ShadowConfig object as a dictionary
        """
        config_dict = {
            "shadow_size": self.shadow_size,
            "measurement_basis": self.measurement_basis,
            "qubits": [qubit.name if isinstance(qubit, Transmon) else qubit for qubit in self.qubits],
            "seed": self.seed,
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict, machine: Optional[Quam] = None):
        """
        Create a ShadowConfig object from a dictionary
        """
        qubits_names = config_dict["qubits"]
        qubits = [machine.qubits[name] if machine is not None else name for name in qubits_names]
        config_dict["qubits"] = qubits
        config_dict["measurement_basis"] = QUAGateSet(config_dict["measurement_basis"])
        return cls(**config_dict)