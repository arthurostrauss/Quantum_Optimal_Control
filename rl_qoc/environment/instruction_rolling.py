from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    Instruction,
    Qubit,
    Clbit,
    QuantumCircuit,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping as gate_map
from .instruction_replacement import _parse_instruction

@dataclass
class InstructionRolling:
    """
    Defines a rolling instruction replacement rule for a single target instruction.
    The number of repetitions can be specified as an integer, a string (the name of the input variable) or a list of integers.
    """
    target_instruction: Union[CircuitInstruction, Tuple[Union[Instruction, Gate], Tuple[Union[int, Qubit], ...], Tuple[Union[int, Clbit], ...]]]
    n_reps: Optional[int|str|List[int]] = None

    parsed_target: Tuple = field(init=False, repr=False)
    functions_to_cycle: List[Union[Callable, Instruction, QuantumCircuit]] = field(init=False, repr=False)
    params_to_cycle: List[Any] = field(init=False, repr=False)
    args_to_cycle: List[Dict] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate and normalize the inputs after the dataclass is created."""
        self.parsed_target = _parse_instruction(self.target_instruction)