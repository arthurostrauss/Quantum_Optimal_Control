from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any, Iterable

from qiskit.circuit import (
    CircuitInstruction,
    ClassicalRegister,
    Gate,
    Instruction,
    Parameter,
    ParameterVector,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    Clbit
)
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping as gate_map,
)

InstructionTuple = Tuple[Union[Instruction, Gate], Tuple[Union[int, Qubit], ...], Tuple[Union[int, Clbit], ...]]
TargetInstructionType = Union[CircuitInstruction, InstructionTuple, Instruction, str]
ReplacementElement = Union[Callable[[QuantumCircuit, Any], QuantumCircuit], Instruction, QuantumCircuit]
ParameterSet = Union[Dict[Union[str, Parameter], Any], List[Parameter], ParameterVector]
ArgumentSet = Union[Dict[str, Any], List[Any]]
ParameterSets = Union[ParameterSet, Iterable[ParameterSet]]
ArgumentSets = Union[ArgumentSet, Iterable[ArgumentSet]]

def format_input(input_val):
    if not isinstance(input_val, list):
        input_val = [input_val]
    return input_val


def _parse_instruction(instruction: TargetInstructionType) -> InstructionTuple:
    if isinstance(instruction, Instruction):
        return (instruction, tuple(range(instruction.num_qubits)), tuple(range(instruction.num_clbits)))
    if isinstance(instruction, CircuitInstruction):
        return (instruction.operation, instruction.qubits, instruction.clbits)
    if isinstance(instruction, Tuple):
        if isinstance(instruction[0], str):
            try:
                op = gate_map()[instruction[0]]
            except KeyError:
                raise ValueError(f"Instruction name '{instruction[0]}' not found in standard gate map. Please provide a valid instruction name or an Instruction object.")
        elif isinstance(instruction[0], Instruction):
            op = instruction[0]
        else:
            raise ValueError(f"Invalid instruction: {instruction[0]}, expected a string or an Instruction object.")
        if len(instruction) > 1:
            if isinstance(instruction[1], (QuantumRegister, Sequence)):
                qargs = tuple(instruction[1])
                if not all(isinstance(q, (Qubit, int)) for q in qargs):
                    raise ValueError(f"Invalid qargs: {instruction[1]}, expected a sequence of Qubits or integers") 
            if len(instruction) > 2:
                if isinstance(instruction[2], (ClassicalRegister, Sequence)):
                    cargs = tuple(instruction[2])
                    if not all(isinstance(c, (Clbit, int)) for c in cargs):
                        raise ValueError(f"Invalid cargs: {instruction[2]}, expected a sequence of Clbits or integers")
                else:
                    raise ValueError(f"Invalid cargs: {instruction[2]}, expected a ClassicalRegister or a sequence of integers/Clbit")
            else:
                cargs = tuple(range(op.num_clbits)) if op.num_clbits > 0 else ()
        else:
            qargs = tuple(range(op.num_qubits)) if op.num_qubits > 0 else ()
            cargs = tuple(range(op.num_clbits)) if op.num_clbits > 0 else ()
        return (op, qargs, cargs)
    elif isinstance(instruction, str):
        try:
            op = gate_map()[instruction]
        except KeyError:
            raise ValueError(f"Instruction name '{instruction}' not found in standard gate map. Please provide a valid instruction name or an Instruction object.")
        qargs = tuple(range(op.num_qubits)) if op.num_qubits > 0 else ()
        cargs = tuple(range(op.num_clbits)) if op.num_clbits > 0 else ()
        return (op, qargs, cargs)
    else:
        raise ValueError(f"Invalid instruction: {instruction}, expected a string, an Instruction object, or a tuple.")


def _parse_function(func):
    if isinstance(func, str):
        try:
            return gate_map()[func]
        except KeyError:
            raise ValueError(f"Instruction name '{func}' not found in standard gate map.")
    return func


@dataclass
class InstructionReplacement:
    """
    Defines a replacement rule for a single target instruction.

    Attributes:
        target_instruction: The instruction to find and replace. This should be specified in general as a tuple containing the instruction to find and the qubits/clbits it acts on.
            Few formats are supported:
            - An instruction object (or instruction name if it is a standard gate). The qubits/clbits are inferred from the number of qubits/clbits of the instruction.
            - A tuple containing the instruction (or instruction name if it is a standard gate) and the qubits/clbits it acts on.
            - A CircuitInstruction object.
            
        custom_instruction: A single replacement element (Instruction, QuantumCircuit, Callable) or a
            list of elements to cycle through for each instance of the target. If a list is provided, then each instance of the target will be replaced by a different element from the list in 
            a cyclic manner.
        parameters: A list of parameter sets (list of parameters) to cycle through and to attach to the custom instruction. If `custom_instruction`
            is a single element, and a list of parameter sets is provided, the parameters will be cycled through for each instance of the target.
            However, if both a list of custom instructions and a list of parameter sets are provided, the lengths of the lists must match. If the custom instruction is a callable, the second argument of the callable should be the parameter set (while the first argument should be the quantum circuit).
        parametrized_circuit_functions_args: Arguments for callable custom instructions. If `custom_instruction`
            is a single element, and a list of argument sets is provided, the arguments will be cycled through for each instance of the target.
            However, if both a list of custom instructions and a list of argument sets are provided, the lengths of the lists must match.
    Example usages:
        - Single function, multiple param sets broadcasted:
            custom_instruction = rx_gate, parameters = [{"theta": 0.1}, {"theta": 0.2}]
        - Multiple functions, single shared args replicated to N functions:
            custom_instruction = [rx_gate, ry_gate], parametrized_circuit_functions_args = {"foo": 1}
    """

    target_instruction: TargetInstructionType
    custom_instruction: Union[ReplacementElement, Iterable[ReplacementElement]]
    parameters: Optional[ParameterSets] = None
    parametrized_circuit_functions_args: Optional[ArgumentSets] = None

    parsed_target: InstructionTuple = field(init=False, repr=False)
    functions_to_cycle: List[ReplacementElement] = field(init=False, repr=False)
    params_to_cycle: ParameterSets = field(init=False, repr=False)
    args_to_cycle: ArgumentSets = field(init=False, repr=False)
    # Exposed metadata
    target_operation: Instruction = field(init=False, repr=False)
    target_qargs: Tuple[Union[int, Qubit], ...] = field(init=False, repr=False)
    target_cargs: Tuple[Union[int, Clbit], ...] = field(init=False, repr=False)

    def __post_init__(self):
        self.parsed_target = _parse_instruction(self.target_instruction)

        op, qargs, cargs = self.parsed_target
        self.target_operation = op
        self.target_qargs = qargs
        self.target_cargs = cargs

        norm_elements = format_input(self.custom_instruction)
        self.functions_to_cycle = [_parse_function(f) for f in norm_elements]

        norm_params = self.parameters if isinstance(self.parameters, Iterable) and all(isinstance(p, Iterable) for p in self.parameters) else [self.parameters]
        norm_args = (
            format_input(self.parametrized_circuit_functions_args)
            if self.parametrized_circuit_functions_args is not None
            else [{}]
        )

        self.params_to_cycle, self.args_to_cycle = self._broadcast_params_and_args(
            len(self.functions_to_cycle), tuple(norm_params), tuple(norm_args), self.target_operation.name
        )

    @staticmethod
    def _broadcast_params_and_args(
        num_funcs: int,
        norm_params: Sequence[Any],
        norm_args: Sequence[Dict],
        target_name: str,
    ) -> Tuple[List[Any], List[Dict]]:
        num_params = len(norm_params)
        num_args = len(norm_args)

        if num_funcs == 1:
            params_to_cycle = list(norm_params)
            args_to_cycle = list(norm_args) if num_args > 1 else [norm_args[0]] * len(params_to_cycle)
            if len(params_to_cycle) > 1 and len(norm_args) > 1 and len(params_to_cycle) != len(norm_args):
                raise ValueError(
                    f"For target '{target_name}', broadcast mismatch: "
                    f"{len(params_to_cycle)} parameter sets vs {len(norm_args)} argument sets."
                )
            return params_to_cycle, args_to_cycle

        if num_params > 1 and num_params != num_funcs:
            raise ValueError(
                f"Mismatch for target '{target_name}': {num_funcs} new elements vs {num_params} parameter sets."
            )
        if num_args > 1 and num_args != num_funcs:
            raise ValueError(
                f"Mismatch for target '{target_name}': {num_funcs} new elements vs {num_args} argument sets."
            )

        params_to_cycle = list(norm_params) if num_params > 1 else [norm_params[0]] * num_funcs
        args_to_cycle = list(norm_args) if num_args > 1 else [norm_args[0]] * num_funcs
        return params_to_cycle, args_to_cycle


