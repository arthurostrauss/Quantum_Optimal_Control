from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any, Iterable

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
from qiskit.providers import BackendV2
from qiskit.transpiler import InstructionProperties, Target

InstructionTuple = Tuple[Union[Instruction, Gate], Tuple[Union[int, Qubit], ...], Tuple[Union[int, Clbit], ...]]
TargetInstructionType = Union[CircuitInstruction, InstructionTuple, Instruction, str]
ReplacementElement = Union[Instruction, QuantumCircuit]
ParameterSet = Union[Dict[Union[str, Parameter], Any], List[Parameter], ParameterVector]
ParameterSets = Union[ParameterSet, Iterable[ParameterSet]]
InstructionPropertiesDict = Dict[Tuple[int, ...], InstructionProperties]
InstructionPropertiesDicts = Union[InstructionPropertiesDict, Iterable[InstructionPropertiesDict]]

def format_input(input_val):
    if not isinstance(input_val, list):
        input_val = [input_val]
    return input_val


def _parse_instruction(instruction: TargetInstructionType) -> InstructionTuple:
    if isinstance(instruction, Instruction):
        return (instruction, tuple(range(instruction.num_qubits)), tuple(range(instruction.num_clbits)))
    if isinstance(instruction, CircuitInstruction):
        return (instruction.operation, instruction.qubits, instruction.clbits)  # type: ignore[union-attr]
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
            else:
                qargs = tuple(range(op.num_qubits)) if op.num_qubits > 0 else ()
            
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
    """
    Parse and validate a replacement function. Only Instruction and QuantumCircuit are allowed.
    If a string is provided, it will be converted to an Instruction from the standard gate map.
    QuantumCircuits will be kept as-is (they will be converted to Instructions later when needed).
    
    Args:
        func: An Instruction, QuantumCircuit, or string (gate name)
        
    Returns:
        The validated Instruction or QuantumCircuit
        
    Raises:
        ValueError: If func is not an Instruction, QuantumCircuit, or valid gate name string
    """
    if isinstance(func, str):
        try:
            return gate_map()[func]
        except KeyError:
            raise ValueError(f"Instruction name '{func}' not found in standard gate map.")
    if isinstance(func, (Instruction, QuantumCircuit)):
        return func
    raise ValueError(
        f"Invalid replacement element type: {type(func)}. "
        f"Only Instruction or QuantumCircuit are allowed."
    )


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
            
        custom_instruction: Optional single replacement element (Instruction or QuantumCircuit) or a
            list of elements to cycle through for each instance of the target. If None, the original instruction
            will be wrapped in a for loop (if n_reps is provided) without replacement. If a list is provided, 
            then each instance of the target will be replaced by a different element from the list in a cyclic manner. 
            QuantumCircuits will be automatically converted to Instructions via `.to_instruction()`.
        parameters: A list of parameter sets (list of parameters) to cycle through and to attach to the custom instruction. 
            If `custom_instruction` is a single element, and a list of parameter sets is provided, the parameters will 
            be cycled through for each instance of the target. However, if both a list of custom instructions and a list 
            of parameter sets are provided, the lengths of the lists must match.
        instruction_properties: Optional dictionary mapping physical qubits to InstructionProperties, or a list of such dictionaries.
            Format: Dict[Tuple[int, ...], InstructionProperties] where keys are tuples of physical qubit indices.
            The number of dictionaries must exactly match the number of unique custom instructions (not counting parameter variations).
            If a single custom_instruction is provided with multiple parameter sets, only one dictionary is needed.
            Example: {(0, 1): InstructionProperties(...), (1, 2): InstructionProperties(...)}
        n_reps: Optional number of repetitions for wrapping the instruction in a for loop. Can be:
            - An integer: fixed number of repetitions
            - A string: name of an input variable for the number of repetitions
            - A list of integers: cycle through different repetition counts
            - None: no for loop wrapping
            If both custom_instruction and n_reps are provided, the custom instruction will be wrapped in a for loop.
            If only n_reps is provided (custom_instruction is None), the original instruction will be wrapped in a for loop.
    Example usages:
        - Single instruction, multiple param sets broadcasted:
            custom_instruction = rx_gate, parameters = [{"theta": 0.1}, {"theta": 0.2}]
        - Multiple instructions, single shared params replicated to N instructions:
            custom_instruction = [rx_gate, ry_gate], parameters = {"theta": 0.1}
        - Wrap original instruction in for loop:
            custom_instruction = None, n_reps = 5
        - Replace and wrap in for loop:
            custom_instruction = rx_gate, n_reps = 3
    """

    target_instruction: TargetInstructionType
    custom_instruction: Optional[Union[ReplacementElement, Iterable[ReplacementElement]]] = None
    parameters: Optional[ParameterSets] = None
    instruction_properties: Optional[InstructionPropertiesDicts] = None
    n_reps: Optional[int|str|List[int]] = None

    parsed_target: InstructionTuple = field(init=False, repr=False)
    functions_to_cycle: List[ReplacementElement] = field(init=False, repr=False)
    params_to_cycle: ParameterSets = field(init=False, repr=False)
    instruction_properties_to_cycle: Optional[List[InstructionPropertiesDict]] = field(init=False, repr=False)
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

        # Validate that at least one of custom_instruction or n_reps is provided
        if self.custom_instruction is None and self.n_reps is None:
            raise ValueError(
                f"At least one of 'custom_instruction' or 'n_reps' must be provided for target '{self.target_operation.name}'."
            )

        # Handle custom_instruction (can be None)
        if self.custom_instruction is not None:
            norm_elements = format_input(self.custom_instruction)
            self.functions_to_cycle = [_parse_function(f) for f in norm_elements]

            norm_params = self.parameters if isinstance(self.parameters, Iterable) and all(isinstance(p, Iterable) for p in self.parameters) else [self.parameters]

            self.params_to_cycle = self._broadcast_params(
                len(self.functions_to_cycle), tuple(norm_params), self.target_operation.name
            )
            
            # Validate and normalize instruction_properties
            self.instruction_properties_to_cycle = self._validate_and_normalize_instruction_properties()
        else:
            # No custom instruction, just for loop wrapping
            self.functions_to_cycle = []
            self.params_to_cycle = []
            self.instruction_properties_to_cycle = None
            
            # Validate that parameters are not provided when custom_instruction is None
            if self.parameters is not None:
                raise ValueError(
                    f"Cannot provide 'parameters' when 'custom_instruction' is None for target '{self.target_operation.name}'."
                )
            
            # Validate that instruction_properties are not provided when custom_instruction is None
            if self.instruction_properties is not None:
                raise ValueError(
                    f"Cannot provide 'instruction_properties' when 'custom_instruction' is None for target '{self.target_operation.name}'."
                )
    def check_qubits(self, circuit: QuantumCircuit) -> bool:
        """
        Check if the qubits of the target instruction are in the given circuit.
        """
        if all(isinstance(q, int) for q in self.target_qargs):
            return all(q < circuit.num_qubits for q in self.target_qargs)
        elif all(isinstance(q, Qubit) for q in self.target_qargs):
            return all(q in circuit.qubits for q in self.target_qargs)
        else:
            return False
    
    def get_qubits(self, circuit: QuantumCircuit) -> List[Qubit]:
        """
        Get the qubits of the target instruction in the given circuit.
        """
        if all(isinstance(q, int) for q in self.target_qargs):
            return [circuit.qubits[q] for q in self.target_qargs]
        elif all(isinstance(q, Qubit) for q in self.target_qargs):
            if all(q in circuit.qubits for q in self.target_qargs):
                return [q for q in self.target_qargs]
            else:
                raise ValueError(f"Circuit does not contain all target qubits: {self.target_qargs}")
        else:
            raise ValueError(f"Invalid target qubits: {self.target_qargs}, expected a sequence of integers or Qubit objects")

    @staticmethod
    def _broadcast_params(
        num_funcs: int,
        norm_params: Sequence[Any],
        target_name: str,
    ) -> List[Any]:
        num_params = len(norm_params)

        if num_funcs == 1:
            params_to_cycle = list(norm_params)
            return params_to_cycle

        if num_params > 1 and num_params != num_funcs:
            raise ValueError(
                f"Mismatch for target '{target_name}': {num_funcs} new elements vs {num_params} parameter sets."
            )

        params_to_cycle = list(norm_params) if num_params > 1 else [norm_params[0]] * num_funcs
        return params_to_cycle

    def _validate_and_normalize_instruction_properties(self) -> Optional[List[InstructionPropertiesDict]]:
        """
        Validate that instruction_properties count matches the number of unique custom instructions.
        If a single custom_instruction is provided with multiple parameter sets, only one dictionary is needed.
        
        Returns:
            Normalized list of dictionaries mapping qubits to InstructionProperties, or None
            
        Raises:
            ValueError: If the number of instruction_properties doesn't match the number of custom instructions,
                or if the format is invalid
        """
        if self.instruction_properties is None:
            return None
        
        # Count unique custom instructions (not counting parameter variations)
        num_unique_instructions = len(self.functions_to_cycle)
        
        # Normalize instruction_properties to a list
        if isinstance(self.instruction_properties, dict):
            # Single dictionary
            norm_props_raw = [self.instruction_properties]
        elif isinstance(self.instruction_properties, Iterable) and not isinstance(self.instruction_properties, (str, bytes)):
            # List/iterable of dictionaries
            norm_props_raw = list(self.instruction_properties)
        else:
            raise ValueError(
                f"Invalid instruction_properties type: {type(self.instruction_properties)}. "
                f"Expected Dict[Tuple[int, ...], InstructionProperties] or Iterable of such dictionaries."
            )
        
        # Validate all elements are dictionaries with correct format
        norm_props: List[InstructionPropertiesDict] = []
        for prop_dict in norm_props_raw:
            if not isinstance(prop_dict, dict):
                raise ValueError(
                    f"Invalid instruction_properties element type: {type(prop_dict)}. "
                    f"Expected Dict[Tuple[int, ...], InstructionProperties]."
                )
            
            # Validate dictionary keys and values
            validated_dict: InstructionPropertiesDict = {}
            for qubits, prop in prop_dict.items():
                # Validate qubits key
                if not isinstance(qubits, tuple):
                    raise ValueError(
                        f"Invalid qubits key type: {type(qubits)}. "
                        f"Expected Tuple[int, ...] representing physical qubit indices."
                    )
                if not all(isinstance(q, int) for q in qubits):
                    raise ValueError(
                        f"Invalid qubits key: {qubits}. "
                        f"All elements must be integers representing physical qubit indices."
                    )
                
                # Validate InstructionProperties value
                if not isinstance(prop, InstructionProperties):
                    raise ValueError(
                        f"Invalid InstructionProperties value type: {type(prop)}. "
                        f"Expected InstructionProperties instance."
                    )
                
                validated_dict[qubits] = prop
            
            norm_props.append(validated_dict)
        
        # Validate count matches
        if len(norm_props) != num_unique_instructions:
            raise ValueError(
                f"Mismatch for target '{self.target_operation.name}': "
                f"{num_unique_instructions} custom instruction(s) vs {len(norm_props)} instruction property dictionary/dictionaries. "
                f"The number of instruction_properties dictionaries must exactly match the number of custom_instruction elements."
            )
        
        return norm_props

    def add_instruction_properties(self, backend: BackendV2) -> None:
        """
        Add the stored instruction properties to the Backend Target object associated to the specified custom instructions.
        Custom instructions must be Instruction objects (or QuantumCircuits that will be converted to Instructions).
        If an instruction with the same name already exists in the target, its properties will be updated instead of added.
        The physical qubits are extracted from the instruction_properties dictionary keys.

        Args:
            backend: The backend to add the instruction properties to.
            
        Raises:
            ValueError: If the custom instructions are not Instruction objects or QuantumCircuits,
                or if there are issues updating the target.
        """
        if self.instruction_properties_to_cycle is None:
            return
        
        target = backend.target
        if target is None:
            raise ValueError("Backend target is None. Cannot add instruction properties.")
        
        # Process each custom instruction with its corresponding properties dictionary
        for i, custom_instr in enumerate(self.functions_to_cycle):
            # Convert QuantumCircuit to Instruction if needed
            if isinstance(custom_instr, QuantumCircuit):
                instr_to_add = custom_instr.to_instruction()
            elif isinstance(custom_instr, Instruction):
                instr_to_add = custom_instr
            else:
                raise ValueError(
                    f"Custom instruction must be an Instruction or QuantumCircuit, "
                    f"got {type(custom_instr)}"
                )
            
            props_dict = self.instruction_properties_to_cycle[i]
            instr_name = instr_to_add.name
            
            # Check if instruction already exists in target
            if instr_name in target:
                # Instruction exists - update properties for each qubit set in the dictionary
                qargs_set = target.qargs_for_operation_name(instr_name)
                
                for physical_qubits, prop in props_dict.items():
                    # qargs_set is None for global instructions, or a set of qargs tuples
                    if qargs_set is None or physical_qubits in qargs_set:
                        # Update existing properties
                        target.update_instruction_properties(instr_name, physical_qubits, prop)
                    else:
                        # Instruction exists but qargs don't - Target doesn't support adding new qargs
                        # to existing instructions easily, so we raise an error
                        raise ValueError(
                            f"Instruction '{instr_name}' already exists in target, but not for qargs {physical_qubits}. "
                            f"Target does not support adding new qargs to existing instructions. "
                            f"Please use a different instruction name or update the existing qargs."
                        )
            else:
                # Add new instruction with all qubit sets from the dictionary
                target.add_instruction(instr_to_add, props_dict)

