from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any, Sequence, Union, get_args

from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    QuantumRegister,
    ClassicalRegister,
    Qubit,
    Clbit,
    Instruction,
    Parameter,
    ParameterVector,
    Gate,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping as gate_map
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, AnalysisPass
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import defaultdict


class MomentAnalysisPass(AnalysisPass):
    """Analysis pass to group operations into moments, storing results in the PropertySet."""

    def __init__(self):
        super().__init__()
        self.property_history = []

    def run(self, dag):
        moments = defaultdict(list)
        moment_index = 0
        for layer in dag.layers():
            for node in layer["graph"].op_nodes():
                if isinstance(node, DAGOpNode):
                    moments[moment_index].append(node)
            moment_index += 1
        self.property_history.append(moments)
        self.property_set["moments"] = (
            self.property_history if len(self.property_history) > 1 else self.property_history[0]
        )


def format_input(input_val, input_type: Union):
    """
    Formats the input to be a list and validates that all its elements
    are of the types specified in the Union.
    """
    if not isinstance(input_val, list):
        input_val = [input_val]
    valid_types = get_args(input_type)
    if not all(isinstance(i, valid_types) for i in input_val):
        raise TypeError(f"Invalid input format. All elements must be one of type {valid_types}.")
    return input_val


def _parse_instruction(instruction):
    """Parses a user-provided instruction into a standardized (operation, qargs, cargs) tuple."""
    if isinstance(instruction, CircuitInstruction):
        return (instruction.operation, instruction.qubits, instruction.clbits)
    # Handle case where instruction[0] is already a gate object (not a string)
    if isinstance(instruction[0], (Gate, Instruction)):
        op = instruction[0]
    else:
        op = gate_map().get(instruction[0], instruction[0])
    qargs = tuple(instruction[1]) if isinstance(instruction[1], (QuantumRegister, Sequence)) else (instruction[1],)
    cargs = tuple(instruction[2]) if len(instruction) > 2 and isinstance(instruction[2], (ClassicalRegister, Sequence)) else ()
    return (op, qargs, cargs)


def _parse_function(func):
    """Parses a user-provided function or name into a callable or Instruction object."""
    if isinstance(func, str):
        try:
            return gate_map()[func]
        except KeyError:
            raise ValueError(f"Instruction name '{func}' not found in standard gate map.")
    return func


class CustomGateReplacementPass(TransformationPass):
    """
    A transpiler pass to dynamically replace target instructions in a circuit.

    This pass replaces occurrences of specified `target_instructions` with `new_elements`.
    It supports dynamic replacement, where multiple instances of the same target gate
    can be replaced by different elements by cycling through provided lists.
    """

    def __init__(
        self,
        target_instructions: List[Union[CircuitInstruction, Tuple]],
        new_elements: List[Union[Callable, Gate, QuantumCircuit, str, List]],
        parameters: Optional[List[Union[Dict, List]]] = None,
        parametrized_circuit_functions_args: Optional[List[Union[Dict, List]]] = None,
    ):
        """
        Initializes the dynamical gate replacement pass.

        Args:
            target_instructions: A list of instructions to be replaced, or a single
                instruction tuple.

            new_elements: A list of replacement elements. For each target instruction,
                you can provide a single replacement or a list to be cycled through.
                If only one target_instruction is given, this can be a flat list of
                replacement elements.

            parameters: Parameter sets for the new elements, mirroring the structure of
                `new_elements`.

            parametrized_circuit_functions_args: Keyword arguments for callable new
                elements, mirroring the structure of `new_elements`.

        Raises:
            ValueError: If input list lengths are inconsistent.
            TypeError: If any input has an invalid format or element type.
        """
        super().__init__()

        # --- Input Parsing and Normalization ---
        self.target_instructions = [_parse_instruction(inst) for inst in format_input(target_instructions, Union[CircuitInstruction, tuple])]
        num_targets = len(self.target_instructions)

        norm_elements = format_input(new_elements, Union[Callable, Gate, QuantumCircuit, str, list])
        
        norm_params = None
        if parameters is not None:
            norm_params = format_input(parameters, Union[dict, list, type(None)])

        norm_args = None
        if parametrized_circuit_functions_args is not None:
            norm_args = format_input(parametrized_circuit_functions_args, Union[dict, list, type(None)])

        # --- Flexibility Improvement for Single Target Instruction ---
        # If one target is passed with a flat list of replacements/parameters,
        # wrap them in another list to match the expected nested structure.
        if num_targets == 1:
            if len(norm_elements) > 1 and not isinstance(norm_elements[0], list):
                norm_elements = [norm_elements]
            if norm_params and len(norm_params) > 1 and not isinstance(norm_params[0], list):
                norm_params = [norm_params]
            if norm_args and len(norm_args) > 1 and not isinstance(norm_args[0], list):
                norm_args = [norm_args]

        # --- Final Normalization to Nested Lists ---
        self.functions = [[_parse_function(f) for f in (item if isinstance(item, list) else [item])] for item in norm_elements]

        if norm_params is None:
            self.parameters = [[None] * len(func_list) for func_list in self.functions]
        else:
            self.parameters = [item if isinstance(item, list) else [item] for item in norm_params]

        if norm_args is None:
            self.args = [[{}] * len(func_list) for func_list in self.functions]
        else:
            self.args = [item if isinstance(item, list) else [item] for item in norm_args]
        
        # --- Validation Checks ---
        if not (len(self.functions) == num_targets and len(self.parameters) == num_targets and len(self.args) == num_targets):
            raise ValueError(
                "The number of target_instructions, new_elements, and parameter sets must align."
                f"Got {num_targets} targets, {len(self.functions)} element sets, {len(self.parameters)} param sets."
            )

        for i in range(num_targets):
            len_f, len_p, len_a = len(self.functions[i]), len(self.parameters[i]), len(self.args[i])
            if len_f > 1 and len_p > 1 and len_f != len_p:
                raise ValueError(f"For target {i}, received {len_f} functions and {len_p} parameter sets. These must be equal.")
            if len_f > 1 and len_a > 1 and len_f != len_a:
                raise ValueError(f"For target {i}, received {len_f} functions and {len_a} argument sets. These must be equal.")

        self.replacement_counters = [0] * num_targets

    def _create_replacement_dag(
        self,
        func: Union[Callable, Gate, QuantumCircuit],
        params: Optional[Union[Dict, List]],
        f_args: Dict,
        qargs: Tuple[Qubit, ...],
        cargs: Tuple[Clbit, ...],
    ) -> DAGCircuit:
        """Creates a DAG for a single replacement instance."""
        qc = QuantumCircuit()
        qc.add_bits(qargs + cargs)
        
        # Remap qargs and cargs to the context of the new small qc
        local_qargs = tuple(qc.qubits[qargs.index(q)] for q in qargs)
        local_cargs = tuple(qc.clbits[cargs.index(c)] for c in cargs)

        if isinstance(func, QuantumCircuit):
            bound_circ = func.assign_parameters(params) if params is not None else func
            qc.compose(bound_circ, qubits=local_qargs, clbits=local_cargs, inplace=True)
        elif isinstance(func, Gate):
            new_gate = func.copy()
            if params is not None:
                new_gate.params = params.params if isinstance(params, ParameterVector) else params
            
            qc.append(new_gate, local_qargs, local_cargs)
        else:  # Callable
            func(qc, params, list(local_qargs) + list(local_cargs), **f_args)
        
        return circuit_to_dag(qc)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the replacement pass on the given DAG.
        """
        self.replacement_counters = [0] * len(self.target_instructions)

        for node in list(dag.op_nodes()):
            if not hasattr(node.op, 'name'):
                continue
                
            for i, (op_template, qargs_template, cargs_template) in enumerate(self.target_instructions):
                # Map integer indices in template to Qubit/Clbit objects from the DAG
                mapped_qargs = tuple(dag.qubits[q] if isinstance(q, int) else q for q in qargs_template)
                mapped_cargs = tuple(dag.clbits[c] if isinstance(c, int) else c for c in cargs_template)

                if node.op.name == op_template.name and node.qargs == mapped_qargs and node.cargs == mapped_cargs:
                    count = self.replacement_counters[i]
                    func = self.functions[i][count % len(self.functions[i])]
                    params = self.parameters[i][count % len(self.parameters[i])]
                    f_args = self.args[i][count % len(self.args[i])]
                    
                    replacement_dag = self._create_replacement_dag(func, params, f_args, node.qargs, node.cargs)
                    dag.substitute_node_with_dag(node, replacement_dag, wires=node.qargs + node.cargs)
                    
                    self.replacement_counters[i] += 1
                    break
        
        for node in list(dag.op_nodes()):
            if isinstance(node.op, ControlFlowOp):
                new_blocks = []
                # Create a new instance for recursion to ensure a fresh state (e.g., counters)
                recursive_pass = CustomGateReplacementPass(
                    self.target_instructions, self.functions, self.parameters, self.args
                )
                for block in node.op.blocks:
                    transformed_block_dag = recursive_pass.run(circuit_to_dag(block))
                    new_blocks.append(dag_to_circuit(transformed_block_dag))
                
                new_op = node.op.replace_blocks(new_blocks)
                dag.substitute_node(node, new_op)
        
        return dag