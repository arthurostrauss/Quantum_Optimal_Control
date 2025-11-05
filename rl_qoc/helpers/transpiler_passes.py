from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from typing import Iterable, List, Tuple, Optional, Callable, Dict, Any, Sequence, Union, get_args

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

__all__ = ["CustomGateReplacementPass", "MomentAnalysisPass", "InstructionReplacement", "ForLoopWrapperPass", "InstructionRolling"]

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


def format_input(input_val):
    """
    Formats the input to be a list and validates that all its elements
    are of the types specified in the Union.
    """
    if not isinstance(input_val, list):
        input_val = [input_val]
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
    qargs = (
        tuple(instruction[1])
        if isinstance(instruction[1], (QuantumRegister, Sequence))
        else (instruction[1],)
    )
    cargs = (
        tuple(instruction[2])
        if len(instruction) > 2 and isinstance(instruction[2], (ClassicalRegister, Sequence))
        else ()
    )
    return (op, qargs, cargs)


def _parse_function(func):
    """Parses a user-provided function or name into a callable or Instruction object."""
    if isinstance(func, str):
        try:
            return gate_map()[func]
        except KeyError:
            raise ValueError(f"Instruction name '{func}' not found in standard gate map.")
    return func

@dataclass
class InstructionRolling:
    """
    Defines a rolling instruction replacement rule for a single target instruction.
    The number of repetitions can be specified as an integer, a string (the name of the input variable) or a list of integers.
    """
    target_instruction: Union[CircuitInstruction, Tuple]
    n_reps: Optional[int|str|List[int]] = None

    parsed_target: Tuple = field(init=False, repr=False)
    functions_to_cycle: List[Union[Callable, Gate, QuantumCircuit]] = field(init=False, repr=False)
    params_to_cycle: List[Optional[Union[Dict, List]]] = field(init=False, repr=False)
    args_to_cycle: List[Dict] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate and normalize the inputs after the dataclass is created."""
        self.parsed_target = _parse_instruction(self.target_instruction)
        

class ForLoopWrapperPass(TransformationPass):
    """
    A transpiler pass that wraps all instructions of a circuit in a for loop.
    """
    def __init__(self, replacements: Union[InstructionRolling, List[InstructionRolling]]):
        super().__init__()
        self.replacements = format_input(replacements)
    
    def _create_for_loop_dag(self, node: DAGOpNode, n_reps: int|str|Iterable[int]|None, n_counter: int) -> DAGCircuit:
        """
        Create a for loop DAG for the given node.
        """
        qc = QuantumCircuit()
        qc.add_bits(node.qargs + node.cargs)
        if n_reps is None or isinstance(n_reps, str):
            from qiskit.circuit.classical import expr, types
            name = f"_n_{n_counter}" if n_reps is None else n_reps
            n = qc.add_input(name, types.Uint(64))
            r = expr.Range(expr.lift(0, types.Uint(64)), n)
        elif isinstance(n_reps, Iterable):
            r = list(n_reps)
        else:
            r = range(n_reps)

        with qc.for_loop(r):
            qc.append(node.op, node.qargs, node.cargs)
        
        return circuit_to_dag(qc)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the pass on the given DAG.
        """
        n_counter = 0
        for node in dag.op_nodes():
            if not hasattr(node.op, "name"):
                continue

            for i, repl_rule in enumerate(self.replacements):
                op_template, qargs_template, cargs_template = repl_rule.parsed_target
                mapped_qargs = tuple(
                    dag.qubits[q] if isinstance(q, int) else q for q in qargs_template
                )
                mapped_cargs = tuple(
                    dag.clbits[c] if isinstance(c, int) else c for c in cargs_template
                )

                if (
                    node.op.name == op_template.name
                    and node.qargs == mapped_qargs
                    and node.cargs == mapped_cargs
                ):
                    for_loop_dag = self._create_for_loop_dag(node, repl_rule.n_reps, n_counter)
                    dag.substitute_node_with_dag(node, for_loop_dag, wires=node.qargs + node.cargs)
                    n_counter += 1
                    break

        return dag
                   

        
@dataclass
class InstructionReplacement:
    """
    Defines a replacement rule for a single target instruction.

    This class validates and normalizes the relationship between the new elements
    and their parameters, handling the "broadcasting" of parameters to a single
    new element.

    Attributes:
        target_instruction: The instruction to find and replace.
        new_elements: A single replacement element (Gate, Circuit, Callable) or a
            list of elements to cycle through for each instance of the target.
        parameters: A list of parameter sets to cycle through. If `new_elements`
            is a single element, these parameters will be broadcast to it.
        parametrized_circuit_functions_args: Arguments for callable new elements.
    """

    target_instruction: Union[CircuitInstruction, Tuple]
    new_elements: Union[Callable, Gate, QuantumCircuit, str, List]
    parameters: Optional[Union[Dict, List[Parameter], ParameterVector]] = None
    parametrized_circuit_functions_args: Optional[Union[Dict, List]] = None

    # These fields are processed and populated after initialization
    parsed_target: Tuple = field(init=False, repr=False)
    functions_to_cycle: List[Union[Callable, Gate, QuantumCircuit]] = field(init=False, repr=False)
    params_to_cycle: List[Optional[Union[Dict, List]]] = field(init=False, repr=False)
    args_to_cycle: List[Dict] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate and normalize the inputs after the dataclass is created."""
        self.parsed_target = _parse_instruction(self.target_instruction)

        # Normalize new_elements into a list of functions/gates/circuits
        norm_elements = (
            self.new_elements if isinstance(self.new_elements, list) else [self.new_elements]
        )
        self.functions_to_cycle = [_parse_function(f) for f in norm_elements]

        # Normalize parameters and args
        norm_params = (
            self.parameters
            if isinstance(self.parameters, list)
            and all(isinstance(p, List) for p in self.parameters)
            else [self.parameters]
        )
        if self.parameters is None:
            norm_params = [None]

        norm_args = self.parametrized_circuit_functions_args
        if norm_args is None:
            norm_args = [{}]
        elif isinstance(norm_args, dict):
            norm_args = [norm_args]

        # --- Broadcasting and Validation Logic ---
        num_funcs = len(self.functions_to_cycle)
        num_params = len(norm_params)
        num_args = len(norm_args)

        # Case 1: Broadcasting (1 function, N parameters/args)
        if num_funcs == 1:
            self.params_to_cycle = norm_params
            self.args_to_cycle = norm_args if num_args > 1 else [norm_args[0]] * num_params
            if num_params > 1 and num_args > 1 and num_params != num_args:
                raise ValueError(
                    f"For target '{self.parsed_target[0].name}', broadcast mismatch: "
                    f"{num_params} parameter sets vs {num_args} argument sets."
                )
            return  # Logic is valid

        # Case 2: Matched Lists (N functions, N parameters/args)
        if num_funcs > 1:
            if num_params > 1 and num_funcs != num_params:
                raise ValueError(
                    f"Mismatch for target '{self.parsed_target[0].name}': "
                    f"{num_funcs} new elements vs {num_params} parameter sets."
                )
            if num_args > 1 and num_funcs != num_args:
                raise ValueError(
                    f"Mismatch for target '{self.parsed_target[0].name}': "
                    f"{num_funcs} new elements vs {num_args} argument sets."
                )

            self.params_to_cycle = norm_params if num_params > 1 else [norm_params[0]] * num_funcs
            self.args_to_cycle = norm_args if num_args > 1 else [norm_args[0]] * num_funcs
            return  # Logic is valid

        # Default case (1 function, 1 param, 1 arg)
        self.params_to_cycle = norm_params
        self.args_to_cycle = norm_args


class CustomGateReplacementPass(TransformationPass):
    """
    A transpiler pass that dynamically replaces instructions in a circuit
    based on a list of replacement rules.
    """

    def __init__(self, replacements: Union[InstructionReplacement, List[InstructionReplacement]], opaque_gates: bool = False):
        """
        Initializes the gate replacement pass.

        Args:
            replacements: A single `InstructionReplacement` object or a list of them,
                each defining a target and its corresponding replacements.
            opaque_gates: Whether to treat the replaced gates as opaque gates (meaning definitions are removed and replaced with a new gate object)
        """
        super().__init__()
        self.replacements = format_input(replacements)
        self.replacement_counters = [0] * len(self.replacements)
        self.opaque_gates = opaque_gates

    def _create_replacement_dag(
        self,
        func: Union[Callable, Gate, QuantumCircuit],
        params: Optional[Union[Dict, List]],
        f_args: Dict,
        qargs: Tuple[Qubit, ...],
        cargs: Tuple[Clbit, ...],
    ) -> DAGCircuit:
        qc = QuantumCircuit()
        qc.add_bits(qargs + cargs)
        # Directly reference bits already present in the new circuit:
        local_qargs = qc.qubits
        local_cargs = qc.clbits

        # Handle QuantumCircuit and Gate types
        if isinstance(func, QuantumCircuit):
            new_instr = func.assign_parameters(params) if params is not None else func
            new_instr = new_instr.to_instruction()
            if self.opaque_gates:
                new_instr.definition = None
            qc.append(new_instr, local_qargs, local_cargs)
        elif isinstance(func, Gate):
            new_gate = func.copy()
            if params is not None:
                # params can be ParameterVector or just values
                new_gate.params = getattr(params, "params", params)
                if new_gate.definition is not None:
                    new_gate.definition.assign_parameters(params, inplace=True)
            if self.opaque_gates:
                new_gate.definition = None
            qc.append(new_gate, local_qargs, local_cargs)
        else:
            func(qc, params, local_qargs + local_cargs, **f_args)
        return circuit_to_dag(qc)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the replacement pass on the given DAG."""
        self.replacement_counters = [0] * len(self.replacements)

        for node in list(dag.op_nodes()):
            if not hasattr(node.op, "name"):
                continue

            for i, repl_rule in enumerate(self.replacements):
                op_template, qargs_template, cargs_template = repl_rule.parsed_target
                mapped_qargs = tuple(
                    dag.qubits[q] if isinstance(q, int) else q for q in qargs_template
                )
                mapped_cargs = tuple(
                    dag.clbits[c] if isinstance(c, int) else c for c in cargs_template
                )

                if (
                    node.op.name == op_template.name
                    and node.qargs == mapped_qargs
                    and node.cargs == mapped_cargs
                ):
                    count = self.replacement_counters[i]

                    # Cycle through the pre-processed lists from the dataclass
                    func = repl_rule.functions_to_cycle[count % len(repl_rule.functions_to_cycle)]
                    params = repl_rule.params_to_cycle[count % len(repl_rule.params_to_cycle)]
                    f_args = repl_rule.args_to_cycle[count % len(repl_rule.args_to_cycle)]

                    replacement_dag = self._create_replacement_dag(
                        func, params, f_args, node.qargs, node.cargs
                    )
                    dag.substitute_node_with_dag(
                        node, replacement_dag, wires=node.qargs + node.cargs
                    )

                    self.replacement_counters[i] += 1
                    break

        # The recursive part for control flow is now cleaner
        for node in list(dag.op_nodes()):
            if isinstance(node.op, ControlFlowOp):
                # The recursive call uses the same, validated list of replacement rules
                recursive_pass = CustomGateReplacementPass(self.replacements)
                new_blocks = [
                    dag_to_circuit(recursive_pass.run(circuit_to_dag(block)))
                    for block in node.op.blocks
                ]
                new_op = node.op.replace_blocks(new_blocks)
                dag.substitute_node(node, new_op)

        return dag


class FilterLocalContext(TransformationPass):

    def __init__(
        self,
        coupling_map: CouplingMap,
        target_instructions: List[
            CircuitInstruction
            | Tuple[
                str,
                Optional[QuantumRegister | Sequence[Qubit | int]],
                Optional[ClassicalRegister | Sequence[Clbit | int]],
            ]
        ],
        occurrence_numbers: List[int] = None,
    ):
        """
        Filter the local context of the circuit by removing all operations that are not involving the target qubits or
        their nearest neighbors. Additionally, the user can specify the number of occurences of the target instructions
        in the circuit context. If not specified, the pass will look for all occurences of the target instructions.

        Args:
            coupling_map: The coupling map of the backend
            target_instructions: The target instructions to be calibrated in the circuit context
            occurrence_numbers: The number of occurences of each target instruction to keep in the circuit context
        """

        if isinstance(target_instructions, CircuitInstruction):
            target_instructions = [target_instructions]
        elif isinstance(target_instructions, tuple):
            target_instructions = [target_instructions]
        elif not isinstance(target_instructions, List):
            raise ValueError("Invalid target instructions format")

        self.target_instructions = []

        for target_instruction in target_instructions:
            if isinstance(target_instruction, CircuitInstruction):
                self.target_instructions.append(
                    (
                        target_instruction.operation,
                        target_instruction.qubits,
                        target_instruction.clbits,
                    )
                )
            else:
                mapping = gate_map()
                if (
                    not isinstance(target_instruction[0], str)
                    or target_instruction[0] not in mapping
                ):
                    raise ValueError("Provided instruction name is not a valid gate name")
                op = mapping[target_instruction[0]]
                qargs = target_instruction[1]
                if isinstance(qargs, QuantumRegister):
                    qargs = (qargs[i] for i in range(len(qargs)))
                elif qargs is None:
                    qargs = ()
                if len(target_instruction) > 2:
                    cargs = target_instruction[2]
                    if isinstance(cargs, ClassicalRegister):
                        cargs = (cargs[i] for i in range(len(cargs)))
                else:
                    cargs = ()

                self.target_instructions.append((op, qargs, cargs))

        if occurrence_numbers is not None:
            if isinstance(occurrence_numbers, int):
                occurrence_numbers = [occurrence_numbers]
            if len(occurrence_numbers) != len(self.target_instructions):
                raise ValueError(
                    "Number of occurence numbers must match number of target instructions"
                )
            self.occurrence_numbers = occurrence_numbers

        else:
            self.occurrence_numbers = [-1 for _ in range(len(self.target_instructions))]
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag: DAGCircuit):
        """
        Run the filtering of local context on the DAG. This step consists in removing all operations that are not
        applied neither on the target qubits nor their nearest neighbors. It also checks the number of occurrences
        of the provided set of instructions and truncate the circuit context accordingly to end up with the desired
        number of occurrences and local context (the rest of the operations are filtered out).
        """
        qargs, cargs = self.get_tgt_args(dag)
        involved_qubits = qargs
        involved_clbits = cargs

        target_nodes = dag.named_nodes(*[op[0].name for op in self.target_instructions])
        target_nodes = list(
            filter(
                lambda node: any(
                    [
                        node.qargs == tgt_instruction[1] and node.cargs == tgt_instruction[2]
                        for tgt_instruction in self.target_instructions
                    ]
                ),
                target_nodes,
            )
        )
        if not target_nodes:
            raise ValueError("Target instructions not found in circuit context")
        isolated_tgt_nodes = list(set(target_nodes))
        node_instruction_mapping = {
            node: target_instruction
            for node, target_instruction in zip(isolated_tgt_nodes, self.target_instructions)
        }
        count_occurrences = [0 for _ in range(len(isolated_tgt_nodes))]
        for node in target_nodes:
            count_occurrences[isolated_tgt_nodes.index(node)] += 1

        current_counts = [0 for _ in range(len(self.target_instructions))]
        keeping_nodes = {node: True for node in dag.op_nodes()}
        for node in dag.op_nodes():

            if (
                node in isolated_tgt_nodes
                and current_counts[isolated_tgt_nodes.index(node)]
                < self.occurrence_numbers[isolated_tgt_nodes.index(node)]
            ):
                current_counts[isolated_tgt_nodes.index(node)] += 1
                keeping_nodes[node] = True

            else:
                for q in node.qargs:
                    if q in involved_qubits:
                        keeping_nodes[node] = True
                        involved_qubits.extend([q for q in node.qargs if q not in qargs])
                        involved_clbits.extend([c for c in node.cargs if c not in cargs])
                        break
                for c in node.cargs:
                    if c in involved_clbits:
                        keeping_nodes[node] = True
                        involved_qubits.extend([q for q in node.qargs if q not in qargs])
                        involved_clbits.extend([c for c in node.cargs if c not in cargs])
                        break
                else:
                    keeping_nodes[node] = False
            for q in node.qargs:
                if q in qargs:
                    involved_qubits.extend([q for q in node.qargs if q not in qargs])
                    break
            for c in node.cargs:
                if c in cargs:
                    involved_clbits.extend([c for c in node.cargs if c not in cargs])
                    break

        def filter_function(node):
            for q in node.qargs:
                if q in involved_qubits and (
                    any([node in dag.ancestors(target_node) for target_node in target_nodes])
                    or node in target_nodes
                ):
                    return True
            for c in node.cargs:
                if c in involved_clbits and (
                    any([node in dag.ancestors(target_node) for target_node in target_nodes])
                    or node in target_nodes
                ):
                    return True
            return False

        filter_op = FilterOpNodes(filter_function)
        dag = filter_op.run(dag)
        dag.remove_qubits(*dag.idle_wires())
        return dag

    def get_tgt_args(self, dag):
        qargs = []
        cargs = []
        current_qargs = ()
        current_cargs = ()
        for i, target_instruction in enumerate(self.target_instructions):
            inst_qargs = target_instruction[1]
            inst_cargs = target_instruction[2]
            if isinstance(inst_qargs, QuantumRegister):
                assert inst_qargs in dag.qregs, "Quantum register not found in DAG"
                current_qargs = tuple([q for q in inst_qargs])
                qargs.extend(current_qargs)
            elif isinstance(inst_qargs, Sequence):
                if isinstance(inst_qargs[0], int):
                    assert all(
                        [q < len(dag.qubits) for q in inst_qargs]
                    ), "Qubit index out of range"
                    current_qargs = tuple([dag.qubits[q] for q in inst_qargs])
                    qargs.extend(current_qargs)
                else:  # Qubit instances
                    for q in inst_qargs:
                        assert q in dag.qubits, "Qubit not found in DAG"
                    current_qargs = tuple(inst_qargs)
                    qargs.extend(inst_qargs)
            if isinstance(inst_cargs, ClassicalRegister):
                assert inst_cargs in dag.cregs, "Classical register not found in DAG"
                current_cargs = tuple([c for c in inst_cargs])
                cargs.extend(current_cargs)
            elif isinstance(inst_cargs, Sequence):
                if isinstance(inst_cargs[0], int):
                    assert all(
                        [c < len(dag.clbits) for c in inst_cargs]
                    ), "Classical bit index out of range"
                    current_cargs = tuple([dag.clbits[c] for c in inst_cargs])
                    cargs.extend(current_cargs)
                else:  # Clbit instances
                    for c in inst_cargs:
                        assert c in dag.clbits, "Classical bit not found in DAG"
                    current_cargs = tuple(inst_cargs)
                    cargs.extend(current_cargs)
            self.target_instructions[i] = (
                target_instruction[0],
                current_qargs,
                current_cargs,
            )
        return list(set(qargs)), list(set(cargs))


class CausalConePass(TransformationPass):

    def __init__(self, qubits: Sequence[int | Qubit] | QuantumRegister):

        self._causal_cone_qubits = qubits
        super().__init__()

    def run(self, dag: DAGCircuit):
        qubits = self._causal_cone_qubits
        if isinstance(qubits, Tuple) and all(isinstance(q, int) for q in qubits):
            qubits = [dag.qubits[q] for q in qubits]
        involved_qubits = [dag.quantum_causal_cone(q) for q in qubits]
        involved_qubits = list(set([q for sublist in involved_qubits for q in sublist]))
        filtered_dag = dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            if all(q in involved_qubits for q in node.qargs):
                filtered_dag.apply_operation_back(node.op, node.qargs)

        filtered_dag.remove_qubits(*[q for q in filtered_dag.qubits if q not in involved_qubits])

        return filtered_dag
