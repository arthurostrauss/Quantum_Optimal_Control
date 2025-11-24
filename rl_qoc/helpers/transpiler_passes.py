from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from typing import Iterable, List, Tuple, Optional, Dict, Any, Sequence, Union, TYPE_CHECKING

from qiskit.circuit import (
    ForLoopOp,
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
from qiskit.providers import BackendV2
if TYPE_CHECKING:
    from ..environment.instruction_replacement import InstructionReplacement
    from ..environment.instruction_rolling import InstructionRolling


__all__ = ["CustomGateReplacementPass", "MomentAnalysisPass", "ForLoopWrapperPass"]

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
                   

class CustomGateReplacementPass(TransformationPass):
    """
    A transpiler pass that dynamically replaces instructions in a circuit
    based on a list of replacement rules. Can also wrap instructions in for loops
    when n_reps is specified.
    """

    def __init__(self, replacements: Union[InstructionReplacement, List[InstructionReplacement]],
                       backend: Optional[BackendV2] = None,
                       opaque_gates: bool = False):
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
        self.n_reps_counters = [0] * len(self.replacements)
        self.opaque_gates = opaque_gates
        self.backend = backend
    def _create_replacement_dag(
        self,
        func: Union[QuantumCircuit, Instruction],
        params: Optional[Union[Dict, List]],
        qargs: Tuple[Qubit, ...],
        cargs: Tuple[Clbit, ...],
    ) -> DAGCircuit:
        dag = DAGCircuit()
        dag.add_qubits(qargs)
        dag.add_clbits(cargs)

        local_qargs = dag.qubits
        local_cargs = dag.clbits
        # Handle QuantumCircuit and Gate types
        if isinstance(func, QuantumCircuit):
            new_instr = func.assign_parameters(params) if params is not None else func
            if self.opaque_gates:
                new_instr = new_instr.to_instruction()
                new_instr.definition = None
            dag.compose(circuit_to_dag(new_instr), local_qargs, local_cargs, inplace=True)
        elif isinstance(func, Instruction):
            new_gate = func.copy()
            if params is not None:
                # params can be ParameterVector or just values
                new_gate.params = getattr(params, "params", params)
                if new_gate.definition is not None:
                    new_gate.definition.assign_parameters(params, inplace=True)
            if self.opaque_gates:
                new_gate.definition = None
            dag.apply_operation_back(new_gate, local_qargs, local_cargs)
        else:
            # This should not happen as only Instruction/QuantumCircuit are allowed
            raise ValueError(
                f"Invalid replacement element type: {type(func)}. "
                f"Only Instruction or QuantumCircuit are allowed."
            )
        return dag

    def _create_for_loop_dag(
        self,
        node: DAGOpNode,
        n_reps: Union[int, str, Iterable[int], None],
        n_counter: int,
    ) -> DAGCircuit:
        """
        Create a for loop DAG for the given node.
        
        Args:
            node: The DAG node to wrap in a for loop
            n_reps: Number of repetitions. Can be:
                - An integer: fixed number of repetitions
                - A string: name of an input variable for the number of repetitions
                - A list of integers: cycle through different repetition counts
                - None: no for loop wrapping (should not happen)
            n_counter: Counter for generating unique variable names when n_reps is None
        """
        dag = DAGCircuit()
        dag.add_qubits(node.qargs)
        dag.add_clbits(node.cargs)
        local_qargs = dag.qubits
        local_cargs = dag.clbits
        if n_reps is None or isinstance(n_reps, str):
            from qiskit.circuit.classical import expr, types
            name = f"_n_{n_counter}" if n_reps is None else n_reps
            n = expr.Var.new(name, types.Uint(64))
            dag.add_input_var(n)
            r = expr.Range(expr.lift(0, types.Uint(64)), n)
        elif isinstance(n_reps, (list, tuple)):
            r = list(n_reps)
        elif isinstance(n_reps, int):
            r = range(n_reps)
        else:
            raise ValueError(f"Invalid n_reps type: {type(n_reps)}. Expected int, str, list, or None.")
        qc = QuantumCircuit()
        qc.add_bits(node.qargs + node.cargs)
        qc.append(node.op, node.qargs, node.cargs)
        for_loop_op = ForLoopOp(r, None, qc)

        dag.apply_operation_back(for_loop_op, local_qargs, local_cargs)
        return dag

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the replacement pass on the given DAG."""
        self.replacement_counters = [0] * len(self.replacements)
        self.n_reps_counters = [0] * len(self.replacements)
        

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
                    # Determine what to do based on custom_instruction and n_reps
                    has_custom_instr = len(repl_rule.functions_to_cycle) > 0
                    has_n_reps = repl_rule.n_reps is not None
                    
                    if has_custom_instr and has_n_reps:
                        # Replace with custom instruction AND wrap in for loop
                        count = self.replacement_counters[i]
                        func = repl_rule.functions_to_cycle[count % len(repl_rule.functions_to_cycle)]
                        params_list = list(repl_rule.params_to_cycle) if isinstance(repl_rule.params_to_cycle, (list, tuple)) else [repl_rule.params_to_cycle]
                        params = params_list[count % len(params_list)] if len(params_list) > 0 else None
                        
                        # Create replacement DAG
                        replacement_dag = self._create_replacement_dag(
                            func, params, node.qargs, node.cargs
                        )
                        
                        # Convert back to circuit to wrap in for loop
                        replacement_circuit = dag_to_circuit(replacement_dag)
                        
                        # Create a temporary node from the replacement circuit to wrap
                        temp_dag = circuit_to_dag(replacement_circuit)
                        temp_node = list(temp_dag.op_nodes())[0]  # Get the first (and only) operation node
                        
                        # Wrap in for loop
                        n_reps_count = self.n_reps_counters[i]
                        n_reps = repl_rule.n_reps
                        if isinstance(n_reps, list):
                            n_reps = n_reps[n_reps_count % len(n_reps)]
                        
                        for_loop_dag = self._create_for_loop_dag(temp_node, n_reps, n_reps_count)
                        dag.substitute_node_with_dag(
                            node, for_loop_dag, wires=node.qargs + node.cargs
                        )
                        
                        self.replacement_counters[i] += 1
                        self.n_reps_counters[i] += 1
                        
                    elif has_custom_instr:
                        # Only replace with custom instruction (no for loop)
                        count = self.replacement_counters[i]
                        func = repl_rule.functions_to_cycle[count % len(repl_rule.functions_to_cycle)]
                        params_list = list(repl_rule.params_to_cycle) if isinstance(repl_rule.params_to_cycle, (list, tuple)) else [repl_rule.params_to_cycle]
                        params = params_list[count % len(params_list)] if len(params_list) > 0 else None

                        replacement_dag = self._create_replacement_dag(
                            func, params, node.qargs, node.cargs
                        )
                        dag.substitute_node_with_dag(
                            node, replacement_dag, wires=node.qargs + node.cargs
                        )

                        self.replacement_counters[i] += 1
                        
                    elif has_n_reps:
                        # Only wrap original instruction in for loop (no replacement)
                        n_reps_count = self.n_reps_counters[i]
                        n_reps = repl_rule.n_reps
                        if isinstance(n_reps, list):
                            n_reps = n_reps[n_reps_count % len(n_reps)]
                        
                        for_loop_dag = self._create_for_loop_dag(node, n_reps, n_reps_count)
                        dag.substitute_node_with_dag(
                            node, for_loop_dag, wires=node.qargs + node.cargs
                        )
                        
                        self.n_reps_counters[i] += 1
                    
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

        if self.backend is not None:
            for repl_rule in self.replacements:
                if repl_rule.instruction_properties is not None:
                    repl_rule.add_instruction_properties(self.backend)
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
        if isinstance(qubits, Sequence) and all(isinstance(q, int) for q in qubits):
            qubits = [dag.qubits[q] for q in qubits]
        involved_qubits = [dag.quantum_causal_cone(q) for q in qubits]
        involved_qubits = list(set([q for sublist in involved_qubits for q in sublist]))
        filtered_dag = dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            if all(q in involved_qubits for q in node.qargs):
                filtered_dag.apply_operation_back(node.op, node.qargs)

        filtered_dag.remove_qubits(*[q for q in filtered_dag.qubits if q not in involved_qubits])

        return filtered_dag
