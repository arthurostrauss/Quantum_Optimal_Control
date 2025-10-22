from __future__ import annotations
from dataclasses import dataclass, field

from typing import List, Tuple, Optional, Callable, Dict, Any, Sequence, Union
from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    QuantumRegister,
    ClassicalRegister,
    Qubit,
    Clbit,
    Instruction,
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
    """
    An analysis pass that groups operations into moments.
    """

    def __init__(self):
        super().__init__()
        self.property_history = []

    def run(self, dag):
        """
        Runs the analysis pass.

        Args:
            dag: The DAG to analyze.
        """
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
    Formats the input to be a list.

    Args:
        input_val: The input to format.

    Returns:
        The formatted input.
    """
    if not isinstance(input_val, list):
        input_val = [input_val]
    return input_val


def _parse_instruction(instruction):
    """
    Parses an instruction.

    Args:
        instruction: The instruction to parse.

    Returns:
        The parsed instruction.
    """
    if isinstance(instruction, CircuitInstruction):
        return (instruction.operation, instruction.qubits, instruction.clbits)
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
    """
    Parses a function.

    Args:
        func: The function to parse.

    Returns:
        The parsed function.
    """
    if isinstance(func, str):
        try:
            return gate_map()[func]
        except KeyError:
            raise ValueError(f"Instruction name '{func}' not found in standard gate map.")
    return func


@dataclass
class InstructionReplacement:
    """
    A class to represent an instruction replacement.

    Attributes:
        target_instruction: The instruction to replace.
        new_elements: The new elements to replace the instruction with.
        parameters: The parameters for the new elements.
        parametrized_circuit_functions_args: The arguments for the parametrized circuit functions.
    """

    target_instruction: Union[CircuitInstruction, Tuple]
    new_elements: Union[Callable, Gate, QuantumCircuit, str, List]
    parameters: Optional[Union[Dict, List[Parameter], ParameterVector]] = None
    parametrized_circuit_functions_args: Optional[Union[Dict, List]] = None

    parsed_target: Tuple = field(init=False, repr=False)
    functions_to_cycle: List[Union[Callable, Gate, QuantumCircuit]] = field(init=False, repr=False)
    params_to_cycle: List[Optional[Union[Dict, List]]] = field(init=False, repr=False)
    args_to_cycle: List[Dict] = field(init=False, repr=False)

    def __post_init__(self):
        self.parsed_target = _parse_instruction(self.target_instruction)

        norm_elements = (
            self.new_elements if isinstance(self.new_elements, list) else [self.new_elements]
        )
        self.functions_to_cycle = [_parse_function(f) for f in norm_elements]

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

        num_funcs = len(self.functions_to_cycle)
        num_params = len(norm_params)
        num_args = len(norm_args)

        if num_funcs == 1:
            self.params_to_cycle = norm_params
            self.args_to_cycle = norm_args if num_args > 1 else [norm_args[0]] * num_params
            if num_params > 1 and num_args > 1 and num_params != num_args:
                raise ValueError(
                    f"For target '{self.parsed_target[0].name}', broadcast mismatch: "
                    f"{num_params} parameter sets vs {num_args} argument sets."
                )
            return

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
            return

        self.params_to_cycle = norm_params
        self.args_to_cycle = norm_args


class CustomGateReplacementPass(TransformationPass):
    """
    A transpiler pass that replaces custom gates with their definitions.
    """

    def __init__(self, replacements: Union[InstructionReplacement, List[InstructionReplacement]]):
        """
        Initializes the pass.

        Args:
            replacements: A list of instruction replacements.
        """
        super().__init__()
        self.replacements = format_input(replacements)
        self.replacement_counters = [0] * len(self.replacements)

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
        local_qargs = tuple(qc.qubits[qargs.index(q)] for q in qargs)
        local_cargs = tuple(qc.clbits[cargs.index(c)] for c in cargs)
        if isinstance(func, QuantumCircuit):
            bound_circ = func.assign_parameters(params) if params is not None else func
            qc.append(bound_circ, qargs=local_qargs, cargs=local_cargs)
        elif isinstance(func, Gate):
            new_gate = func.copy()
            if params is not None:
                new_gate.params = params.params if isinstance(params, ParameterVector) else params
            qc.append(new_gate, local_qargs, local_cargs)
        else:
            func(qc, params, list(local_qargs) + list(local_cargs), **f_args)
        return circuit_to_dag(qc)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Runs the pass.

        Args:
            dag: The DAG to run the pass on.

        Returns:
            The modified DAG.
        """
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

        for node in list(dag.op_nodes()):
            if isinstance(node.op, ControlFlowOp):
                recursive_pass = CustomGateReplacementPass(self.replacements)
                new_blocks = [
                    dag_to_circuit(recursive_pass.run(circuit_to_dag(block)))
                    for block in node.op.blocks
                ]
                new_op = node.op.replace_blocks(new_blocks)
                dag.substitute_node(node, new_op)

        return dag


class FilterLocalContext(TransformationPass):
    """
    A transpiler pass that filters the local context of a circuit.
    """

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
        Initializes the pass.

        Args:
            coupling_map: The coupling map of the backend.
            target_instructions: The target instructions.
            occurrence_numbers: The occurrence numbers of the target instructions.
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
        Runs the pass.

        Args:
            dag: The DAG to run the pass on.

        Returns:
            The modified DAG.
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
        """
        Gets the target arguments.

        Args:
            dag: The DAG.

        Returns:
            A tuple of the target qargs and cargs.
        """
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
                else:
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
                else:
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
    """
    A transpiler pass that extracts the causal cone of a circuit.
    """

    def __init__(self, qubits: Sequence[int | Qubit] | QuantumRegister):
        """
        Initializes the pass.

        Args:
            qubits: The qubits to extract the causal cone for.
        """

        self._causal_cone_qubits = qubits
        super().__init__()

    def run(self, dag: DAGCircuit):
        """
        Runs the pass.

        Args:
            dag: The DAG to run the pass on.

        Returns:
            The modified DAG.
        """
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
