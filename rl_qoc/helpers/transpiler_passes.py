from __future__ import annotations

from typing import List, Tuple, Optional, Callable, Dict, Any, Sequence, Union, get_args

from qiskit.circuit import *
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping as gate_map,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import FilterOpNodes
from qiskit.transpiler import TransformationPass
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag

from qiskit.transpiler import AnalysisPass
from qiskit.dagcircuit import DAGOpNode
from collections import defaultdict


class MomentAnalysisPass(AnalysisPass):
    """Analysis pass to group operations into moments, storing results in the PropertySet."""

    def __init__(self):
        super().__init__()
        self.property_history = []

    def run(self, dag):
        # Initialize dictionary to store moments for the current DAG
        moments = defaultdict(list)
        moment_index = 0

        # Use the layers method to get sequentially executable groups
        for layer in dag.layers():
            for node in layer["graph"].op_nodes():
                if isinstance(node, DAGOpNode):
                    moments[moment_index].append(node)
            moment_index += 1

        self.property_history.append(moments)
        self.property_set["moments"] = (
            self.property_history
            if len(self.property_history) > 1
            else self.property_history[0]
        )


def format_input(input, input_type):
    """
    Format the input to the required type

    Args:
        input: The input to be formatted
        input_type: The required type of the input

    Returns:
        The formatted input
    """
    if not isinstance(input, List):
        input = [input]
    valid_types = get_args(input_type)
    assert all([isinstance(i, valid_types) for i in input]), "Invalid input format"
    return input


def _parse_instruction(instruction):
    if isinstance(instruction, CircuitInstruction):
        return (instruction.operation, instruction.qubits, instruction.clbits)
    op = gate_map().get(instruction[0], instruction[0])
    qargs = (
        tuple(instruction[1])
        if isinstance(instruction[1], QuantumRegister)
        else instruction[1]
    )
    if len(instruction) > 2:
        cargs = (
            tuple(instruction[2])
            if isinstance(instruction[2], ClassicalRegister)
            else instruction[2]
        )
    else:
        cargs = ()
    return (op, qargs, cargs)


def _parse_function(func):
    try:
        return gate_map()[func] if isinstance(func, str) else func
    except KeyError:
        raise ValueError(
            "Provided instruction name not part of standard instruction set"
        )


class CustomGateReplacementPass(TransformationPass):
    def __init__(
        self,
        target_instructions: List[
            CircuitInstruction
            | Tuple[
                str | Instruction,
                Optional[QuantumRegister | Sequence[Qubit | int]],
                Optional[ClassicalRegister | Sequence[Clbit | int]],
            ]
        ],
        new_elements: List[Callable | Gate | QuantumCircuit | str],
        parameters: Optional[List[ParameterVector | List[Parameter]]] = None,
        parametrized_circuit_functions_args: List[Dict[str, Any]] = None,
    ):
        """
        Custom transformation pass to replace target instructions in the DAG with custom parametrized circuits

        Args:
            target_instructions: The target instructions to be replaced in the DAG
            new_elements: The new elements to replace the target instructions
            parameters: The parameters to be used in the parametrized circuits (if
            parametrized_circuit_functions_args: The arguments to be passed to the parametrized circuit functions
        """
        super().__init__()

        target_instructions = format_input(
            target_instructions, Union[CircuitInstruction, Tuple, str, Instruction]
        )
        new_elements = format_input(
            new_elements, Union[Callable, Gate, QuantumCircuit, str]
        )
        parameters = format_input(
            parameters,
            Optional[Union[ParameterVector, List]],
        )

        parametrized_circuit_functions_args = format_input(
            parametrized_circuit_functions_args, Optional[Dict]
        )
        if all([param is None for param in parameters]):
            parameters = [None] * len(target_instructions)
        if all([args is None for args in parametrized_circuit_functions_args]):
            parametrized_circuit_functions_args = [{}] * len(target_instructions)
        assert (
            len(target_instructions)
            == len(new_elements)
            == len(parameters)
            == len(parametrized_circuit_functions_args)
        ), "Number of target instructions, parametrized circuit functions, and parameters must match"
        self.target_instructions = [
            _parse_instruction(inst) for inst in target_instructions
        ]
        self.functions = [_parse_function(func) for func in new_elements]
        self.parameters = parameters
        self.parametrized_circuit_functions_args = parametrized_circuit_functions_args

    def run(self, dag: DAGCircuit):
        """Run the custom transformation on the DAG."""
        for i, (op, qargs, cargs) in enumerate(self.target_instructions):

            qc = QuantumCircuit()
            qargs = tuple(dag.qubits[q] if isinstance(q, int) else q for q in qargs)
            cargs = tuple(dag.clbits[c] if isinstance(c, int) else c for c in cargs)
            qc.add_bits(qargs + cargs)

            func = self.functions[i]
            args = list(qargs) + list(cargs)
            qargs2 = qargs if qargs else None
            cargs2 = cargs if cargs else None

            if isinstance(func, QuantumCircuit):
                qc.compose(func, qubits=qargs2, clbits=cargs2, inplace=True)
                if self.parameters[i] is not None:
                    qc.assign_parameters(self.parameters[i], inplace=True)
            elif isinstance(func, Gate):
                qc.append(func, qargs2, cargs2)
            else:  # Callable
                func(
                    qc,
                    self.parameters[i],
                    args,
                    **self.parametrized_circuit_functions_args[i],
                )

            instruction_nodes = dag.named_nodes(op.name)
            instruction_nodes = list(
                filter(
                    lambda node: node.qargs == qargs and node.cargs == cargs,
                    instruction_nodes,
                )
            )
            for node in instruction_nodes:
                dag.substitute_node_with_dag(
                    node, circuit_to_dag(qc), wires=args if args else None
                )

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
                    raise ValueError(
                        "Provided instruction name is not a valid gate name"
                    )
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
                        node.qargs == tgt_instruction[1]
                        and node.cargs == tgt_instruction[2]
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
            for node, target_instruction in zip(
                isolated_tgt_nodes, self.target_instructions
            )
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
                        involved_qubits.extend(
                            [q for q in node.qargs if q not in qargs]
                        )
                        involved_clbits.extend(
                            [c for c in node.cargs if c not in cargs]
                        )
                        break
                for c in node.cargs:
                    if c in involved_clbits:
                        keeping_nodes[node] = True
                        involved_qubits.extend(
                            [q for q in node.qargs if q not in qargs]
                        )
                        involved_clbits.extend(
                            [c for c in node.cargs if c not in cargs]
                        )
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
                    any(
                        [
                            node in dag.ancestors(target_node)
                            for target_node in target_nodes
                        ]
                    )
                    or node in target_nodes
                ):
                    return True
            for c in node.cargs:
                if c in involved_clbits and (
                    any(
                        [
                            node in dag.ancestors(target_node)
                            for target_node in target_nodes
                        ]
                    )
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
