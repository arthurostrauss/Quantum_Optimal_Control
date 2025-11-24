"""
Target classes for the quantum environment

This module contains the classes to represent the target states and gates for the quantum environment

Author: Arthur Strauss
Created: 08/11/2024
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Statevector,
    pauli_basis,
    state_fidelity,
    average_gate_fidelity,
)
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.transpiler import Layout
import numpy as np
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    Gate,
    CircuitInstruction,
    Qubit,
    Parameter,
)
from itertools import product
from typing import Any, Dict, List, Optional, Literal, Sequence, Union
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
    Pauli6PreparationBasis,
)
from ..helpers.circuit_utils import (
    density_matrix_to_statevector,
    get_gate,
    causal_cone_circuit,
    get_2design_input_states,
)
from .instruction_replacement import InstructionReplacement
import warnings


def _calculate_chi_target(target: DensityMatrix | Operator | QuantumCircuit | Gate):
    """
    Calculate characteristic function for the given target. Based on DFE scheme
    :param target: Density matrix of the target state or Operator of the target gate
    :return: Characteristic function of the target state
    """

    if not isinstance(target, (DensityMatrix, Operator)):
        try:  # Try to convert to Operator (in case Gate or QuantumCircuit is provided)
            target = Operator(target)
        except Exception as e:
            raise ValueError(
                "Target should be a DensityMatrix or an Operator (Gate or QuantumCircuit) object"
            ) from e
    d = 2**target.num_qubits
    basis = pauli_basis(num_qubits=target.num_qubits)
    if isinstance(target, DensityMatrix):
        chi = np.real([target.expectation_value(basis[k]) for k in range(d**2)]) / np.sqrt(d)
    else:
        dms = [DensityMatrix(pauli).evolve(target) for pauli in basis]
        chi = (
            np.real(
                [dms[k_].expectation_value(basis[k]) for k_, k in product(range(d**2), repeat=2)]
            )
            / d
        )
        # dms = [target.to_matrix() @ basis[k].to_matrix() @ target.adjoint().to_matrix() for k in range(len(basis))]
        # chi = np.real([np.trace(dms[k_] @ basis[k].to_matrix()) for k_, k in product(range(d**2), repeat=2)])/d

    # Real part is taken to convert it in good format,
    # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
    return chi


class BaseTarget(ABC):
    """
    Base class for the target of the quantum environment
    """

    def __init__(
        self,
        physical_qubits: Sequence[int] | int | Sequence[Sequence[int]],
        tgt_register: QuantumRegister | Sequence[Qubit] | Sequence[Sequence[Qubit]],
        layout: Layout | Sequence[Layout],
    ):
        """
        Initialize the base target for the quantum environment.
        :param physical_qubits: Physical qubits on which the target is defined.
        :param tgt_register: Optional existing QuantumRegister for the target.
        :param layout: Optional existing layout for the target (used when target touches a subset of
        all physical qubits present in a circuit)

        """
        self._physical_qubits = (
            list(range(physical_qubits)) if isinstance(physical_qubits, int) else physical_qubits
        )
        self._tgt_register = tgt_register
        self._layout = layout
        self._n_qubits = len(self._physical_qubits) if isinstance(self._physical_qubits, Sequence) and all(isinstance(q, int) for q in self._physical_qubits) else len(self._physical_qubits[0]) 

    @property
    def physical_qubits(self) -> Sequence[int]:
        """
        Physical qubits on which the target is defined
        """
        raise NotImplementedError("Physical qubits not implemented")

    @property
    def tgt_register(self) -> Sequence[Qubit]:
        """
        Target register on which the target is defined
        """
        raise NotImplementedError("Target register not implemented")

    @property
    def layout(self) -> Layout:
        """
        Layout for the target
        """
        raise NotImplementedError("Layout not implemented")

    @layout.setter
    def layout(self, layout: Layout):
        if not isinstance(layout, Layout):
            raise ValueError("Layout should be a Layout object")
        self._layout = layout

    @property
    def n_qubits(self) -> int:
        """
        Number of qubits on which the target is defined
        """
        return self._n_qubits

    @property
    def target_type(self) -> str:
        """
        Type of the target (state / gate)
        """
        raise NotImplementedError("Target type not implemented")


class StateTarget(BaseTarget):
    """
    Class to represent the state target for the quantum environment
    """

    def __init__(
        self,
        state: DensityMatrix | Statevector | QuantumCircuit | str | np.ndarray,
        physical_qubits: Optional[Sequence[int]] = None,
    ):
        """
        Initialize the state target for the quantum environment
        :param physical_qubits: Physical qubits forming the target state
        """
        if isinstance(state, str):
            self.dm = DensityMatrix.from_label(state)
            tgt_register = QuantumRegister(self.dm.num_qubits, "tgt")
            self.circuit = QuantumCircuit(tgt_register)
            self.circuit.prepare_state(Statevector.from_label(state))
        elif isinstance(state, (QuantumCircuit, Statevector)):
            self.dm = DensityMatrix(state)
            if isinstance(state, QuantumCircuit):
                tgt_register = state.qregs[0] if state.qregs else None
                self.circuit = state
            else:
                tgt_register = QuantumRegister(state.num_qubits, "tgt")
                self.circuit = QuantumCircuit(tgt_register)
                self.circuit.prepare_state(state)
        elif isinstance(state, DensityMatrix):
            self.dm = state
            tgt_register = QuantumRegister(self.dm.num_qubits, "tgt")
            self.circuit = QuantumCircuit(tgt_register)
            self.circuit.prepare_state(density_matrix_to_statevector(state))
        else:
            try:  # Try to convert to DensityMatrix
                state = DensityMatrix(state)
                if physical_qubits is not None and state.num_qubits != len(physical_qubits):
                    raise ValueError(
                        "Number of qubits in the state should match the number of physical qubits"
                    )
                if state.purity() - 1 > 1e-6:
                    raise ValueError("Density matrix should be pure")
                self.dm = state
                tgt_register = QuantumRegister(self.dm.num_qubits, "tgt")
                self.circuit = QuantumCircuit(tgt_register)
                self.circuit.prepare_state(density_matrix_to_statevector(state))
            except Exception as e:
                raise ValueError("Input could not be converted to DensityMatrix") from e
        if physical_qubits is None:
            physical_qubits = list(range(self.dm.num_qubits))

        super().__init__(
            physical_qubits=physical_qubits,
            tgt_register=tgt_register,
            layout=Layout(
                {tgt_register[i]: physical_qubits[i] for i in range(len(physical_qubits))}
            ),
        )

        self.Chi = _calculate_chi_target(self.dm)
        layout = Layout(
            {self.circuit.qubits[i]: physical_qubits[i] for i in range(len(physical_qubits))}
        )
        super().__init__(
            physical_qubits=physical_qubits,
            tgt_register=tgt_register,
            layout=layout,
        )
    
    @property
    def physical_qubits(self) -> Sequence[int]:
        """
        Physical qubits on which the target is defined
        """
        return self._physical_qubits

    @property
    def tgt_register(self) -> Sequence[Qubit]:
        """
        Target register on which the target is defined
        """
        return self._tgt_register

    @property
    def layout(self) -> Layout:
        """
        Layout for the target
        """
        return self._layout
    def target_instruction(self) -> CircuitInstruction:
        """
        Get the target instruction
        """
        return self.circuit.data[0]

    @property
    def target_instruction_counts(self) -> int:
        """
        Get the number of target instructions in the circuit
        """
        return self.circuit.data.count(self.target_instruction)

    @property
    def target_instructions(self) -> List[CircuitInstruction]:
        """
        Get the target instructions in the circuit
        """
        return [self.target_instruction]

    @property
    def target_type(self):
        """
        Type of the target (state)
        """
        return "state"

    def fidelity(self, state: QuantumState | QuantumCircuit, n_reps: int = 1, validate=True):
        """
        Compute the fidelity of the state with the target
        :param state: State to compare with the target state
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if isinstance(state, QuantumCircuit):
            try:
                state = DensityMatrix(state)
            except Exception as e:
                raise ValueError("Input could not be converted to state") from e
        if not isinstance(state, (Statevector, DensityMatrix)):
            raise ValueError("Input should be a Statevector or DensityMatrix object")
        return state_fidelity(state, self.dm, validate=validate)

    def __repr__(self):
        return f"StateTarget({self.dm} on qubits {self.physical_qubits})"

    def as_dict(self):
        """
        Convert the target state to a dictionary representation
        """
        return {
            "state": np.array2string(self.dm.data, separator=","),
            "physical_qubits": self.physical_qubits,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a StateTarget from a dictionary representation
        :param data: Dictionary representation of the target state
        :return: StateTarget object
        :raises ValueError: If the data does not contain a valid density matrix or physical qubits
        """
        dm = DensityMatrix(np.array(data["dm"]))
        if dm.num_qubits is None:
            raise ValueError("DensityMatrix num_qubits is None")
        physical_qubits = data.get("physical_qubits", list(range(dm.num_qubits)))
        return cls(state=dm, physical_qubits=physical_qubits)


class InputState(StateTarget):
    """
    Class to represent the input state for the quantum environment
    """

    def __init__(
        self,
        input_circuit: QuantumCircuit,
        target_op: Gate | QuantumCircuit,
    ):
        """
        Initialize the input state for the quantum environment
        :param input_circuit: Quantum circuit representing the input state
        :param target_op: Gate to be calibrated (or circuit context)
        :param tgt_register: Quantum register for the target gate
        """
        super().__init__(input_circuit)
        self._target_op = target_op

    def target_state(self, n_reps: int = 1):
        """
        Get the target state
        """
        if isinstance(self._target_op, Gate):
            circ = QuantumCircuit(self.tgt_register)
            circ.append(self._target_op, self.tgt_register)
        else:
            circ = self._target_op
        circ = circ.repeat(n_reps).compose(self.circuit, front=True, inplace=False)
        return StateTarget(circ)

    @property
    def layout(self):
        raise AttributeError("Input state does not have a layout")

    @layout.setter
    def layout(self, layout: Layout):
        raise AttributeError("Input state does not have a layout")

    @property
    def tgt_register(self):
        raise AttributeError("Input state does not have a target register")

    @property
    def input_circuit(self) -> QuantumCircuit:
        """
        Get the input state preparation circuit
        """
        return self.circuit

    @property
    def target_circuit(self, n_reps: int = 1) -> QuantumCircuit:
        """
        Get the target circuit for the input state
        """
        return self.target_state(n_reps).circuit

    @property
    def target_dm(self, n_reps: int = 1) -> DensityMatrix:
        """
        Get the target density matrix for the input state
        """
        return self.target_state(n_reps).dm


def _normalize_circuit_context(
    circuit_context: Optional[QuantumCircuit | List[QuantumCircuit]], gate: Gate
) -> List[QuantumCircuit]:
    """
    Normalize circuit_context to a list of circuits.
    If None, create a default circuit with just the gate.
    """
    if circuit_context is None:
        tgt_register = QuantumRegister(gate.num_qubits, "tgt")
        circ = QuantumCircuit(tgt_register)
        circ.append(gate, tuple(q for q in tgt_register))
        return [circ]
    elif isinstance(circuit_context, QuantumCircuit):
        return [circuit_context]
    else:
        return list(circuit_context)


def _normalize_virtual_target_qubits(
    virtual_target_qubits: Optional[Sequence[int | Qubit] | Sequence[Sequence[int | Qubit]]],
    circuit_context: List[QuantumCircuit],
    gate: Gate,
    instruction_replacement: Optional[InstructionReplacement],
) -> List[List[Qubit]]:
    """
    Normalize virtual_target_qubits to a list of lists of Qubit objects, one per circuit context.
    Validates that all qubits are present in their respective circuit contexts.
    """
    num_circuits = len(circuit_context)
    
    # Validate circuit contexts have enough qubits
    if any(circ.num_qubits < gate.num_qubits for circ in circuit_context):
        raise ValueError(
            "Circuit context must have at least as many qubits as the target gate"
        )
    
    if virtual_target_qubits is None:
        # Auto-determine virtual target qubits
        if any(circ.num_qubits > gate.num_qubits for circ in circuit_context):
            if instruction_replacement is None:
                raise ValueError(
                    "If circuit context is larger than target gate, virtual_target_qubits must be provided"
                )
            # Use instruction replacement target qubits
            target_qargs = instruction_replacement.target_qargs
            if all(isinstance(q, Qubit) for q in target_qargs):
                # Check all circuits have these qubits
                if not all(
                    all(q in circ.qubits for q in target_qargs) for circ in circuit_context
                ):
                    raise ValueError(
                        "Instruction replacement target qubits must be in all circuit contexts"
                    )
                return [list(target_qargs) for _ in circuit_context]
            elif all(isinstance(q, int) for q in target_qargs):
                return [[circ.qubits[q] for q in target_qargs] for circ in circuit_context]
            else:
                raise ValueError(
                    "Instruction replacement target qubits must be Qubit objects or integers"
                )
        # Use all qubits in each circuit
        return [[q for q in circ.qubits] for circ in circuit_context]
    
    # Check if it's a sequence of sequences (one per circuit)
    if isinstance(virtual_target_qubits, Sequence) and len(virtual_target_qubits) > 0:
        if isinstance(virtual_target_qubits[0], Sequence) and all(isinstance(q, Sequence) for q in virtual_target_qubits):
            # Sequence of sequences - one per circuit
            if len(virtual_target_qubits) != num_circuits:
                raise ValueError(
                    f"Number of virtual qubit sequences ({len(virtual_target_qubits)}) "
                    f"must match number of circuit contexts ({num_circuits})"
                )
            result = []
            for i, vq_seq in enumerate(virtual_target_qubits):
                circ = circuit_context[i]
                if not isinstance(vq_seq, Sequence):
                    raise ValueError(f"Virtual qubit sequence {i} must be a sequence")
                vq_list = list(vq_seq)
                if len(vq_list) == 0:
                    raise ValueError(f"Virtual qubit sequence {i} cannot be empty")
                
                if all(isinstance(q, Qubit) for q in vq_list):
                    if not all(q in circ.qubits for q in vq_list):
                        raise ValueError(
                            f"Virtual target qubits for circuit {i} must be in the circuit context."
                        )
                    result.append(vq_list)
                elif all(isinstance(q, int) for q in vq_list):
                    # Validate indices
                    if any(q < 0 or q >= circ.num_qubits for q in vq_list):
                        raise ValueError(
                            f"Virtual target qubit indices for circuit {i} are out of range"
                        )
                    result.append([circ.qubits[q] for q in vq_list])
                else:
                    raise ValueError(
                        f"Virtual target qubits for circuit {i} must be a sequence of "
                        f"Qubit objects or a sequence of integers"
                    )
            return result
        else:
            # Single sequence - broadcast to all circuits
            if len(virtual_target_qubits) == 0:
                raise ValueError("Virtual target qubits cannot be empty")
            
            first_item = virtual_target_qubits[0]
            if isinstance(first_item, Qubit):
                if not all(isinstance(q, Qubit) for q in virtual_target_qubits):
                    raise ValueError(
                        "Virtual target qubits must be a sequence of Qubit objects, "
                        "a sequence of integers, or a sequence of such sequences"
                    )
                # Validate all circuits have these qubits
                if not all(
                    all(q in circ.qubits for q in virtual_target_qubits) for circ in circuit_context
                ):
                    raise ValueError("Virtual target qubits must be in all circuit contexts.")
                return [list(virtual_target_qubits) for _ in circuit_context]
            elif isinstance(first_item, int):
                if not all(isinstance(q, int) for q in virtual_target_qubits):
                    raise ValueError(
                        "Virtual target qubits must be a sequence of Qubit objects, "
                        "a sequence of integers, or a sequence of such sequences"
                    )
                # Cast to Sequence[int] for type checker
                virt_int_seq: Sequence[int] = virtual_target_qubits  # type: ignore
                # Validate indices for all circuits
                for i, circ in enumerate(circuit_context):
                    if any(q < 0 or q >= circ.num_qubits for q in virt_int_seq):
                        raise ValueError(
                            f"Virtual target qubit indices for circuit {i} are out of range"
                        )
                return [
                    [circ.qubits[q] for q in virt_int_seq] for circ in circuit_context
                ]
            else:
                raise ValueError(
                    "Virtual target qubits must be a sequence of Qubit objects, "
                    "a sequence of integers, or a sequence of such sequences"
                )
    
    raise ValueError(
        "Virtual target qubits must be a sequence of Qubit objects, "
        "a sequence of integers, or a sequence of such sequences"
    )


def _normalize_physical_qubits(
    physical_qubits: Optional[Sequence[int] | Sequence[Sequence[int]] | int],
    gate: Gate,
    num_circuits: int,
) -> List[List[int]]:
    """
    Normalize physical_qubits to a list of lists of integers, one per circuit context.
    """
    if physical_qubits is None:
        # Default: use range based on gate size
        return [list(range(gate.num_qubits)) for _ in range(num_circuits)]
    elif isinstance(physical_qubits, int):
        # Single integer - create range and broadcast
        return [list(range(physical_qubits)) for _ in range(num_circuits)]
    elif isinstance(physical_qubits, Sequence):
        if len(physical_qubits) == 0:
            raise ValueError("Physical qubits cannot be empty")
        
        if isinstance(physical_qubits[0], Sequence) and all(isinstance(q, Sequence) for q in physical_qubits):
            # Sequence of sequences - one per circuit
            if len(physical_qubits) != num_circuits:
                raise ValueError(
                    f"Number of physical qubit sequences ({len(physical_qubits)}) "
                    f"must match number of circuit contexts ({num_circuits})"
                )
            result = []
            for i, phys_seq in enumerate(physical_qubits):
                if not isinstance(phys_seq, Sequence):
                    raise ValueError(f"Physical qubit entry {i} must be a sequence")
                phys_list = list(phys_seq)
                if len(phys_list) == 0:
                    raise ValueError(f"Physical qubit sequence {i} cannot be empty")
                if not all(isinstance(q, int) for q in phys_list):
                    raise ValueError(f"Each physical qubit entry {i} must be an integer")
                result.append(phys_list)
            return result
        else:
            # Single sequence - broadcast to all circuits
            if not all(isinstance(q, int) for q in physical_qubits):
                raise ValueError("Physical qubits must be integers")
            # Cast to Sequence[int] for type checker
            phys_int_seq: Sequence[int] = physical_qubits  # type: ignore
            return [list(phys_int_seq) for _ in range(num_circuits)]
    
    raise ValueError(f"Invalid physical_qubits type: {type(physical_qubits)}")


def _normalize_layout(
    layout: Optional[Layout | List[Layout]],
    virtual_target_qubits: List[List[Qubit]],
    physical_qubits: List[List[int]],
    circuit_context: List[QuantumCircuit],
    gate: Gate,
) -> List[Layout]:
    """
    Normalize layout to a list of Layout objects, one per circuit context.
    Validates that all qubits in layout are present in circuit contexts.
    If layout is None, creates layouts from virtual_target_qubits and physical_qubits.
    """
    num_circuits = len(circuit_context)
    
    if layout is not None:
        # Normalize to list
        if isinstance(layout, Layout):
            layout_list = [layout for _ in range(num_circuits)]
        else:
            if len(layout) != num_circuits:
                raise ValueError(
                    f"Number of layouts ({len(layout)}) "
                    f"must match number of circuit contexts ({num_circuits})"
                )
            layout_list = list(layout)
        
        # Validate all qubits in layouts are in circuit contexts
        for i, (lay, circ) in enumerate(zip(layout_list, circuit_context)):
            if not isinstance(lay, Layout):
                raise ValueError(f"Layout {i} must be a Layout object")
            # Check all virtual qubits in layout are in circuit
            for virt_q in lay.get_virtual_bits():
                if virt_q not in circ.qubits:
                    raise ValueError(
                        f"Virtual qubit {virt_q} in layout {i} is not in circuit context {i}"
                    )
        
        return layout_list
    
    # No layout provided - create from virtual_target_qubits and physical_qubits
    if len(virtual_target_qubits) != num_circuits or len(physical_qubits) != num_circuits:
        raise ValueError(
            "When circuit context is larger than target gate, "
            "layout must be provided or virtual_target_qubits and physical_qubits "
            "must match number of circuit contexts"
        )
    
    layout_list = []
    for virt_qs, phys_qs in zip(virtual_target_qubits, physical_qubits):
        if len(virt_qs) != len(phys_qs):
            raise ValueError(
                f"Length mismatch: virtual qubits ({len(virt_qs)}) vs physical qubits ({len(phys_qs)})"
            )
        layout_list.append(Layout({virt_qs[i]: phys_qs[i] for i in range(len(virt_qs))}))
    return layout_list



class GateTarget(BaseTarget):
    """
    Class to represent the gate target for the quantum environment
    """

    def __init__(
        self,
        gate: Gate | str | InstructionReplacement,
        physical_qubits: Optional[Sequence[int] | Sequence[Sequence[int]] | int] = None,
        circuit_context: Optional[QuantumCircuit | List[QuantumCircuit]] = None,
        virtual_target_qubits: Optional[Sequence[int | Qubit] | Sequence[Sequence[int | Qubit]]] = None,
        layout: Optional[Layout | List[Layout]] = None,
    ):
        """
        Initialize the gate target for the quantum environment.
        :param gate: Gate to be calibrated. It can be a Gate object or a string representing the gate name.
        :param physical_qubits: Physical qubits forming the target gate. Can be:
            - None: defaults to range(gate.num_qubits) for each circuit
            - int: creates range(physical_qubits) and broadcasts to all circuits
            - Sequence[int]: single sequence broadcast to all circuits
            - Sequence[Sequence[int]]: one sequence per circuit context
        :param circuit_context: Circuit to be used for context-aware calibration (default is the gate to be calibrated). 
            Can also be a list of circuits. If None, creates a default circuit with just the gate.
        :param virtual_target_qubits: Virtual target qubits to be used for the context-aware calibration. 
            Can be a single sequence (applied to all circuits) or a sequence of sequences (one per circuit).
            Those are the qubits within the virtual circuit context (not necessarily the same as the physical qubits after transpilation).
        :param layout: Specify layout if already declared. Can be a single Layout (broadcast to all circuits) 
            or a list of Layouts (one per circuit context).
        """
        # Parse gate
        if isinstance(gate, InstructionReplacement):
            if not isinstance(gate.target_operation, Gate):
                raise ValueError("Target operation must be a Gate object")
            self._gate = gate.target_operation
            self._instruction_replacement = gate
        else:
            self._gate = get_gate(gate)
            self._instruction_replacement = None
        
        # Normalize circuit_context to list of circuits
        circuit_context_list = _normalize_circuit_context(circuit_context, self._gate)
        num_circuits = len(circuit_context_list)
        
        # Determine if we have context (not just the default gate circuit)
        self._has_context = circuit_context is not None
        self._circuit_choice = 0
        
        # Normalize virtual_target_qubits to list of lists of Qubit objects
        self._virtual_target_qubits = _normalize_virtual_target_qubits(
            virtual_target_qubits, circuit_context_list, self._gate, self._instruction_replacement
        )
        
        # Compute virtual target qubit indices
        self._virtual_target_qubits_indices = [
            [circ.find_bit(q).index for q in vq]
            for circ, vq in zip(circuit_context_list, self._virtual_target_qubits)
        ]
        
        # Normalize physical_qubits to list of lists of integers
        physical_qubits_list = _normalize_physical_qubits(
            physical_qubits, self._gate, num_circuits
        )
        
        # Normalize layout to list of Layout objects
        layout_list = _normalize_layout(
            layout, self._virtual_target_qubits, physical_qubits_list, 
            circuit_context_list, self._gate
        )
        
        # Initialize base class with list of lists
        super().__init__(
            physical_qubits=physical_qubits_list,
            tgt_register=self._virtual_target_qubits,
            layout=layout_list
        )
        
        # Store circuit contexts and parameters
        self._unbound_circuit_contexts = circuit_context_list
        self._bound_circuit_contexts = [
            circ if not circ.parameters else None for circ in circuit_context_list
        ]
        self._context_parameters: List[Dict[Parameter, float | None]] = [
            {p: None for p in circ.parameters} for circ in circuit_context_list
        ]

    @property
    def gate(self) -> Gate:
        """
        Get the target gate
        """
        return self._gate
    
    @property
    def instruction_replacement(self) -> Optional[InstructionReplacement]:
        """
        Get the instruction replacement for the target gate if defined.
        """
        return self._instruction_replacement

    def Chi(self, n_reps: int = 1):
        """
        Compute the characteristic function of the target gate.
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if self.causal_cone_size <= 3:
            if n_reps == 1:
                return _calculate_chi_target(self.target_operator)
            else:
                return _calculate_chi_target(self.target_operator.power(n_reps))
        else:
            warnings.warn("Chi is not computed for more than 3 qubits")
            return None

    def input_states(self, input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"):
        """
        Get the input states for the target
        :param input_states_choice: Type of input states to be used for the calibration
            (relevant only for state reward)
        """
        n_qubits = self.causal_cone_size
        if input_states_choice == "pauli4":
            input_circuits = [
                PauliPreparationBasis().circuit(s) for s in product(range(4), repeat=n_qubits)
            ]
        elif input_states_choice == "pauli6":
            input_circuits = [
                Pauli6PreparationBasis().circuit(s) for s in product(range(6), repeat=n_qubits)
            ]
        elif input_states_choice == "2-design":  # 2-design
            d = 2**n_qubits
            states = get_2design_input_states(d)
            input_circuits = [QuantumCircuit(n_qubits) for _ in range(len(states))]
            for circ, state in zip(input_circuits, states):
                circ.prepare_state(state)
        else:
            raise ValueError(
                f"Input states choice {input_states_choice} not recognized. Should be 'pauli4', 'pauli6' or '2-design'"
            )
        input_states = [
            InputState(
                input_circuit=circ,
                target_op=self.causal_cone_circuit,
            )
            for circ in input_circuits
        ]
        return input_states

    def gate_fidelity(
        self,
        channel: QuantumChannel | Operator | Gate | QuantumCircuit,
        n_reps: int = 1,
    ) -> float:
        """
        Compute the average gate fidelity of the gate with the target gate
        If the target has a circuit context, the fidelity is computed with respect to the channel derived from the
        target circuit
        :param channel: channel to compare with the target gate
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if isinstance(channel, (QuantumCircuit, Gate)):
            try:
                channel = Operator(channel)
            except Exception as e:
                raise ValueError("Input could not be converted to channel") from e
        if not isinstance(channel, (Operator, QuantumChannel)):
            raise ValueError("Input should be an Operator object")

        if channel.num_qubits == self.causal_cone_size:
            circuit = self.causal_cone_circuit
        else:
            circuit = self.circuit

        return average_gate_fidelity(channel, Operator(circuit).power(n_reps))

    def state_fidelity(self, state: QuantumState, n_reps: int = 1, validate: bool = True):
        """
        Compute the fidelity of the state with the target state derived from the application of the target gate/circuit
        context to |0...0>
        :param state: State to compare with target state
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if not isinstance(state, (Statevector, DensityMatrix)):
            raise ValueError("Input should be a Statevector or DensityMatrix object")
        if np.linalg.norm(state) != 1 and not validate:
            warnings.warn(f"Input state is not normalized (norm = {np.linalg.norm(state)})")
        if state.num_qubits == self.causal_cone_size:
            circuit = self.causal_cone_circuit
        else:
            circuit = self.circuit
        return state_fidelity(
            state,
            Statevector(circuit.power(n_reps, True, True)),
            validate=validate,
        )

    def fidelity(
        self,
        op: QuantumState | QuantumChannel | Operator,
        n_reps: int = 1,
        validate: bool = True,
    ) -> float:
        """
        Compute the fidelity of the input op with respect to the target circuit context channel/output state.
        :param op: Object to compare with the target. If QuantumState, computes state fidelity, if QuantumChannel or
            Operator, computes gate fidelity
        :param n_reps: Specify number of repetitions of the target operation (default is 1)
        :param validate: Validate the input state (default is True)
        """
        if isinstance(op, (QuantumCircuit, Gate)):
            try:
                op = Operator(op)
            except Exception as e:
                raise ValueError("Input could not be converted to channel") from e
        if isinstance(op, (Operator, QuantumChannel)):
            return self.gate_fidelity(op, n_reps)
        elif isinstance(op, (Statevector, DensityMatrix)):
            return self.state_fidelity(op, n_reps, validate)
        else:
            raise ValueError(
                f"Input should be a Statevector, DensityMatrix, Operator or QuantumChannel object, "
                f"received {type(op)}"
            )

    @property
    def target_instruction(self) -> CircuitInstruction:
        """
        Get the target instruction
        """
        return CircuitInstruction(self.gate, (q for q in self._tgt_register[self._circuit_choice]))

    @property
    def target_instructions(self) -> List[CircuitInstruction]:
        """
        Get the target instructions in the circuit
        """
        return [CircuitInstruction(self.gate, (q for q in r)) for r in self._tgt_register]

    @property
    def target_operator(self) -> Operator:
        """
        Get the target unitary operator
        """
        return Operator(self.causal_cone_circuit)

    @property
    def circuit(self) -> QuantumCircuit:
        """
        Get the target circuit (with context)
        """
        return (
            self._unbound_circuit_contexts[self._circuit_choice]
            if not self._bound_circuit_contexts[self._circuit_choice]
            else self._bound_circuit_contexts[self._circuit_choice]
        )

    @circuit.setter
    def circuit(self, circuit: QuantumCircuit):
        """
        Set the target circuit (with context)
        """
        self._unbound_circuit_contexts[self._circuit_choice] = circuit
        if not circuit.parameters:
            self._bound_circuit_contexts[self._circuit_choice] = circuit
        else:
            self._bound_circuit_contexts[self._circuit_choice] = None

    @property
    def unbound_circuit(self) -> QuantumCircuit:
        """
        Get the unbound circuit context for the target gate
        """
        return self._unbound_circuit_contexts[self._circuit_choice]

    @property
    def has_context(self) -> bool:
        """
        Check if the target has a circuit context attached or if only composed of the target gate
        """
        return self._has_context
    
    @property
    def physical_qubits(self) -> Sequence[int]:
        """
        Physical qubits on which the target is defined
        """
        return self._physical_qubits[self._circuit_choice]

    @property
    def virtual_target_qubits(self) -> List[Qubit]:
        """
        Get the virtual target qubits for the context-aware calibration
        """
        return self._virtual_target_qubits[self._circuit_choice]

    @property
    def context_parameters(self) -> Dict[Parameter, float | None]:
        """
        Get the context parameters for the target
        """
        return self._context_parameters[self._circuit_choice]

    @property
    def causal_cone_qubits(self) -> List[Qubit]:
        """
        Get the qubits forming the causal cone of the target gate
        (i.e., the qubits that are logically entangled with the target qubits)
        """
        return self.causal_cone_circuit.qubits

    @property
    def causal_cone_qubits_indices(self) -> List[int]:
        """
        Get the indices of the qubits forming the causal cone of the target gate
        """
        return [self.circuit.find_bit(q).index for q in self.causal_cone_qubits]

    @property
    def causal_cone_circuit(self) -> QuantumCircuit:
        """
        Get the circuit forming the causal cone of the target gate
        """
        if self.has_context:
            circuit, _ = causal_cone_circuit(self.circuit, self.virtual_target_qubits)
            return circuit
        else:
            return self.circuit

    @property
    def causal_cone_size(self) -> int:
        """
        Get the size of the causal cone of the target gate
        """
        return self.causal_cone_circuit.num_qubits

    def __repr__(self):
        return (
            f"GateTarget({self.gate.name}, on qubits {self.physical_qubits}"
            f" with{'' if self.has_context else 'out'} context)"
        )

    @property
    def target_type(self):
        """
        Type of the target (gate)
        """
        return "gate"

    def __str__(self):
        """
        String representation of the GateTarget
        """
        return f"GateTarget({self.gate.name}, on qubits {self.physical_qubits})"

    @property
    def circuit_choice(self):
        """
        Get the current circuit choice for the target
        """
        return self._circuit_choice

    @circuit_choice.setter
    def circuit_choice(self, choice: int):
        """
        Set the current circuit choice for the target
        :param choice: Index of the circuit to be used as the target
        """
        if not (0 <= choice < len(self._circuit_contexts)):
            raise ValueError(
                f"Invalid circuit choice {choice}, should be in range [0, {len(self._circuit_contexts)})"
            )
        self._circuit_choice = choice

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """
        Get the available circuit contexts for the target gate
        """
        return (
            self._unbound_circuit_contexts
            if any(circ is None for circ in self._bound_circuit_contexts)
            else self._bound_circuit_contexts
        )

    @property
    def layout(self) -> Layout:
        """
        Get the layout of the target gate
        """
        return self._layout[self._circuit_choice]

    @layout.setter
    def layout(self, layout: Layout):
        """
        Set the layout of the target gate
        """
        self._layout[self._circuit_choice] = layout

    @property
    def tgt_register(self) -> List[Qubit]:
        """
        Get the target register for the target gate
        """
        return self._tgt_register[self._circuit_choice]

    @circuits.setter
    def circuits(self, circuits: List[QuantumCircuit]):
        """
        Set the available circuit contexts for the target gate
        """
        self._circuit_contexts = circuits

    def bind_parameters(self, params: dict[Parameter, float]):
        """
        Assign parameters to the target circuit context
        """

        self._bound_circuit_contexts[self._circuit_choice] = self._unbound_circuit_contexts[
            self._circuit_choice
        ].assign_parameters(params)
        self._context_parameters[self._circuit_choice] = {
            p: params[p.name] for p in self._context_parameters[self._circuit_choice]
        }

    def clear_parameters(self):
        """
        Clear the parameters of the target circuit contexts
        """
        self._bound_circuit_contexts = [None for _ in self._bound_circuit_contexts]

    @property
    def all_bound_circuits(self) -> Sequence[Optional[QuantumCircuit]]:
        """
        Get all bound circuit contexts regardless of circuit choice
        """
        return self._bound_circuit_contexts

    @property
    def all_unbound_circuits(self) -> Sequence[QuantumCircuit]:
        """
        Get all unbound circuit contexts regardless of circuit choice
        """
        return self._unbound_circuit_contexts

    @property
    def all_virtual_target_qubits(self) -> Sequence[Sequence[Qubit]]:
        """
        Get all virtual target qubits lists regardless of circuit choice
        """
        return self._virtual_target_qubits

    @property
    def all_virtual_target_qubits_indices(self) -> Sequence[Sequence[int]]:
        """
        Get all virtual target qubit indices lists regardless of circuit choice
        """
        return self._virtual_target_qubits_indices

    @property
    def all_layouts(self) -> Sequence[Layout]:
        """
        Get all layouts regardless of circuit choice
        """
        return self._layout

    @property
    def all_target_registers(self) -> Sequence[Sequence[Qubit]]:
        """
        Get all target registers regardless of circuit choice
        """
        return self._tgt_register

    @property
    def all_context_parameters(self) -> Sequence[Dict[Parameter, float | None]]:
        """
        Get all symbolic parameters associated to each circuit context
        """
        return self._context_parameters

    @property
    def causal_cone_circuits(self) -> List[QuantumCircuit]:
        """
        Get all causal cone circuits for each circuit context
        """
        return [
            causal_cone_circuit(circ, vq)[0] if self.has_context else circ
            for circ, vq in zip(self._unbound_circuit_contexts, self._virtual_target_qubits)
        ]

    @property
    def all_causal_cone_qubit_indices(self) -> Sequence[Sequence[int]]:
        """
        Get all causal cone qubit indices for each circuit context
        """
        return [
            (
                [circ.find_bit(q).index for q in causal_cone_circuit(circ, vq)[0].qubits]
                if self.has_context
                else list(range(circ.num_qubits))
            )
            for circ, vq in zip(self._unbound_circuit_contexts, self._virtual_target_qubits)
        ]

    def get(self, item: str, default: Any = None):
        """
        Get method for dictionary-like access to the target
        """
        if item == "gate":
            return self.gate
        elif item == "physical_qubits":
            return self.physical_qubits
        elif item == "layout":
            return self.layout
        else:
            return default

    def __getitem__(self, item: str):
        """
        Get item method for dictionary-like access to the target
        """
        return self.get(item)

    def as_dict(self):
        """
        Convert the target gate to a dictionary representation
        """
        return {
            "gate": self.gate.name,
            "physical_qubits": self.physical_qubits,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a GateTarget from a dictionary representation
        :param data: Dictionary representation of the target gate
        :return: GateTarget object
        :raises ValueError: If the data does not contain a valid gate or physical qubits
        """

        return cls(**data)


Target = Union[StateTarget, GateTarget]

class MultiTarget:
    """
    Class to represent a collection of targets that can be defined simultaneously for a single collection of circuit contexts.
    """

    def __init__(self, target_instructions: List[InstructionReplacement], 
                       circuit_contexts: Union[QuantumCircuit, List[QuantumCircuit]],
                       layout: Optional[Union[Layout, List[int], Sequence[Union[Layout, List[int]]]]] = None,
                       ):
        """
        Initialize the multi-target for the quantum environment
        :param targets: List of instruction replacements to be defined simultaneously. The targets must act on different qubits subsets to be simultaneously considered.
        :param circuit_contexts: List of circuit contexts to be used for the multi-target. All specified circuit contexts
            should contain the same qubit objects as the ones specified in the InstructionReplacement objects. The layout of the circuit contexts
            will be used to map the virtual target qubits to physical qubits. 
        :param layout: Physical Layout for all qubits present in the circuit contexts. If not provided, a trivial layout will be used for each circuit context.
        """
        # Validate the target instructions
        if not all(isinstance(target_instruction, InstructionReplacement) for target_instruction in target_instructions):
            raise ValueError("All target instructions must be InstructionReplacement objects")
        
        # Validate the circuit contexts
        if isinstance(circuit_contexts, QuantumCircuit):
            circuit_contexts = [circuit_contexts]
        if not all(isinstance(circuit_context, QuantumCircuit) for circuit_context in circuit_contexts):
            raise ValueError("All circuit contexts must be QuantumCircuit objects")

        # Validate that the circuit contexts contain all the target qubits for each instruction and that they do not overlap
        for circ in circuit_contexts:
            used_qubits = set()
            for target_instruction in target_instructions:
                # Validate qargs based on their type
                qargs = target_instruction.target_qargs
                
                if isinstance(qargs, QuantumRegister):
                    # Qubits from QuantumRegister should be in the circuit
                    if qargs not in circ.qregs:
                        raise ValueError(
                            f"QuantumRegister {qargs} from target instruction {target_instruction} "
                            f"not found in circuit context {circ}"
                        )
                    if not all(q in circ.qubits for q in qargs):
                        raise ValueError(
                            f"Not all qubits from QuantumRegister {qargs} are in circuit context {circ}"
                        )
                elif isinstance(qargs, Sequence) and len(qargs) > 0:
                    if all(isinstance(q, Qubit) for q in qargs):
                        # Qubit objects should be directly in the circuit
                        if not all(q in circ.qubits for q in qargs):
                            raise ValueError(
                                f"Not all Qubit objects from target instruction {target_instruction} "
                                f"are in circuit context {circ}"
                            )
                    elif all(isinstance(q, int) for q in qargs):
                        # Integer indices - check circuit width
                        max_index = max(qargs)
                        if max_index >= circ.num_qubits:
                            raise ValueError(
                                f"Target instruction {target_instruction} has qubit index {max_index} "
                                f"but circuit context {circ} only has {circ.num_qubits} qubits"
                            )
                    else:
                        raise ValueError(
                            f"Invalid qargs type in target instruction {target_instruction}: "
                            f"expected Qubit objects or integers"
                        )
                
                # Check for overlap with other instructions (disjoint check)
                if not target_instruction.check_qubits(circ):
                    raise ValueError(
                        f"Circuit context {circ} does not contain all target qubits for instruction {target_instruction}"
                    )
                current_qubits = target_instruction.get_qubits(circ)
                if not used_qubits.isdisjoint(current_qubits):
                    raise ValueError(
                        f"Target qubits for instruction {target_instruction} overlap with other instructions qubits: "
                        f"{used_qubits.intersection(current_qubits)}"
                    )
                used_qubits.update(current_qubits)
        
        # Validate and normalize layouts
        num_circuits = len(circuit_contexts)
        
        # Normalize layout to a list
        if layout is None:
            # Generate trivial layouts for each circuit
            layouts = [Layout.generate_trivial_layout(*circ.qregs) for circ in circuit_contexts]
        elif isinstance(layout, Layout):
            # Single layout provided - use for all circuits
            if num_circuits > 1:
                raise ValueError(f"Single Layout provided but {num_circuits} circuit contexts given. Provide a list of layouts.")
            layouts = [layout]
        elif isinstance(layout, list) and len(layout) > 0:
            # List of layouts or list of integers
            if len(layout) != num_circuits:
                raise ValueError(f"Layout list length ({len(layout)}) must match number of circuit contexts ({num_circuits})")
            
            layouts = []
            for i, layout_item in enumerate(layout):
                circ = circuit_contexts[i]
                
                if isinstance(layout_item, Layout):
                    # Already a Layout object
                    # Validate qubit count matches
                    if len(layout_item) != circ.num_qubits:
                        raise ValueError(
                            f"Layout {i} has {len(layout_item)} qubits but circuit {i} has {circ.num_qubits} qubits"
                        )
                    layouts.append(layout_item)
                elif isinstance(layout_item, list) and all(isinstance(x, int) for x in layout_item):
                    # List of integers - convert to Layout
                    if len(layout_item) != circ.num_qubits:
                        raise ValueError(
                            f"Layout {i} (list of integers) has length {len(layout_item)} but circuit {i} has {circ.num_qubits} qubits"
                        )
                    # Convert list of integers to Layout using from_intlist
                    layout_obj = Layout.from_intlist(layout_item, *circ.qregs)
                    layouts.append(layout_obj)
                else:
                    raise ValueError(
                        f"Invalid layout type at index {i}: {type(layout_item)}. "
                        f"Expected Layout or List[int]"
                    )
        elif isinstance(layout, Sequence):
            # Sequence (tuple, etc.) - convert to list and process
            layout_list = list(layout)
            if len(layout_list) != num_circuits:
                raise ValueError(f"Layout sequence length ({len(layout_list)}) must match number of circuit contexts ({num_circuits})")
            
            layouts = []
            for i, layout_item in enumerate(layout_list):
                circ = circuit_contexts[i]
                
                if isinstance(layout_item, Layout):
                    if len(layout_item) != circ.num_qubits:
                        raise ValueError(
                            f"Layout {i} has {len(layout_item)} qubits but circuit {i} has {circ.num_qubits} qubits"
                        )
                    layouts.append(layout_item)
                elif isinstance(layout_item, (list, tuple)) and all(isinstance(x, int) for x in layout_item):
                    layout_item = list(layout_item)
                    if len(layout_item) != circ.num_qubits:
                        raise ValueError(
                            f"Layout {i} (list of integers) has length {len(layout_item)} but circuit {i} has {circ.num_qubits} qubits"
                        )
                    layout_obj = Layout.from_intlist(layout_item, *circ.qregs)
                    layouts.append(layout_obj)
                else:
                    raise ValueError(
                        f"Invalid layout type at index {i}: {type(layout_item)}. "
                        f"Expected Layout or List[int]"
                    )
        else:
            raise ValueError(
                f"Invalid layout type: {type(layout)}. "
                f"Expected Layout, List[int], List[Layout], or Sequence[Union[Layout, List[int]]]"
            )
        
        # Store validated layouts
        self.layouts = layouts
        self.target_instructions = target_instructions
        self.circuit_contexts = circuit_contexts
        
        # Create GateTarget objects for each target instruction
        self.gate_targets: List[GateTarget] = []
        
        for target_instruction in target_instructions:
            # Extract virtual qubits for each circuit context
            virtual_qubits_per_circuit = []
            physical_qubits_per_circuit = []
            
            for circ, layout_obj in zip(circuit_contexts, layouts):
                # Get virtual qubits from the circuit
                virtual_qubits = target_instruction.get_qubits(circ)
                virtual_qubits_per_circuit.append(virtual_qubits)
                
                # Map virtual qubits to physical qubits using the layout
                physical_qubits = []
                for vq in virtual_qubits:
                    # Find the physical qubit index for this virtual qubit
                    if vq in layout_obj:
                        physical_qubit = layout_obj[vq]
                        physical_qubits.append(physical_qubit)
                    else:
                        raise ValueError(
                            f"Virtual qubit {vq} not found in layout for circuit context. "
                            f"Ensure all target instruction qubits are mapped in the layout."
                        )
                physical_qubits_per_circuit.append(physical_qubits)
            
            # Check if physical qubits are consistent across circuits
            # If consistent, use single sequence; otherwise use sequence of sequences
            reference_physical_qubits = physical_qubits_per_circuit[0]
            physical_qubits_consistent = all(
                phys_q == reference_physical_qubits for phys_q in physical_qubits_per_circuit[1:]
            )
            
            if physical_qubits_consistent:
                # All circuits use the same physical qubits - use single sequence
                final_physical_qubits: Sequence[int] | Sequence[Sequence[int]] = reference_physical_qubits
            else:
                # Different physical qubits per circuit - use sequence of sequences
                final_physical_qubits = physical_qubits_per_circuit
            
            # Check if virtual qubits are consistent across circuits
            reference_virtual_qubits = virtual_qubits_per_circuit[0]
            virtual_qubits_consistent = all(
                vq == reference_virtual_qubits for vq in virtual_qubits_per_circuit[1:]
            )
            
            if virtual_qubits_consistent:
                # All circuits use the same virtual qubits - use single sequence
                final_virtual_qubits: Sequence[int | Qubit] | Sequence[Sequence[int | Qubit]] = reference_virtual_qubits
            else:
                # Different virtual qubits per circuit - use sequence of sequences
                final_virtual_qubits = virtual_qubits_per_circuit
            
            # Create GateTarget with all circuit contexts
            gate_target = GateTarget(
                gate=target_instruction,
                physical_qubits=final_physical_qubits,
                circuit_context=circuit_contexts,
                virtual_target_qubits=final_virtual_qubits,
                layout=layouts,
            )
            self.gate_targets.append(gate_target)
            
