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
import warnings


def _calculate_chi_target(target: DensityMatrix | Operator | QuantumCircuit | Gate):
    """
    Calculates the characteristic function for the given target.

    Args:
        target: The target state or gate.

    Returns:
        The characteristic function of the target.
    """

    if not isinstance(target, (DensityMatrix, Operator)):
        try:
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

    return chi


class BaseTarget(ABC):
    """
    An abstract base class for targets in a quantum environment.
    """

    def __init__(
        self,
        physical_qubits: Sequence[int] | int,
        tgt_register: QuantumRegister | Sequence[Qubit] | Sequence[Sequence[Qubit]],
        layout: Layout | List[Layout],
    ):
        """
        Initializes the BaseTarget.

        Args:
            physical_qubits: The physical qubits on which the target is defined.
            tgt_register: The quantum register for the target.
            layout: The layout for the target.
        """
        self.physical_qubits = (
            list(range(physical_qubits)) if isinstance(physical_qubits, int) else physical_qubits
        )
        self._tgt_register = tgt_register
        self._layout: Layout = layout
        self._n_qubits = len(self.physical_qubits)

    @property
    def tgt_register(self):
        """The quantum register for the target."""
        return self._tgt_register

    @property
    def layout(self) -> Layout:
        """The layout for the target."""
        return self._layout

    @layout.setter
    def layout(self, layout: Layout):
        if not isinstance(layout, Layout):
            raise ValueError("Layout should be a Layout object")
        self._layout = layout

    @property
    def n_qubits(self):
        """The number of qubits in the target."""
        return self._n_qubits

    @property
    @abstractmethod
    def target_type(self):
        """The type of the target."""
        pass


class StateTarget(BaseTarget):
    """
    A class to represent a state target in a quantum environment.
    """

    def __init__(
        self,
        state: DensityMatrix | Statevector | QuantumCircuit | str | np.ndarray,
        physical_qubits: Optional[Sequence[int]] = None,
    ):
        """
        Initializes the StateTarget.

        Args:
            state: The target state.
            physical_qubits: The physical qubits on which the target is defined.
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
            try:
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
    def circuits(self) -> List[QuantumCircuit]:
        """The circuits for the target state."""
        return [self.circuit]

    @property
    def target_instruction(self) -> CircuitInstruction:
        """The target instruction."""
        return self.circuit.data[0]

    @property
    def target_instruction_counts(self) -> int:
        """The number of target instructions."""
        return self.circuit.data.count(self.target_instruction)

    @property
    def target_instructions(self) -> List[CircuitInstruction]:
        """The target instructions."""
        return [self.target_instruction]

    @property
    def target_type(self):
        """The type of the target."""
        return "state"

    def fidelity(self, state: QuantumState | QuantumCircuit, n_reps: int = 1, validate=True):
        """
        Calculates the fidelity between the given state and the target state.

        Args:
            state: The state to compare with the target.
            n_reps: The number of repetitions of the target gate.
            validate: Whether to validate the input state.

        Returns:
            The fidelity between the two states.
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
        Returns a dictionary representation of the StateTarget.

        Returns:
            A dictionary representation of the StateTarget.
        """
        return {
            "state": np.array2string(self.dm.data, separator=","),
            "physical_qubits": self.physical_qubits,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a StateTarget from a dictionary.

        Args:
            data: The dictionary to create the StateTarget from.

        Returns:
            A StateTarget object.
        """
        dm = DensityMatrix(np.array(data["dm"]))
        if dm.num_qubits is None:
            raise ValueError("DensityMatrix num_qubits is None")
        physical_qubits = data.get("physical_qubits", list(range(dm.num_qubits)))
        return cls(state=dm, physical_qubits=physical_qubits)


class InputState(StateTarget):
    """
    A class to represent an input state in a quantum environment.
    """

    def __init__(
        self,
        input_circuit: QuantumCircuit,
        target_op: Gate | QuantumCircuit,
    ):
        """
        Initializes the InputState.

        Args:
            input_circuit: The circuit that prepares the input state.
            target_op: The target gate or circuit.
        """
        super().__init__(input_circuit)
        self._target_op = target_op

    def target_state(self, n_reps: int = 1):
        """
        Returns the target state after applying the target operation to the input state.

        Args:
            n_reps: The number of repetitions of the target operation.

        Returns:
            A StateTarget object representing the target state.
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
        """The circuit that prepares the input state."""
        return self.circuit

    @property
    def target_circuit(self, n_reps: int = 1) -> QuantumCircuit:
        """The circuit that prepares the target state."""
        return self.target_state(n_reps).circuit

    @property
    def target_dm(self, n_reps: int = 1) -> DensityMatrix:
        """The density matrix of the target state."""
        return self.target_state(n_reps).dm


class GateTarget(BaseTarget):
    """
    A class to represent a gate target in a quantum environment.
    """

    def __init__(
        self,
        gate: Gate | str,
        physical_qubits: Optional[Sequence[int]] = None,
        circuit_context: Optional[QuantumCircuit | List[QuantumCircuit]] = None,
        virtual_target_qubits: Optional[Sequence[int | Qubit]] = None,
        layout: Optional[Layout | List[Layout]] = None,
    ):
        """
        Initializes the GateTarget.

        Args:
            gate: The target gate.
            physical_qubits: The physical qubits on which the target is defined.
            circuit_context: The circuit context for the target gate.
            virtual_target_qubits: The virtual target qubits for the target gate.
            layout: The layout for the target gate.
        """
        gate = get_gate(gate)
        if physical_qubits is None:
            physical_qubits = list(range(gate.num_qubits))
        self.gate = gate
        self._circuit_choice = 0
        if circuit_context is None:
            self._has_context = False
            tgt_register = QuantumRegister(gate.num_qubits, "tgt")
            circuit_context = QuantumCircuit(tgt_register)
            circuit_context.append(gate, tuple(q for q in tgt_register))
            circuit_context = [circuit_context]
            self._virtual_target_qubits = [tgt_register]
            self._virtual_target_qubits_indices = list(range(gate.num_qubits))
        else:
            self._has_context = True
            if isinstance(circuit_context, QuantumCircuit):
                circuit_context = [circuit_context]
            if any(circ.num_qubits < gate.num_qubits for circ in circuit_context):
                raise ValueError(
                    "Circuit context must have at least as many qubits as the target gate"
                )
            if virtual_target_qubits is None:
                if any(circ.num_qubits > gate.num_qubits for circ in circuit_context):
                    raise ValueError(
                        "If circuit context is larger than target gate, virtual_target_qubits must be provided"
                    )
                self._virtual_target_qubits = [[q for q in circ.qubits] for circ in circuit_context]
                self._virtual_target_qubits_indices = [
                    [circ.find_bit(q).index for q in circ.qubits] for circ in circuit_context
                ]
            else:
                if all(isinstance(q, Qubit) for q in virtual_target_qubits):
                    if not all(
                        q in circ.qubits for circ in circuit_context for q in virtual_target_qubits
                    ):
                        raise ValueError("Virtual target qubits must be in the circuit context")
                    self._virtual_target_qubits = [virtual_target_qubits for _ in circuit_context]

                else:
                    if not all(isinstance(q, int) for q in virtual_target_qubits):
                        raise ValueError(
                            "Virtual target qubits must be a list of Qubit objects or a list of integers"
                        )
                    self._virtual_target_qubits = [
                        [circ.qubits[q] for q in virtual_target_qubits] for circ in circuit_context
                    ]

        self._virtual_target_qubits_indices = [
            [circ.find_bit(q).index for q in vq]
            for circ, vq in zip(circuit_context, self._virtual_target_qubits)
        ]
        if layout is not None:
            if isinstance(layout, Layout):
                layout = [layout]
            if len(layout) != len(circuit_context):
                raise ValueError("Layout should be provided for each circuit in the context")
        else:
            if any(circ.num_qubits > gate.num_qubits for circ in circuit_context):
                raise ValueError(
                    "If circuit context is larger than target gate, layout must be provided"
                )
            layout = [
                Layout({tgt_reg[i]: physical_qubits[i] for i in range(len(physical_qubits))})
                for tgt_reg in self._virtual_target_qubits
            ]
        super().__init__(
            physical_qubits=physical_qubits, tgt_register=self._virtual_target_qubits, layout=layout
        )
        self._unbound_circuit_contexts = circuit_context
        self._bound_circuit_contexts = [
            circ if not circ.parameters else None for circ in circuit_context
        ]
        self._context_parameters: List[Dict[Parameter, float | None]] = [
            {p: None for p in circ.parameters} for circ in circuit_context
        ]

    def Chi(self, n_reps: int = 1):
        """
        Calculates the characteristic function for the target gate.

        Args:
            n_reps: The number of repetitions of the target gate.

        Returns:
            The characteristic function of the target gate.
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
        Returns the input states for the target gate.

        Args:
            input_states_choice: The type of input states to use.

        Returns:
            A list of InputState objects.
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
        elif input_states_choice == "2-design":
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
        Calculates the average gate fidelity between the given channel and the target gate.

        Args:
            channel: The channel to compare with the target.
            n_reps: The number of repetitions of the target gate.

        Returns:
            The average gate fidelity.
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
        Calculates the state fidelity between the given state and the target state.

        Args:
            state: The state to compare with the target.
            n_reps: The number of repetitions of the target gate.
            validate: Whether to validate the input state.

        Returns:
            The state fidelity.
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
        Calculates the fidelity between the given operator and the target.

        Args:
            op: The operator to compare with the target.
            n_reps: The number of repetitions of the target gate.
            validate: Whether to validate the input state.

        Returns:
            The fidelity.
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
        """The target instruction."""
        return CircuitInstruction(self.gate, (q for q in self._tgt_register[self._circuit_choice]))

    @property
    def target_instructions(self) -> List[CircuitInstruction]:
        """The target instructions."""
        return [CircuitInstruction(self.gate, (q for q in r)) for r in self._tgt_register]

    @property
    def target_operator(self) -> Operator:
        """The target operator."""
        return Operator(self.causal_cone_circuit)

    @property
    def circuit(self) -> QuantumCircuit:
        """The target circuit."""
        return (
            self._unbound_circuit_contexts[self._circuit_choice]
            if not self._bound_circuit_contexts[self._circuit_choice]
            else self._bound_circuit_contexts[self._circuit_choice]
        )

    @circuit.setter
    def circuit(self, circuit: QuantumCircuit):
        self._unbound_circuit_contexts[self._circuit_choice] = circuit
        if not circuit.parameters:
            self._bound_circuit_contexts[self._circuit_choice] = circuit
        else:
            self._bound_circuit_contexts[self._circuit_choice] = None

    @property
    def unbound_circuit(self) -> QuantumCircuit:
        """The unbound circuit."""
        return self._unbound_circuit_contexts[self._circuit_choice]

    @property
    def has_context(self) -> bool:
        """Whether the target has a circuit context."""
        return self._has_context

    @property
    def virtual_target_qubits(self) -> List[Qubit]:
        """The virtual target qubits."""
        return self._virtual_target_qubits[self._circuit_choice]

    @property
    def context_parameters(self) -> Dict[Parameter, float | None]:
        """The context parameters."""
        return self._context_parameters[self._circuit_choice]

    @property
    def causal_cone_qubits(self) -> List[Qubit]:
        """The qubits in the causal cone of the target."""
        return self.causal_cone_circuit.qubits

    @property
    def causal_cone_qubits_indices(self) -> List[int]:
        """The indices of the qubits in the causal cone of the target."""
        return [self.circuit.find_bit(q).index for q in self.causal_cone_qubits]

    @property
    def causal_cone_circuit(self) -> QuantumCircuit:
        """The causal cone circuit."""
        if self.has_context:
            circuit, _ = causal_cone_circuit(self.circuit, self.virtual_target_qubits)
            return circuit
        else:
            return self.circuit

    @property
    def causal_cone_size(self) -> int:
        """The size of the causal cone."""
        return self.causal_cone_circuit.num_qubits

    def __repr__(self):
        return (
            f"GateTarget({self.gate.name}, on qubits {self.physical_qubits}"
            f" with{'' if self.has_context else 'out'} context)"
        )

    @property
    def target_type(self):
        """The type of the target."""
        return "gate"

    def __str__(self):
        return f"GateTarget({self.gate.name}, on qubits {self.physical_qubits})"

    @property
    def circuit_choice(self):
        """The circuit choice."""
        return self._circuit_choice

    @circuit_choice.setter
    def circuit_choice(self, choice: int):
        if not (0 <= choice < len(self._circuit_contexts)):
            raise ValueError(
                f"Invalid circuit choice {choice}, should be in range [0, {len(self._circuit_contexts)})"
            )
        self._circuit_choice = choice

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """The circuits."""
        return (
            self._unbound_circuit_contexts
            if any(circ is None for circ in self._bound_circuit_contexts)
            else self._bound_circuit_contexts
        )

    @property
    def layout(self) -> Layout:
        """The layout."""
        return self._layout[self._circuit_choice]

    @property
    def tgt_register(self) -> List[Qubit]:
        """The target register."""
        return self._tgt_register[self._circuit_choice]

    @circuits.setter
    def circuits(self, circuits: List[QuantumCircuit]):
        self._circuit_contexts = circuits

    def bind_parameters(self, params: dict[Parameter, float]):
        """
        Binds the parameters to the circuit.

        Args:
            params: The parameters to bind.
        """

        self._bound_circuit_contexts[self._circuit_choice] = self._unbound_circuit_contexts[
            self._circuit_choice
        ].assign_parameters(params)
        self._context_parameters[self._circuit_choice] = {
            p: params[p.name] for p in self._context_parameters[self._circuit_choice]
        }

    def clear_parameters(self):
        """Clears the parameters of the circuit."""
        self._bound_circuit_contexts = [None for _ in self._bound_circuit_contexts]

    @property
    def all_bound_circuits(self) -> Sequence[Optional[QuantumCircuit]]:
        """All bound circuits."""
        return self._bound_circuit_contexts

    @property
    def all_unbound_circuits(self) -> Sequence[QuantumCircuit]:
        """All unbound circuits."""
        return self._unbound_circuit_contexts

    @property
    def all_virtual_target_qubits(self) -> Sequence[Sequence[Qubit]]:
        """All virtual target qubits."""
        return self._virtual_target_qubits

    @property
    def all_virtual_target_qubits_indices(self) -> Sequence[Sequence[int]]:
        """All virtual target qubit indices."""
        return self._virtual_target_qubits_indices

    @property
    def all_layouts(self) -> Sequence[Layout]:
        """All layouts."""
        return self._layout

    @property
    def all_target_registers(self) -> Sequence[Sequence[Qubit]]:
        """All target registers."""
        return self._tgt_register

    @property
    def all_context_parameters(self) -> Sequence[Dict[Parameter, float | None]]:
        """All context parameters."""
        return self._context_parameters

    @property
    def causal_cone_circuits(self) -> List[QuantumCircuit]:
        """All causal cone circuits."""
        return [
            causal_cone_circuit(circ, vq)[0] if self.has_context else circ
            for circ, vq in zip(self._unbound_circuit_contexts, self._virtual_target_qubits)
        ]

    @property
    def all_causal_cone_qubit_indices(self) -> Sequence[Sequence[int]]:
        """All causal cone qubit indices."""
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
        Gets an item from the target.

        Args:
            item: The item to get.
            default: The default value to return if the item is not found.

        Returns:
            The value of the item.
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
        return self.get(item)

    def as_dict(self):
        """
        Returns a dictionary representation of the GateTarget.

        Returns:
            A dictionary representation of the GateTarget.
        """
        return {
            "gate": self.gate.name,
            "physical_qubits": self.physical_qubits,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a GateTarget from a dictionary.

        Args:
            data: The dictionary to create the GateTarget from.

        Returns:
            A GateTarget object.
        """

        return cls(**data)


Target = Union[StateTarget, GateTarget]
