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
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate, CircuitInstruction, Qubit
from itertools import product
from typing import List, Optional, Literal, Sequence
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
        physical_qubits: Sequence[int] | int,
        tgt_register: QuantumRegister | Sequence[Qubit],
        layout: Layout | List[Layout],
    ):
        """
        Initialize the base target for the quantum environment.
        :param physical_qubits: Physical qubits on which the target is defined.
        :param tgt_register: Optional existing QuantumRegister for the target.
        :param layout: Optional existing layout for the target (used when target touches a subset of
        all physical qubits present in a circuit)

        """
        self.physical_qubits = (
            list(range(physical_qubits)) if isinstance(physical_qubits, int) else physical_qubits
        )
        self._tgt_register = tgt_register
        self._layout: Layout = layout
        self._n_qubits = len(self.physical_qubits)

    @property
    def tgt_register(self):
        return self._tgt_register

    @property
    def layout(self) -> Layout:
        """
        Layout for the target
        """
        return self._layout

    @layout.setter
    def layout(self, layout: Layout):
        if not isinstance(layout, Layout):
            raise ValueError("Layout should be a Layout object")
        self._layout = layout

    @property
    def n_qubits(self):
        return self._n_qubits

    @abstractmethod
    @property
    def target_type(self):
        """
        Type of the target (state / gate)
        """
        pass


class StateTarget(BaseTarget):
    """
    Class to represent the state target for the quantum environment
    """

    def __init__(
        self,
        state: DensityMatrix | Statevector | QuantumCircuit | str,
        physical_qubits: Optional[List[int]] = None,
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
        else:
            if not isinstance(state, DensityMatrix):
                raise ValueError(
                    "State should be a DensityMatrix, Statevector or QuantumCircuit object"
                )
            if state.num_qubits != len(physical_qubits):
                raise ValueError(
                    "Number of qubits in the state should match the number of physical qubits"
                )
            if state.purity() - 1 > 1e-6:
                raise ValueError("Density matrix should be pure")
            self.dm = state
            tgt_register = QuantumRegister(self.dm.num_qubits, "tgt")
            self.circuit = QuantumCircuit(tgt_register)
            self.circuit.prepare_state(density_matrix_to_statevector(state))
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
        """
        Get the circuits for the target state
        """
        return [self.circuit]

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


class GateTarget(BaseTarget):
    """
    Class to represent the gate target for the quantum environment
    """

    def __init__(
        self,
        gate: Gate | str,
        physical_qubits: Optional[List[int]] = None,
        circuit_context: Optional[QuantumCircuit | List[QuantumCircuit]] = None,
        virtual_target_qubits: Optional[Sequence[int]] = None,
        layout: Optional[Layout | List[Layout]] = None,
        input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4",
    ):
        """
        Initialize the gate target for the quantum environment.
        :param gate: Gate to be calibrated. It can be a Gate object or a string representing the gate name.
        :param physical_qubits: Physical qubits forming the target gate.
        :param circuit_context: Circuit to be used for context-aware calibration (default is the gate to be calibrated).
        :param virtual_target_qubits: Virtual target qubits to be used for the context-aware calibration.
        :param layout: Specify layout if already declared
        :param input_states_choice: Type of input states to be used for
            the calibration (relevant only for state and CAFE rewards)
        """
        gate = get_gate(gate)
        if physical_qubits is None:
            physical_qubits = list(range(gate.num_qubits))
        self.gate = gate
        # super().__init__(
        #     gate.num_qubits if physical_qubits is None else physical_qubits,
        #     "gate",
        #     tgt_register,
        #     layout,
        # )
        self._circuit_choice = 0
        if circuit_context is None:  # If no context is provided, use the gate itself
            self._has_context = False
            tgt_register = QuantumRegister(gate.num_qubits, "tgt")
            circuit_context = QuantumCircuit(tgt_register)
            circuit_context.append(gate, tgt_register)
            self._circuit_context = [circuit_context]
        else:
            self._has_context = True
            if isinstance(circuit_context, QuantumCircuit):
                circuit_context = [circuit_context]
            if isinstance(circuit_context, list):
                if any(circ.num_qubits < gate.num_qubits for circ in circuit_context):
                    raise ValueError(
                        "Circuit context must have at least as many qubits as the target gate"
                    )
                if virtual_target_qubits is None:
                    if any(circ.num_qubits > gate.num_qubits for circ in circuit_context):
                        raise ValueError(
                            "If circuit context is larger than target gate, virtual_target_qubits must be provided"
                        )
                    self._virtual_target_qubits = list(range(gate.num_qubits))
            else:
                self._virtual_target_qubits = virtual_target_qubits
            tgt_register = [circuit_context[0].qubits[q] for q in self._virtual_target_qubits]
            if layout is None:
                if any(circ.num_qubits > gate.num_qubits for circ in circuit_context):
                    raise ValueError(
                        "If circuit context is larger than target gate, layout must be provided"
                    )
                layout = [
                    Layout(
                        {tgt_register[i]: physical_qubits[i] for i in range(len(physical_qubits))}
                    )
                ]
            else:
                if isinstance(layout, Layout):
                    layout = [layout]
                if len(layout) != len(circuit_context):
                    raise ValueError("Layout should be provided for each circuit in the context")
        super().__init__(physical_qubits=physical_qubits, tgt_register=tgt_register, layout=layout)
        self._circuit_contexts = circuit_context

        if self.has_context:
            # Filter context to get causal cone of the target gate
            target_qubits = [
                [circ.qubits[i] for i in self.virtual_target_qubits]
                for circ in self._circuit_contexts
            ]
            filtered_contexts = []
            filtered_qubits = []
            for circ in self._circuit_contexts:
                filtered_context, filtered_qbs = causal_cone_circuit(
                    circ,
                    target_qubits,
                )
                filtered_contexts.append(filtered_context)
                filtered_qubits.append(filtered_qbs)

            self._causal_cone_qubits = filtered_qubits
            self._causal_cone_circuit = filtered_contexts

        else:  # If no context is provided, the causal cone is the target qubits
            self._causal_cone_qubits = [self.target_circuit.qubits]
            self._causal_cone_circuit = [self.target_circuit]

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
    ):
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
            circuit = self.target_circuit

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
            circuit = self.target_circuit
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
    ):
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
    def target_instruction(self):
        """
        Get the target instruction
        """
        return CircuitInstruction(self.gate, (q for q in self.tgt_register))

    @property
    def target_operator(self):
        """
        Get the target unitary operator
        """
        return Operator(self.causal_cone_circuit)

    @property
    def target_circuit(self):
        """
        Get the target circuit (with context)
        """
        return self._circuit_contexts[self._circuit_choice]

    @property
    def has_context(self):
        """
        Check if the target has a circuit context attached or if only composed of the target gate
        """
        return self._has_context

    @property
    def virtual_target_qubits(self):
        """
        Get the virtual target qubits for the context-aware calibration
        """
        return self._virtual_target_qubits

    @property
    def causal_cone_qubits(self) -> List[Qubit]:
        """
        Get the qubits forming the causal cone of the target gate
        (i.e., the qubits that are logically entangled with the target qubits)
        """
        return self._causal_cone_qubits[self._circuit_choice]

    @property
    def causal_cone_qubits_indices(self) -> List[int]:
        """
        Get the indices of the qubits forming the causal cone of the target gate
        """
        return [self.target_circuit.find_bit(q).index for q in self.causal_cone_qubits]

    @property
    def causal_cone_circuit(self) -> QuantumCircuit:
        """
        Get the circuit forming the causal cone of the target gate
        """
        return self._causal_cone_circuit[self._circuit_choice]

    @property
    def causal_cone_size(self) -> int:
        """
        Get the size of the causal cone of the target gate
        """
        return self._causal_cone_circuit[self._circuit_choice].num_qubits

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
        return self._circuit_contexts
