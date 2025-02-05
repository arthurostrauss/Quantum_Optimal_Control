"""
Target classes for the quantum environment

This module contains the classes to represent the target states and gates for the quantum environment

Author: Arthur Strauss
Created: 08/11/2024
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Statevector,
    pauli_basis,
    state_fidelity,
    average_gate_fidelity,
    random_unitary,
)
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.transpiler import Layout
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate, CircuitInstruction
from itertools import product
from typing import List, Optional, Literal
from qiskit_experiments.library.tomography.basis import (
    PauliPreparationBasis,
    Pauli6PreparationBasis,
)
from ..helpers.circuit_utils import (
    density_matrix_to_statevector,
    get_gate,
    causal_cone_circuit,
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
        chi = np.real(
            [target.expectation_value(basis[k]) for k in range(d**2)]
        ) / np.sqrt(d)
    else:
        dms = [DensityMatrix(pauli).evolve(target) for pauli in basis]
        chi = (
            np.real(
                [
                    dms[k_].expectation_value(basis[k])
                    for k, k_ in product(range(d**2), repeat=2)
                ]
            )
            / d
        )
    # Real part is taken to convert it in good format,
    # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
    return chi


def get_2design_input_states(d: int = 4) -> List[Statevector]:
    """
    Function that return the 2-design input states (used for CAFE reward scheme)
    Follows this Reference: https://arxiv.org/pdf/1008.1138 (see equations 2 and 13)
    """
    # Define constants
    golden_ratio = (np.sqrt(5) - 1) / 2
    omega = np.exp(2 * np.pi * 1j / d)

    # Define computational basis states
    e0 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    e1 = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
    e2 = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
    e3 = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩

    # Create the Z matrix (diagonal matrix with powers of omega)
    Z = np.diag([omega**r for r in range(d)])

    # Create the X matrix (shift matrix)
    X = np.zeros((d, d), dtype=complex)
    for r in range(d - 1):
        X[r, r + 1] = 1
    X[d - 1, 0] = 1  # Wrap-around to satisfy the condition for |e_0>

    # Define the fiducial state from Eq. (13)
    coefficients = (
        1
        / (2 * np.sqrt(3 + golden_ratio))
        * np.array(
            [
                1 + np.exp(-1j * np.pi / 4),
                np.exp(1j * np.pi / 4) + 1j * golden_ratio ** (-3 / 2),
                1 - np.exp(-1j * np.pi / 4),
                np.exp(1j * np.pi / 4) - 1j * golden_ratio ** (-3 / 2),
            ]
        )
    )

    fiducial_state = (
        coefficients[0] * e0
        + coefficients[1] * e1
        + coefficients[2] * e2
        + coefficients[3] * e3
    ).reshape(d, 1)
    # Prepare all 16 states
    states = []
    for k in range(0, d):
        for l in range(0, d):
            # state = apply_hw_group(p1, p2, coefficients)
            state = (
                np.linalg.matrix_power(X, k)
                @ np.linalg.matrix_power(Z, l)
                @ fiducial_state
            )
            states.append(Statevector(state))

    return states


@dataclass
class BaseTarget(ABC):
    """
    Base class for the target of the quantum environment
    """

    def __init__(
        self,
        physical_qubits: int | List[int],
        target_type: str,
        tgt_register: Optional[QuantumRegister] = None,
        layout: Optional[Layout] = None,
    ):
        """
        Initialize the base target for the quantum environment
        :param physical_qubits: Physical qubits on which the target is defined
        :param target_type: Type of the target (state / gate)
        :param tgt_register: Optional existing QuantumRegister for the target
        :param layout: Optional existing layout for the target (used when target touches a subset of
        all physical qubits present in a circuit)

        """
        self.physical_qubits = (
            list(range(physical_qubits))
            if isinstance(physical_qubits, int)
            else physical_qubits
        )
        self.target_type = target_type
        self._tgt_register = (
            QuantumRegister(len(self.physical_qubits), "tgt")
            if tgt_register is None
            else tgt_register
        )
        self._layout: Layout = (
            Layout(
                {
                    self._tgt_register[i]: self.physical_qubits[i]
                    for i in range(len(self.physical_qubits))
                }
            )
            if layout is None
            else layout
        )
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

    @n_qubits.setter
    def n_qubits(self, n_qubits: int):
        self._n_qubits = n_qubits


class StateTarget(BaseTarget):
    """
    Class to represent the state target for the quantum environment
    """

    def __init__(
        self,
        state: Optional[DensityMatrix | Statevector] = None,
        circuit: Optional[QuantumCircuit] = None,
        physical_qubits: Optional[List[int]] = None,
    ):
        """
        Initialize the state target for the quantum environment
        :param state: State to be calibrated (either DensityMatrix or Statevector)
        :param circuit: Circuit generating the target state (optional)
        :param physical_qubits: Physical qubits forming the target state
        """

        if isinstance(state, DensityMatrix):
            if state.purity() != 1:
                raise ValueError("Density matrix should be pure")
            self.dm = state
        elif isinstance(state, Statevector):
            self.dm = DensityMatrix(state)

        elif state is None:
            if circuit is None:
                raise ValueError("Either state or circuit should be provided")
            self.dm = DensityMatrix(circuit)
            self.circuit = circuit if isinstance(circuit, QuantumCircuit) else None
        if circuit is None:
            qc = QuantumCircuit(self.dm.num_qubits)
            if not isinstance(state, Statevector):
                state = density_matrix_to_statevector(state)
            qc.prepare_state(state)
            self.circuit = qc

        self.Chi = _calculate_chi_target(self.dm)
        super().__init__(
            self.dm.num_qubits if physical_qubits is None else physical_qubits, "state"
        )

    def fidelity(
        self, state: QuantumState | QuantumCircuit, n_reps: int = 1, validate=True
    ):
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
        tgt_register: QuantumRegister,
    ):
        """
        Initialize the input state for the quantum environment
        :param input_circuit: Quantum circuit representing the input state
        :param target_op: Gate to be calibrated (or circuit context)
        :param tgt_register: Quantum register for the target gate
        """
        super().__init__(circuit=input_circuit)
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
        circ.repeat(n_reps).compose(self.circuit, front=True, inplace=True)
        return StateTarget(circuit=circ)

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
        circuit_context: Optional[QuantumCircuit] = None,
        tgt_register: Optional[QuantumRegister] = None,
        layout: Optional[Layout] = None,
        input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4",
    ):
        """
        Initialize the gate target for the quantum environment
        :param gate: Gate to be calibrated
        :param physical_qubits: Physical qubits forming the target gate
        :param circuit_context: circuit to be used for context-aware calibration (default is the gate to be calibrated)
        :param tgt_register: Specify target QuantumRegister if already declared
        :param layout: Specify layout if already declared
        :param input_states_choice: Type of input states to be used for
            the calibration (relevant only for state and CAFE rewards)
        """
        gate = get_gate(gate)
        self.gate = gate
        super().__init__(
            gate.num_qubits if physical_qubits is None else physical_qubits,
            "gate",
            tgt_register,
            layout,
        )

        if circuit_context is None:  # If no context is provided, use the gate itself
            circuit_context = QuantumCircuit(self.tgt_register)
            circuit_context.append(gate, self.tgt_register)
        elif not isinstance(circuit_context, QuantumCircuit):
            raise ValueError("circuit_context should be a QuantumCircuit")

        self._circuit_context: QuantumCircuit = circuit_context

        if self.has_context:
            # Filter context to get causal cone of the target gate
            target_qubits = [
                self._circuit_context.qubits[i] for i in self.physical_qubits
            ]
            filtered_context, filtered_qubits = causal_cone_circuit(
                self._circuit_context,
                target_qubits,
            )

            self._causal_cone_qubits = filtered_qubits
            self._causal_cone_size = len(filtered_qubits)
            self._causal_cone_circuit = filtered_context

        else:  # If no context is provided, the causal cone is the target qubits
            self._causal_cone_qubits = self._circuit_context.qubits
            self._causal_cone_circuit = self._circuit_context
            self._causal_cone_size = self.n_qubits

        n_qubits = self.causal_cone_size

        if input_states_choice == "pauli4":
            input_circuits = [
                PauliPreparationBasis().circuit(s)
                for s in product(range(4), repeat=n_qubits)
            ]
        elif input_states_choice == "pauli6":
            input_circuits = [
                Pauli6PreparationBasis().circuit(s)
                for s in product(range(6), repeat=n_qubits)
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
        self._input_states_choice = input_states_choice

        self.input_states = [
            InputState(
                input_circuit=circ,
                target_op=self.causal_cone_circuit,
                tgt_register=self.tgt_register,
            )
            for circ in input_circuits
        ]

    def Chi(self, n_reps: int = 1):
        """
        Compute the characteristic function of the target gate
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if self.causal_cone_size <= 3:
            return _calculate_chi_target(
                self.target_operator.power(n_reps, assume_unitary=True)
            )
        else:
            warnings.warn("Chi is not computed for more than 3 qubits")
            return None

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
        return average_gate_fidelity(
            channel, Operator(self._circuit_context).power(n_reps)
        )

    def state_fidelity(
        self, state: QuantumState, n_reps: int = 1, validate: bool = True
    ):
        """
        Compute the fidelity of the state with the target state derived from the application of the target gate/circuit
        context to |0...0>
        :param state: State to compare with target state
        :param n_reps: Number of repetitions of the target gate (default is 1)
        """
        if not isinstance(state, (Statevector, DensityMatrix)):
            raise ValueError("Input should be a Statevector or DensityMatrix object")
        if np.linalg.norm(state) != 1 and not validate:
            warnings.warn(
                f"Input state is not normalized (norm = {np.linalg.norm(state)})"
            )
        return state_fidelity(
            state,
            Statevector(self._circuit_context.power(n_reps, True, True)),
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
        return self._circuit_context

    @target_circuit.setter
    def target_circuit(self, target_op: QuantumCircuit):
        """
        Set the target circuit
        """
        self._circuit_context = target_op
        self.input_states = [
            InputState(input_state.circuit, target_op, self.tgt_register)
            for input_state in self.input_states
        ]

    @property
    def has_context(self):
        """
        Check if the target has a circuit context attached or if only composed of the target gate
        """
        gate_qc = QuantumCircuit(self.gate.num_qubits)
        gate_qc.append(self.gate, list(range(self.gate.num_qubits)))

        return Operator(self._circuit_context) != Operator(gate_qc)

    @property
    def causal_cone_qubits(self):
        """
        Get the qubits forming the causal cone of the target gate
        (i.e. the qubits that are logically entangled with the target qubits)
        """
        return self._causal_cone_qubits

    @property
    def causal_cone_qubits_indices(self):
        """
        Get the indices of the qubits forming the causal cone of the target gate
        """
        return [self.target_circuit.find_bit(q).index for q in self.causal_cone_qubits]

    @property
    def causal_cone_circuit(self):
        """
        Get the circuit forming the causal cone of the target gate
        """
        return self._causal_cone_circuit

    @property
    def causal_cone_size(self):
        """
        Get the size of the causal cone of the target gate
        """
        return self._causal_cone_size

    @property
    def input_states_choice(self):
        """
        Get the choice of input states for the calibration
        """
        return self._input_states_choice

    def __repr__(self):
        return (
            f"GateTarget({self.gate.name}, on qubits {self.physical_qubits}"
            f" with{'' if self.has_context else 'out'} context)"
        )
