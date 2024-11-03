from __future__ import annotations

from typing import Optional, List
import numpy as np

from rl_qoc.helper_functions import (
    generate_default_instruction_durations_dict,
)
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.transpiler import InstructionDurations
from qiskit.providers import BackendV2


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: ParameterVector, q_reg: QuantumRegister, **kwargs
):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit_pulse ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param q_reg: Quantum Register formed of target qubits
    :return:
    """
    target, backend = kwargs["target"], kwargs["backend"]
    gate, physical_qubits = target.get("gate", None), target["physical_qubits"]
    my_qc = QuantumCircuit(q_reg, name=f"{gate.name if gate is not None else 'G'}_cal")
    optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
    # optimal_params = np.pi * np.zeros(len(params))

    # my_qc.rx(params[0], q_reg[0])
    my_qc.u(
        optimal_params[0] + params[0],
        optimal_params[1] + params[1],
        optimal_params[2] + params[2],
        q_reg[0],
    )
    my_qc.u(
        optimal_params[3] + params[3],
        optimal_params[4] + params[4],
        optimal_params[5] + params[5],
        q_reg[1],
    )

    my_qc.rzx(optimal_params[6] + params[6], q_reg[0], q_reg[1])

    qc.append(my_qc.to_gate(label=my_qc.name), q_reg)


def get_circuit_context(
    backend: Optional[BackendV2] = None, initial_layout: Optional[List[int]] = None
):
    """
    Define the context of the circuit to be used in the training
    :param backend: Backend instance
    :param initial_layout: Initial layout of the qubits
    :return: QuantumCircuit instance
    """
    from qiskit.transpiler import CouplingMap, Layout

    coupling_map = CouplingMap.from_full(2)
    tgt_reg = QuantumRegister(2, name="tgt")
    # nn_reg = QuantumRegister(3, name="nn")
    layout = Layout(
        input_dict=(
            {tgt_reg[i]: initial_layout[i] for i in range(len(initial_layout))}
            if initial_layout is not None
            else {tgt_reg[i]: i for i in range(2)}
        )
    )
    # layout.add_register(nn_reg)
    circuit = QuantumCircuit(tgt_reg)
    circuit.cx(0, 1)

    transpile_input = (
        {"backend": backend} if backend is not None else {"coupling_map": coupling_map}
    )
    circuit = transpile(
        circuit,
        **transpile_input,
        initial_layout=layout,
        optimization_level=1,
        seed_transpiler=42,
    )

    print("Circuit context")
    circuit.draw("mpl")
    return circuit


def custom_instruction_durations(num_qubits: int):
    # User input for default gate durations
    single_qubit_gate_time = 1.6e-7
    two_qubit_gate_time = 5.3e-7
    readout_time = 1.2e-6
    reset_time = 1.0e-6
    virtual_gates = ["rz", "s", "t"]

    circuit_gate_times = {
        "x": single_qubit_gate_time,
        "sx": single_qubit_gate_time,
        "h": single_qubit_gate_time,
        "u": single_qubit_gate_time,
        "cx": two_qubit_gate_time,
        "rzx": two_qubit_gate_time,
        "measure": readout_time,
        "reset": reset_time,
    }
    circuit_gate_times.update({gate: 0.0 for gate in virtual_gates})
    instruction_durations_dict = generate_default_instruction_durations_dict(
        n_qubits=num_qubits,
        single_qubit_gate_time=single_qubit_gate_time,
        two_qubit_gate_time=two_qubit_gate_time,
        circuit_gate_times=circuit_gate_times,
        virtual_gates=virtual_gates,
    )

    instruction_durations = InstructionDurations()
    instruction_durations.dt = 2.2222222222222221e-10
    instruction_durations.duration_by_name_qubits = instruction_durations_dict

    return instruction_durations
