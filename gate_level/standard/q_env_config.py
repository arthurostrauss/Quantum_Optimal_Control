from __future__ import annotations

import warnings
from typing import Optional, Dict, List
import os
import numpy as np

from rl_qoc.helper_functions import (
    generate_default_instruction_durations_dict,
    select_backend,
    get_q_env_config,
)
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.transpiler import InstructionDurations
from qiskit.providers import BackendV2

current_dir = os.path.dirname(os.path.realpath(__file__))
config_file_name = "q_env_gate_config.yml"
config_file_address = os.path.join(current_dir, config_file_name)


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


def get_backend(
    real_backend: Optional[bool] = None,
    backend_name: Optional[str] = None,
    use_dynamics: Optional[bool] = None,
    physical_qubits: Optional[list] = None,
    channel: Optional[str] = None,
    instance: Optional[str] = None,
    solver_options: Optional[Dict] = None,
    calibration_files: Optional[str] = None,
):
    """
    Define backend on which the calibration is performed.
    This function uses data from the yaml file to define the backend.
    If provided parameters on the backend are null, then the user should provide the backend instance.
    :param real_backend: If True, then calibration is performed on real quantum hardware, otherwise on simulator
    :param backend_name: Name of the backend to be used, if None, then least busy backend is used
    :param use_dynamics: If True, then DynamicsBackend is used, otherwise standard backend is used
    :param physical_qubits: Physical qubits indices to be used for the calibration
    :param channel: Qiskit Runtime Channel (for real backend)
    :param instance: Qiskit Runtime Instance (for real backend)
    :param solver_options: Options for the solver (for DynamicsBackend)
    :param calibration_files: Path to the calibration files (for DynamicsBackend)
    :return: Backend instance
    """
    # Real backend initialization

    backend = select_backend(
        real_backend,
        channel,
        instance,
        backend_name,
        use_dynamics,
        physical_qubits,
        solver_options,
        calibration_files,
    )

    if backend is None:
        # TODO: Add here your custom backend
        pass

        # backend = FakeTorontoV2()
    if backend is None:
        warnings.warn("No backend was provided, State vector simulation will be used")
    return backend


def get_circuit_context(
    backend: Optional[BackendV2], initial_layout: Optional[List[int]] = None
):
    """
    Define the context of the circuit to be used in the training
    :param backend: Backend instance
    :param initial_layout: Initial layout of the qubits
    :return: QuantumCircuit instance
    """
    circuit = QuantumCircuit(5)
    circuit.h(0)
    for i in range(1, 5):
        circuit.cx(0, i)

    if backend is not None and backend.target.has_calibration("x", (0,)):
        circuit = transpile(
            circuit,
            backend,
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


# Do not touch part below, just retrieve in your notebook training_config and circuit_context
q_env_config = get_q_env_config(
    config_file_address,
    get_backend,
    apply_parametrized_circuit,
)
q_env_config.parametrized_circuit_kwargs = {
    "target": q_env_config.target,
    "backend": q_env_config.backend,
}
q_env_config.instruction_durations_dict = custom_instruction_durations(
    q_env_config.backend.num_qubits if q_env_config.backend is not None else 5
)
circuit_context = get_circuit_context(
    q_env_config.backend, q_env_config.physical_qubits
)
