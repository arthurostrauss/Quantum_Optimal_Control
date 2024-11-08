from __future__ import annotations
from typing import Optional, List
import warnings
import os
import numpy as np
from rl_qoc.helpers.helper_functions import (
    generate_default_instruction_durations_dict,
    get_q_env_config,
)
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers import BackendV2
import qiskit_aer.noise as noise
from qiskit_aer import AerSimulator
from qiskit_aer.noise import ReadoutError, reset_error
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RXGate
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler import CouplingMap

current_dir = os.path.dirname(os.path.realpath(__file__))
config_file_name = "noise_q_env_gate_config.yml"
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
    target = kwargs["target"]
    my_qc = QuantumCircuit(q_reg, name=f"{target['gate'].name}_cal")
    optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
    # optimal_params = np.pi * np.zeros(len(params))

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
    qc.append(my_qc.to_instruction(label=my_qc.name), q_reg)


def get_backend(ϕ=np.pi / 2, γ=0.05, custom_rx_gate_label="rx_custom"):
    """
    Define backend on which the calibration is performed.
    :return: Backend instance
    """
    # Real backend initialization
    # TODO: Add here your custom backend

    ### Random noise with Kraus operators ###
    # dim = 4  # For a 4x4 system (e.g., 2 qubits)
    # num_ops = 3  # Number of Kraus operators
    # epsilon = 0.01  # Noise strength parameter
    # kraus_ops_eps = generate_random_cptp_map(dim, num_ops, epsilon)
    # noise_model = NoiseModel()
    # noise_model.add_all_qubit_quantum_error(kraus_ops_eps, ['rzx'])

    ### Custom spillover noise model ###

    n_qubits = 1

    rx_phi_gamma_op = Operator(RXGate(γ * ϕ))
    ident_rx_op = rx_phi_gamma_op ^ Operator.from_label("I")

    noise_model = noise.NoiseModel(["unitary", "rzx", "cx", "u", "h", "x", "s", "z"])
    coherent_rx_noise = noise.coherent_unitary_error(ident_rx_op)
    noise_model.add_quantum_error(coherent_rx_noise, [custom_rx_gate_label], [0, 1])
    noise_model.add_quantum_error(coherent_rx_noise, [custom_rx_gate_label], [1, 0])

    p0given1 = 0.0138  # IBM Sherbrooke
    p1given0 = 0.0116  # IBM Sherbrooke
    readout_error_matrix = ReadoutError(
        [[1 - p1given0, p1given0], [p0given1, 1 - p0given1]]
    )
    noise_model.add_all_qubit_readout_error(readout_error_matrix, "measure")
    noise_model.add_all_qubit_quantum_error(reset_error(0.01), "reset")

    backend = AerSimulator(
        noise_model=noise_model,
        coupling_map=CouplingMap.from_full(n_qubits),
        max_parallel_experiments=0,
        # runtime_parameter_bind_enable=True,
    )

    # backend = FakeTorontoV2()
    if backend is None:
        warnings.warn("No backend was provided, State vector simulation will be used")
    return backend


# Custom spillover noise model
phi = np.pi / 2  # rotation angle
gamma = 0.05  # spillover rate for the CRX gate
custom_rx_gate_label = "rx_custom"


def get_circuit_context(
    backend: Optional[BackendV2], initial_layout: Optional[List[int]] = None
):
    """
    Define the context of the circuit to be used in the training
    :param backend: Backend instance
    :param initial_layout: Initial layout of the qubits
    :return: QuantumCircuit instance
    """
    global phi, gamma, custom_rx_gate_label

    circuit = QuantumCircuit(2)
    sub_context = QuantumCircuit(2)
    sub_context.rx(phi, 0)
    circuit.unitary(Operator(sub_context), [0, 1], label=custom_rx_gate_label)
    circuit.cx(0, 1)
    print(Operator(circuit))
    print(circuit)

    if backend is not None:
        circuit = transpile(
            circuit,
            backend,
            optimization_level=1,
            seed_transpiler=42,
            initial_layout=initial_layout,
        )
    # print("Circuit context")
    # print(circuit)
    return circuit


def get_instruction_durations(backend: Optional[BackendV2] = None):
    if backend is not None and backend.instruction_durations.duration_by_name_qubits:
        instruction_durations = backend.instruction_durations
    else:
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

        n_qubits = backend.num_qubits if backend else 10
        instruction_durations_dict = generate_default_instruction_durations_dict(
            n_qubits=n_qubits,
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
    apply_parametrized_circuit,
    get_backend(ϕ=phi, γ=gamma, custom_rx_gate_label=custom_rx_gate_label),
)
q_env_config.backend_config.parametrized_circuit_kwargs = {
    "target": q_env_config.target,
    "backend": q_env_config.backend,
}
q_env_config.backend_config.instruction_durations_dict = get_instruction_durations(
    q_env_config.backend
)
circuit_context = get_circuit_context(
    q_env_config.backend, q_env_config.physical_qubits
)
