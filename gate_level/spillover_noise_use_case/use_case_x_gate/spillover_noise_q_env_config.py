from __future__ import annotations
from typing import Optional, Dict, List
import warnings
import os
import numpy as np
from rl_qoc.helpers.helper_functions import (
    generate_default_instruction_durations_dict,
    select_backend,
    get_q_env_config,
)
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.providers import BackendV2
import qiskit_aer.noise as noise
from qiskit_aer import AerSimulator
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
    optimal_param = [np.pi]
    my_qc.rx(optimal_param[0] + params[0], q_reg[0])
    qc.append(my_qc.to_instruction(label=my_qc.name), q_reg)


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

        ### Random noise with Kraus operators ###
        # dim = 4  # For a 4x4 system (e.g., 2 qubits)
        # num_ops = 3  # Number of Kraus operators
        # epsilon = 0.01  # Noise strength parameter
        # kraus_ops_eps = generate_random_cptp_map(dim, num_ops, epsilon)
        # noise_model = NoiseModel()
        # noise_model.add_all_qubit_quantum_error(kraus_ops_eps, ['rzx'])

        ### Custom spillover noise model ###
        global phi, gamma, custom_rx_gate_label

        n_qubits = 1

        rx_phi_gamma_op = Operator(RXGate(gamma * phi))

        noise_model = noise.NoiseModel(["unitary", "u", "h", "x", "s", "z", "sdg"])
        coherent_rx_noise = noise.coherent_unitary_error(rx_phi_gamma_op)
        noise_model.add_quantum_error(coherent_rx_noise, [custom_rx_gate_label], [0])

        from qiskit_aer.noise import ReadoutError

        p0given1 = 0.0138  # IBM Sherbrooke
        p1given0 = 0.0116  # IBM Sherbrooke
        readout_error_matrix = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
        # noise_model.add_all_qubit_readout_error(readout_error_matrix, "measure")
        #
        # noise_model.add_all_qubit_quantum_error(reset_error(0.01), "reset")
        # backend = AerSimulator.from_backend(generic_backend, noise_model=noise_model)

        backend = AerSimulator(
            noise_model=noise_model,
            coupling_map=CouplingMap.from_full(n_qubits),
            max_parallel_experiments=0,
            runtime_parameter_bind_enable=True,
        )

        # backend = FakeTorontoV2()
    if backend is None:
        warnings.warn("No backend was provided, State vector simulation will be used")
    return backend


# Custom spillover noise model
phi = np.pi  # rotation angle
phi_in_degree = int(phi * 180 / np.pi)
gamma = 0.0  # spillover rate for the CRX gate
custom_rx_gate_label = "x" + str(phi_in_degree)


def get_circuit_context(backend: Optional[BackendV2], initial_layout: Optional[List[int]] = None):
    """
    Define the context of the circuit to be used in the training
    :param backend: Backend instance
    :param initial_layout: Initial layout of the qubits
    :return: QuantumCircuit instance
    """
    global phi, gamma, custom_rx_gate_label
    circuit = QuantumCircuit(1)
    context_circuit = QuantumCircuit(1)
    context_circuit.rx(phi, 0)
    circuit.unitary(Operator(context_circuit), [0], label=custom_rx_gate_label)
    circuit.x(0)
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
    get_backend,
    apply_parametrized_circuit,
)
q_env_config.backend_config.parametrized_circuit_kwargs = {
    "target": q_env_config.target,
    "backend": q_env_config.backend,
}
q_env_config.backend_config.instruction_durations_dict = get_instruction_durations(
    q_env_config.backend
)
circuit_context = get_circuit_context(q_env_config.backend, q_env_config.physical_qubits)
