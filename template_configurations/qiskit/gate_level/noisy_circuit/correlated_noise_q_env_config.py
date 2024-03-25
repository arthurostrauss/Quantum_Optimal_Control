from __future__ import annotations
import warnings
from typing import Optional, Dict
import os
import numpy as np
from helper_functions import (
    load_q_env_from_yaml_file,
    select_backend,
)
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import IBMBackend as RuntimeBackend
from qiskit.providers import BackendV1, BackendV2

import qiskit_aer.noise as noise
from qiskit_aer.noise.passes.local_noise_pass import LocalNoisePass

from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RXGate, SXGate, IGate, CRXGate
from scipy.linalg import sqrtm

from qconfig import QiskitConfig, QEnvConfig
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from helper_functions import create_circuit_from_own_unitaries

from qiskit.providers.fake_provider import GenericBackendV2

current_dir = os.path.dirname(os.path.realpath(__file__))
config_file_name = "noise_q_env_gate_config.yml"
config_file_address = os.path.join(current_dir, config_file_name)


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: ParameterVector, q_reg: QuantumRegister, **kwargs
):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param q_reg: Quantum Register formed of target qubits
    :return:
    """
    target = kwargs["target"]
    my_qc = QuantumCircuit(q_reg, name=f"custom_{target['gate'].name}")
    optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
    optimal_params += np.array(
        [
            -0.00020222,
            -0.00018466,
            0.00075005,
            0.00248492,
            -0.00792428,
            -0.00582522,
            -0.00161892,
        ]
    )
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

    ### Random noise with Kraus operators ###
    # dim = 4  # For a 4x4 system (e.g., 2 qubits)
    # num_ops = 3  # Number of Kraus operators
    # epsilon = 0.01  # Noise strength parameter
    # kraus_ops_eps = generate_random_cptp_map(dim, num_ops, epsilon)
    # noise_model = NoiseModel()
    # noise_model.add_all_qubit_quantum_error(kraus_ops_eps, ['rzx'])

    ### Custom spillover noise model ###
    global phi, gamma, custom_rx_gate_label

    noise_model = noise.NoiseModel()
    coherent_crx_noise = noise.coherent_unitary_error(CRXGate(gamma * phi))
    noise_model.add_quantum_error(coherent_crx_noise, [custom_rx_gate_label], [0, 1])
    noise_model.add_basis_gates(["unitary"])
    print("\n", noise_model, "\n")

    generic_backend = GenericBackendV2(
        num_qubits=2,
        dtm=2.2222 * 1e-10,
        basis_gates=["cx", "id", "rz", "sx", "x", "crx"],
    )
    backend = AerSimulator.from_backend(generic_backend, noise_model=noise_model)

    if backend is None:
        # TODO: Add here your custom backend
        # For now use FakeJakartaV2 as a safe working custom backend
        # backend = FakeProvider().get_backend("fake_jakarta")
        from qiskit_ibm_runtime.fake_provider import FakeTorontoV2

        # backend = FakeTorontoV2()
    if backend is None:
        warnings.warn("No backend was provided, State vector simulation will be used")
    return backend


### Custom spillover noise model
phi = np.pi / 4  # rotation angle
gamma = 0.01  # spillover rate for the CRX gate
custom_rx_gate_label = "custom_kron(rx,ident)_gate"


def get_circuit_context(backend: Optional[BackendV2]):
    """
    Define the context of the circuit to be used in the training
    :param backend: Backend instance
    :return: QuantumCircuit instance
    """
    global phi, gamma, custom_rx_gate_label

    circuit = QuantumCircuit(2)
    rx_op = Operator(RXGate(phi))
    identity_op = Operator(IGate())
    rx_op_2q = Operator(identity_op.tensor(rx_op))
    circuit.unitary(rx_op_2q, [0, 1], label=custom_rx_gate_label)

    circuit.cx(0, 1)

    if backend is not None and backend.target.has_calibration("x", (0,)):
        circuit = transpile(circuit, backend, optimization_level=1, seed_transpiler=42)
    print("Circuit context")
    print(circuit)
    return circuit


# Do not touch part below, just retrieve in your notebook training_config and circuit_context
(
    env_params,
    backend_params,
    estimator_options,
    check_on_exp,
    channel_estimator,
) = load_q_env_from_yaml_file(config_file_address)
backend = get_backend(**backend_params)
backend_config = QiskitConfig(
    apply_parametrized_circuit,
    backend,
    estimator_options=(
        estimator_options if isinstance(backend, RuntimeBackend) else None
    ),
    parametrized_circuit_kwargs={"target": env_params["target"], "backend": backend},
)

QuantumEnvironment.check_on_exp = ContextAwareQuantumEnvironment.check_on_exp = (
    check_on_exp
)
QuantumEnvironment.channel_estimator = channel_estimator
q_env_config = QEnvConfig(backend_config=backend_config, **env_params)
circuit_context = get_circuit_context(backend)
