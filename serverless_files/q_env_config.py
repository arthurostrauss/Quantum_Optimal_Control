from __future__ import annotations

import warnings
from typing import Optional, Dict
import os
import numpy as np
from rl_qoc.helpers.helper_functions import (
    load_q_env_from_yaml_file,
    select_backend,
)
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import IBMBackend as RuntimeBackend
from qiskit.providers import BackendV2

from rl_qoc.qconfig import QiskitConfig, QEnvConfig

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
    # optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
    optimal_params = np.pi * np.zeros(len(params))

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

    if backend is None:
        # TODO: Add here your custom backend
        # For now use FakeJakartaV2 as a safe working custom backend
        # backend = FakeProvider().get_backend("fake_jakarta")
        pass

        # backend = FakeTorontoV2()
    if backend is None:
        warnings.warn("No backend was provided, State vector simulation will be used")
    return backend


def get_circuit_context(backend: Optional[BackendV2]):
    """
    Define the context of the circuit to be used in the training
    :param backend: Backend instance
    :return: QuantumCircuit instance
    """
    circuit = QuantumCircuit(5)
    circuit.h(0)
    for i in range(1, 5):
        circuit.cx(0, i)

    if backend is not None and backend.target.has_calibration("x", (0,)):
        circuit = transpile(circuit, backend, optimization_level=1, seed_transpiler=42)
    print("Circuit context")
    circuit.draw("mpl")
    return circuit


# Do not touch part below, just retrieve in your notebook training_config and circuit_context
(
    env_params,
    backend_params,
    estimator_options,
) = load_q_env_from_yaml_file(config_file_address)
backend = get_backend(**backend_params)

backend_config = QiskitConfig(
    apply_parametrized_circuit,
    backend,
    estimator_options=(estimator_options if isinstance(backend, RuntimeBackend) else None),
    parametrized_circuit_kwargs={"target": env_params["target"], "backend": backend},
)

q_env_config = QEnvConfig(backend_config=backend_config, **env_params)
circuit_context = get_circuit_context(backend)
