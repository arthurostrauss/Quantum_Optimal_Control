from __future__ import annotations

from typing import Optional, Dict
import os
import sys
import yaml
from gymnasium.spaces import Box
import numpy as np
from basis_gate_library import FixedFrequencyTransmon, EchoedCrossResonance
from helper_functions import (
    get_ecr_params,
    load_q_env_from_yaml_file,
    perform_standard_calibrations,
)
from qiskit import pulse, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector, Gate
from qiskit_dynamics import Solver, DynamicsBackend
from custom_jax_sim.jax_solver import JaxSolver
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend as RuntimeBackend
from qiskit.providers.fake_provider import FakeProvider
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.fake_provider import FakeJakartaV2
from qiskit_experiments.calibration_management import Calibrations
from qconfig import QiskitConfig, QEnvConfig
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from template_configurations.qiskit.dynamics_config import dynamics_backend

current_dir = os.path.dirname(os.path.realpath(__file__))
config_file_name = "q_env_gate_config.yml"
config_file_address = os.path.join(current_dir, config_file_name)


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: ParameterVector, q_reg: QuantumRegister, **kwargs
):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param tgt_register: Quantum Register formed of target qubits
    :return:
    """

    parametrized_qc = QuantumCircuit(q_reg)
    my_qc = QuantumCircuit(q_reg, name="custom_cx")
    optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])

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
    # my_qc.u(2 * np.pi * params[0], 2 *  np.pi *params[1], 2 * np.pi * params[2], 0)
    # my_qc.u(2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], 1)
    # my_qc.rzx(2 * np.pi * params[6], 0, 1)
    qc.append(my_qc.to_instruction(label="custom_cx"), q_reg)


def get_backend(
    real_backend: Optional[bool] = None,
    backend_name: Optional[str] = None,
    use_dynamics: Optional[bool] = None,
    physical_qubits: Optional[list] = None,
    channel: Optional[str] = None,
    instance: Optional[str] = None,
):
    """
    Define backend on which the calibration is performed.
    This function uses data from the yaml file to define the backend.
    If provided parameters on the backend are null, then the user should provide the backend instance.
    :param real_backend: If True, then calibration is performed on real quantum hardware, otherwise on simulator
    :param backend_name: Name of the backend to be used, if None, then least busy backend is used
    :param use_dynamics: If True, then DynamicsBackend is used, otherwise standard backend is used
    :param n_qubits: Number of qubits to be used for the calibration
    :param channel: Qiskit Runtime Channel
    :param instance: Qiskit Runtime Instance
    :return: Backend instance
    """
    # Real backend initialization

    backend = None
    if real_backend is not None:
        if real_backend:
            service = QiskitRuntimeService(channel=channel, instance=instance)
            if backend_name is None:
                backend = service.least_busy(min_num_qubits=2)
            else:
                backend = service.get_backend(backend_name)

            # Specify options below if needed
            # backend.set_options(**options)
        else:
            # Fake backend initialization (Aer Simulator)
            if backend_name is None:
                backend_name = "fake_jakarta"
            backend = FakeProvider().get_backend(backend_name)

        if use_dynamics:
            if backend is not None:
                backend = DynamicsBackend.from_backend(
                    backend, subsystem_list=list(physical_qubits)
                )
                _, _ = perform_standard_calibrations(backend)
    else:
        # Propose here your custom backend, for Dynamics we take for instance the configuration from dynamics_config.py
        if use_dynamics is not None and use_dynamics:
            backend = dynamics_backend
            _, _ = perform_standard_calibrations(backend)
        else:
            # TODO: Add here your custom backend
            # For now use FakeJakartaV2 as a safe working custom backend
            backend = FakeJakartaV2()

    if backend is None:
        Warning("No backend was provided, Statevector simulation will be used")
    return backend


def get_circuit_context(backend: BackendV1 | BackendV2):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    transpiled_circ = transpile(circuit, backend)

    return transpiled_circ


# Do not touch part below, just retrieve in your notebook training_config and circuit_context
(params, backend_params, estimator_options, check_on_exp) = load_q_env_from_yaml_file(
    config_file_address
)
backend = get_backend(**backend_params)
backend_config = QiskitConfig(
    apply_parametrized_circuit,
    backend,
    estimator_options=estimator_options
    if isinstance(backend, RuntimeBackend)
    else None,
    parametrized_circuit_kwargs={"target": params["target"], "backend": backend},
)
QuantumEnvironment.check_on_exp = (
    ContextAwareQuantumEnvironment.check_on_exp
) = check_on_exp
q_env_config = QEnvConfig(backend_config=backend_config, **params)
circuit_context = get_circuit_context(backend)