from __future__ import annotations

from typing import Optional, Dict
import os
import yaml
from gymnasium.spaces import Box
import numpy as np
from basis_gate_library import FixedFrequencyTransmon, EchoedCrossResonance
from helper_functions import (
    determine_ecr_params,
    load_q_env_from_yaml_file,
    perform_standard_calibrations,
)
from qiskit import pulse, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector, Gate
from qiskit_dynamics import Solver, DynamicsBackend
from custom_jax_sim import JaxSolver
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend as RuntimeBackend
from qiskit_ibm_runtime.fake_provider import FakeProvider
from qiskit.providers import BackendV1, BackendV2
from qiskit_experiments.calibration_management import Calibrations
from qconfig import QiskitConfig, QEnvConfig
from dynamics_config import dynamics_backend

current_dir = os.path.dirname(os.path.realpath(__file__))
config_file_name = "q_env_config.yml"
config_file_address = os.path.join(current_dir, config_file_name)


def custom_schedule(
    backend: BackendV1 | BackendV2,
    physical_qubits: list,
    params: ParameterVector,
    keep_symmetry: bool = True,
):
    """
    Define parametrization of the pulse schedule characterizing the target gate.
    This function can be customized at will, however one shall recall to make sure that number of actions match the
    number of pulse parameters used within the function (through the params argument).
        :param backend: IBM Backend on which schedule shall be added
        :param physical_qubits: Physical qubits on which custom gate is applied on
        :param params: Parameters of the Schedule/Custom gate
        :param keep_symmetry: Choose if the two parts of the ECR tone shall be jointly parametrized or not

        :return: Parametrized Schedule
    """
    # Example of custom schedule for Echoed Cross Resonance gate

    # Load here all pulse parameters names that should be tuned during model-free calibration.
    # Here we focus on real time tunable pulse parameters (amp, angle, duration)
    pulse_features = ["amp", "angle", "tgt_amp", "tgt_angle"]

    # Uncomment line below to include pulse duration as tunable parameter
    # pulse_features.append("duration")
    duration_window = 0

    new_params, _, _ = determine_ecr_params(backend, physical_qubits)

    qubits = tuple(physical_qubits)

    if keep_symmetry:  # Maintain symmetry between the two GaussianSquare pulses
        for sched in ["cr45p", "cr45m"]:
            for i, feature in enumerate(pulse_features):
                if feature != "duration":
                    new_params[(feature, qubits, sched)] += params[i]
                else:
                    new_params[
                        (feature, qubits, sched)
                    ] += pulse.builder.seconds_to_samples(duration_window * params[i])
    else:
        num_features = len(pulse_features)
        for i, sched in enumerate(["cr45p", "cr45m"]):
            for j, feature in enumerate(pulse_features):
                if feature != "duration":
                    new_params[(feature, qubits, sched)] += params[i * num_features + j]
                else:
                    new_params[
                        (feature, qubits, sched)
                    ] += pulse.builder.seconds_to_samples(
                        duration_window * params[i * num_features + j]
                    )

    cals = Calibrations.from_backend(
        backend,
        [
            FixedFrequencyTransmon(["x", "sx"]),
            EchoedCrossResonance(["cr45p", "cr45m", "ecr"]),
        ],
        add_parameter_defaults=True,
    )

    # Retrieve schedule (for now, works only with ECRGate(), as no library yet available for CX)
    parametrized_schedule = cals.get_schedule("ecr", qubits, assign_params=new_params)
    return parametrized_schedule


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: ParameterVector, tgt_register: QuantumRegister, **kwargs
):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param tgt_register: Quantum Register formed of target qubits
    :return:
    """
    target = kwargs["target"]
    backend = kwargs["backend"]

    gate, physical_qubits = target["gate"], target["register"]
    parametrized_gate = Gate("custom_ecr", 2, params=params.params)
    parametrized_schedule = custom_schedule(
        backend=backend, physical_qubits=physical_qubits, params=params
    )
    qc.add_calibration(parametrized_gate, physical_qubits, parametrized_schedule)
    qc.append(parametrized_gate, tgt_register)


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
    If provided parameters on the backend are null, then the user should provide a custom backend instance.
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
        if use_dynamics:
            backend = dynamics_backend
            _, _ = perform_standard_calibrations(backend)
        else:
            # TODO: Add here your custom backend
            pass

    return backend


def get_circuit_context(backend: BackendV1 | BackendV2):
    """
    Define here the circuit context to be used for the calibration
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    transpiled_circ = transpile(circuit, backend)

    return transpiled_circ


# Do not touch part below, just retrieve in your notebook training_config and circuit_context
(params, backend_params, estimator_options) = load_q_env_from_yaml_file(
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
q_env_config = QEnvConfig(backend_config=backend_config, **params)
circuit_context = get_circuit_context(backend)
