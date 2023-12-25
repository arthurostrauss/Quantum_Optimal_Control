from __future__ import annotations

from typing import Optional, Dict
import os
import yaml
from gymnasium.spaces import Box
import numpy as np
import warnings
from basis_gate_library import FixedFrequencyTransmon, EchoedCrossResonance
from helper_functions import (
    get_ecr_params,
    get_pulse_params,
    load_q_env_from_yaml_file,
    perform_standard_calibrations,
)
from qiskit import pulse, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector, Gate
from qiskit_dynamics import DynamicsBackend
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend as RuntimeBackend
from qiskit_ibm_runtime.fake_provider import FakeProvider
from qiskit.providers import BackendV1, BackendV2
from qiskit_experiments.calibration_management import Calibrations
from qconfig import QiskitConfig, QEnvConfig
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from dynamics_config import dynamics_backend
from typing import List, Sequence

current_dir = os.path.dirname(os.path.realpath(__file__))
config_file_name = "q_env_pulse_config.yml"
config_file_address = os.path.join(current_dir, config_file_name)


def new_params_ecr(
    params: ParameterVector,
    qubits: Sequence[int],
    backend: BackendV1 | BackendV2,
    pulse_features: List[str],
    keep_symmetry: bool = True,
    duration_window: float = 0.1,
):
    new_params, available_features, _, _ = get_ecr_params(backend, qubits)

    if keep_symmetry:  # Maintain symmetry between the two GaussianSquare pulses
        if len(pulse_features) != len(params):
            raise ValueError(
                f"Number of pulse features ({len(pulse_features)} and number of parameters ({len(params)}"
                f"do not match"
            )
        for sched in ["cr45p", "cr45m"]:
            for i, feature in enumerate(pulse_features):
                if feature != "duration" and feature in available_features:
                    new_params[(feature, qubits, sched)] += params[i]
                else:
                    new_params[
                        (feature, qubits, sched)
                    ] += pulse.builder.seconds_to_samples(duration_window * params[i])
    else:
        if 2 * len(pulse_features) != len(params):
            raise ValueError(
                f"Number of pulse features ({len(pulse_features)}) and number of parameters ({len(params)} do not "
                f"match"
            )
        num_features = len(pulse_features)
        for i, sched in enumerate(["cr45p", "cr45m"]):
            for j, feature in enumerate(pulse_features):
                if feature != "duration" and feature in available_features:
                    new_params[(feature, qubits, sched)] += params[i * num_features + j]
                else:
                    new_params[
                        (feature, qubits, sched)
                    ] += pulse.builder.seconds_to_samples(
                        duration_window * params[i * num_features + j]
                    )

    return new_params


def new_params_x(
    params: ParameterVector,
    qubits: Sequence[int],
    backend: BackendV1 | BackendV2,
    pulse_features: List[str],
    duration_window: float,
):
    new_params, available_features, _, _ = get_pulse_params(backend, qubits, "x")
    if len(pulse_features) != len(params):
        raise ValueError(
            f"Number of pulse features ({len(pulse_features)}) and number of parameters ({len(params)}"
            f" do not match"
        )
    for i, feature in enumerate(pulse_features):
        if feature != "duration" and feature in available_features:
            new_params[(feature, qubits, "x")] += params[i]
        else:
            new_params[(feature, qubits, "x")] += pulse.builder.seconds_to_samples(
                duration_window * params[i]
            )
    return new_params


def custom_schedule(
    backend: BackendV1 | BackendV2,
    physical_qubits: list,
    params: ParameterVector,
    keep_symmetry: bool = True,
) -> pulse.ScheduleBlock:
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
    ecr_pulse_features = ["amp", "angle", "tgt_amp", "tgt_angle"]
    x_pulse_features = ["amp", "angle"]
    # Uncomment line below to include pulse duration as tunable parameter
    # ecr_pulse_features.append("duration")
    # x_pulse_features.append("duration")

    duration_window = 0
    qubits = tuple(physical_qubits)

    if len(qubits) == 2:  # Retrieve schedule for ECR gate
        new_params = new_params_ecr(
            params, qubits, backend, ecr_pulse_features, keep_symmetry, duration_window
        )
    elif len(qubits) == 1:  # Retrieve schedule for X gate
        new_params = new_params_x(
            params,
            qubits,
            backend,
            x_pulse_features,
            duration_window,
        )
    else:
        raise ValueError(
            f"Number of physical qubits ({len(physical_qubits)}) not supported by current pulse macro, "
            f"adapt it to your needs"
        )
    cals = Calibrations.from_backend(
        backend,
        [
            FixedFrequencyTransmon(["x", "sx"]),
            EchoedCrossResonance(["cr45p", "cr45m", "ecr"]),
        ],
        add_parameter_defaults=True,
    )

    gate_name = "ecr" if len(physical_qubits) == 2 else "x"
    # Retrieve schedule (for now, works only with ECRGate(), as no library yet available for CX)

    return cals.get_schedule(gate_name, qubits, assign_params=new_params)


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
    parametrized_gate = Gate(
        f"custom_{gate.name}", len(tgt_register), params=params.params
    )
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

    if use_dynamics is not None and use_dynamics:
        if backend is not None:
            backend = DynamicsBackend.from_backend(
                backend, subsystem_list=list(physical_qubits)
            )
        else:
            # TODO: Add here your custom DynamicsBackend
            backend = dynamics_backend

        _, _ = perform_standard_calibrations(backend)
    else:
        # TODO: Add here your custom backend
        pass
    if backend is None:
        warnings.warn("No backend was provided, Statevector simulation will be used")

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
(
    env_params,
    backend_params,
    estimator_options,
    check_on_exp,
) = load_q_env_from_yaml_file(config_file_address)
env_backend = get_backend(**backend_params)
backend_config = QiskitConfig(
    apply_parametrized_circuit,
    env_backend,
    estimator_options=estimator_options
    if isinstance(env_backend, RuntimeBackend)
    else None,
    parametrized_circuit_kwargs={
        "target": env_params["target"],
        "backend": env_backend,
    },
)
QuantumEnvironment.check_on_exp = (
    ContextAwareQuantumEnvironment.check_on_exp
) = check_on_exp
q_env_config = QEnvConfig(backend_config=backend_config, **env_params)
circuit_context = get_circuit_context(env_backend)