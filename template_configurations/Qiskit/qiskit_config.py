from __future__ import annotations

from typing import Optional, Dict

import yaml
from gymnasium.spaces import Box
import numpy as np
from basis_gate_library import FixedFrequencyTransmon, EchoedCrossResonance
from helper_functions import determine_ecr_params, load_from_yaml_file
from qiskit import pulse, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Gate
from qiskit_dynamics import Solver, DynamicsBackend
from custom_jax_sim import JaxSolver
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeJakarta, FakeProvider
from qiskit.providers import BackendV1, BackendV2
from qiskit_experiments.calibration_management import Calibrations
from qconfig import QiskitConfig, TrainingConfig

def custom_schedule(backend: BackendV1 | BackendV2, physical_qubits: list, params: ParameterVector,
                    keep_symmetry: bool = True):
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
                    new_params[(feature, qubits, sched)] += pulse.builder.seconds_to_samples(
                        duration_window * params[i])
    else:
        num_features = len(pulse_features)
        for i, sched in enumerate(["cr45p", "cr45m"]):
            for j, feature in enumerate(pulse_features):
                if feature != "duration":
                    new_params[(feature, qubits, sched)] += params[i * num_features + j]
                else:
                    new_params[(feature, qubits, sched)] += pulse.builder.seconds_to_samples(
                        duration_window * params[i * num_features + j])

    cals = Calibrations.from_backend(backend, [FixedFrequencyTransmon(["x", "sx"]),
                                               EchoedCrossResonance(["cr45p", "cr45m", "ecr"])],
                                     add_parameter_defaults=True)

    # Retrieve schedule (for now, works only with ECRGate(), as no library yet available for CX)
    parametrized_schedule = cals.get_schedule("ecr", qubits, assign_params=new_params)
    return parametrized_schedule


def apply_parametrized_circuit(qc: QuantumCircuit, params: ParameterVector, tgt_register: QuantumRegister,
                               target: Dict, backend: BackendV1 | BackendV2):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param tgt_register: Quantum Register formed of target qubits
    :param target: Target Gate to calibrate
    :param backend: Backend on which the calibration is performed
    :return:
    """

    gate, physical_qubits = target["gate"], target["register"]
    parametrized_gate = Gate("custom_ecr", 2, params=params.params)
    parametrized_schedule = custom_schedule(backend=backend, physical_qubits=physical_qubits, params=params)
    qc.add_calibration(parametrized_gate, physical_qubits, parametrized_schedule)
    qc.append(parametrized_gate, tgt_register)


def retrieve_backend(backend_name: str, real_backend: bool = False):

    """
    Real backend initialization:
    Run this cell only if intending to use a real backend, where Qiskit Runtime is enabled
    """
    if real_backend:
        service = QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q-nus/default/default')
        runtime_backend = service.get_backend(backend_name)
        # Specify options below if needed
        return runtime_backend
    elif not real_backend:
        backend = FakeProvider().get_backend(backend_name)
        return backend
    else:
        from qiskit_dynamics.array import Array
        import jax
        jax.config.update("jax_enable_x64", True)
        # tell JAX we are using CPU
        jax.config.update("jax_platform_name", "cpu")
        # import Array and set default backend

        Array.set_default_backend('jax')
        dim = 3
        v0 = 4.86e9
        anharm0 = -0.32e9
        r0 = 0.22e9

        v1 = 4.97e9
        anharm1 = -0.32e9
        r1 = 0.26e9

        J = 0.002e9

        a = np.diag(np.sqrt(np.arange(1, dim)), 1)
        adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
        N = np.diag(np.arange(dim))

        ident = np.eye(dim, dtype=complex)
        full_ident = np.eye(dim**2, dtype=complex)

        N0 = np.kron(ident, N)
        N1 = np.kron(N, ident)

        a0 = np.kron(ident, a)
        a1 = np.kron(a, ident)

        a0dag = np.kron(ident, adag)
        a1dag = np.kron(adag, ident)


        static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
        static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)

        static_ham_full = static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))

        drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
        drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)

        # build solver
        dt = 1/4.5e9

        solver = Solver(
            static_hamiltonian=static_ham_full,
            hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1, drive_op1, drive_op0],
            rotating_frame=static_ham_full,
            hamiltonian_channels=["d0", "d1", "u0", "u1", "u2", "u3"],
            channel_carrier_freqs={"d0": v0, "d1": v1, "u0": v1, "u1": v0, "u2":v0, "u3": v1},
            dt=dt,
            evaluation_mode="sparse"
        )
        # Consistent solver option to use throughout notebook
        solver_options = {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8}

        custom_backend= DynamicsBackend(
            solver=solver,
            #target = fake_backend_v2.target,
            subsystem_dims=[dim, dim],  # for computing measurement data
            solver_options=solver_options,  # to be used every time run is called
        )

        return custom_backend

params = load_from_yaml_file("q_env_config.yml")
backend_config = QiskitConfig(retrieve_backend("fake_jakarta", real_backend=False),
                               apply_parametrized_circuit,
                                parametrized_circuit_kwargs={"target": target, "backend": backend},
