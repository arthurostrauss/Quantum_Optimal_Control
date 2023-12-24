from __future__ import annotations

from typing import Optional

from qconfig import QiskitConfig, QEnvConfig
from qiskit_dynamics import Solver, DynamicsBackend
from custom_jax_sim import JaxSolver
from qiskit.circuit import QuantumCircuit, ParameterVector, Gate
from qiskit.circuit.library import ECRGate
from qiskit import pulse, QuantumRegister

import numpy as np
from qiskit.providers import BackendV1, BackendV2
from qiskit_experiments.calibration_management import Calibrations
from basis_gate_library import FixedFrequencyTransmon, EchoedCrossResonance
from helper_functions import get_ecr_params
from gymnasium.spaces import Box


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
    # Load here all pulse parameters names that should be tuned during model-free calibration.
    # Here we focus on real time tunable pulse parameters (amp, angle, duration)
    pulse_features = ["amp", "angle", "tgt_amp", "tgt_angle"]

    # Uncomment line below to include pulse duration as tunable parameter
    # pulse_features.append("duration")
    duration_window = 0

    global n_actions
    assert n_actions == len(
        params
    ), f"Number of actions ({n_actions}) does not match length of ParameterVector {params.name} ({len(params)})"

    new_params, _, _ = get_ecr_params(backend, physical_qubits)

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


# Pulse gate ansatz


def apply_parametrized_circuit(
    qc: QuantumCircuit,
    params: Optional[ParameterVector] = None,
    tgt_register: Optional[QuantumRegister] = None,
):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param tgt_register: Quantum Register formed of target qubits
    :return:
    """
    global n_actions, backend, target

    gate, physical_qubits = target["gate"], target["register"]

    if params is None:
        params = ParameterVector("theta", n_actions)
    if tgt_register is None:
        tgt_register = qc.qregs[0]

    # Choose below which target gate you'd like to calibrate
    parametrized_gate = Gate("custom_ecr", 2, params=params.params)
    # parametrized_gate = gate.copy()
    # parametrized_gate.params = params.params
    parametrized_schedule = custom_schedule(
        backend=backend, physical_qubits=physical_qubits, params=params
    )
    qc.add_calibration(parametrized_gate, physical_qubits, parametrized_schedule)
    qc.append(parametrized_gate, tgt_register)


physical_qubits = [0, 1]
sampling_Paulis = 50
N_shots = 200
n_actions = 4  # Cf number of parameters in custom_schedule function above
action_space = Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)
obs_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)

target = {"gate": ECRGate(), "register": physical_qubits}
