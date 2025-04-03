from __future__ import annotations

from typing import Optional, List
import os

from qiskit.transpiler import Target, InstructionProperties
from qiskit_experiments.calibration_management import (
    FixedFrequencyTransmon,
    EchoedCrossResonance,
)
from rl_qoc.helpers import (
    to_python_identifier,
    new_params_ecr,
    new_params_sq_gate,
    get_gate,
)
from qiskit import pulse, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector, Gate, Parameter
from qiskit.providers import BackendV1, BackendV2, BackendV2Converter
from qiskit_experiments.calibration_management import Calibrations
import jax

jax.config.update("jax_enable_x64", True)
# tell JAX we are using CPU
jax.config.update("jax_platform_name", "cpu")
current_dir = os.path.dirname(os.path.realpath(__file__))


def custom_schedule(
    backend: BackendV1 | BackendV2,
    physical_qubits: List[int],
    params: ParameterVector,
) -> pulse.ScheduleBlock:
    """
    Define parametrization of the pulse schedule characterizing the target gate.
    This function can be customized at will, however one shall recall to make sure that number of actions match the
    number of pulse parameters used within the function (through the params argument).
        :param backend: IBM Backend on which schedule shall be added
        :param physical_qubits: Physical qubits on which custom gate is applied on
        :param params: Parameters of the Schedule/Custom gate

        :return: Parametrized Schedule
    """

    # Load here all pulse parameters names that should be tuned during model-free calibration.
    # Here we focus on real time tunable pulse parameters (amp, angle, duration)

    ecr_pulse_features = ["amp", "angle", "tgt_amp", "tgt_angle"]  # For ECR gate
    sq_pulse_features = ["amp"]  # For single qubit gates
    sq_name = "x"  # Name of the single qubit gate baseline to pick ("x" or "sx")
    keep_symmetry = True  # Choose if the two parts of the ECR tone shall be jointly parametrized or not
    include_baseline = False  # Choose if original calibration shall be included as baseline in parametrization
    include_duration = (
        False  # Choose if pulse duration shall be included in parametrization
    )
    duration_window = 1  # Duration window for the pulse duration
    if include_duration:
        ecr_pulse_features.append("duration")
        sq_pulse_features.append("duration")

    qubits = tuple(physical_qubits)

    if len(qubits) == 2:  # Retrieve schedule for ECR gate
        new_params = new_params_ecr(
            params,
            qubits,
            backend,
            ecr_pulse_features,
            keep_symmetry,
            duration_window,
            include_baseline,
        )
    elif len(qubits) == 1:  # Retrieve schedule for single qubit gate
        new_params = new_params_sq_gate(
            params,
            qubits,
            backend,
            sq_pulse_features,
            duration_window,
            include_baseline,
            gate_name=sq_name,
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

    gate_name = "ecr" if len(physical_qubits) == 2 else sq_name

    # Retrieve schedule (for now, works only with ECRGate(), as no library yet available for CX)

    basis_gate_sched = cals.get_schedule(gate_name, qubits, assign_params=new_params)

    if isinstance(
        backend, BackendV1
    ):  # Convert to BackendV2 if needed (to access Target)
        backend = BackendV2Converter(backend)

    # Choose which gate to build here
    with pulse.build(backend, name="custom_sched") as custom_sched:
        # pulse.call(backend.target.get_calibration("s", qubits))
        pulse.call(basis_gate_sched)
        # pulse.call(backend.target.get_calibration("s", qubits))

    return custom_sched


def validate_pulse_kwargs(
    **kwargs,
) -> tuple[Optional[Gate], list[int], BackendV1 | BackendV2]:
    """
    Validate the kwargs passed to the parametrized circuit function for pulse level calibration
    :param kwargs: Additional arguments to be passed to the function (here target dict and backend object)
        Target dict should contain the gate and physical qubits to be calibrated in the form of a dictionary
        e.g. {"gate": Gate, "physical_qubits": [0, 1]}
    :return: Gate, Physical qubits, Backend
    """
    if "target" not in kwargs or "backend" not in kwargs:
        raise ValueError("Missing target and backend in kwargs.")
    target, backend = kwargs["target"], kwargs["backend"]
    assert isinstance(
        backend, (BackendV1, BackendV2)
    ), "Backend should be a valid Qiskit Backend instance"
    assert isinstance(
        target, dict
    ), "Target should be a dictionary with 'physical_qubits' keys."

    gate, physical_qubits = target.get("gate", None), target["physical_qubits"]
    if gate is not None:
        gate = get_gate(gate)
        assert isinstance(gate, Gate), "Gate should be a valid Qiskit Gate instance"
    assert isinstance(
        physical_qubits, list
    ), "Physical qubits should be a list of integers"
    assert all(
        isinstance(qubit, int) for qubit in physical_qubits
    ), "Physical qubits should be a list of integers"

    return gate, physical_qubits, backend


def apply_parametrized_circuit(
    qc: QuantumCircuit, params: ParameterVector, tgt_register: QuantumRegister, **kwargs
) -> None:
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    This function is used to run the QuantumCircuit instance on a Runtime backend
    :param qc: Quantum Circuit instance to add the gate on
    :param params: Parameters of the custom Gate
    :param tgt_register: Quantum Register formed of target qubits
    :param kwargs: Additional arguments to be passed to the function (here target dict and backend object)
    :return:
    """

    gate, physical_qubits, backend = validate_pulse_kwargs(**kwargs)
    parametrized_gate_name = f"{gate.name if gate is not None else 'G'}_cal"

    param_vec_name = params.name
    if param_vec_name[-1].isdigit():
        parametrized_gate_name += param_vec_name[-1]

    # Create a custom gate with the same name as the original gate
    # Create new set of parameters that are valid Python identifiers (no brackets from the ParameterVector)
    params2 = [Parameter(to_python_identifier(p.name)) for p in params]
    parametrized_gate = Gate(
        parametrized_gate_name,
        len(physical_qubits),
        params=params2,
    )

    # Create custom schedule with original parameters
    parametrized_schedule = custom_schedule(
        backend=backend,
        physical_qubits=physical_qubits,
        params=params,
    )

    # Add the custom calibration to the circuit with current parameters
    qc.append(parametrized_gate, tgt_register)
    qc.add_calibration(
        parametrized_gate_name, physical_qubits, parametrized_schedule, params.params
    )

    # Add the custom calibration to the backend for enabling transpilation with the custom gate
    if parametrized_gate_name not in backend.operation_names:
        target: Target = backend.target
        properties = InstructionProperties(
            duration=parametrized_schedule.duration * backend.dt,
            calibration=parametrized_schedule.assign_parameters(
                {params: params2}, inplace=False
            ),
        )
        target.add_instruction(parametrized_gate, {tuple(physical_qubits): properties})


def get_circuit_context(
    backend: Optional[BackendV1 | BackendV2] = None,
    initial_layout: Optional[List[int]] = None,
):
    """
    Define here the circuit context to be used for the calibration (relevant only for context-aware calibration)
    :param backend: Backend instance
    :param initial_layout: Initial layout of the qubits
    :return: QuantumCircuit instance
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.barrier()
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.cx(0, 1)

    try:
        circuit = transpile(circuit, backend, initial_layout=initial_layout)
    except Exception as e:
        pass
    print("Circuit context: ")
    print(circuit)

    return circuit
