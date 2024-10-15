from __future__ import annotations

import os
import sys
import pickle
import gzip
from dataclasses import asdict

from qiskit import pulse, schedule, transpile, QuantumRegister
from qiskit.circuit import (
    QuantumCircuit,
    Gate,
    Parameter,
    CircuitInstruction,
    ParameterVector,
    Delay,
)
from qiskit.circuit.library import get_standard_gate_name_mapping as gate_map, RZGate
from qiskit.exceptions import QiskitError
from qiskit.primitives import (
    BackendEstimator,
    Estimator,
    Sampler,
    BackendSampler,
    StatevectorEstimator,
    StatevectorSampler,
    BaseEstimatorV1,
    BaseEstimatorV2,
    BackendSamplerV2,
    BackendEstimatorV2,
)
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import (
    Operator,
    Statevector,
    DensityMatrix,
    average_gate_fidelity,
    state_fidelity,
    SuperOp,
)
from qiskit.transpiler import (
    CouplingMap,
    InstructionProperties,
    Target,
)

from qiskit.providers import (
    BackendV1,
    Backend,
    BackendV2,
    Options as AerOptions,
    QiskitBackendNotFoundError,
)
from qiskit_ibm_runtime.fake_provider import FakeProvider, FakeProviderForBackendV2
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2, FakeBackend
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_ibm_runtime import (
    Session,
    IBMBackend as RuntimeBackend,
    EstimatorV2 as RuntimeEstimatorV2,
    Options as RuntimeOptions,
    EstimatorOptions as RuntimeEstimatorOptions,
    SamplerV2 as RuntimeSamplerV2,
    QiskitRuntimeService,
)

from qiskit_dynamics import Solver, RotatingFrame, ArrayLike
from qiskit_dynamics.backend.backend_string_parser.hamiltonian_string_parser import (
    parse_backend_hamiltonian_dict,
)
from qiskit_dynamics.backend.dynamics_backend import (
    _get_backend_channel_freqs,
    DynamicsBackend,
)

from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import BatchExperiment, BaseAnalysis
from qiskit_experiments.library import (
    StateTomography,
    ProcessTomography,
    RoughXSXAmplitudeCal,
    RoughDragCal,
)
from qiskit_experiments.calibration_management.basis_gate_library import (
    FixedFrequencyTransmon,
    EchoedCrossResonance,
)

from itertools import chain
from typing import Optional, Tuple, List, Union, Dict, Sequence, Callable, Any
import yaml

import numpy as np

from gymnasium.spaces import Box
import optuna

from scipy.optimize import minimize, OptimizeResult

import keyword
import re

from .custom_jax_sim.pulse_estimator_v2 import PulseEstimatorV2
from .qconfig import (
    BackendConfig,
    ExecutionConfig,
    BenchmarkConfig,
    StateConfig,
    QiskitConfig,
    QEnvConfig,
    CAFEConfig,
    ChannelConfig,
    XEBConfig,
    ORBITConfig,
    FidelityConfig,
)
from .custom_jax_sim import (
    JaxSolver,
    DynamicsBackendEstimator,
    PauliToQuditOperator,
)

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

Estimator_type = Union[
    RuntimeEstimatorV2,
    Estimator,
    BackendEstimator,
    BackendEstimatorV2,
    DynamicsBackendEstimator,
    StatevectorEstimator,
]
Sampler_type = Union[
    RuntimeSamplerV2,
    Sampler,
    BackendSampler,
    BackendSamplerV2,
    StatevectorSampler,
]
Backend_type = Union[BackendV1, BackendV2]

reward_configs = {
    "channel": ChannelConfig,
    "xeb": XEBConfig,
    "orbit": ORBITConfig,
    "state": StateConfig,
    "cafe": CAFEConfig,
    "fidelity": FidelityConfig,
}


def to_python_identifier(s):
    # Prepend underscore if the string starts with a digit
    if s[0].isdigit():
        s = "_" + s

    # Replace non-alphanumeric characters with underscore
    s = re.sub("\W|^(?=\d)", "_", s)

    # Append underscore if the string is a Python keyword
    if keyword.iskeyword(s):
        s += "_"

    return s


def count_gates(qc: QuantumCircuit):
    """
    Count number of gates in a Quantum Circuit
    """
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            if not isinstance(gate.operation, Delay):
                gate_count[qubit] += 1
    return gate_count


def remove_unused_wires(qc: QuantumCircuit):
    """
    Remove unused wires from a Quantum Circuit
    """
    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if count == 0:
            for instr in qc.data:
                if qubit in instr.qubits:
                    qc.data.remove(instr)
            qc.qubits.remove(qubit)
    return qc


def get_instruction_timings(circuit: QuantumCircuit):
    # Initialize the timings for each qubit
    qubit_timings = {i: 0 for i in range(circuit.num_qubits)}

    # Initialize the list of start times
    start_times = []

    # Loop over each instruction in the circuit
    for inst, qubits, _ in circuit.data:
        # Find the maximum time among the qubits involved in the instruction
        qubit_indices = [circuit.qubits.index(qubit) for qubit in qubits]
        start_time = max(qubit_timings[i] for i in qubit_indices)

        # Add the start time to the list of start times
        start_times.append(start_time)

        # Update the time for each qubit involved in the instruction
        for i in qubit_indices:
            qubit_timings[i] = start_time + 1

    return start_times


def perform_standard_calibrations(
    backend: DynamicsBackend, calibration_files: Optional[str] = None
):
    """
    Generate baseline single qubit gates (X, SX, RZ, H) for all qubits using traditional calibration experiments
    :param backend: Dynamics Backend on which calibrations should be run
    :param calibration_files: Optional calibration files containing single qubit gate calibrations for provided
        DynamicsBackend instance (Qiskit Experiments does not support this feature yet)

    """
    if not isinstance(backend, DynamicsBackend):
        raise TypeError(
            "Backend must be a DynamicsBackend instance (given: {type(backend)})"
        )

    target, qubits, dt = backend.target, range(backend.num_qubits), backend.dt
    num_qubits = len(qubits)
    single_qubit_properties = {(qubit,): None for qubit in qubits}
    single_qubit_errors = {(qubit,): 0.0 for qubit in qubits}
    two_qubit_properties = None

    control_channel_map = backend.options.control_channel_map
    coupling_map = None
    physical_control_channel_map = None
    if num_qubits > 1:
        if control_channel_map is not None:
            physical_control_channel_map = {
                (qubit_pair[0], qubit_pair[1]): backend.control_channel(
                    (qubit_pair[0], qubit_pair[1])
                )
                for qubit_pair in control_channel_map
            }
        else:
            all_to_all_connectivity = CouplingMap.from_full(num_qubits).get_edges()
            control_channel_map = {
                (q[0], q[1]): index for index, q in enumerate(all_to_all_connectivity)
            }
            physical_control_channel_map = {
                (q[0], q[1]): [pulse.ControlChannel(index)]
                for index, q in enumerate(all_to_all_connectivity)
            }
        backend.set_options(control_channel_map=control_channel_map)
        coupling_map = [list(qubit_pair) for qubit_pair in control_channel_map]
        two_qubit_properties = {qubits: None for qubits in control_channel_map}

    standard_gates: Dict[str, Gate] = gate_map()  # standard gate library
    fixed_phase_gates, fixed_phases = ["z", "s", "sdg", "t", "tdg"], np.pi * np.array(
        [1, 0.5, -0.5, 0.25, -0.25]
    )
    other_gates = ["rz", "id", "h", "x", "sx", "reset", "delay"]
    single_qubit_gates = fixed_phase_gates + other_gates
    two_qubit_gates = ["ecr"]
    exp_results = {}
    existing_cals = calibration_files is not None

    phi: Parameter = standard_gates["rz"].params[0]
    if existing_cals:
        cals = Calibrations.load(calibration_files)
    else:
        cals = Calibrations(
            coupling_map=coupling_map,
            control_channel_map=physical_control_channel_map,
            libraries=(
                [
                    FixedFrequencyTransmon(basis_gates=["x", "sx"]),
                    EchoedCrossResonance(basis_gates=["cr45p", "cr45m", "ecr"]),
                ]
                if num_qubits > 1
                else [FixedFrequencyTransmon(basis_gates=["x", "sx"])]
            ),
            backend_name=backend.name,
            backend_version=backend.backend_version,
        )
    if (
        len(target.instruction_schedule_map().instructions) <= 1
    ):  # Check if instructions have already been added
        for gate in single_qubit_gates:
            target.add_instruction(
                standard_gates[gate], properties=single_qubit_properties
            )
        if num_qubits > 1:
            for gate in two_qubit_gates:
                target.add_instruction(
                    standard_gates[gate], properties=two_qubit_properties
                )
            backend._coupling_map = target.build_coupling_map(two_qubit_gates[0])

    for qubit in qubits:  # Add calibrations for each qubit
        control_channels = (
            list(
                filter(
                    lambda x: x is not None,
                    [control_channel_map.get((i, qubit), None) for i in qubits],
                )
            )
            if num_qubits > 1
            else []
        )
        # Calibration of RZ gate, virtual Z-rotation
        with pulse.build(backend, name=f"rz{qubit}") as rz_cal:
            pulse.shift_phase(-phi, pulse.DriveChannel(qubit))
            for q in control_channels:
                pulse.shift_phase(-phi, pulse.ControlChannel(q))
        # Identity gate
        id_cal = pulse.Schedule(
            pulse.Delay(20, pulse.DriveChannel(qubit))
        )  # Wait 20 cycles for identity gate

        delay_param = standard_gates["delay"].params[0]
        with pulse.build(backend, name=f"delay{qubit}") as delay_cal:
            pulse.delay(delay_param, pulse.DriveChannel(qubit))

        # Update backend Target by adding calibrations for all phase gates (fixed angle virtual Z-rotations)
        for name, cal, duration in zip(
            ["rz", "id", "delay", "reset"],
            [rz_cal, id_cal, delay_cal, id_cal],
            [0, 20 * dt, None, 1000 * dt],
        ):
            target.update_instruction_properties(
                name, (qubit,), InstructionProperties(duration, 0.0, cal)
            )

        for phase, gate in zip(fixed_phases, fixed_phase_gates):
            gate_cal = rz_cal.assign_parameters({phi: phase}, inplace=False)
            instruction_prop = InstructionProperties(
                gate_cal.duration * dt, 0.0, gate_cal
            )
            target.update_instruction_properties(gate, (qubit,), instruction_prop)

        # Perform calibration experiments (Rabi/Drag) for calibrating X and SX gates
        if not existing_cals and backend.options.subsystem_dims[qubit] > 1:
            rabi_exp = RoughXSXAmplitudeCal(
                [qubit], cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 100)
            )
            drag_exp = RoughDragCal(
                [qubit], cals, backend=backend, betas=np.linspace(-20, 20, 15)
            )
            drag_exp.set_experiment_options(reps=[3, 5, 7])
            print(f"Starting Rabi experiment for qubit {qubit}...")
            rabi_result = rabi_exp.run().block_for_results()
            print(f"Rabi experiment for qubit {qubit} done.")
            print(f"Starting Drag experiment for qubit {qubit}...")
            drag_result = drag_exp.run().block_for_results()
            print(f"Drag experiments done for qubit {qubit} done.")
            exp_results[qubit] = [rabi_result, drag_result]

        # Build Hadamard gate schedule from following equivalence: H = S @ SX @ S
        sx_schedule = cals.get_schedule("sx", (qubit,))
        s_schedule = target.get_calibration("s", (qubit,))
        with pulse.build(backend, name="h") as h_schedule:
            pulse.call(s_schedule)
            pulse.call(sx_schedule)
            pulse.call(s_schedule)

        target.update_instruction_properties(
            "h",
            (qubit,),
            properties=InstructionProperties(h_schedule.duration * dt, 0.0, h_schedule),
        )
        measure_cal = target.get_calibration("measure", (qubit,))
        target.update_instruction_properties(
            "measure", (qubit,), InstructionProperties(1000 * dt, 0.0, measure_cal)
        )

    print("All single qubit calibrations are done")
    if calibration_files is None:
        cals.save(overwrite=True, file_prefix="Custom" + backend.name)
    error_dict = {"x": single_qubit_errors, "sx": single_qubit_errors}
    target.update_from_instruction_schedule_map(
        cals.get_inst_map(), error_dict=error_dict
    )
    # for qubit_pair in control_channel_map:
    #     print(qubit_pair)
    #     cr_ham_exp = CrossResonanceHamiltonian(physical_qubits=qubit_pair, flat_top_widths=np.linspace(0, 5000, 17),
    #                                            backend=backend)
    #     print("Calibrating CR for qubits", qubit_pair, "...")
    #     data_cr = cr_ham_exp.run().block_for_results()
    #     exp_results[qubit_pair] = data_cr

    print("Updated Instruction Schedule Map", target.instruction_schedule_map())

    return cals, exp_results


def add_ecr_gate(
    backend: BackendV2, basis_gates: Optional[List[str]] = None, coupling_map=None
):
    """
    Add ECR gate to basis gates if not present
    :param backend: Backend instance
    :param basis_gates: Basis gates of the backend
    :param coupling_map: Coupling map of the backend
    """
    if "ecr" not in basis_gates and backend.num_qubits > 1:
        target = backend.target
        target.add_instruction(
            gate_map()["ecr"],
            properties={qubits: None for qubits in coupling_map.get_edges()},
        )
        cals = Calibrations.from_backend(
            backend,
            [
                FixedFrequencyTransmon(["x", "sx"]),
                EchoedCrossResonance(["cr45p", "cr45m", "ecr"]),
            ],
            add_parameter_defaults=True,
        )

        for qubit_pair in coupling_map.get_edges():
            if target.has_calibration("cx", qubit_pair):
                default_params, _, _ = get_ecr_params(backend, qubit_pair)
                error = backend.target["cx"][qubit_pair].error
                target.update_instruction_properties(
                    "ecr",
                    qubit_pair,
                    InstructionProperties(
                        error=error,
                        calibration=cals.get_schedule(
                            "ecr", qubit_pair, default_params
                        ),
                    ),
                )
        basis_gates.append("ecr")
        for i, gate in enumerate(basis_gates):
            if gate == "cx":
                basis_gates.pop(i)
        # raise ValueError("Backend must carry 'ecr' as basis_gate for transpilation, will change in the
        # future")


def get_ecr_params(backend: Backend_type, physical_qubits: Sequence[int]):
    """
    Determine default parameters for ECR gate on provided backend (works even if basis gate of the IBM Backend is CX)

    Args:
        backend: Backend instance
        physical_qubits: Physical qubits on which ECR gate is to be performed
    Returns:
        default_params: Default parameters for ECR gate
        pulse_features: Features of the pulse
        basis_gate_instructions: Instructions for the basis gate
        instructions_array: Array of instructions for the basis gate
    """
    if not isinstance(backend, (BackendV1, BackendV2)):
        raise TypeError("Backend must be defined")
    basis_gates = (
        backend.configuration().basis_gates
        if isinstance(backend, BackendV1)
        else backend.operation_names
    )
    if "cx" in basis_gates:
        basis_gate = "cx"
    elif "ecr" in basis_gates:
        basis_gate = "ecr"
    else:
        raise ValueError("No identifiable two-qubit gate found, must be 'cx' or 'ecr'")
    if isinstance(backend, BackendV1):
        instruction_schedule_map = backend.defaults().instruction_schedule_map
    else:
        instruction_schedule_map = backend.target.instruction_schedule_map()

    q_c, q_t = (physical_qubits[0],), (physical_qubits[1],)
    physical_qubits = tuple(physical_qubits)
    basis_gate_instructions = instruction_schedule_map.get(
        basis_gate, qubits=physical_qubits
    )
    instructions_array = np.array(basis_gate_instructions.instructions)[:, 1]
    control_pulse = target_pulse = x_pulse = None

    if isinstance(backend, DynamicsBackend):
        x_pulse = instruction_schedule_map.get("x", q_c).instructions[0][1].pulse
        cr45p_instructions = np.array(
            instruction_schedule_map.get("cr45p", physical_qubits).instructions
        )[:, 1]
        for op in cr45p_instructions:
            if isinstance(op.channel, pulse.DriveChannel):
                target_pulse = op.pulse
            elif isinstance(op.channel, pulse.ControlChannel):
                control_pulse = op.pulse

    else:
        for instruction in list(instructions_array):
            if bool(x_pulse and target_pulse and control_pulse):
                break
            name = str(instruction.name)
            if "Xp_d" in name:
                x_pulse = instruction.pulse
                continue
            elif "CR90p_d" in name:
                target_pulse = instruction.pulse
                continue
            elif "CR90p_u" in name:
                control_pulse = instruction.pulse
                continue
            elif "CX_u" in name:
                control_pulse = instruction.pulse
                continue
            elif "CX_d" in name:
                target_pulse = instruction.pulse

        if x_pulse is None:
            x_pulse = instruction_schedule_map.get("x", q_c).instructions[0][1].pulse
    default_params = {
        ("amp", q_c, "x"): x_pulse.amp,
        ("σ", q_c, "x"): x_pulse.sigma,
        ("β", q_c, "x"): x_pulse.beta,
        ("duration", q_c, "x"): x_pulse.duration,
        ("angle", q_c, "x"): x_pulse.angle,
    }
    for sched in ["cr45p", "cr45m"]:
        rise_fall = (control_pulse.duration - control_pulse.width) / (
            2 * control_pulse.sigma
        )
        default_params.update(
            {
                ("amp", physical_qubits, sched): control_pulse.amp,
                ("tgt_amp", physical_qubits, sched): (
                    target_pulse.amp
                    if hasattr(target_pulse, "amp")
                    else np.linalg.norm(np.max(target_pulse.samples))
                ),
                ("angle", physical_qubits, sched): control_pulse.angle,
                ("tgt_angle", physical_qubits, sched): (
                    target_pulse.angle
                    if hasattr(target_pulse, "angle")
                    else np.angle(np.max(target_pulse.samples))
                ),
                ("duration", physical_qubits, sched): control_pulse.duration,
                ("σ", physical_qubits, sched): control_pulse.sigma,
                ("risefall", physical_qubits, sched): rise_fall,
            }
        )
    pulse_features = [
        "amp",
        "angle",
        "duration",
        "σ",
        "β",
        "risefall",
        "tgt_amp",
        "tgt_angle",
    ]
    return default_params, pulse_features, basis_gate_instructions, instructions_array


def get_pulse_params(
    backend: Backend_type, physical_qubit: Sequence[int], gate_name: str = "x"
):
    """
    Determine default parameters for SX or X gate on provided backend

    Args:
        backend: Backend instance
        physical_qubit: Physical qubit on which gate is to be performed
        gate_name: Name of the gate (X or SX)
    Returns:
        default_params: Default parameters for X or SX gate
        pulse_features: Features of the pulse
        basis_gate_instructions: Instructions for the basis gate
        instructions_array: Array of instructions for the basis gate
    """
    if not isinstance(backend, (BackendV1, BackendV2)):
        raise TypeError("Backend must be defined")
    if isinstance(backend, BackendV1):
        instruction_schedule_map = backend.defaults().instruction_schedule_map
    else:
        instruction_schedule_map = backend.target.instruction_schedule_map()
    basis_gate_inst = instruction_schedule_map.get(gate_name, physical_qubit)
    basis_gate_instructions = np.array(basis_gate_inst.instructions)[:, 1]
    ref_pulse = basis_gate_inst.instructions[0][1].pulse
    default_params = {
        ("amp", physical_qubit, gate_name): ref_pulse.amp,
        ("σ", physical_qubit, gate_name): ref_pulse.sigma,
        ("β", physical_qubit, gate_name): ref_pulse.beta,
        ("duration", physical_qubit, gate_name): ref_pulse.duration,
        ("angle", physical_qubit, gate_name): ref_pulse.angle,
    }
    pulse_features = ["amp", "angle", "duration", "σ", "β"]
    return default_params, pulse_features, basis_gate_inst, basis_gate_instructions


def new_params_ecr(
    params: ParameterVector,
    qubits: Sequence[int],
    backend: BackendV1 | BackendV2,
    pulse_features: List[str],
    keep_symmetry: bool = True,
    duration_window: float = 0.1,
    include_baseline: bool = False,
):
    """
    Helper function to parametrize a custom ECR gate using Qiskit Experiments Calibrations syntax
    :param params: Parameters of the Schedule/Custom gate
    :param qubits: Physical qubits on which custom gate is applied on
    :param backend: IBM Backend on which schedule shall be added
    :param pulse_features: List of pulse features to be parametrized
    :param keep_symmetry: Choose if the two parts of the ECR tone shall be jointly parametrized or not
    :param duration_window: Duration window for the pulse duration
    :param include_baseline: Include baseline calibration in the parameters
    :return: Dictionary of updated ECR parameters
    """
    qubits = tuple(qubits)
    new_params, available_features, _, _ = get_ecr_params(backend, qubits)

    if keep_symmetry:  # Maintain symmetry between the two GaussianSquare pulses
        if len(pulse_features) != len(params):
            raise ValueError(
                f"Number of pulse features ({len(pulse_features)}) and number of parameters ({len(params)})"
                f" do not match"
            )
        for sched in ["cr45p", "cr45m"]:
            for i, feature in enumerate(pulse_features):
                if feature != "duration" and feature in available_features:
                    if (
                        include_baseline
                    ):  # Add the parameter to the pulse baseline calibration
                        new_params[(feature, qubits, sched)] += params[i]
                    else:  # Replace baseline calibration with the parameter
                        new_params[(feature, qubits, sched)] = 0.0 + params[i]

                else:
                    if include_baseline:
                        new_params[(feature, qubits, sched)] += (
                            duration_window * params[i]
                        )
                    else:
                        new_params[(feature, qubits, sched)] = (
                            duration_window * params[i]
                        )

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
                    new_params[(feature, qubits, sched)] += (
                        duration_window * params[i * num_features + j]
                    )

    return new_params


def new_params_sq_gate(
    params: ParameterVector,
    qubits: Sequence[int],
    backend: BackendV1 | BackendV2,
    pulse_features: List[str],
    duration_window: float,
    include_baseline: bool = False,
    gate_name: str = "x",
):
    """
    Helper function to parametrize a custom X or SX gate using Qiskit Experiments Calibrations syntax
    :param params: Parameters of the Schedule/Custom gate
    :param qubits: Physical qubits on which custom gate is applied on
    :param backend: IBM Backend on which schedule shall be added
    :param pulse_features: List of pulse features to be parametrized
    :param duration_window: Duration window for the pulse duration
    :param include_baseline: Include baseline calibration in the parameters
    :param gate_name: Name of the gate ('x' or 'sx')
    :return: Dictionary of updated X parameters
    """
    new_params, available_features, _, _ = get_pulse_params(backend, qubits, gate_name)
    if len(pulse_features) != len(params):
        raise ValueError(
            f"Number of pulse features ({len(pulse_features)}) and number of parameters ({len(params)}"
            f" do not match"
        )
    for i, feature in enumerate(pulse_features):
        if feature != "duration" and feature in available_features:
            if include_baseline:  # Add the parameter to the pulse baseline calibration
                new_params[(feature, qubits, gate_name)] += params[i]
            else:  # Replace baseline calibration with the parameter
                new_params[(feature, qubits, gate_name)] = 0.0 + params[i]

        else:
            if include_baseline:
                new_params[(feature, qubits, gate_name)] += duration_window * params[i]
            else:
                new_params[(feature, qubits, gate_name)] = duration_window * params[i]

    return new_params


def simulate_pulse_schedule(
    solver_instance: DynamicsBackend | Solver | JaxSolver,
    sched: pulse.Schedule | pulse.ScheduleBlock,
    solver_options: Optional[Dict] = None,
    target_unitary: Optional[Operator] = None,
    initial_state: Optional[Statevector | DensityMatrix] = None,
    target_state: Optional[Statevector | DensityMatrix] = None,
    normalize: bool = True,
) -> Dict[str, Union[Operator, Statevector, float]]:
    """
    Simulate pulse schedule on provided backend

    :param solver_instance: DynamicsBackend or Solver instance
    :param sched: Pulse schedule to simulate
    :param solver_options: Optional solver options
    :param target_unitary: Optional target unitary for gate fidelity calculation
    :param initial_state: Optional initial state for state fidelity calculation  (if None and target_state is not None,
    then initial state is assumed to be |0..0>)
    :param target_state: Optional target state for state fidelity calculation
    :param normalize: Normalize the projected statevector or not
    :return: Dictionary containing simulated unitary, statevector, projected unitary, projected statevector, gate fidelity, state fidelity
    """

    if isinstance(solver_instance, DynamicsBackend):
        solver = solver_instance.options.solver
        solver_options = solver_instance.options.solver_options
        dt = solver_instance.dt
        subsystem_dims = list(
            filter(lambda x: x > 1, solver_instance.options.subsystem_dims)
        )
    elif isinstance(solver_instance, (Solver, JaxSolver)):
        solver = solver_instance
        dt = solver._dt
        subsystem_dims = solver.model.dim
    else:
        raise TypeError(
            "Solver instance must be defined. Backend is not DynamicsBackend or Solver instance"
        )

    results = solver.solve(
        t_span=[0, sched.duration * dt],
        y0=np.eye(solver.model.dim),
        signals=sched,
        **solver_options,
    )

    output_unitary = np.array(results.y[-1])

    output_op = Operator(
        output_unitary,
        input_dims=tuple(subsystem_dims),
        output_dims=tuple(subsystem_dims),
    )
    projected_unitary = qubit_projection(output_unitary, subsystem_dims)

    initial_state = (
        DensityMatrix.from_int(0, subsystem_dims)
        if initial_state is None
        else initial_state
    )

    final_state = initial_state.evolve(output_op)
    projected_statevec = projected_state(final_state, subsystem_dims, normalize)
    rotated_state = None

    final_results = {
        "unitary": output_op,
        "state": final_state,
        "projected_unitary": projected_unitary,
        "projected_state": projected_statevec,
    }
    if target_unitary is not None:
        optimal_rots = get_optimal_z_rotation(
            projected_unitary, target_unitary, len(subsystem_dims)
        )
        rotated_unitary = rotate_unitary(optimal_rots.x, projected_unitary)
        try:
            rotated_state = initial_state.evolve(rotated_unitary)
        except QiskitError:
            pass

        gate_fid = average_gate_fidelity(projected_unitary, target_unitary)
        optimal_gate_fid = average_gate_fidelity(rotated_unitary, target_unitary)
        final_results["gate_fidelity"] = {
            "raw": gate_fid,
            "optimal": optimal_gate_fid,
            "rotations": optimal_rots.x,
            "rotated_unitary": rotated_unitary,
        }

    if target_state is not None:
        state_fid1 = state_fidelity(projected_statevec, target_state, validate=False)
        state_fid2 = None
        if rotated_state is not None:
            state_fid2 = state_fidelity(rotated_state, target_state, validate=False)
        final_results["state_fidelity"] = {
            "raw": state_fid1,
            "optimal": state_fid2,
            "rotated_state": rotated_state,
        }
    return final_results


def density_matrix_to_statevector(density_matrix: DensityMatrix):
    """
    Convert a density matrix to a statevector (if the density matrix represents a pure state)

    Args:
        density_matrix: DensityMatrix object representing the pure state

    Returns:
        Statevector: Statevector object representing the pure state

    Raises:
        ValueError: If the density matrix does not represent a pure state
    """
    # Check if the state is pure by examining if Tr(rho^2) is 1
    if np.isclose(density_matrix.purity(), 1):
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix.data)

        # Find the eigenvector corresponding to the eigenvalue 1 (pure state)
        # The statevector is the eigenvector corresponding to the maximum eigenvalue
        max_eigenvalue_index = np.argmax(eigenvalues)
        statevector = eigenvectors[:, max_eigenvalue_index]

        # Return the Statevector object
        return Statevector(statevector)
    else:
        raise ValueError("The density matrix does not represent a pure state.")


def run_jobs(session: Session, circuits: List[QuantumCircuit], **run_options):
    """
    Run batch of Quantum Circuits on provided backend

    Args:
        session: Runtime session
        circuits: List of Quantum Circuits
        run_options: Optional run options
    """
    jobs = []
    runtime_inputs = {"circuits": circuits, "skip_transpilation": True, **run_options}
    jobs.append(session.run("circuit_runner", inputs=runtime_inputs))

    return jobs


def fidelity_from_tomography(
    qc_input: List[QuantumCircuit] | QuantumCircuit,
    backend: Optional[Backend],
    target: Operator | QuantumState,
    physical_qubits: Optional[Sequence[int]],
    analysis: Union[BaseAnalysis, None, str] = "default",
    sampler: RuntimeSamplerV2 = None,
    shots: int = 8192,
):
    """
    Extract average state or gate fidelity from batch of Quantum Circuit for target state or gate

    Args:
        qc_input: Quantum Circuit input to benchmark
        backend: Backend instance
        physical_qubits: Physical qubits on which state or process tomography is to be performed
        analysis: Analysis instance
        target: Target state or gate for fidelity calculation (must be either Operator or QuantumState)
        sampler: Runtime Sampler
    Returns:
        avg_fidelity: Average state or gate fidelity (over the batch of Quantum Circuits)
    """
    if isinstance(qc_input, QuantumCircuit):
        qc_input = [qc_input]
    if isinstance(target, Operator):
        tomo = ProcessTomography
        fidelity = "process_fidelity"
    elif isinstance(target, QuantumState):
        tomo = StateTomography
        fidelity = "state_fidelity"
    else:
        raise TypeError("Target must be either Operator or QuantumState")

    process_tomo = BatchExperiment(
        [
            tomo(
                qc,
                physical_qubits=physical_qubits,
                analysis=analysis,
                target=target,
            )
            for qc in qc_input
        ],
        backend=backend,
        flatten_results=True,
    )

    if isinstance(backend, RuntimeBackend):
        circuits = process_tomo._transpiled_circuits()
        jobs = sampler.run([(circ,) for circ in circuits])
        exp_data = process_tomo._initialize_experiment_data()
        exp_data.add_data()
        results = process_tomo.analysis.run(exp_data).block_for_results()
    else:
        results = process_tomo.run(shots=shots).block_for_results()

    if len(qc_input) == 1:
        process_results = [results.analysis_results(fidelity).value]
    else:
        process_results = [
            results.analysis_results(fidelity)[i].value for i in range(len(qc_input))
        ]
    if isinstance(target, Operator) and target.is_unitary():
        dim, _ = target.dim
        avg_gate_fids = [(dim * f_pro + 1) / (dim + 1) for f_pro in process_results]

        return avg_gate_fids if len(avg_gate_fids) > 1 else avg_gate_fids[0]
    else:  # target is QuantumState
        return process_results


def get_control_channel_map(backend: BackendV1, qubit_tgt_register: List[int]):
    """
    Get reduced control_channel_map from Backend configuration (needs to be of type BackendV1)
    :param backend: IBM Backend instance, must carry a configuration method
    :param qubit_tgt_register: Subsystem of interest from which to build control_channel_map

    Returns:
    control_channel_map: Reduced control channel map for the qubit_tgt_register
    """
    control_channel_map = {}
    control_channel_map_backend = {
        qubits: backend.configuration().control_channels[qubits][0].index
        for qubits in backend.configuration().control_channels
    }
    for qubits in control_channel_map_backend:
        if qubits[0] in qubit_tgt_register and qubits[1] in qubit_tgt_register:
            control_channel_map[qubits] = control_channel_map_backend[qubits]
    return control_channel_map


def retrieve_primitives(
    backend: Backend_type,
    config: BackendConfig,
    estimator_options: Optional[
        Dict | AerOptions | RuntimeOptions | RuntimeEstimatorOptions
    ] = None,
    circuit: Optional[QuantumCircuit] = None,
) -> Tuple[Estimator_type, Sampler_type]:
    """
    Retrieve appropriate Qiskit primitives (estimator and sampler) from backend and layout

    Args:
        backend: Backend instance
        config: Configuration dictionary
        estimator_options: Estimator options
        circuit: QuantumCircuit instance implementing the custom gate (for DynamicsBackend)
    """
    if isinstance(backend, DynamicsBackend):
        assert isinstance(config, QiskitConfig), "Configuration must be a QiskitConfig"
        estimator = PulseEstimatorV2(backend=backend, options=estimator_options)
        sampler = BackendSamplerV2(backend=backend)

        if config.do_calibrations and not backend.target.has_calibration("x", (0,)):
            calibration_files = config.calibration_files
            _, _ = perform_standard_calibrations(backend, calibration_files)
    elif isinstance(backend, (FakeBackend, FakeBackendV2, AerBackend)):
        from qiskit_aer.primitives import (
            EstimatorV2 as AerEstimatorV2,
            SamplerV2 as AerSamplerV2,
        )

        estimator = AerEstimatorV2.from_backend(backend=backend)
        sampler = AerSamplerV2.from_backend(backend=backend)
    elif backend is None:  # No backend specified, ideal state-vector simulation
        sampler = StatevectorSampler()
        estimator = StatevectorEstimator()

    elif isinstance(backend, RuntimeBackend):
        estimator = RuntimeEstimatorV2(
            mode=Session(backend=backend),
            options=estimator_options,
        )
        sampler = RuntimeSamplerV2(mode=estimator.session)

    # elif isinstance(backend, QMBackend):
    #     estimator = QMEstimator(backend=backend, options=estimator_options)
    #     sampler = QMSampler(backend=backend)
    else:
        estimator = BackendEstimatorV2(backend=backend, options=estimator_options)
        sampler = BackendSamplerV2(backend=backend)

    return estimator, sampler


def substitute_target_gate(
    circuit: QuantumCircuit,
    target_gate: Gate,
    custom_gate: Gate,
):
    """
    Substitute target gate in Quantum Circuit with a parametrized version of the gate.
    The parametrized_circuit function signature should match the expected one for a QiskitConfig instance.

    Args:
        circuit: Quantum Circuit instance
        target_gate: Target gate to be substituted
        custom_gate: Custom gate to be substituted with
    """
    ref_label = target_gate.label
    qc = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.label != ref_label:
            qc.append(instruction)
        else:
            qc.append(custom_gate, instruction.qubits)
    return qc


def handle_session(
    estimator: BaseEstimatorV1 | BaseEstimatorV2,
    backend: Backend_type,
    counter: Optional[int] = None,
    qc: Optional[QuantumCircuit] = None,
    input_state_circ: Optional[QuantumCircuit] = None,
):
    """
    Handle session reopening for RuntimeEstimator or load necessary data for custom DynamicsBackendEstimator
    Args:
        estimator: Estimator instance
        backend: Backend instance
        counter: Optional session counter (for RuntimeEstimator) or circuit macro counter (for DynamicsBackendEstimator)
        qc: Optional QuantumCircuit instance (for DynamicsBackendEstimator)
        input_state_circ: Optional input state QuantumCircuit instance (for DynamicsBackendEstimator)

    Returns:
        Updated Estimator instance
    """
    if (
        isinstance(estimator, RuntimeEstimatorV2)
        and estimator.mode.status() == "Closed"
    ):
        old_session = estimator.mode
        counter += 1
        print(f"New Session opened (#{counter})")
        session, options = (
            Session(old_session.service, backend),
            estimator.options,
        )
        estimator = type(estimator)(mode=session, options=options)

    elif isinstance(estimator, DynamicsBackendEstimator):
        if not isinstance(backend, DynamicsBackend) or not isinstance(
            backend.options.solver, JaxSolver
        ):
            raise TypeError(
                "DynamicsBackendEstimator can only be used with DynamicsBackend and JaxSolver"
            )
        # Update callable within the jit compiled function
        if counter != backend.options.solver.circuit_macro_counter:
            backend.options.solver.circuit_macro_counter = counter
            backend.options.solver.circuit_macro = lambda: schedule(qc, backend)

        # Update initial state of DynamicsBackend with input state circuit
        # The initial state is adapted to match the dimensions of the HamiltonianModel
        new_circ = transpile(input_state_circ, backend)
        subsystem_dims = backend.options.subsystem_dims
        initial_state = Statevector.from_int(
            0, dims=tuple(filter(lambda x: x > 1, subsystem_dims))
        )
        initial_rotations = [
            Operator.from_label("I") for _ in range(new_circ.num_qubits)
        ]
        qubit_counter, qubit_list = 0, []
        for instruction in new_circ.data:
            assert (
                len(instruction.qubits) == 1
            ), "Input state circuit must be in a tensor product form"
            if instruction.qubits[0] not in qubit_list:
                qubit_list.append(instruction.qubits[0])
                qubit_counter += 1
            initial_rotations[qubit_counter - 1] = initial_rotations[
                qubit_counter - 1
            ].compose(Operator(instruction.operation))

        operation = PauliToQuditOperator(initial_rotations, subsystem_dims)
        initial_state = initial_state.evolve(operation)
        backend.set_options(initial_state=initial_state)

    return estimator


def select_backend(
    real_backend: Optional[bool] = None,
    channel: Optional[str] = None,
    instance: Optional[str] = None,
    backend_name: Optional[str] = None,
    use_dynamics: Optional[bool] = None,
    physical_qubits: Optional[List[int]] = None,
    solver_options: Optional[Dict] = None,
    calibration_files: Optional[str] = None,
) -> Optional[BackendV2]:
    """
    Select backend to use for training among real backend or fake backend (Aer Simulator)

    Args:
        real_backend: Boolean indicating if real backend should be used
        channel: Channel to use for Runtime Service
        instance: Instance to use for Runtime Service
        backend_name: Name of the backend to use for training
        use_dynamics: Boolean indicating if DynamicsBackend should be used
        physical_qubits: Physical qubits on which DynamicsBackend should be used
        solver_options: Solver options for DynamicsBackend
        calibration_files: Calibration files for DynamicsBackend

    Returns:
        backend: Backend instance
    """

    backend = None
    if real_backend is not None:
        if real_backend:
            service = QiskitRuntimeService(channel=channel, instance=instance)
            if backend_name is None:
                backend = service.least_busy(
                    min_num_qubits=2, simulator=False, operational=True, open_pulse=True
                )
            else:
                backend = service.get_backend(backend_name)

            # Specify options below if needed
            # backend.set_options(**options)
        else:
            # Fake backend initialization (Aer Simulator)
            try:
                if not use_dynamics:
                    backend = FakeProviderForBackendV2().backend(
                        backend_name if backend_name is not None else "fake_jakarta"
                    )
                else:
                    backend = FakeProvider().get_backend(
                        backend_name if backend_name is not None else "fake_jakarta"
                    )
            except QiskitBackendNotFoundError:
                raise QiskitError(
                    "Backend not found. Please check the backend name and try again."
                )

    if backend is not None:
        if use_dynamics:
            solver_options = convert_solver_options(solver_options)
            assert isinstance(
                backend, BackendV1
            ), "DynamicsBackend can only be used with BackendV1 instances"
            backend = custom_dynamics_from_backend(
                backend,
                subsystem_list=list(physical_qubits),
                solver_options=solver_options,
                jax_solver=False,
            )
            _, _ = perform_standard_calibrations(
                backend, calibration_files=calibration_files
            )

    return backend


def convert_solver_options(
    solver_options: Optional[Dict], dt: Optional[float | int] = None
) -> Optional[Dict]:
    """
    Convert solver options passed from YAML to correct format
    """
    if solver_options["hmax"] == "auto" and dt is not None:
        solver_options["hmax"] = dt
    if solver_options["hmax"] == "auto" and dt is None:
        raise ValueError("dt must be specified for hmax='auto'")
    for key in ["atol", "rtol"]:
        solver_options[key] = float(solver_options[key])
    return solver_options


def has_noise_model(backend: AerBackend):
    """
    Check if Aer backend has noise model or not

    Args:
        backend: AerBackend instance
    """
    if (
        backend.options.noise_model is None
        or backend.options.noise_model.to_dict() == {}
        or len(backend.options.noise_model.to_dict()["errors"]) == 0
    ):
        return False
    else:
        return True


def custom_dynamics_from_backend(
    backend: BackendV1,
    subsystem_list: Optional[List[int]] = None,
    rotating_frame: Optional[Union[ArrayLike, RotatingFrame, str]] = "auto",
    array_library: Optional[str] = "jax",
    vectorized: Optional[bool] = None,
    rwa_cutoff_freq: Optional[float] = None,
    static_dissipators: Optional[ArrayLike] = None,
    dissipator_operators: Optional[ArrayLike] = None,
    dissipator_channels: Optional[List[str]] = None,
    jax_solver: Optional[bool] = True,
    **options,
) -> DynamicsBackend:
    """
    Method to retrieve custom DynamicsBackend instance from IBMBackend instance
    added with potential dissipation operators, inspired from DynamicsBackend.from_backend() method.
    Contrary to the original method, the Solver instance can be created with the custom JaxSolver
    tailormade for fast simulation with the Estimator primitive.

    :param backend: IBMBackend instance from which Hamiltonian parameters are extracted
    :param subsystem_list: The list of qubits in the backend to include in the model.
    :param rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`
    :param array_library: Array library to use for storing operators of underlying model. See the
        :ref:`model evaluation section of the Models API documentation <model evaluation>`
        for a more detailed description of this argument.
    :param vectorized: If including dissipator terms, whether or not to construct the
        :class:`.LindbladModel` in vectorized form. See the
        :ref:`model evaluation section of the Models API documentation <model evaluation>`
        for a more detailed description of this argument.
    :param rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
            ``"auto"``, allowing this method to pick a rotating frame.
    :param backend: IBMBackend instance from which Hamiltonian parameters are extracted
    :param static_dissipators: static_dissipators: Constant dissipation operators.
    :param dissipator_operators: Dissipation operators with time-dependent coefficients.
    :param dissipator_channels: List of channel names in pulse schedules corresponding to dissipator operators.
    :param jax_solver: Boolean indicating if the custom JaxSolver should be used
    :return: Solver instance carrying Hamiltonian information extracted from the IBMBackend instance
    """
    # get available target, config, and defaults objects
    backend_target = getattr(backend, "target", None)

    if not hasattr(backend, "configuration"):
        raise QiskitError(
            "DynamicsBackend.from_backend requires that the backend argument has a "
            "configuration method."
        )
    backend_config = backend.configuration()

    backend_defaults = None
    if hasattr(backend, "defaults"):
        backend_defaults = backend.defaults()

    # get and parse Hamiltonian string dictionary
    if backend_target is not None:
        backend_num_qubits = backend_target.num_qubits
    else:
        backend_num_qubits = backend_config.n_qubits

    if subsystem_list is not None:
        subsystem_list = sorted(subsystem_list)
        if subsystem_list[-1] >= backend_num_qubits:
            raise QiskitError(
                f"subsystem_list contained {subsystem_list[-1]}, which is out of bounds for "
                f"backend with {backend_num_qubits} qubits."
            )
    else:
        subsystem_list = list(range(backend_num_qubits))

    if backend_config.hamiltonian is None:
        raise QiskitError(
            "DynamicsBackend.from_backend requires that backend.configuration() has a "
            "hamiltonian."
        )

    (
        static_hamiltonian,
        hamiltonian_operators,
        hamiltonian_channels,
        subsystem_dims_dict,
    ) = parse_backend_hamiltonian_dict(backend_config.hamiltonian, subsystem_list)
    subsystem_dims = [
        subsystem_dims_dict.get(idx, 1) for idx in range(backend_num_qubits)
    ]

    # construct model frequencies dictionary from backend
    channel_freqs = _get_backend_channel_freqs(
        backend_target=backend_target,
        backend_config=backend_config,
        backend_defaults=backend_defaults,
        channels=hamiltonian_channels,
    )

    # Add control_channel_map from backend (only if not specified before by user)
    if "control_channel_map" not in options:
        if hasattr(backend, "control_channels"):
            control_channel_map_backend = {
                qubits: backend.control_channels[qubits][0].index
                for qubits in backend.control_channels
            }

        elif hasattr(backend.configuration(), "control_channels"):
            control_channel_map_backend = {
                qubits: backend.configuration().control_channels[qubits][0].index
                for qubits in backend.configuration().control_channels
            }

        else:
            control_channel_map_backend = {}

        # Reduce control_channel_map based on which channels are in the model
        if bool(control_channel_map_backend):
            control_channel_map = {}
            for label, idx in control_channel_map_backend.items():
                if f"u{idx}" in hamiltonian_channels:
                    control_channel_map[label] = idx
            options["control_channel_map"] = control_channel_map

    # build the solver
    if rotating_frame == "auto":
        if array_library is not None and "sparse" in array_library:
            rotating_frame = np.diag(static_hamiltonian)
        else:
            rotating_frame = static_hamiltonian

    # get time step size
    if backend_target is not None and backend_target.dt is not None:
        dt = backend_target.dt
    else:
        # config is guaranteed to have a dt
        dt = backend_config.dt
    if jax_solver:
        solver = JaxSolver(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            dt=dt,
            rotating_frame=rotating_frame,
            array_library=array_library,
            vectorized=vectorized,
            rwa_cutoff_freq=rwa_cutoff_freq,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
            dissipator_channels=dissipator_channels,
        )
    else:
        solver = Solver(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            dt=dt,
            rotating_frame=rotating_frame,
            array_library=array_library,
            vectorized=vectorized,
            rwa_cutoff_freq=rwa_cutoff_freq,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
            dissipator_channels=dissipator_channels,
        )

    return DynamicsBackend(
        solver=solver,
        target=Target(dt=dt),
        subsystem_dims=subsystem_dims,
        **options,
    )


def rotate_frame(yf: Any, tf: Any, backend: DynamicsBackend):
    """
    Rotate the frame of the state or operator to the lab frame

    Args:
        yf: State or operator in the frame of the rotating frame
        tf: Time at which the frame is rotated
        backend: DynamicsBackend object containing the backend information
    """
    # Take state out of frame, put in dressed basis, and normalize
    if isinstance(yf, Statevector):
        yf = np.array(
            backend.options.solver.model.rotating_frame.state_out_of_frame(t=tf, y=yf)
        )
        yf = backend._dressed_states_adjoint @ yf
        yf = Statevector(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.linalg.norm(yf.data)
    elif isinstance(yf, DensityMatrix):
        yf = np.array(
            backend.options.solver.model.rotating_frame.operator_out_of_frame(
                t=tf, operator=yf
            )
        )
        yf = backend._dressed_states_adjoint @ yf @ backend._dressed_states
        yf = DensityMatrix(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.diag(yf.data).sum()
    elif isinstance(yf, Operator):
        yf = np.array(
            backend.options.solver.model.rotating_frame.operator_out_of_frame(
                t=tf, operator=yf
            )
        )
        yf = backend._dressed_states_adjoint @ yf @ backend._dressed_states
        yf = Operator(
            yf,
            input_dims=backend.options.subsystem_dims,
            output_dims=backend.options.subsystem_dims,
        )

    return yf


def build_qubit_space_projector(initial_subsystem_dims: list) -> Operator:
    """
    Build projector on qubit space from initial subsystem dimensions
    The returned operator is a non-square matrix mapping the qudit space to the qubit space.
    It can be applied to convert multi-qudit states/unitaries to multi-qubit states/unitaries.

    Args:
        initial_subsystem_dims: Initial subsystem dimensions

    Returns: Projector on qubit space as a Qiskit Operator object
    """
    total_dim = np.prod(initial_subsystem_dims)
    output_dims = (2,) * len(initial_subsystem_dims)
    total_qubit_dim = np.prod(output_dims)
    projector = Operator(
        np.zeros((total_qubit_dim, total_dim), dtype=np.complex128),
        input_dims=tuple(initial_subsystem_dims),
        output_dims=output_dims,
    )  # Projector initialized in the qudit space
    for i in range(
        total_dim
    ):  # Loop over all computational basis states in the qudit space
        s = Statevector.from_int(i, initial_subsystem_dims)  # Computational qudit state
        for key in s.to_dict().keys():  # Loop over all computational basis states
            if all(
                c in "01" for c in key
            ):  # Check if basis state is in the qubit space
                s_qubit = Statevector.from_label(key)  # Computational qubit state
                projector += Operator(
                    s_qubit.data.reshape(total_qubit_dim, 1)
                    @ s.data.reshape(total_dim, 1).conj().T,
                    input_dims=tuple(initial_subsystem_dims),
                    output_dims=output_dims,
                )  # Add |s_qubit><s_qudit| to projector
                break
            else:
                continue
    return projector


def projected_state(
    state: np.ndarray | Statevector | DensityMatrix,
    subsystem_dims: List[int],
    normalize: bool = True,
) -> Statevector | DensityMatrix:
    """
    Project statevector on qubit space

    Args:
        state: State, given as numpy array or QuantumState object
        subsystem_dims: Subsystem dimensions
        normalize: Normalize statevector
    """
    if not isinstance(state, (np.ndarray, QuantumState)):
        raise TypeError("State must be either numpy array or QuantumState object")
    proj = build_qubit_space_projector(
        subsystem_dims
    )  # Projector on qubit space (in qudit space)
    if isinstance(state, np.ndarray):
        state_type = DensityMatrix if state.ndim == 2 else Statevector
        output_state: Statevector | DensityMatrix = state_type(state)
    else:
        output_state: Statevector | DensityMatrix = state
    qubitized_state = output_state.evolve(proj)

    if (
        normalize
    ) and qubitized_state.trace() != 0:  # Normalize the projected state (which is for now unnormalized due to selection of components)
        qubitized_state = (
            qubitized_state / qubitized_state.trace()
            if isinstance(qubitized_state, DensityMatrix)
            else qubitized_state / np.linalg.norm(qubitized_state.data)
        )

    return qubitized_state


def qubit_projection(
    unitary: np.ndarray | Operator, subsystem_dims: List[int]
) -> Operator:
    """
    Project unitary on qubit space

    Args:
        unitary: Unitary, given as numpy array or Operator object
        subsystem_dims: Subsystem dimensions

    Returns: unitary projected on qubit space as a Qiskit Operator object
    """

    proj = build_qubit_space_projector(
        subsystem_dims
    )  # Projector on qubit space (in qudit space)
    unitary_op = (
        Operator(
            unitary, input_dims=tuple(subsystem_dims), output_dims=tuple(subsystem_dims)
        )
        if isinstance(unitary, np.ndarray)
        else unitary
    )  # Unitary operator (in qudit space)

    qubitized_op = (
        proj @ unitary_op @ proj.adjoint()
    )  # Projected unitary (in qubit space)
    # (Note that is actually not unitary at this point, it's a Channel on the multi-qubit system)
    return qubitized_op


def rotate_unitary(x, op: Operator) -> Operator:
    """
    Rotate input unitary with virtual Z rotations on all qubits
    x: Rotation parameters
    unitary: Rotated unitary
    """
    assert len(x) % 2 == 0, "Rotation parameters should be a pair"
    ops = [
        Operator(RZGate(x[i])) for i in range(len(x))
    ]  # Virtual Z rotations to be applied on all qubits
    pre_rot, post_rot = (
        ops[0],
        ops[-1],
    )  # Degrees of freedom before and after the unitary
    for i in range(1, len(x) // 2):  # Apply virtual Z rotations on all qubits
        pre_rot = pre_rot.expand(ops[i])
        post_rot = post_rot.tensor(ops[-i - 1])

    return pre_rot @ op @ post_rot


def get_optimal_z_rotation(
    unitary: Operator, target_gate: Gate | Operator, n_qubits: int
) -> OptimizeResult:
    """
    Get optimal Z rotation angles for input unitary to match target gate (minimize gate infidelity)
    Args:
        unitary: Unitary to be rotated
        target_gate: Target gate
        n_qubits: Number of qubits
    """

    def cost_function(x):
        rotated_unitary = rotate_unitary(x, unitary)
        return 1 - average_gate_fidelity(
            rotated_unitary,
            target_gate if isinstance(target_gate, Operator) else Operator(target_gate),
        )

    x0 = np.zeros(2**n_qubits)
    res = minimize(cost_function, x0, method="Nelder-Mead")
    return res


def load_q_env_from_yaml_file(file_path: str):
    """
    Load Qiskit Quantum Environment from yaml file

    Args:
        file_path: File path
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    if "ENV" not in config:
        raise KeyError("ENV section must be present in the configuration")
    elif "BACKEND" not in config:
        raise KeyError("BACKEND section must be present in the configuration")
    elif "TARGET" not in config:
        raise KeyError("TARGET section must be present in the configuration")

    env_config = config["ENV"]

    if "ACTION_SPACE" not in env_config:
        raise KeyError("ACTION_SPACE section must be present in the configuration")
    action_space_config = env_config["ACTION_SPACE"]
    if "LOW" not in action_space_config or "HIGH" not in action_space_config:
        raise KeyError("LOW and HIGH must be present in the ACTION_SPACE section")
    if not all(isinstance(val, (int, float)) for val in action_space_config["LOW"]):
        try:
            action_space_config["LOW"] = [
                float(val) for val in action_space_config["LOW"]
            ]
        except ValueError:
            raise ValueError("LOW values in action space must be numeric")
    if not all(isinstance(val, (int, float)) for val in action_space_config["HIGH"]):
        try:
            action_space_config["HIGH"] = [
                float(val) for val in action_space_config["HIGH"]
            ]
        except ValueError:
            raise ValueError("HIGH values in action space must be numeric")
    low = np.array(action_space_config["LOW"], dtype=np.float32)
    high = np.array(action_space_config["HIGH"], dtype=np.float32)
    if low.shape != high.shape:
        raise ValueError(
            "Low and high arrays in action space should have the same shape"
        )
    action_shape = low.shape

    try:
        execution_config = env_config["EXECUTION"]
    except KeyError:
        raise KeyError("EXECUTION section must be present in the configuration")

    params = {
        "action_space": Box(low=low, high=high, shape=action_shape, dtype=np.float32),
        "execution_config": ExecutionConfig(**get_lower_keys_dict(execution_config)),
        "benchmark_config": BenchmarkConfig(
            **(
                get_lower_keys_dict(env_config["BENCHMARKING"])
                if "BENCHMARKING" in env_config
                else {}
            )
        ),
        "reward_config": reward_configs[env_config["REWARD"]["REWARD_METHOD"]](
            **remove_none_values(
                get_lower_keys_dict(
                    env_config.get(
                        "REWARD_PARAMS", env_config["REWARD"].get("REWARD_PARAMS", {})
                    )
                )
            )
        ),
        "training_with_cal": env_config.get("TRAINING_WITH_CAL", False),
        "target": {
            "physical_qubits": config["TARGET"]["PHYSICAL_QUBITS"],
        },
    }
    if "GATE" in config["TARGET"] and config["TARGET"]["GATE"] is not None:
        try:
            params["target"]["gate"] = gate_map()[config["TARGET"]["GATE"].lower()]
        except KeyError:
            raise KeyError("Specified gate not found in standard gate set of Qiskit")
    else:
        try:
            params["target"]["state"] = Statevector.from_label(
                config["TARGET"]["STATE"]
            )
        except KeyError:
            raise KeyError(
                "Target gate or state must be specified in the configuration"
            )

    backend_config = config["BACKEND"]
    dynamics_config = backend_config.get(
        "DYNAMICS",
        {
            "USE_DYNAMICS": None,
            "PHYSICAL_QUBITS": None,
            "SOLVER_OPTIONS": {
                "hmax": "auto",
                "atol": 1e-6,
                "rtol": 1e-8,
                "method": "jax_odeint",
            },
            "CALIBRATION_FILES": None,
        },
    )
    service_config = config.get("SERVICE", {"CHANNEL": None, "INSTANCE": None})
    backend_params = {
        "real_backend": backend_config.get("REAL_BACKEND", None),
        "backend_name": backend_config.get("NAME", None),
        "use_dynamics": dynamics_config["USE_DYNAMICS"],
        "physical_qubits": dynamics_config["PHYSICAL_QUBITS"],
        "channel": service_config["CHANNEL"],
        "instance": service_config["INSTANCE"],
        "solver_options": dynamics_config["SOLVER_OPTIONS"],
        "calibration_files": dynamics_config["CALIBRATION_FILES"],
    }

    runtime_options = config.get("RUNTIME_OPTIONS", {})

    if backend_params["real_backend"]:
        print("Runtime Options:", runtime_options)

    return (
        params,
        backend_params,
        remove_none_values(runtime_options),
    )


def get_lower_keys_dict(dictionary: Dict[str, Any]):
    """
    Get dictionary with lower keys

    Args:
        dictionary: Dictionary
    """
    return {key.lower(): value for key, value in dictionary.items()}


def get_q_env_config(
    config_file_path: str,
    get_backend_func: Callable[[Any], Optional[BackendV2]],
    parametrized_circ_func: Callable[
        [QuantumCircuit, ParameterVector, QuantumRegister, Dict[str, Any]], None
    ],
    **parametrized_circ_args,
):
    """
    Get Qiskit Quantum Environment configuration

    Args:
        config_file_path: Configuration file path
        get_backend_func: Function to get backend (should be defined in your Python config)
        parametrized_circ_func: Function to get parametrized circuit (should be defined in your Python config)
        parametrized_circ_args: Additional arguments for parametrized circuit function
    """
    params, backend_params, runtime_options = load_q_env_from_yaml_file(
        config_file_path
    )
    backend = get_backend_func(**backend_params)
    backend_config = QiskitConfig(
        parametrized_circ_func,
        backend,
        estimator_options=(
            runtime_options if isinstance(backend, RuntimeBackend) else None
        ),
        parametrized_circuit_kwargs=parametrized_circ_args,
    )
    q_env_config = QEnvConfig(backend_config=backend_config, **params)
    return q_env_config


def remove_none_values(dictionary: Dict):
    """
    Remove None values from dictionary

    Args:
        dictionary: Dictionary
    """
    new_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v = remove_none_values(v)
        if v is not None:
            new_dict[k] = v
    return new_dict


def load_from_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_from_pickle(file_path: str):
    """Load data from a pickle or gzip file."""
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rb") as file:
                data = pickle.load(file)
        else:
            with open(file_path, "rb") as file:
                data = pickle.load(file)
    except Exception as e:
        logging.warning(f"Failed to open file {file_path}")
        logging.warning(f"Error Message: {e}")
        return None
    return data


def save_to_pickle(data, file_path: str) -> None:
    """Save data as a pickle or gzip file."""
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "wb") as file:
                pickle.dump(data, file)
        else:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
    except Exception as e:
        logging.warning(f"Failed to save file {file_path}")
        logging.warning(f"Error Message: {e}")


def create_hpo_agent_config(
    trial: optuna.trial.Trial, hpo_config: Dict, path_to_agent_config: str
):
    hyper_params = {}
    hyperparams_in_scope = []

    # Loop through hpo_config and decide whether to optimize or use the provided value
    for param, values in hpo_config.items():
        if isinstance(values, list):
            if len(values) == 2:  # If values is a list of length 2, optimize
                if isinstance(values[0], int):
                    hyper_params[param] = trial.suggest_int(param, values[0], values[1])
                elif isinstance(values[0], float):
                    if param == "LR":  # If learning rate, suggest in log scale
                        hyper_params[param] = trial.suggest_float(
                            param, values[0], values[1], log=True
                        )
                    else:
                        hyper_params[param] = trial.suggest_float(
                            param, values[0], values[1]
                        )
                hyperparams_in_scope.append(param)
            elif (
                len(values) > 2
            ):  # If values is a list of more than 2, choose from list and optimize
                hyper_params[param] = trial.suggest_categorical(param, values)
                hyperparams_in_scope.append(param)
        else:
            hyper_params[param] = values

    # Dynamically calculate batchsize from minibatch_size and num_minibatches
    print("MINIBATCH_SIZE", hyper_params["MINIBATCH_SIZE"])
    print("NUM_MINIBATCHES", hyper_params["NUM_MINIBATCHES"])
    hyper_params["BATCHSIZE"] = (
        hyper_params["MINIBATCH_SIZE"] * hyper_params["NUM_MINIBATCHES"]
    )

    # Print hyperparameters considered for HPO
    print("Hyperparameters considered for HPO:", hyperparams_in_scope)

    # Print hyperparameters NOT considered for HPO
    hyperparams_not_in_scope = [
        param for param in hpo_config if param not in hyperparams_in_scope
    ]
    print("Hyperparameters NOT in scope of HPO:", hyperparams_not_in_scope)

    # Take over attributes from agent_config and populate hyper_params
    agent_config = load_from_yaml_file(path_to_agent_config)
    final_config = hyper_params.copy()
    final_config.update(agent_config)
    final_config.update(hyper_params)

    return final_config, hyperparams_in_scope


def get_hardware_runtime_single_circuit(
    qc: QuantumCircuit,
    instruction_durations_dict: Dict[Tuple[str, Tuple[int, ...]], Tuple[float, str]],
):
    total_time_per_qubit = {qubit: 0.0 for qubit in qc.qubits}

    for instruction in qc.data:
        qubits_involved = instruction.qubits
        gate_name: str = (
            instruction.operation.name
            if not instruction.operation.label
            else instruction.operation.label
        )

        if len(qubits_involved) == 1:
            qbit1 = qubits_involved[0]
            qbit1_index = qc.find_bit(qbit1)[0]
            key = (gate_name, (qbit1_index,))
            if key in instruction_durations_dict:
                gate_time = instruction_durations_dict[key][0]
                total_time_per_qubit[qbit1] += gate_time

        elif len(qubits_involved) == 2:
            qbit1, qbit2 = qubits_involved
            qbit1_index = qc.find_bit(qbit1)[0]
            qbit2_index = qc.find_bit(qbit2)[0]
            key = (gate_name, (qbit1_index, qbit2_index))
            if key in instruction_durations_dict:
                gate_time = instruction_durations_dict[key][0]
                for qbit in qubits_involved:
                    total_time_per_qubit[qbit] += gate_time

        else:
            raise NotImplementedError(
                "Hardware runtimes of 3-qubit gates are not implemented currently."
            )

    # Find the maximum execution time among all qubits
    total_execution_time = (
        max(total_time_per_qubit.values())
        + instruction_durations_dict[("reset", (0,))][
            0
        ]  # Reset time is the same for all qubits
        + instruction_durations_dict[("measure", (0,))][
            0
        ]  # Reset time is the same for all qubits
    )

    return total_execution_time


def get_hardware_runtime_cumsum(
    qc: QuantumCircuit, circuit_gate_times: Dict, total_shots: List[int]
) -> np.array:
    return np.cumsum(
        get_hardware_runtime_single_circuit(qc, circuit_gate_times)
        * np.array(total_shots)
    )


def retrieve_neighbor_qubits(coupling_map: CouplingMap, target_qubits: List):
    """
    Retrieve neighbor qubits of target qubits

    Args:
        coupling_map: Coupling map
        target_qubits: Target qubits

    Returns:
        neighbor_qubits: List of neighbor qubits indices for specified target qubits
    """

    return list(
        set(
            filter(
                lambda x: x not in target_qubits,
                chain(
                    *[
                        list(coupling_map.neighbors(target_qubit))
                        for target_qubit in target_qubits
                    ]
                ),
            )
        )
    )


def retrieve_tgt_instruction_count(qc: QuantumCircuit, target: Dict):
    """
    Retrieve count of target instruction in Quantum Circuit

    Args:
        qc: Quantum Circuit (ideally already transpiled)
        target: Target in form of {"gate": "X", "physical_qubits": [0, 1]}
    """
    tgt_instruction = CircuitInstruction(
        target["gate"], [qc.qubits[i] for i in target["physical_qubits"]]
    )
    return qc.data.count(tgt_instruction)


def generate_default_instruction_durations_dict(
    n_qubits: int,
    single_qubit_gate_time: float,
    two_qubit_gate_time: float,
    circuit_gate_times: Dict,
    virtual_gates: Optional[List] = None,
):
    """
    Generates a dictionary of default instruction durations for each gate and qubit combination. This allows for calculating the total execution time of a quantum circuit.
    In particular, the metric of hardware runtime becomes relevant to benchmark the performance of different methods for the same calibration task.

    Args:
        n_qubits (int): The number of qubits in the quantum circuit.
        single_qubit_gate_time (float): The duration of a single-qubit gate.
        two_qubit_gate_time (float): The duration of a two-qubit gate.
        circuit_gate_times (dict): A dictionary mapping gate names to their respective durations.
        virtual_gates (list): A list of gates that are performed by software and have zero duration.

    Returns:
        dict: A dictionary where the keys are tuples of the form (gate, qubits) and the values are tuples of the form (duration, unit).
              The duration is the default duration for the gate and qubit combination, and the unit is the time unit (e.g., 's' for seconds).

    """
    default_instruction_durations_dict = {}

    # Identify single-qubit and two-qubit gates
    single_qubit_gates = []
    two_qubit_gates = []

    for gate in circuit_gate_times:
        if virtual_gates is not None and gate in virtual_gates:
            continue
        if gate == "measure" or gate == "reset":
            continue
        if circuit_gate_times[gate] == single_qubit_gate_time:
            single_qubit_gates.append(gate)
        elif circuit_gate_times[gate] == two_qubit_gate_time:
            two_qubit_gates.append(gate)

    # Single qubit gates
    for gate in single_qubit_gates:
        for qubit in range(n_qubits):
            default_instruction_durations_dict[(gate, (qubit,))] = (
                circuit_gate_times[gate],
                "s",
            )

    # Two qubit gates (assuming all-to-all connectivity)
    for gate in two_qubit_gates:
        for qubit1 in range(n_qubits):
            for qubit2 in range(n_qubits):
                if qubit1 != qubit2:
                    default_instruction_durations_dict[(gate, (qubit1, qubit2))] = (
                        two_qubit_gate_time,
                        "s",
                    )

    # Reset and Measure operations
    for qubit in range(n_qubits):
        default_instruction_durations_dict[("measure", (qubit,))] = (
            circuit_gate_times["measure"],
            "s",
        )
        default_instruction_durations_dict[("reset", (qubit,))] = (
            circuit_gate_times["reset"],
            "s",
        )

    # Gates done by software
    if virtual_gates is not None:
        for gate in virtual_gates:
            for qubit in range(n_qubits):
                default_instruction_durations_dict[(gate, (qubit,))] = (0.0, "s")

    return default_instruction_durations_dict
