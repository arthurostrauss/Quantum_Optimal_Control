from __future__ import annotations

import os
import sys
import pickle
import gzip
import warnings

from qiskit import pulse, QuantumRegister
from qiskit.circuit import (
    QuantumCircuit,
    Gate,
    Parameter,
    CircuitInstruction,
    ParameterVector,
    Delay,
    Qubit,
)
from qiskit.circuit.library import get_standard_gate_name_mapping as gate_map, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
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
from qiskit.qobj import QobjExperimentHeader
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import (
    Operator,
    Statevector,
    DensityMatrix,
    average_gate_fidelity,
    state_fidelity,
)
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import (
    CouplingMap,
    InstructionProperties,
    PassManager,
    InstructionDurations,
)

from qiskit.providers import (
    BackendV1,
    Backend,
    BackendV2,
    Options as AerOptions,
    QiskitBackendNotFoundError,
)
from qiskit_dynamics.backend.backend_utils import (
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
    _get_iq_data,
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

from qiskit_dynamics import Solver, DynamicsBackend

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
from typing import Optional, Tuple, List, Union, Dict, Sequence, Callable, Any, Set
import yaml

import numpy as np

from gymnasium.spaces import Box
import optuna
from scipy.integrate._ivp.ivp import OdeResult

from scipy.optimize import minimize, OptimizeResult

import keyword
import re

from .transpiler_passes import CustomGateReplacementPass
from ..custom_jax_sim import PulseEstimatorV2
from ..environment.qconfig import (
    BackendConfig,
    ExecutionConfig,
    BenchmarkConfig,
    StateConfig,
    QiskitRuntimeConfig,
    DynamicsConfig,
    QEnvConfig,
    CAFEConfig,
    ChannelConfig,
    XEBConfig,
    ORBITConfig,
    FidelityConfig,
    QiboConfig,
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
QuantumTarget = Union[Statevector, Operator, DensityMatrix]
QuantumInput = Union[QuantumCircuit, pulse.Schedule, pulse.ScheduleBlock]
PulseInput = Union[pulse.Schedule, pulse.ScheduleBlock]

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
    if "ecr" in basis_gates:
        basis_gate = "ecr"
    elif "cx" in basis_gates:
        basis_gate = "cx"
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
    basis_gate_inst: PulseInput = instruction_schedule_map.get(
        gate_name, physical_qubit
    )
    basis_gate_instructions = np.array(basis_gate_inst.instructions)[:, 1]

    play_instructions = basis_gate_inst.filter(instruction_types=[pulse.Play])
    if len(play_instructions) == 0:
        raise ValueError(f"No Play instructions found for {gate_name} gate")
    if len(play_instructions) > 1:
        warnings.warn(
            f"Multiple Play instructions found for {gate_name} gate, using the first one"
        )

    ref_pulse = play_instructions.instructions[0][1].pulse

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


def custom_experiment_result_function(
    experiment_name: str,
    solver_result: OdeResult,
    measurement_subsystems: List[int],
    memory_slot_indices: List[int],
    num_memory_slots: Union[None, int],
    backend: DynamicsBackend,
    seed: Optional[int] = None,
    metadata: Optional[Dict] = None,
) -> ExperimentResult:
    """Default routine for generating ExperimentResult object.

    To generate the results for a given experiment, this method takes the following steps:

    * The final state is transformed out of the rotating frame and into the lab frame using
      ``backend.options.solver``.
    * If ``backend.options.normalize_states==True``, the final state is normalized.
    * Measurement results are computed, in the dressed basis, based on both the measurement-related
      options in ``backend.options`` and the measurement specification extracted from the specific
      experiment.

    Args:
        experiment_name: Name of experiment.
        solver_result: Result object from :class:`Solver.solve`.
        measurement_subsystems: Labels of subsystems in the model being measured.
        memory_slot_indices: Indices of memory slots to store the results in for each subsystem.
        num_memory_slots: Total number of memory slots in the returned output. If ``None``,
            ``max(memory_slot_indices)`` will be used.
        backend: The backend instance that ran the simulation. Various options and properties
            are utilized.
        seed: Seed for any random number generation involved (e.g. when computing outcome samples).
        metadata: Metadata to add to the header of the
            :class:`~qiskit.result.models.ExperimentResult` object.

    Returns:
        :class:`~qiskit.result.models.ExperimentResult` object containing results.

    Raises:
        QiskitError: If a specified option is unsupported.
    """

    yf = solver_result.y[-1]
    tf = solver_result.t[-1]

    # Take state out of frame, put in dressed basis, and normalize
    yf = rotate_frame(yf, tf, backend)

    if backend.options.meas_level == MeasLevel.CLASSIFIED:
        memory_slot_probabilities = _get_memory_slot_probabilities(
            probability_dict=yf.probabilities_dict(qargs=measurement_subsystems),
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            max_outcome_value=backend.options.max_outcome_level,
        )

        # sample
        memory_samples = _sample_probability_dict(
            memory_slot_probabilities,
            shots=backend.options.shots,
            normalize_probabilities=backend.options.normalize_states,
            seed=seed,
        )
        counts = _get_counts_from_samples(memory_samples)

        # construct results object
        exp_data = ExperimentResultData(
            counts=counts,
            memory=memory_samples if backend.options.memory else None,
            statevector=projected_state(
                yf,
                backend.options.subsystem_dims,
                normalize=backend.options.normalize_states,
            ),
            raw_statevector=yf,
        )

        return ExperimentResult(
            shots=backend.options.shots,
            success=True,
            data=exp_data,
            meas_level=MeasLevel.CLASSIFIED,
            seed=seed,
            header=QobjExperimentHeader(name=experiment_name, metadata=metadata),
        )
    elif backend.options.meas_level == MeasLevel.KERNELED:
        iq_centers = backend.options.iq_centers
        if iq_centers is None:
            # Default iq_centers
            iq_centers = []
            for sub_dim in backend.options.subsystem_dims:
                theta = 2 * np.pi / sub_dim
                iq_centers.append(
                    [
                        (np.cos(idx * theta), np.sin(idx * theta))
                        for idx in range(sub_dim)
                    ]
                )

        # generate IQ
        measurement_data = _get_iq_data(
            yf,
            measurement_subsystems=measurement_subsystems,
            iq_centers=iq_centers,
            iq_width=backend.options.iq_width,
            shots=backend.options.shots,
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            seed=seed,
        )

        if backend.options.meas_return == MeasReturnType.AVERAGE:
            measurement_data = np.average(measurement_data, axis=0)

        # construct results object
        exp_data = ExperimentResultData(
            memory=measurement_data,
            statevector=projected_state(
                yf,
                backend.options.subsystem_dims,
                normalize=backend.options.normalize_states,
            ),
        )
        return ExperimentResult(
            shots=backend.options.shots,
            success=True,
            data=exp_data,
            meas_level=MeasLevel.KERNELED,
            seed=seed,
            header=QobjHeader(name=experiment_name, metadata=metadata),
        )

    else:
        raise QiskitError(f"meas_level=={backend.options.meas_level} not implemented.")


def simulate_pulse_input(
    backend: DynamicsBackend,
    qc_input: QuantumInput | List[QuantumInput],
    target: Optional[List[QuantumTarget] | QuantumTarget | Tuple[QuantumTarget]] = None,
    initial_state: Optional[QuantumState] = None,
    normalize: bool = True,
) -> Dict[str, Union[Operator, Statevector, float]]:
    """
    Function extending the functionality of the DynamicsBackend to simulate pulse input (backend.solve)
    by computing unitary simulation and state/gate fidelity calculation for the provided input.

    :param backend: DynamicsBackend or Solver instance
    :param qc_input: (List of) Quantum Circuit(s) or Pulse Schedule(s) input to be simulated
    :param target: Optional target unitary/state (or list thereof) for gate/state fidelity calculation (if input
        is QuantumCircuit, target gate and state are automatically inferred).
    :param initial_state: Optional initial state for state fidelity calculation  (if None and target_state is not None,
    then initial state is assumed to be |0..0>)
    :param normalize: Normalize the projected statevector or not
    :return: Dictionary containing simulated unitary, statevector, projected unitary, projected statevector, gate fidelity, state fidelity
    """
    solver: Solver = backend.options.solver
    subsystem_dims = list(filter(lambda x: x > 1, backend.options.subsystem_dims))
    if not isinstance(qc_input, list):
        qc_input = [qc_input]
    if not all(isinstance(qc, QuantumInput) for qc in qc_input):
        raise TypeError("Input must be a Quantum Circuit or Pulse Schedule")
    qc_input = [
        qc.remove_final_measurements(False) if isinstance(qc, QuantumCircuit) else qc
        for qc in qc_input
    ]

    results = backend.solve(
        y0=np.eye(solver.model.dim),
        solve_input=qc_input,
    )

    output_unitaries = [np.array(result.y[-1]) for result in results]

    output_ops = [
        Operator(
            output_unitary,
            input_dims=tuple(subsystem_dims),
            output_dims=tuple(subsystem_dims),
        )
        for output_unitary in output_unitaries
    ]

    # Rotate the frame of the output unitaries to lab frame
    output_ops = [
        rotate_frame(output_op, result.t[-1], backend)
        for output_op, result in zip(output_ops, results)
    ]

    projected_unitaries = [
        qubit_projection(output_unitary, subsystem_dims)
        for output_unitary in output_unitaries
    ]

    initial_state = (
        DensityMatrix.from_int(0, subsystem_dims)
        if initial_state is None
        else initial_state
    )

    final_states = [initial_state.evolve(output_op) for output_op in output_ops]
    projected_statevectors = [
        projected_state(final_state, subsystem_dims, normalize)
        for final_state in final_states
    ]
    rotated_state = None

    if len(final_states) > 1:
        final_results = {
            "unitary": output_ops,
            "state": final_states,
            "projected_unitary": projected_unitaries,
            "projected_state": projected_statevectors,
        }
    else:
        final_results = {
            "unitary": output_ops[0],
            "state": final_states[0],
            "projected_unitary": projected_unitaries[0],
            "projected_state": projected_statevectors[0],
        }

    target_unitaries = []
    target_states = []
    rotated_states = None
    if target is not None:
        if isinstance(target, List):
            if len(target) != len(qc_input):
                raise ValueError(
                    "Number of target states/gates does not match the number of input circuits"
                )
            for target_op in target:
                if isinstance(target_op, Tuple):
                    for op in target_op:
                        if isinstance(op, Operator):
                            target_unitaries.append(op)
                        elif isinstance(op, QuantumState):
                            target_states.append(op)
                        else:
                            raise TypeError(
                                "Target must be either Operator or Statevector"
                            )
                elif isinstance(target_op, Operator):
                    target_unitaries.append(target_op)
                elif isinstance(target_op, QuantumState):
                    target_states.append(target_op)
                else:
                    raise TypeError("Target must be either Operator or Statevector")
        else:
            if isinstance(target, Tuple):
                for op in target:
                    if isinstance(op, Operator):
                        target_unitaries.extend([op] * len(qc_input))
                    elif isinstance(op, QuantumState):
                        target_states.extend([op] * len(qc_input))
                    else:
                        raise TypeError("Target must be either Operator or Statevector")
            elif isinstance(target, Operator):
                target_unitaries.extend([target] * len(qc_input))
            elif isinstance(target, QuantumState):
                target_states.extend([target] * len(qc_input))
            else:
                raise TypeError(
                    "Target must be either Operator or Statevector or a Tuple of them"
                )
    else:
        for qc in qc_input:
            if isinstance(qc, QuantumCircuit):
                target_unitaries.append(Operator(qc.remove_final_measurements(False)))
                target_states.append(Statevector(qc.remove_final_measurements(False)))
            else:
                target_unitaries.append(None)
                target_states.append(None)

    if target_unitaries:
        optimal_rots = [
            (
                get_optimal_z_rotation(
                    projected_unitary, target_unitary, len(subsystem_dims)
                )
                if target_unitary is not None
                else None
            )
            for projected_unitary, target_unitary in zip(
                projected_unitaries, target_unitaries
            )
        ]
        rotated_unitaries = [
            (
                rotate_unitary(optimal_rot.x, projected_unitary)
                if optimal_rot is not None
                else None
            )
            for optimal_rot, projected_unitary in zip(optimal_rots, projected_unitaries)
        ]
        try:
            rotated_states = [
                (
                    initial_state.evolve(rotated_unitary)
                    if rotated_unitary is not None
                    else None
                )
                for rotated_unitary in rotated_unitaries
            ]
        except QiskitError:
            pass

        gate_fids = [
            (
                average_gate_fidelity(projected_unitary, target_unitary)
                if target_unitary is not None
                else None
            )
            for projected_unitary, target_unitary in zip(
                projected_unitaries, target_unitaries
            )
        ]
        optimal_gate_fids = [
            (
                average_gate_fidelity(rotated_unitary, target_unitary)
                if rotated_unitary is not None
                else None
            )
            for rotated_unitary, target_unitary in zip(
                rotated_unitaries, target_unitaries
            )
        ]

        final_results["gate_fidelity"] = {
            "raw": gate_fids if len(gate_fids) > 1 else gate_fids[0],
            "optimal": (
                optimal_gate_fids
                if len(optimal_gate_fids) > 1
                else optimal_gate_fids[0]
            ),
            "rotations": (
                [optimal_rot.x for optimal_rot in optimal_rots]
                if len(optimal_rots) > 1
                else optimal_rots[0].x
            ),
            "rotated_unitary": (
                rotated_unitaries
                if len(rotated_unitaries) > 1
                else rotated_unitaries[0]
            ),
        }

    if target_states:
        state_fid1 = [
            (
                state_fidelity(projected_statevec, target_state, validate=False)
                if target_state is not None
                else None
            )
            for projected_statevec, target_state in zip(
                projected_statevectors, target_states
            )
        ]
        state_fid2 = []
        if rotated_states is not None:
            for rotated_state, target_state in zip(rotated_states, target_states):
                if rotated_state is not None and target_state is not None:
                    state_fid2.append(
                        state_fidelity(rotated_state, target_state, validate=False)
                    )
                else:
                    state_fid2.append(None)

        final_results["state_fidelity"] = {
            "raw": state_fid1 if len(state_fid1) > 1 else state_fid1[0],
        }
        if state_fid2:
            final_results["state_fidelity"]["optimal"] = (
                state_fid2 if len(state_fid2) > 1 else state_fid2[0]
            )
        if rotated_states:
            final_results["state_fidelity"]["rotated_states"] = (
                rotated_states if len(rotated_states) > 1 else rotated_states[0]
            )
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


def causal_cone_circuit(
    circuit: QuantumCircuit, qubits: Sequence[int | Qubit] | QuantumRegister
) -> Tuple[QuantumCircuit, List[Qubit]]:
    """
    Get the causal cone circuit of the specified qubits as well as the qubits involved in the causal cone
    """
    dag = circuit_to_dag(circuit)
    if isinstance(qubits, List) and all(isinstance(q, int) for q in qubits):
        qubits = [dag.qubits[q] for q in qubits]
    involved_qubits = [dag.quantum_causal_cone(q) for q in qubits]
    involved_qubits = list(set([q for sublist in involved_qubits for q in sublist]))
    filtered_dag = DAGCircuit()
    filtered_dag.add_qubits(involved_qubits)
    for node in dag.topological_op_nodes():
        if any(q in involved_qubits for q in node.qargs):
            filtered_dag.apply_operation_back(node.op, node.qargs)
    return dag_to_circuit(filtered_dag, False), involved_qubits


def fidelity_from_tomography(
    qc_input: List[QuantumCircuit] | QuantumCircuit,
    backend: Optional[Backend],
    physical_qubits: Optional[Sequence[int]],
    target: Optional[QuantumTarget | List[QuantumTarget]] = None,
    analysis: Union[BaseAnalysis, None, str] = "default",
    **run_options,
):
    """
    Extract average state or gate fidelity from batch of Quantum Circuit for target state or gate

    Args:
        qc_input: Quantum Circuit input to benchmark (Note that we handle removing final measurements if any)
        backend: Backend instance
        physical_qubits: Physical qubits on which state or process tomography is to be performed
        analysis: Analysis instance
        target: Target state or gate for fidelity calculation (must be either Operator or QuantumState)
        sampler: Runtime Sampler
    Returns:
        avg_fidelity: Average state or gate fidelity (over the batch of Quantum Circuits)
    """
    if isinstance(qc_input, QuantumCircuit):
        qc_input = [qc_input.remove_final_measurements(False)]
    else:
        qc_input = [qc.remove_final_measurements(False) for qc in qc_input]
    if isinstance(target, QuantumTarget):
        target = [target] * len(qc_input)
    elif target is not None:
        if len(target) != len(qc_input):
            raise ValueError(
                "Number of target states/gates does not match the number of input circuits"
            )
    else:
        target = [Statevector(qc) for qc in qc_input]

    exps = []
    fids = []
    for qc, tgt in zip(qc_input, target):
        if isinstance(tgt, Operator):
            exps.append(
                ProcessTomography(
                    qc, physical_qubits=physical_qubits, analysis=analysis, target=tgt
                )
            )
            fids.append("process_fidelity")
        elif isinstance(tgt, QuantumState):
            exps.append(
                StateTomography(
                    qc, physical_qubits=physical_qubits, analysis=analysis, target=tgt
                )
            )
            fids.append("state_fidelity")
        else:
            raise TypeError("Target must be either Operator or QuantumState")
    batch_exp = BatchExperiment(exps, backend=backend, flatten_results=False)

    exp_data = batch_exp.run(**run_options).block_for_results()
    results = []
    for fid, tgt, child_data in zip(fids, target, exp_data.child_data()):
        result = child_data.analysis_results(fid).value
        if fid == "process_fidelity" and tgt.is_unitary():
            # Convert to average gate fidelity metric
            dim, _ = tgt.dim
            result = dim * result / (dim + 1)
        results.append(result)

    return results if len(results) > 1 else results[0]


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
        assert isinstance(
            config, DynamicsConfig
        ), "Configuration must be a DynamicsConfig"
        dummy_param = Parameter("dummy")
        if hasattr(dummy_param, "jax_compat"):
            estimator = PulseEstimatorV2(backend=backend, options=estimator_options)
        else:
            estimator = BackendEstimatorV2(backend=backend, options=estimator_options)
        sampler = BackendSamplerV2(backend=backend)

        if config.do_calibrations and not backend.target.has_calibration("x", (0,)):
            calibration_files = config.calibration_files
            _, _ = perform_standard_calibrations(backend, calibration_files)
    elif isinstance(backend, str) and isinstance(config, QiboConfig):
        from ..qibo import QiboEstimatorV2

        estimator = QiboEstimatorV2(
            platform=config.platform, options={"qubits": config.qubit_pair}
        )
        sampler = StatevectorSampler()  # Dummy sampler
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
        sampler = RuntimeSamplerV2(mode=estimator.mode)

    # elif isinstance(backend, QMBackend):
    #     estimator = QMEstimator(backend=backend, options=estimator_options)
    #     sampler = QMSampler(backend=backend)
    else:
        estimator = BackendEstimatorV2(backend=backend, options=estimator_options)
        sampler = BackendSamplerV2(backend=backend)

    return estimator, sampler


def substitute_target_gate(
    circuit: QuantumCircuit,
    target_gate: Gate | str,
    custom_gate: Gate | str,
    qubits: Optional[Sequence[int]] = None,
    parameters: ParameterVector | List[Parameter] | List[float] = None,
):
    """
    Substitute target gate in Quantum Circuit with a parametrized version of the gate.
    The parametrized_circuit function signature should match the expected one for a QiskitConfig instance.

    Args:
        circuit: Quantum Circuit instance
        target_gate: Target gate to be substituted
        custom_gate: Custom gate to be substituted with
        qubits: Physical qubits on which the gate is to be applied (if None, all qubits of input circuit are considered)
    """

    if isinstance(custom_gate, str):
        try:
            custom_gate2 = gate_map()[custom_gate]
        except KeyError:
            raise ValueError(f"Custom gate {custom_gate} not found in gate map")
        if custom_gate2.params and parameters is not None:
            assert len(custom_gate2.params) == len(parameters), (
                f"Number of parameters ({len(parameters)}) does not match number of parameters "
                f"required by the custom gate ({len(custom_gate2.params)})"
            )

    pass_manager = PassManager(
        [
            CustomGateReplacementPass(
                (target_gate, qubits), custom_gate, parameters=parameters
            )
        ]
    )

    return pass_manager.run(circuit)


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
            backend = DynamicsBackend.from_backend(
                backend,
                subsystem_list=list(physical_qubits),
                solver_options=solver_options,
            )
            _, _ = perform_standard_calibrations(
                backend, calibration_files=calibration_files
            )

    if backend is None:
        warnings.warn(
            "No backend selected. Training will be performed on Statevector simulator"
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

    backend_config = config.get("BACKEND", {})
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
    parametrized_circ_func: Callable[
        [QuantumCircuit, ParameterVector, QuantumRegister, Dict[str, Any]], None
    ],
    backend: Optional[Backend_type | Callable[[Any], Backend_type]] = None,
    pass_manager: Optional[PassManager] = None,
    instruction_durations: Optional[InstructionDurations] = None,
    **backend_callable_args,
):
    """
    Get Qiskit Quantum Environment configuration from yaml file

    Args:
        config_file_path: Configuration file path (yaml, should contain at least ENV and TARGET section)
        parametrized_circ_func: Function to applying parametrized gate (should be defined in your Python config)
        backend: Optional custom backend instance
            (if None, backend will be selected based on configuration set in yaml file)

        backend_callable_args: Additional arguments for backend if it was passed as a callable

    """
    params, backend_params, runtime_options = load_q_env_from_yaml_file(
        config_file_path
    )
    if isinstance(backend, Callable):
        backend = backend(**backend_callable_args)
    elif backend is None:
        backend = select_backend(**backend_params)

    if isinstance(backend, DynamicsBackend):
        backend_config = DynamicsConfig(
            parametrized_circ_func,
            backend,
            pass_manager=pass_manager,
            instruction_durations=instruction_durations,
        )
    else:
        backend_config = QiskitRuntimeConfig(
            parametrized_circ_func,
            backend,
            pass_manager=pass_manager,
            instruction_durations=instruction_durations,
            estimator_options=(
                runtime_options if isinstance(backend, RuntimeBackend) else None
            ),
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
