from __future__ import annotations

import os
import sys
import pickle
import warnings
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
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
from qiskit.quantum_info import (
    Operator,
    Statevector,
    DensityMatrix,
    average_gate_fidelity,
    state_fidelity,
)
from qiskit.transpiler import (
    CouplingMap,
    InstructionDurations,
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
from qiskit_ibm_runtime import (
    Session,
    IBMBackend as RuntimeBackend,
    EstimatorV1 as RuntimeEstimatorV1,
    EstimatorV2 as RuntimeEstimatorV2,
    Options as RuntimeOptions,
    EstimatorOptions as RuntimeEstimatorOptions,
    SamplerV1 as RuntimeSamplerV1,
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
from qiskit_experiments.framework import BatchExperiment, BaseAnalysis, BackendData
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

from itertools import permutations, chain
from typing import Optional, Tuple, List, Union, Dict, Sequence, Callable, Any
import yaml

import numpy as np

from gymnasium.spaces import Box
import optuna

import tensorflow as tf
from scipy.optimize import minimize
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense

import keyword
import re
from rl_qoc.qconfig import (
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
from rl_qoc.custom_jax_sim import (
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
    AerEstimator,
    RuntimeEstimatorV1,
    RuntimeEstimatorV2,
    Estimator,
    BackendEstimator,
    BackendEstimatorV2,
    DynamicsBackendEstimator,
    StatevectorEstimator,
]
Sampler_type = Union[
    AerSampler,
    RuntimeSamplerV1,
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

    target, qubits = backend.target, range(backend.num_qubits)
    num_qubits = len(qubits)
    single_qubit_properties = {(qubit,): None for qubit in qubits}
    single_qubit_errors = {(qubit,): 0.0 for qubit in qubits}

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
            all_to_all_connectivity = tuple(permutations(qubits, 2))
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
        target.update_instruction_properties(
            "rz", (qubit,), InstructionProperties(calibration=rz_cal, error=0.0)
        )
        target.update_instruction_properties(
            "id", (qubit,), InstructionProperties(calibration=id_cal, error=0.0)
        )
        target.update_instruction_properties(
            "reset", (qubit,), InstructionProperties(calibration=id_cal, error=0.0)
        )
        target.update_instruction_properties(
            "delay", (qubit,), InstructionProperties(calibration=delay_cal, error=0.0)
        )
        for phase, gate in zip(fixed_phases, fixed_phase_gates):
            gate_cal = rz_cal.assign_parameters({phi: phase}, inplace=False)
            instruction_prop = InstructionProperties(calibration=gate_cal, error=0.0)
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
            properties=InstructionProperties(calibration=h_schedule, error=0.0),
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
        Statevector.from_int(0, subsystem_dims)
        if initial_state is None
        else initial_state
    )
    final_state = initial_state.evolve(output_op)
    projected_statevec = projected_statevector(final_state, subsystem_dims, normalize)
    rotated_state = None

    final_results = {
        "unitary": output_op,
        "statevector": final_state,
        "projected_unitary": projected_unitary,
        "projected_statevector": projected_statevec,
    }
    if target_unitary is not None:
        optimal_rots = get_optimal_z_rotation(
            projected_unitary, target_unitary, len(subsystem_dims)
        )
        rotated_unitary = rotate_unitary(optimal_rots.x, projected_unitary)
        rotated_state = initial_state.evolve(Operator(rotated_unitary))
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
        state_fid2 = state_fidelity(rotated_state, target_state, validate=False)
        final_results["state_fidelity"] = {
            "raw": state_fid1,
            "optimal": state_fid2,
            "rotated_state": rotated_state,
        }
    return final_results


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
    qc_list: List[QuantumCircuit],
    backend: Optional[Backend],
    target: Operator | QuantumState,
    physical_qubits: Optional[Sequence[int]],
    analysis: Union[BaseAnalysis, None, str] = "default",
    sampler: RuntimeSamplerV2 = None,
):
    """
    Extract average state or gate fidelity from batch of Quantum Circuit for target state or gate

    Args:
        qc_list: List of Quantum Circuits
        backend: Backend instance
        physical_qubits: Physical qubits on which state or process tomography is to be performed
        analysis: Analysis instance
        target: Target state or gate for fidelity calculation
        sampler: Runtime Sampler
    Returns:
        avg_fidelity: Average state or gate fidelity (over the batch of Quantum Circuits)
    """
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
            for qc in qc_list
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
        results = process_tomo.run().block_for_results()

    process_results = [
        results.analysis_results(fidelity)[i].value for i in range(len(qc_list))
    ]
    if isinstance(target, Operator):
        dim, _ = target.dim
        avg_gate_fids = [(dim * f_pro + 1) / (dim + 1) for f_pro in process_results]

        return avg_gate_fids
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
    config: Union[Dict, BackendConfig],
    estimator_options: Optional[
        Dict | AerOptions | RuntimeOptions | RuntimeEstimatorOptions
    ] = None,
    circuit: Optional[QuantumCircuit] = None,
) -> Tuple[Estimator_type, Sampler_type]:
    """
    Retrieve appropriate Qiskit primitives (estimator and sampler) from backend and layout

    Args:
        backend: Backend instance
        layout: Layout instance
        config: Configuration dictionary
        abstraction_level: Abstraction level ("circuit" or "pulse")
        estimator_options: Estimator options
        circuit: QuantumCircuit instance implementing the custom gate (for DynamicsBackend)
    """
    if isinstance(backend, DynamicsBackend) and isinstance(
        backend.options.solver, JaxSolver
    ):
        estimator = DynamicsBackendEstimator(
            backend, options=estimator_options, skip_transpilation=True
        )
        sampler = BackendSamplerV2(backend=backend)
        backend.options.solver.circuit_macro = lambda: schedule(circuit, backend)

        if config.do_calibrations and not backend.target.has_calibration("x", (0,)):
            calibration_files = config.calibration_files
            _, _ = perform_standard_calibrations(backend, calibration_files)
    # elif isinstance(backend, (FakeBackend, FakeBackendV2, AerBackend)):
    #     from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2, SamplerV2 as AerSamplerV2
    #     print("Aer Backend created out of backend", backend)
    #     estimator = AerEstimatorV2.from_backend(backend=backend)
    #     sampler = AerSamplerV2.from_backend(backend=backend)
    elif backend is None:  # No backend specified, ideal state-vector simulation
        sampler = StatevectorSampler()
        estimator = StatevectorEstimator()

    elif isinstance(backend, RuntimeBackend):
        # TODO: Next update: will switch to keyword 'mode' instead of 'session'
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
        isinstance(estimator, (RuntimeEstimatorV1, RuntimeEstimatorV2))
        and estimator.session.status() == "Closed"
    ):
        old_session = estimator.session
        counter += 1
        print(f"New Session opened (#{counter})")
        session, options = (
            Session(old_session.service, backend),
            estimator.options,
        )
        estimator = type(estimator)(session=session, options=asdict(options))

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
            Operator.from_label("I") for i in range(new_circ.num_qubits)
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
        token: Token to use for Runtime Service
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
            if solver_options["hmax"] == "auto":
                solver_options["hmax"] = backend.configuration().dt
            for key in ["atol", "rtol"]:
                solver_options[key] = float(solver_options[key])
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


def build_qubit_space_projector(initial_subsystem_dims: list):
    """
    Build projector on qubit space from initial subsystem dimensions

    Args:
        initial_subsystem_dims: Initial subsystem dimensions

    Returns: Projector on qubit space as a Qiskit Operator object
    """
    total_dim = np.prod(initial_subsystem_dims)
    projector = Operator(
        np.zeros((total_dim, total_dim), dtype=np.complex128),
        input_dims=tuple(initial_subsystem_dims),
        output_dims=tuple(initial_subsystem_dims),
    )  # Projector initialized in the qudit space
    for i in range(
        total_dim
    ):  # Loop over all computational basis states in the qudit space
        s = Statevector.from_int(
            i, initial_subsystem_dims
        )  # Statevector initialized in the qudit space
        for key in s.to_dict().keys():
            if all(
                c in "01" for c in key
            ):  # Check if the statevector is in the qubit space
                projector += (
                    s.to_operator()
                )  # Add the statevector to the projector if it is in the qubit space
                break
            else:
                continue
    return projector


def projected_statevector(
    statevector: np.array, subsystem_dims: List[int], normalize: bool = True
):
    """
    Project statevector on qubit space

    Args:
        statevector: Statevector, given as numpy array
        subsystem_dims: Subsystem dimensions
        normalize: Normalize statevector
    """
    proj = build_qubit_space_projector(
        subsystem_dims
    )  # Projector on qubit space (in qudit space)
    new_dim = 2 ** len(subsystem_dims)  # Dimension of the qubit space
    qubitized_statevector = np.zeros(new_dim, dtype=np.complex128)
    qubit_count = 0
    new_statevector = Statevector(statevector, dims=subsystem_dims).evolve(
        proj
    )  # Projected statevec (in qudit space)
    for i in range(np.prod(subsystem_dims)):
        if (
            new_statevector.data[i] != 0
        ):  # All zeros components correspond to qudit states (due to projection)
            qubitized_statevector[qubit_count] = new_statevector.data[i]
            qubit_count += 1
    if (
        normalize
    ):  # Normalize the projected statevector (which is for now unnormalized due to selection of components)
        qubitized_statevector = qubitized_statevector / np.linalg.norm(
            qubitized_statevector
        )
    qubitized_statevector = Statevector(qubitized_statevector)
    return qubitized_statevector


def qubit_projection(unitary: np.array, subsystem_dims: List[int]):
    """
    Project unitary on qubit space

    Args:
        unitary: Unitary, given as numpy array
        subsystem_dims: Subsystem dimensions

    Returns: unitary projected on qubit space as a Qiskit Operator object
    """

    proj = build_qubit_space_projector(
        subsystem_dims
    )  # Projector on qubit space (in qudit space)
    new_dim = 2 ** len(subsystem_dims)  # Dimension of the qubit space
    unitary_op = Operator(
        unitary, input_dims=tuple(subsystem_dims), output_dims=tuple(subsystem_dims)
    )  # Unitary operator (in qudit space)
    qubitized_unitary = np.zeros((new_dim, new_dim), dtype=np.complex128)
    qubit_count1 = qubit_count2 = 0
    new_unitary = proj @ unitary_op @ proj  # Projected unitary (in qudit space)

    for i in range(
        np.prod(subsystem_dims)
    ):  # Select elements in the projection to build a qubitized unitary
        for j in range(np.prod(subsystem_dims)):
            if (
                new_unitary.data[i, j] != 0
            ):  # All zeros components correspond to qudit states (due to projection)
                qubitized_unitary[qubit_count1, qubit_count2] = new_unitary.data[
                    i, j
                ]  # Fill the qubitized unitary
                qubit_count2 += 1
                if (
                    qubit_count2 == new_dim
                ):  # Reset qubit_count2 when it reaches the dimension of the qubit space
                    qubit_count2 = 0
                    qubit_count1 += 1
                    break
    qubitized_unitary = Operator(
        qubitized_unitary,
        input_dims=(2,) * len(subsystem_dims),
        output_dims=(2,) * len(subsystem_dims),
    )  # Qubitized unitary as a Qiskit Operator object (Note that is actually not unitary at this point, it's a Channel)
    return qubitized_unitary


def rotate_unitary(x, unitary: Operator):
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
        pre_rot = pre_rot.tensor(ops[i])
        post_rot = post_rot.expand(ops[-i - 1])

    return pre_rot @ unitary @ post_rot


def get_optimal_z_rotation(
    unitary: Operator, target_gate: Gate | Operator, n_qubits: int
):
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

    low = np.array(config["ENV"]["ACTION_SPACE"]["LOW"], dtype=np.float32)
    high = np.array(config["ENV"]["ACTION_SPACE"]["HIGH"], dtype=np.float32)
    if low.shape != high.shape:
        raise ValueError(
            "Low and high arrays in action space should have the same shape"
        )
    action_shape = low.shape
    env_config = config["ENV"]
    params = {
        "action_space": Box(low=low, high=high, shape=action_shape, dtype=np.float32),
        "execution_config": ExecutionConfig(
            **get_lower_keys_dict(env_config["EXECUTION"])
        ),
        "benchmark_config": BenchmarkConfig(
            **get_lower_keys_dict(env_config["BENCHMARKING"])
        ),
        "reward_config": reward_configs[env_config["REWARD"]["REWARD_METHOD"]](
            **remove_none_values(
                get_lower_keys_dict(env_config.get("REWARD_PARAMS", {}))
            )
        ),
        "training_with_cal": env_config["TRAINING_WITH_CAL"],
        "target": {
            "physical_qubits": config["TARGET"]["PHYSICAL_QUBITS"],
        },
    }
    if "GATE" in config["TARGET"]:
        params["target"]["gate"] = gate_map()[config["TARGET"]["GATE"].lower()]
    else:
        params["target"]["dm"] = DensityMatrix.from_label(config["TARGET"]["STATE"])

    backend_params = {
        "real_backend": config["BACKEND"]["REAL_BACKEND"],
        "backend_name": config["BACKEND"]["NAME"],
        "use_dynamics": config["BACKEND"]["DYNAMICS"]["USE_DYNAMICS"],
        "physical_qubits": config["BACKEND"]["DYNAMICS"]["PHYSICAL_QUBITS"],
        "channel": config["SERVICE"]["CHANNEL"],
        "instance": config["SERVICE"]["INSTANCE"],
        "solver_options": config["BACKEND"]["DYNAMICS"]["SOLVER_OPTIONS"],
        "calibration_files": config["BACKEND"]["DYNAMICS"]["CALIBRATION_FILES"],
    }
    runtime_options = config["RUNTIME_OPTIONS"]
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
    config_file_address: str,
    get_backend_func: Callable,
    parametrized_circ_func: Callable[
        [QuantumCircuit, ParameterVector, QuantumRegister, Dict[str, Any]], None
    ],
    **parametrized_circ_args,
):
    """
    Get Qiskit Quantum Environment configuration

    Args:
        config_file_address: Configuration file address
        get_backend_func: Function to get backend (should be defined in your Python config)
        parametrized_circ_func: Function to get parametrized circuit (should be defined in your Python config)
        parametrized_circ_args: Additional arguments for parametrized circuit function
    """
    params, backend_params, runtime_options = load_q_env_from_yaml_file(
        config_file_address
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


def remove_none_values(dictionary):
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
    qc: QuantumCircuit, instruction_durations_dict: Dict[str, float]
):
    total_time_per_qubit = {qubit: 0.0 for qubit in qc.qubits}

    for instruction in qc.data:
        qubits_involved = instruction.qubits
        gate_name = (
            instruction.operation.name
            if not instruction.operation.label
            else instruction.operation.label
        )

        if len(qubits_involved) == 1:
            qbit1 = qubits_involved[0]
            qbit1_index = qc.find_bit(qbit1).index
            key = (gate_name, (qbit1_index,))
            if key in instruction_durations_dict:
                gate_time = instruction_durations_dict[key][0]
                total_time_per_qubit[qbit1] += gate_time

        elif len(qubits_involved) == 2:
            qbit1, qbit2 = qubits_involved
            qbit1_index = qc.find_bit(qbit1).index
            qbit2_index = qc.find_bit(qbit2).index
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


# def get_hardware_runtime_single_circuit(
#     qc: QuantumCircuit, circuit_gate_times: Optional[Dict]=None, backend: Optional[BackendV2]=None
# ) -> float:
#     """
#     Return a worst-case estimate of the runtime for a single execution of the circuit on the hardware

#     :param qc: QuantumCircuit to be executed
#     """

#     if backend is not None and backend.target.InstructionDurations is not None:
#         instruction_durations = backend.target.InstructionDurations
#         total_execution_time = sum(instruction_durations.get(instr[0].name, instr[1]) for instr in qc.data)
#         return total_execution_time

#     if not circuit_gate_times:
#         raise ValueError(
#             "Empty circuit_gate_time dictionary received. Please add durations for your gates."
#         )

#     total_time_per_qubit = {qubit: 0.0 for qubit in qc.qubits}
#     for instruction in qc.decompose().data:
#         for qubit in instruction.qubits:
#             # Custom gates only appear in the circuit context for the CAQEnv case
#             if instruction.operation.label in circuit_gate_times:
#                 total_time_per_qubit[qubit] += circuit_gate_times[
#                     instruction.operation.label
#                 ]
#             else:
#                 total_time_per_qubit[qubit] += circuit_gate_times[
#                     instruction.operation.name
#                 ]

#     # Find the maximum execution time among all qubits
#     total_execution_time = (
#         max(total_time_per_qubit.values())
#         + circuit_gate_times["reset"]
#         + circuit_gate_times["measure"]
#     )

#     return total_execution_time


def get_hardware_runtime_cumsum(
    qc: QuantumCircuit, circuit_gate_times: Dict, total_shots: List[int]
) -> List[float]:
    return np.cumsum(
        get_hardware_runtime_single_circuit(qc, circuit_gate_times)
        * np.array(total_shots)
    )


def retrieve_backend_info(
    backend: Optional[Backend_type] = None,
    instruction_durations_dict: Optional[Dict[str, float]] = None,
):
    """
    Retrieve useful Backend data to run context aware gate calibration

    Args:
        backend: Backend instance
        instruction_durations_dict: Instruction duration dictionary


    Returns:
    dt: Time step
    coupling_map: Coupling map
    basis_gates: Basis gates
    instruction_durations: Instruction durations

    """

    if isinstance(backend, Backend_type):
        backend_data = BackendData(backend)
        dt = backend_data.dt if backend_data.dt is not None else 2.2222222222222221e-10
        coupling_map = CouplingMap(backend_data.coupling_map)
        # Check basis_gates and their respective durations of backend (for identifying timing context)
        if isinstance(backend, BackendV1):
            instruction_durations = InstructionDurations.from_backend(backend)
            basis_gates = backend.configuration().basis_gates.copy()
        elif isinstance(backend, BackendV2):
            instruction_durations = backend.instruction_durations
            basis_gates = backend.operation_names.copy()
        else:
            instruction_durations = instruction_durations_dict
            basis_gates = None
    else:
        warnings.warn(
            "No Backend was provided, using default values for dt, coupling_map, basis_gates and instruction_durations"
        )

        return 2.222e-10, CouplingMap.from_full(5), ["x", "sx", "cx", "rz"], None

    return dt, coupling_map, basis_gates, instruction_durations


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
        qc: Quantum Circuit
        target: Target in form of {"gate": "X", "register": [0, 1]}
    """
    tgt_instruction = CircuitInstruction(
        target["gate"], [qc.qubits[i] for i in target["register"]]
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


def select_optimizer(
    lr: float,
    optimizer: str = "Adam",
    grad_clip: Optional[float] = None,
    concurrent_optimization: bool = True,
    lr2: Optional[float] = None,
):
    if concurrent_optimization:
        if optimizer == "Adam":
            return tf.optimizers.Adam(learning_rate=lr, clipvalue=grad_clip)
        elif optimizer == "SGD":
            return tf.optimizers.SGD(learning_rate=lr, clipvalue=grad_clip)
    else:
        if optimizer == "Adam":
            return tf.optimizers.Adam(learning_rate=lr), tf.optimizers.Adam(
                learning_rate=lr2, clipvalue=grad_clip
            )
        elif optimizer == "SGD":
            return tf.optimizers.SGD(learning_rate=lr), tf.optimizers.SGD(
                learning_rate=lr2, clipvalue=grad_clip
            )


def constrain_mean_value(mu_var):
    return [tf.clip_by_value(m, -1.0, 1.0) for m in mu_var]


def constrain_std_value(std_var):
    return [tf.clip_by_value(std, 1e-3, 3) for std in std_var]


def generate_model(
    input_shape: Tuple,
    hidden_units: Union[List, Tuple],
    n_actions: int,
    actor_critic_together: bool = True,
    hidden_units_critic: Optional[Union[List, Tuple]] = None,
):
    """
    Helper function to generate fully connected NN
    :param input_shape: Input shape of the NN
    :param hidden_units: List containing number of neurons per hidden layer
    :param n_actions: Output shape of the NN on the actor part, i.e. dimension of action space
    :param actor_critic_together: Decide if actor and critic network should be distinct or should be sharing layers
    :param hidden_units_critic: If actor_critic_together set to False, List containing number of neurons per hidden
           layer for critic network
    :return: Model or Tuple of two Models for actor critic network
    """
    input_layer = Input(shape=input_shape)
    Net = Dense(
        hidden_units[0],
        activation="relu",
        input_shape=input_shape,
        kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
        bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
        name=f"hidden_{0}",
    )(input_layer)
    for i in range(1, len(hidden_units)):
        Net = Dense(
            hidden_units[i],
            activation="relu",
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
            name=f"hidden_{i}",
        )(Net)

    mean_param = Dense(n_actions, activation="tanh", name="mean_vec")(
        Net
    )  # Mean vector output
    sigma_param = Dense(n_actions, activation="softplus", name="sigma_vec")(
        Net
    )  # Diagonal elements of cov matrix
    # output

    if actor_critic_together:
        critic_output = Dense(1, activation="linear", name="critic_output")(Net)
        return Model(
            inputs=input_layer, outputs=[mean_param, sigma_param, critic_output]
        )
    else:
        assert (
            hidden_units_critic is not None
        ), "Network structure for critic network not provided"
        input_critic = Input(shape=input_shape)
        Critic_Net = Dense(
            hidden_units_critic[0],
            activation="relu",
            input_shape=input_shape,
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
            name=f"hidden_{0}",
        )(input_critic)
        for i in range(1, len(hidden_units)):
            Critic_Net = Dense(
                hidden_units[i],
                activation="relu",
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.initializers.RandomNormal(stddev=0.5),
                name=f"hidden_{i}",
            )(Critic_Net)
            critic_output = Dense(1, activation="linear", name="critic_output")(
                Critic_Net
            )
            return Model(inputs=input_layer, outputs=[mean_param, sigma_param]), Model(
                inputs=input_critic, outputs=critic_output
            )
