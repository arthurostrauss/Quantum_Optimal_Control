from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple, Union, Dict, Optional, Sequence, Any
import numpy as np
import warnings
from qiskit import pulse, QiskitError
from qiskit.circuit import (
    Gate,
    Parameter,
    ParameterVector,
    QuantumCircuit,
    SwitchCaseOp,
    ForLoopOp,
    IfElseOp,
    WhileLoopOp,
)
from qiskit.circuit.library import get_standard_gate_name_mapping as gate_map, RZGate
from qiskit.providers import BackendV1, BackendV2
from qiskit.qobj import QobjExperimentHeader
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.utils import MeasLevel
from qiskit.quantum_info import (
    average_gate_fidelity,
    state_fidelity,
    Operator,
    Statevector,
    DensityMatrix,
)
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit.transpiler import CouplingMap, InstructionProperties
from qiskit_dynamics.backend.backend_utils import (
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
    _get_iq_data,
)
from qiskit_experiments.calibration_management import (
    Calibrations,
    FixedFrequencyTransmon,
    EchoedCrossResonance,
)
from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal, RoughDragCal
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import minimize, OptimizeResult

Backend_type = Optional[Union[BackendV1, BackendV2]]
QuantumTarget = Union[Statevector, Operator, DensityMatrix]
QuantumInput = Union[QuantumCircuit, pulse.Schedule, pulse.ScheduleBlock]
PulseInput = Union[pulse.Schedule, pulse.ScheduleBlock]


def perform_standard_calibrations(
    backend: DynamicsBackend,
    calibration_files: Optional[str] = None,
    control_flow: bool = True,
):
    """
    Performs standard calibrations for a given backend.

    Args:
        backend: The backend to perform the calibrations on.
        calibration_files: The calibration files to use.
        control_flow: Whether to use control flow.

    Returns:
        A tuple containing the calibrations and the experiment results.
    """
    if not isinstance(backend, DynamicsBackend):
        raise TypeError("Backend must be a DynamicsBackend instance (given: {type(backend)})")

    target, qubits, dt = backend.target, range(backend.num_qubits), backend.dt
    num_qubits = len(qubits)
    single_qubit_properties = {(qubit,): InstructionProperties() for qubit in qubits}
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
        two_qubit_properties = {qubits: InstructionProperties() for qubits in control_channel_map}

    standard_gates: Dict[str, Gate] = gate_map()
    fixed_phase_gates = {
        "z": np.pi,
        "s": np.pi / 2,
        "sdg": -np.pi / 2,
        "t": np.pi / 4,
        "tdg": -np.pi / 4,
    }

    other_gates = ["rz", "id", "h", "x", "sx", "reset", "delay"]
    single_qubit_gates = list(fixed_phase_gates.keys()) + other_gates
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
    ):
        for gate in single_qubit_gates:
            target.add_instruction(
                standard_gates[gate], properties=deepcopy(single_qubit_properties)
            )
        if num_qubits > 1:
            for gate in two_qubit_gates:
                target.add_instruction(
                    standard_gates[gate], properties=deepcopy(two_qubit_properties)
                )
            backend._coupling_map = target.build_coupling_map(two_qubit_gates[0])

    for qubit in qubits:
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
        with pulse.build(backend, name=f"rz{qubit}") as rz_cal:
            pulse.shift_phase(-phi, pulse.DriveChannel(qubit))
            for q in control_channels:
                pulse.shift_phase(-phi, pulse.ControlChannel(q))
        id_cal = pulse.Schedule(
            pulse.Delay(20, pulse.DriveChannel(qubit)),
            name=f"id{qubit}",
        )
        reset_cal = pulse.Schedule(
            pulse.Delay(1000, pulse.DriveChannel(qubit)), name=f"reset{qubit}"
        )

        delay_param = standard_gates["delay"].params[0]
        with pulse.build(backend, name=f"delay{qubit}") as delay_cal:
            pulse.delay(delay_param, pulse.DriveChannel(qubit))

        for name, cal, duration in zip(
            ("rz", "id", "delay", "reset"),
            (rz_cal, id_cal, delay_cal, id_cal),
            (0, 20 * dt, None, 1000 * dt),
        ):

            new_prop = InstructionProperties(duration, 0.0, cal)
            target.update_instruction_properties(name, (qubit,), new_prop)

        for gate, phase in fixed_phase_gates.items():
            gate_cal = rz_cal.assign_parameters({phi: phase}, inplace=False)
            instruction_prop = InstructionProperties(gate_cal.duration * dt, 0.0, gate_cal)
            target.update_instruction_properties(gate, (qubit,), instruction_prop)

        if not existing_cals and backend.options.subsystem_dims[qubit] > 1:
            backend_run = True
            sampler = None
            rabi_exp = RoughXSXAmplitudeCal(
                [qubit], cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 100)
            )
            drag_exp = RoughDragCal([qubit], cals, backend=backend, betas=np.linspace(-20, 20, 15))
            drag_exp.set_experiment_options(reps=[3, 5, 7])
            print(f"Starting Rabi experiment for qubit {qubit}...")
            rabi_result = rabi_exp.run(sampler=sampler, backend_run=backend_run).block_for_results()
            print(f"Rabi experiment for qubit {qubit} done.")
            print(f"Starting Drag experiment for qubit {qubit}...")
            drag_result = drag_exp.run(sampler=sampler, backend_run=backend_run).block_for_results()
            print(f"Drag experiments done for qubit {qubit} done.")
            exp_results[qubit] = [rabi_result, drag_result]

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
    target.update_from_instruction_schedule_map(cals.get_inst_map(), error_dict=error_dict)

    if control_flow:
        for control_flow_op, control_op_name in zip(
            [SwitchCaseOp, ForLoopOp, IfElseOp, WhileLoopOp],
            ["switch_case", "for_loop", "if_else", "while_loop"],
        ):
            target.add_instruction(control_flow_op, name=control_op_name)

    print("Updated Instruction Schedule Map", target.instruction_schedule_map())

    return cals, exp_results


def add_ecr_gate(backend: BackendV2, basis_gates: Optional[List[str]] = None, coupling_map=None):
    """
    Adds the ECR gate to the basis gates of a backend.

    Args:
        backend: The backend to add the ECR gate to.
        basis_gates: The basis gates of the backend.
        coupling_map: The coupling map of the backend.
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
                        calibration=cals.get_schedule("ecr", qubit_pair, default_params),
                    ),
                )
        basis_gates.append("ecr")
        for i, gate in enumerate(basis_gates):
            if gate == "cx":
                basis_gates.pop(i)


def get_ecr_params(backend: Backend_type, physical_qubits: Sequence[int]):
    """
    Gets the parameters for the ECR gate.

    Args:
        backend: The backend to get the parameters from.
        physical_qubits: The physical qubits to get the parameters for.

    Returns:
        A tuple containing the default parameters, pulse features, basis gate instructions, and instructions array.
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
    basis_gate_instructions = instruction_schedule_map.get(basis_gate, qubits=physical_qubits)
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
        rise_fall = (control_pulse.duration - control_pulse.width) / (2 * control_pulse.sigma)
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


def get_pulse_params(backend: Backend_type, physical_qubit: Sequence[int], gate_name: str = "x"):
    """
    Gets the parameters for a pulse.

    Args:
        backend: The backend to get the parameters from.
        physical_qubit: The physical qubit to get the parameters for.
        gate_name: The name of the gate to get the parameters for.

    Returns:
        A tuple containing the default parameters, pulse features, basis gate instructions, and instructions array.
    """
    if not isinstance(backend, (BackendV1, BackendV2)):
        raise TypeError("Backend must be defined")
    if isinstance(backend, BackendV1):
        instruction_schedule_map = backend.defaults().instruction_schedule_map
    else:
        instruction_schedule_map = backend.target.instruction_schedule_map()
    basis_gate_inst: PulseInput = instruction_schedule_map.get(gate_name, physical_qubit)
    basis_gate_instructions = np.array(basis_gate_inst.instructions)[:, 1]

    play_instructions = basis_gate_inst.filter(instruction_types=[pulse.Play])
    if len(play_instructions) == 0:
        raise ValueError(f"No Play instructions found for {gate_name} gate")
    if len(play_instructions) > 1:
        warnings.warn(f"Multiple Play instructions found for {gate_name} gate, using the first one")

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
    Creates a new set of parameters for the ECR gate.

    Args:
        params: The parameters to use.
        qubits: The qubits to apply the gate to.
        backend: The backend to use.
        pulse_features: The pulse features to use.
        keep_symmetry: Whether to keep the symmetry of the gate.
        duration_window: The duration window to use.
        include_baseline: Whether to include the baseline parameters.

    Returns:
        A dictionary of the new parameters.
    """
    qubits = tuple(qubits)
    new_params, available_features, _, _ = get_ecr_params(backend, qubits)

    if keep_symmetry:
        if len(pulse_features) != len(params):
            raise ValueError(
                f"Number of pulse features ({len(pulse_features)}) and number of parameters ({len(params)})"
                f" do not match"
            )
        for sched in ["cr45p", "cr45m"]:
            for i, feature in enumerate(pulse_features):
                if feature != "duration" and feature in available_features:
                    if include_baseline:
                        new_params[(feature, qubits, sched)] += params[i]
                    else:
                        new_params[(feature, qubits, sched)] = 0.0 + params[i]

                else:
                    if include_baseline:
                        new_params[(feature, qubits, sched)] += duration_window * params[i]
                    else:
                        new_params[(feature, qubits, sched)] = duration_window * params[i]

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
    Creates a new set of parameters for a single-qubit gate.

    Args:
        params: The parameters to use.
        qubits: The qubits to apply the gate to.
        backend: The backend to use.
        pulse_features: The pulse features to use.
        duration_window: The duration window to use.
        include_baseline: Whether to include the baseline parameters.
        gate_name: The name of the gate.

    Returns:
        A dictionary of the new parameters.
    """
    new_params, available_features, _, _ = get_pulse_params(backend, qubits, gate_name)
    if len(pulse_features) != len(params):
        raise ValueError(
            f"Number of pulse features ({len(pulse_features)}) and number of parameters ({len(params)}"
            f" do not match"
        )
    for i, feature in enumerate(pulse_features):
        if feature != "duration" and feature in available_features:
            if include_baseline:
                new_params[(feature, qubits, gate_name)] += params[i]
            else:
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
    """
    Creates a custom experiment result.

    Args:
        experiment_name: The name of the experiment.
        solver_result: The result of the solver.
        measurement_subsystems: The measurement subsystems.
        memory_slot_indices: The memory slot indices.
        num_memory_slots: The number of memory slots.
        backend: The backend used for the experiment.
        seed: The seed for the random number generator.
        metadata: The metadata for the experiment.

    Returns:
        The custom experiment result.
    """

    yf = solver_result.y[-1]
    tf = solver_result.t[-1]

    yf = rotate_frame(yf, tf, backend)

    if backend.options.meas_level == MeasLevel.CLASSIFIED:
        memory_slot_probabilities = _get_memory_slot_probabilities(
            probability_dict=yf.probabilities_dict(qargs=measurement_subsystems),
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            max_outcome_value=backend.options.max_outcome_level,
        )

        memory_samples = _sample_probability_dict(
            memory_slot_probabilities,
            shots=backend.options.shots,
            normalize_probabilities=backend.options.normalize_states,
            seed=seed,
        )
        counts = _get_counts_from_samples(memory_samples)

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
            iq_centers = []
            for sub_dim in backend.options.subsystem_dims:
                theta = 2 * np.pi / sub_dim
                iq_centers.append(
                    [(np.cos(idx * theta), np.sin(idx * theta)) for idx in range(sub_dim)]
                )

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
    Simulates a pulse input.

    Args:
        backend: The backend to use for the simulation.
        qc_input: The quantum circuit or pulse schedule to simulate.
        target: The target state or gate.
        initial_state: The initial state for the simulation.
        normalize: Whether to normalize the output state.

    Returns:
        A dictionary containing the simulation results.
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

    output_ops = [
        rotate_frame(output_op, result.t[-1], backend)
        for output_op, result in zip(output_ops, results)
    ]

    projected_unitaries = [
        qubit_projection(output_unitary, subsystem_dims) for output_unitary in output_unitaries
    ]

    initial_state = (
        DensityMatrix.from_int(0, subsystem_dims) if initial_state is None else initial_state
    )

    final_states = [initial_state.evolve(output_op) for output_op in output_ops]
    projected_statevectors = [
        projected_state(final_state, subsystem_dims, normalize) for final_state in final_states
    ]

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
                            raise TypeError("Target must be either Operator or Statevector")
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
                raise TypeError("Target must be either Operator or Statevector or a Tuple of them")
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
                get_optimal_z_rotation(projected_unitary, target_unitary, len(subsystem_dims))
                if target_unitary is not None
                else None
            )
            for projected_unitary, target_unitary in zip(projected_unitaries, target_unitaries)
        ]
        rotated_unitaries = [
            (rotate_unitary(optimal_rot.x, projected_unitary) if optimal_rot is not None else None)
            for optimal_rot, projected_unitary in zip(optimal_rots, projected_unitaries)
        ]
        try:
            rotated_states = [
                (initial_state.evolve(rotated_unitary) if rotated_unitary is not None else None)
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
            for projected_unitary, target_unitary in zip(projected_unitaries, target_unitaries)
        ]
        optimal_gate_fids = [
            (
                average_gate_fidelity(rotated_unitary, target_unitary)
                if rotated_unitary is not None
                else None
            )
            for rotated_unitary, target_unitary in zip(rotated_unitaries, target_unitaries)
        ]

        final_results["gate_fidelity"] = {
            "raw": gate_fids if len(gate_fids) > 1 else gate_fids[0],
            "optimal": (optimal_gate_fids if len(optimal_gate_fids) > 1 else optimal_gate_fids[0]),
            "rotations": (
                [optimal_rot.x for optimal_rot in optimal_rots]
                if len(optimal_rots) > 1
                else optimal_rots[0].x
            ),
            "rotated_unitary": (
                rotated_unitaries if len(rotated_unitaries) > 1 else rotated_unitaries[0]
            ),
        }

    if target_states:
        state_fid1 = [
            (
                state_fidelity(projected_statevec, target_state, validate=False)
                if target_state is not None
                else None
            )
            for projected_statevec, target_state in zip(projected_statevectors, target_states)
        ]
        state_fid2 = []
        if rotated_states is not None:
            for rotated_state, target_state in zip(rotated_states, target_states):
                if rotated_state is not None and target_state is not None:
                    state_fid2.append(state_fidelity(rotated_state, target_state, validate=False))
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


def get_control_channel_map(backend: BackendV2, qubit_tgt_register: List[int]):
    """
    Gets the control channel map for a backend.

    Args:
        backend: The backend to get the control channel map from.
        qubit_tgt_register: The qubit target register.

    Returns:
        The control channel map.
    """
    if not hasattr(backend, "channels_map"):
        raise AttributeError("Backend must have channels_map attribute")
    control_channel_map = {}
    control_channel_map_backend = {
        qubits: control_channels[0].index
        for qubits, control_channels in backend.channels_map["control"].items()
    }
    for qubits in control_channel_map_backend:
        if qubits[0] in qubit_tgt_register and qubits[1] in qubit_tgt_register:
            control_channel_map[qubits] = control_channel_map_backend[qubits]
    return control_channel_map


def rotate_frame(yf: Any, tf: Any, backend: DynamicsBackend):
    """
    Rotates the frame of a state or operator.

    Args:
        yf: The state or operator to rotate.
        tf: The time to rotate to.
        backend: The backend to use for the rotation.

    Returns:
        The rotated state or operator.
    """
    if isinstance(yf, Statevector):
        yf = np.array(backend.options.solver.model.rotating_frame.state_out_of_frame(t=tf, y=yf))
        yf = backend._dressed_states_adjoint @ yf
        yf = Statevector(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.linalg.norm(yf.data)
    elif isinstance(yf, DensityMatrix):
        yf = np.array(
            backend.options.solver.model.rotating_frame.operator_out_of_frame(t=tf, operator=yf)
        )
        yf = backend._dressed_states_adjoint @ yf @ backend._dressed_states
        yf = DensityMatrix(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.diag(yf.data).sum()
    elif isinstance(yf, Operator):
        yf = np.array(
            backend.options.solver.model.rotating_frame.operator_out_of_frame(t=tf, operator=yf)
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
    Builds a projector onto the qubit space.

    Args:
        initial_subsystem_dims: The initial subsystem dimensions.

    Returns:
        The projector.
    """
    total_dim = np.prod(initial_subsystem_dims)
    output_dims = (2,) * len(initial_subsystem_dims)
    total_qubit_dim = np.prod(output_dims)
    projector = Operator(
        np.zeros((total_qubit_dim, total_dim), dtype=np.complex128),
        input_dims=tuple(initial_subsystem_dims),
        output_dims=output_dims,
    )
    for i in range(total_dim):
        s = Statevector.from_int(i, initial_subsystem_dims)
        for key in s.to_dict().keys():
            if all(c in "01" for c in key):
                s_qubit = Statevector.from_label(key)
                projector += Operator(
                    s_qubit.data.reshape(total_qubit_dim, 1)
                    @ s.data.reshape(total_dim, 1).conj().T,
                    input_dims=tuple(initial_subsystem_dims),
                    output_dims=output_dims,
                )
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
    Projects a state onto the qubit space.

    Args:
        state: The state to project.
        subsystem_dims: The subsystem dimensions.
        normalize: Whether to normalize the projected state.

    Returns:
        The projected state.
    """
    if not isinstance(state, (np.ndarray, QuantumState)):
        raise TypeError("State must be either numpy array or QuantumState object")
    proj = build_qubit_space_projector(subsystem_dims)
    if isinstance(state, np.ndarray):
        state_type = DensityMatrix if state.ndim == 2 else Statevector
        output_state: Statevector | DensityMatrix = state_type(state)
    else:
        output_state: Statevector | DensityMatrix = state
    qubitized_state = output_state.evolve(proj)

    if (
        normalize
    ) and qubitized_state.trace() != 0:
        qubitized_state = (
            qubitized_state / qubitized_state.trace()
            if isinstance(qubitized_state, DensityMatrix)
            else qubitized_state / np.linalg.norm(qubitized_state.data)
        )

    return qubitized_state


def qubit_projection(unitary: np.ndarray | Operator, subsystem_dims: List[int]) -> Operator:
    """
    Projects a unitary onto the qubit space.

    Args:
        unitary: The unitary to project.
        subsystem_dims: The subsystem dimensions.

    Returns:
        The projected unitary.
    """

    proj = build_qubit_space_projector(subsystem_dims)
    unitary_op = (
        Operator(unitary, input_dims=tuple(subsystem_dims), output_dims=tuple(subsystem_dims))
        if isinstance(unitary, np.ndarray)
        else unitary
    )

    qubitized_op = proj @ unitary_op @ proj.adjoint()
    return qubitized_op


def rotate_unitary(x, op: Operator) -> Operator:
    """
    Rotates a unitary by a given angle.

    Args:
        x: The angle to rotate by.
        op: The unitary to rotate.

    Returns:
        The rotated unitary.
    """
    assert len(x) % 2 == 0, "Rotation parameters should be a pair"
    ops = [
        Operator(RZGate(x[i])) for i in range(len(x))
    ]
    pre_rot, post_rot = (
        ops[0],
        ops[-1],
    )
    for i in range(1, len(x) // 2):
        pre_rot = pre_rot.expand(ops[i])
        post_rot = post_rot.tensor(ops[-i - 1])

    return pre_rot @ op @ post_rot


def get_optimal_z_rotation(
    unitary: Operator, target_gate: Gate | Operator, n_qubits: int
) -> OptimizeResult:
    """
    Gets the optimal Z rotation for a given unitary.

    Args:
        unitary: The unitary to get the optimal Z rotation for.
        target_gate: The target gate.
        n_qubits: The number of qubits.

    Returns:
        The optimal Z rotation.
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


def handle_virtual_rotations(operations, fidelities, subsystem_dims, n_reps, target):
    """
    Handles virtual rotations.

    Args:
        operations: The operations to handle.
        fidelities: The fidelities of the operations.
        subsystem_dims: The subsystem dimensions.
        n_reps: The number of repetitions.
        target: The target.

    Returns:
        The fidelities of the operations with virtual rotations.
    """
    best_op = operations[np.argmax(fidelities)]
    res = get_optimal_z_rotation(best_op, target.target_operator.power(n_reps), len(subsystem_dims))
    rotated_unitaries = [rotate_unitary(res.x, op) for op in operations]
    fidelities = [target.fidelity(op, n_reps) for op in rotated_unitaries]

    return fidelities


def custom_schedule(
    backend: BackendV1 | BackendV2,
    physical_qubits: List[int],
    params: ParameterVector,
) -> pulse.ScheduleBlock:
    """
    Creates a custom pulse schedule.

    Args:
        backend: The backend to use.
        physical_qubits: The physical qubits to apply the schedule to.
        params: The parameters for the schedule.

    Returns:
        The custom pulse schedule.
    """
    ecr_pulse_features = ["amp", "angle", "tgt_amp", "tgt_angle"]
    sq_pulse_features = ["amp"]
    sq_name = "x"
    keep_symmetry = (
        True
    )
    include_baseline = (
        False
    )
    include_duration = False
    duration_window = 1
    if include_duration:
        ecr_pulse_features.append("duration")
        sq_pulse_features.append("duration")

    qubits = tuple(physical_qubits)

    if len(qubits) == 2:
        new_params = new_params_ecr(
            params,
            qubits,
            backend,
            ecr_pulse_features,
            keep_symmetry,
            duration_window,
            include_baseline,
        )
    elif len(qubits) == 1:
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

    basis_gate_sched = cals.get_schedule(gate_name, qubits, assign_params=new_params)

    if isinstance(backend, BackendV1):
        backend = BackendV2Converter(backend)

    with pulse.build(backend, name="custom_sched") as custom_sched:
        pulse.call(basis_gate_sched)

    return custom_sched


def validate_pulse_kwargs(
    **kwargs,
) -> tuple[Optional[Gate], list[int], BackendV1 | BackendV2]:
    """
    Validates the kwargs for a pulse calibration.

    Args:
        **kwargs: The kwargs to validate.

    Returns:
        A tuple containing the gate, physical qubits, and backend.
    """
    if "target" not in kwargs or "backend" not in kwargs:
        raise ValueError("Missing target and backend in kwargs.")
    target, backend = kwargs["target"], kwargs["backend"]
    assert isinstance(
        backend, (BackendV1, BackendV2)
    ), "Backend should be a valid Qiskit Backend instance"
    assert isinstance(target, dict), "Target should be a dictionary with 'physical_qubits' keys."

    gate, physical_qubits = target.get("gate", None), target["physical_qubits"]
    if gate is not None:
        from .circuit_utils import get_gate

        gate = get_gate(gate)
        assert isinstance(gate, Gate), "Gate should be a valid Qiskit Gate instance"
    assert isinstance(physical_qubits, list), "Physical qubits should be a list of integers"
    assert all(
        isinstance(qubit, int) for qubit in physical_qubits
    ), "Physical qubits should be a list of integers"

    return gate, physical_qubits, backend
