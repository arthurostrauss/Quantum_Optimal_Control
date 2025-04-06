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
from qiskit.pulse.transforms import block_to_schedule
from qiskit.qobj import QobjExperimentHeader
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.utils import MeasLevel, MeasReturnType
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
    Generate baseline single qubit gates (X, SX, RZ, H) for all qubits using traditional calibration experiments
    :param backend: Dynamics Backend on which calibrations should be run
    :param calibration_files: Optional calibration files containing single qubit gate calibrations for provided
        DynamicsBackend instance (Qiskit Experiments does not support this feature yet)
    :param control_flow: Include control flow instructions in the backend

    """
    if not isinstance(backend, DynamicsBackend):
        raise TypeError(
            "Backend must be a DynamicsBackend instance (given: {type(backend)})"
        )

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
        two_qubit_properties = {
            qubits: InstructionProperties() for qubits in control_channel_map
        }

    standard_gates: Dict[str, Gate] = gate_map()  # standard gate library
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
    ):  # Check if instructions have already been added
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
            pulse.Delay(20, pulse.DriveChannel(qubit)),
            name=f"id{qubit}",
        )  # Wait 20 cycles for identity gate
        reset_cal = pulse.Schedule(
            pulse.Delay(1000, pulse.DriveChannel(qubit)), name=f"reset{qubit}"
        )

        delay_param = standard_gates["delay"].params[0]
        with pulse.build(backend, name=f"delay{qubit}") as delay_cal:
            pulse.delay(delay_param, pulse.DriveChannel(qubit))

        # Update backend Target by adding calibrations for all phase gates (fixed angle virtual Z-rotations)
        for name, cal, duration in zip(
            ("rz", "id", "delay", "reset"),
            (rz_cal, id_cal, delay_cal, id_cal),
            (0, 20 * dt, None, 1000 * dt),
        ):

            new_prop = InstructionProperties(duration, 0.0, cal)
            target.update_instruction_properties(name, (qubit,), new_prop)

        for gate, phase in fixed_phase_gates.items():
            gate_cal = rz_cal.assign_parameters({phi: phase}, inplace=False)
            instruction_prop = InstructionProperties(
                gate_cal.duration * dt, 0.0, gate_cal
            )
            target.update_instruction_properties(gate, (qubit,), instruction_prop)

        # Perform calibration experiments (Rabi/Drag) for calibrating X and SX gates
        if not existing_cals and backend.options.subsystem_dims[qubit] > 1:
            backend_run = True
            sampler = None  # BackendSamplerV2(backend=backend)
            rabi_exp = RoughXSXAmplitudeCal(
                [qubit], cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 100)
            )
            drag_exp = RoughDragCal(
                [qubit], cals, backend=backend, betas=np.linspace(-20, 20, 15)
            )
            drag_exp.set_experiment_options(reps=[3, 5, 7])
            print(f"Starting Rabi experiment for qubit {qubit}...")
            rabi_result = rabi_exp.run(
                sampler=sampler, backend_run=backend_run
            ).block_for_results()
            print(f"Rabi experiment for qubit {qubit} done.")
            print(f"Starting Drag experiment for qubit {qubit}...")
            drag_result = drag_exp.run(
                sampler=sampler, backend_run=backend_run
            ).block_for_results()
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
    if control_flow:
        for control_flow_op, control_op_name in zip(
            [SwitchCaseOp, ForLoopOp, IfElseOp, WhileLoopOp],
            ["switch_case", "for_loop", "if_else", "while_loop"],
        ):
            target.add_instruction(control_flow_op, name=control_op_name)

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


def get_control_channel_map(backend: BackendV2, qubit_tgt_register: List[int]):
    """
    Get reduced control_channel_map from Backend configuration (needs to be of type BackendV1)
    :param backend: IBM Backend instance, must carry a channels_map attribute
    :param qubit_tgt_register: Subsystem of interest from which to build control_channel_map

    Returns:
    control_channel_map: Reduced control channel map for the qubit_tgt_register
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


def handle_virtual_rotations(operations, fidelities, subsystem_dims, n_reps, target):
    """
    Optimize gate fidelity by finding optimal Z-rotations before and after gate
    """
    best_op = operations[np.argmax(fidelities)]
    res = get_optimal_z_rotation(
        best_op, target.target_operator.power(n_reps), len(subsystem_dims)
    )
    rotated_unitaries = [rotate_unitary(res.x, op) for op in operations]
    fidelities = [target.fidelity(op, n_reps) for op in rotated_unitaries]

    return fidelities

def generate_schedule_macro(sched: pulse.ScheduleBlock|pulse.Schedule):
    """
    Generate a new function replacing symbolic Qiskit Parameters by 'to-be-fed' JAX traced values
    Args:
        sched: ScheduleBlock or Schedule

    Returns:
        New function with arguments to be JAX traced values

    """
    sched = block_to_schedule(sched) if isinstance(sched, pulse.ScheduleBlock) else sched
    def sched_macro(*args):
        new_sched = pulse.Schedule()
        for instruction in sched.instructions:
            pass
            
        
