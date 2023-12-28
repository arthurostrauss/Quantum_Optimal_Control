from itertools import permutations
from typing import Optional, Tuple, List, Union, Dict, Sequence

import numpy as np
import tensorflow as tf
import yaml
from gymnasium.spaces import Box
from qiskit import pulse, schedule, transpile
from qiskit.circuit import QuantumCircuit, Gate, Parameter, CircuitInstruction
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.primitives import BackendEstimator, Estimator, Sampler, BackendSampler
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
from qiskit_aer.backends.aerbackend import AerBackend

from qiskit.providers import BackendV1, Backend, BackendV2, Options as AerOptions
from qiskit.providers.fake_provider.fake_backend import FakeBackend, FakeBackendV2
from qiskit.quantum_info import Operator, Statevector, DensityMatrix
from qiskit.transpiler import (
    CouplingMap,
    InstructionDurations,
    InstructionProperties,
    Layout,
)
from qiskit_dynamics import Solver, RotatingFrame
from qiskit_dynamics.array import Array
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
from qiskit_ibm_provider import IBMBackend

from qiskit_ibm_runtime import (
    Session,
    IBMBackend as RuntimeBackend,
    Estimator as RuntimeEstimator,
    Options as RuntimeOptions,
    Sampler as RuntimeSampler,
)

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

import optuna

from basis_gate_library import EchoedCrossResonance, FixedFrequencyTransmon
from custom_jax_sim.jax_solver import PauliToQuditOperator
from qconfig import QiskitConfig
from custom_jax_sim import JaxSolver, DynamicsBackendEstimator

# from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon, EchoedCrossResonance

Estimator_type = Union[
    AerEstimator,
    RuntimeEstimator,
    Estimator,
    BackendEstimator,
    DynamicsBackendEstimator,
]
Sampler_type = Union[AerSampler, RuntimeSampler, Sampler, BackendSampler]
Backend_type = Union[BackendV1, BackendV2]


def constrain_mean_value(mu_var):
    return [tf.clip_by_value(m, -1.0, 1.0) for m in mu_var]


def constrain_std_value(std_var):
    return [tf.clip_by_value(std, 1e-3, 3) for std in std_var]


def count_gates(qc: QuantumCircuit):
    """
    Count number of gates in a Quantum Circuit
    """
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_unused_wires(qc: QuantumCircuit):
    """
    Remove unused wires from a Quantum Circuit
    """
    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if count == 0:
            qc.qubits.remove(qubit)
    return qc


def perform_standard_calibrations(
        backend: DynamicsBackend, calibration_files: Optional[List[str]] = None
):
    """
    Generate baseline single qubit gates (X, SX, RZ, H) for all qubits using traditional calibration experiments
    :param backend: Dynamics Backend on which calibrations should be run
    :param calibration_files: Optional calibration files containing single qubit gate calibrations for provided
        DynamicsBackend instance (Qiskit Experiments does not support this feature yet)

    """

    target, qubits = backend.target, range(backend.num_qubits)
    num_qubits = len(qubits)
    single_qubit_properties = {(qubit,): None for qubit in qubits}
    single_qubit_errors = {(qubit,): 0.0 for qubit in qubits}

    control_channel_map = backend.options.control_channel_map
    physical_control_channel_map = None
    if control_channel_map is not None:
        physical_control_channel_map = {
            (qubit_pair[0], qubit_pair[1]): backend.control_channel(
                (qubit_pair[0], qubit_pair[1])
            )
            for qubit_pair in control_channel_map
        }
    elif num_qubits > 1:
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
    two_qubit_errors = {qubits: 0.0 for qubits in control_channel_map}
    standard_gates: Dict[
        str, Gate
    ] = get_standard_gate_name_mapping()  # standard gate library
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
        cals = Calibrations.load(files=calibration_files)
    else:
        cals = Calibrations(
            coupling_map=coupling_map,
            control_channel_map=physical_control_channel_map,
            libraries=[
                FixedFrequencyTransmon(basis_gates=["x", "sx"]),
                EchoedCrossResonance(basis_gates=["cr45p", "cr45m", "ecr"]),
            ],
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
        control_channels = list(
            filter(
                lambda x: x is not None,
                [control_channel_map.get((i, qubit), None) for i in qubits],
            )
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

        # sx_schedule = block_to_schedule(cals.get_schedule("sx", (qubit,)))
        # s_schedule = block_to_schedule(target.get_calibration("s", (qubit,)))
        # h_schedule = pulse.Schedule(s_schedule, sx_schedule, s_schedule, name="h")
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
    # cals.save(file_type="csv", overwrite=True, file_prefix="Custom" + backend.name)
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


def get_ecr_params(backend: Backend_type, physical_qubits: Sequence[int]):
    """
    Determine default parameters for ECR gate on provided backend (works even if basis gate of the IBM Backend is CX)
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
        default_params.update(
            {
                ("amp", physical_qubits, sched): control_pulse.amp,
                ("tgt_amp", physical_qubits, sched): target_pulse.amp
                if hasattr(target_pulse, "amp")
                else np.linalg.norm(np.max(target_pulse.samples)),
                ("angle", physical_qubits, sched): control_pulse.angle,
                ("tgt_angle", physical_qubits, sched): target_pulse.angle
                if hasattr(target_pulse, "angle")
                else np.angle(np.max(target_pulse.samples)),
                ("duration", physical_qubits, sched): control_pulse.duration,
                ("σ", physical_qubits, sched): control_pulse.sigma,
                ("risefall", physical_qubits, sched): (
                                                              control_pulse.duration - control_pulse.width
                                                      )
                                                      / (2 * control_pulse.sigma),
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
        backend: Backend_type, physical_qubit: Sequence[int], name: str = "x"
):
    """
    Determine default parameters for SX or X gate on provided backend
    """
    if not isinstance(backend, (BackendV1, BackendV2)):
        raise TypeError("Backend must be defined")
    if isinstance(backend, BackendV1):
        instruction_schedule_map = backend.defaults().instruction_schedule_map
    else:
        instruction_schedule_map = backend.target.instruction_schedule_map()
    basis_gate_inst = instruction_schedule_map.get(name, physical_qubit)
    basis_gate_instructions = np.array(basis_gate_inst.instructions)[:, 1]
    ref_pulse = basis_gate_inst.instructions[0][1].pulse
    default_params = {
        ("amp", physical_qubit, "x"): ref_pulse.amp,
        ("σ", physical_qubit, "x"): ref_pulse.sigma,
        ("β", physical_qubit, "x"): ref_pulse.beta,
        ("duration", physical_qubit, "x"): ref_pulse.duration,
        ("angle", physical_qubit, "x"): ref_pulse.angle,
    }
    pulse_features = ["amp", "angle", "duration", "σ", "β"]
    return default_params, pulse_features, basis_gate_inst, basis_gate_instructions


def state_fidelity_from_state_tomography(
        qc_list: List[QuantumCircuit],
        backend: Backend,
        physical_qubits: Optional[Sequence[int]],
        analysis: Union[BaseAnalysis, None, str] = "default",
        target_state: Optional[QuantumState] = None,
        session: Optional[Session] = None,
):
    state_tomo = BatchExperiment(
        [
            StateTomography(
                qc,
                physical_qubits=physical_qubits,
                analysis=analysis,
                target=target_state,
            )
            for qc in qc_list
        ],
        backend=backend,
        flatten_results=True,
    )
    if isinstance(backend, RuntimeBackend):
        jobs = run_jobs(session, state_tomo._transpiled_circuits())
        exp_data = state_tomo._initialize_experiment_data()
        exp_data.add_jobs(jobs)
        exp_data = state_tomo.analysis.run(exp_data).block_for_results()
    else:
        exp_data = state_tomo.run().block_for_results()

    fidelities = [
        exp_data.analysis_result("state_fidelity")[i].value for i in range(len(qc_list))
    ]
    avg_fidelity = np.mean(fidelities)
    return avg_fidelity


def run_jobs(session: Session, circuits: List[QuantumCircuit], run_options=None):
    jobs = []
    runtime_inputs = {"circuits": circuits, "skip_transpilation": True, **run_options}
    jobs.append(session.run("circuit_runner", inputs=runtime_inputs))

    return jobs


def gate_fidelity_from_process_tomography(
        qc_list: List[QuantumCircuit],
        backend: Backend,
        target_gate: Gate,
        physical_qubits: Optional[Sequence[int]],
        analysis: Union[BaseAnalysis, None, str] = "default",
        session: Optional[Session] = None,
):
    """
    Extract average gate and process fidelities from batch of Quantum Circuit for target gate
    """
    # Process tomography
    process_tomo = BatchExperiment(
        [
            ProcessTomography(
                qc,
                physical_qubits=physical_qubits,
                analysis=analysis,
                target=Operator(target_gate),
            )
            for qc in qc_list
        ],
        backend=backend,
        flatten_results=True,
    )

    if isinstance(backend, RuntimeBackend):
        circuits = process_tomo._transpiled_circuits()
        jobs = run_jobs(session, circuits)
        exp_data = process_tomo._initialize_experiment_data()
        exp_data.add_jobs(jobs)
        results = process_tomo.analysis.run(exp_data).block_for_results()
    else:
        results = process_tomo.run().block_for_results()

    process_results = [
        results.analysis_results("process_fidelity")[i].value
        for i in range(len(qc_list))
    ]
    dim, _ = Operator(target_gate).dim
    avg_gate_fid = np.mean([(dim * f_pro + 1) / (dim + 1) for f_pro in process_results])
    return avg_gate_fid


def get_control_channel_map(backend: BackendV1, qubit_tgt_register: List[int]):
    """
    Get reduced control_channel_map from Backend configuration (needs to be of type BackendV1)
    :param backend: IBM Backend instance, must carry a configuration method
    :param qubit_tgt_register: Subsystem of interest from which to build control_channel_map
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
        layout: Layout,
        config: Union[Dict, QiskitConfig],
        abstraction_level: str = "circuit",
        estimator_options: Optional[Union[Dict, AerOptions, RuntimeOptions]] = None,
) -> (Estimator_type, Sampler_type):
    """
    Retrieve appropriate Qiskit primitives (estimator and sampler) from backend and layout
    """
    if isinstance(
            backend, RuntimeBackend
    ):  # Real backend, or Simulation backend from Runtime Service
        estimator: Estimator_type = RuntimeEstimator(
            session=Session(backend.service, backend),
            options=estimator_options,
        )
        sampler: Sampler_type = RuntimeSampler(
            session=estimator.session, options=estimator_options
        )

        if estimator.options.transpilation["initial_layout"] is None:
            estimator.options.transpilation[
                "initial_layout"
            ] = layout.get_physical_bits()
            sampler.options.transpilation["initial_layout"] = layout.get_physical_bits()

    else:
        if isinstance(estimator_options, RuntimeOptions):
            # estimator_options = asdict(estimator_options)
            estimator_options = None
        if isinstance(backend, (AerBackend, FakeBackend, FakeBackendV2)):
            if abstraction_level != "circuit":
                raise ValueError(
                    "AerSimulator only works at circuit level, and a pulse gate calibration is provided"
                )
            # Estimator taking noise model into consideration, have to provide an AerSimulator backend
            estimator = AerEstimator(
                backend_options=backend.options,
                transpile_options={"initial_layout": layout},
                approximation=True,
            )
            sampler = AerSampler(
                backend_options=backend.options,
                transpile_options={"initial_layout": layout},
            )
        elif backend is None:  # No backend specified, ideal state-vector simulation
            if abstraction_level != "circuit":
                raise ValueError("Statevector simulation only works at circuit level")
            estimator = Estimator(options={"initial_layout": layout})
            sampler = Sampler(options={"initial_layout": layout})

        elif isinstance(backend, DynamicsBackend):
            assert (
                    abstraction_level == "pulse"
            ), "DynamicsBackend works only with pulse level abstraction"
            if isinstance(backend.options.solver, JaxSolver):
                estimator: Estimator_type = DynamicsBackendEstimator(
                    backend, options=estimator_options, skip_transpilation=False
                )
            else:
                estimator: Estimator_type = BackendEstimator(
                    backend, options=estimator_options, skip_transpilation=False
                )
            estimator.set_transpile_options(initial_layout=layout)
            sampler = BackendSampler(
                backend, options=estimator_options, skip_transpilation=False
            )
            if config.do_calibrations and not backend.target.has_calibration("x", (0,)):
                calibration_files: List[str] = config.calibration_files
                print("3")
                _, _ = perform_standard_calibrations(backend, calibration_files)

        else:
            raise TypeError("Backend not recognized")
    return estimator, sampler


def set_primitives_transpile_options(
        estimator, sampler, layout, skip_transpilation, physical_qubits
):
    if isinstance(estimator, RuntimeEstimator):
        # TODO: Could change resilience level
        estimator.set_options(
            optimization_level=0,
            resilience_level=0,
            skip_transpilation=skip_transpilation,
        )
        estimator.options.transpilation["initial_layout"] = physical_qubits
        sampler.set_options(**estimator.options)

    elif isinstance(estimator, AerEstimator):
        estimator._transpile_options = AerOptions(
            initial_layout=layout, optimization_level=0
        )
        estimator._skip_transpilation = skip_transpilation
        sampler_transpile_options = AerOptions(
            initial_layout=layout, optimization_level=0
        )
        sampler._skip_transpilation = skip_transpilation

    elif isinstance(estimator, BackendEstimator):
        estimator.set_transpile_options(initial_layout=layout, optimization_level=0)
        estimator._skip_transpilation = skip_transpilation
        sampler.set_transpile_options(initial_layout=layout, optimization_level=0)
        sampler._skip_transpilation = skip_transpilation

    else:
        raise TypeError(
            "Estimator primitive not recognized (must be either BackendEstimator, Aer or Runtime"
        )


def handle_session(qc, input_state_circ, estimator, backend, session_count):
    """
    Handle session reopening for RuntimeEstimator and load necessary data for DynamicsBackendEstimator
    """
    if isinstance(estimator, RuntimeEstimator):
        """Open a new Session if time limit of the ongoing one is reached"""
        if estimator.session.status() == "Closed":
            old_session = estimator.session
            session_count += 1
            print(f"New Session opened (#{session_count})")
            session, options = (
                Session(old_session.service, backend),
                estimator.options,
            )
            estimator = RuntimeEstimator(session=session, options=dict(options))
    elif isinstance(
            backend, IBMBackend
    ):  # Soon deprecated (backend.run also available in Qiskit Runtime)
        if not backend.session.active:
            session_count += 1
            print(f"New Session opened (#{session_count})")
            backend.open_session()
    elif isinstance(estimator, DynamicsBackendEstimator):
        if not isinstance(backend, DynamicsBackend) or not isinstance(
                backend.options.solver, JaxSolver
        ):
            raise TypeError(
                "DynamicsBackendEstimator can only be used with DynamicsBackend and JaxSolver"
            )
        # Update callable within the jit compiled function
        backend.options.solver.circuit_macro = lambda: schedule(qc, backend)
        # Update initial state of DynamicsBackend with input state circuit
        # The initial state is adapted to match the dimensions of the HamiltonianModel
        new_circ = transpile(input_state_circ, backend)
        subsystem_dims = backend.options.subsystem_dims
        initial_state = Statevector.from_int(0, dims=subsystem_dims)
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


def get_solver_and_freq_from_backend(
        backend: BackendV1,
        subsystem_list: Optional[List[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        static_dissipators: Optional[Array] = None,
        dissipator_operators: Optional[Array] = None,
        dissipator_channels: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Solver]:
    """
    Method to retrieve solver instance and relevant freq channels information from an IBM
    backend added with potential dissipation operators, inspired from DynamicsBackend.from_backend() method
    :param subsystem_list: The list of qubits in the backend to include in the model.
    :param rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`
    :param evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
    :param rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
            ``"auto"``, allowing this method to pick a rotating frame.
    :param backend: IBMBackend instance from which Hamiltonian parameters are extracted
    :param static_dissipators: static_dissipators: Constant dissipation operators.
    :param dissipator_operators: Dissipation operators with time-dependent coefficients.
    :param dissipator_channels: List of channel names in pulse schedules corresponding to dissipator operators.

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
            "get_solver_from_backend requires that backend.configuration() has a "
            "hamiltonian."
        )

    (
        static_hamiltonian,
        hamiltonian_operators,
        hamiltonian_channels,
        subsystem_dims,
    ) = parse_backend_hamiltonian_dict(backend_config.hamiltonian, subsystem_list)

    # construct model frequencies dictionary from backend
    channel_freqs = _get_backend_channel_freqs(
        backend_target=backend_target,
        backend_config=backend_config,
        backend_defaults=backend_defaults,
        channels=hamiltonian_channels,
    )

    # build the solver
    if rotating_frame == "auto":
        if "dense" in evaluation_mode:
            rotating_frame = static_hamiltonian
        else:
            rotating_frame = np.diag(static_hamiltonian)

    # get time step size
    if backend_target is not None and backend_target.dt is not None:
        dt = backend_target.dt
    else:
        # config is guaranteed to have a dt
        dt = backend_config.dt

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=hamiltonian_operators,
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_freqs,
        dt=dt,
        rotating_frame=rotating_frame,
        evaluation_mode=evaluation_mode,
        rwa_cutoff_freq=rwa_cutoff_freq,
        static_dissipators=static_dissipators,
        dissipator_operators=dissipator_operators,
        dissipator_channels=dissipator_channels,
    )

    return channel_freqs, solver


def build_qubit_space_projector(initial_subsystem_dims: list):
    """
    Build projector on qubit space from initial subsystem dimensions
    """
    total_dim = np.prod(initial_subsystem_dims)
    projector = Operator(
        np.zeros((total_dim, total_dim), dtype=np.complex128),
        input_dims=tuple(initial_subsystem_dims),
        output_dims=tuple(initial_subsystem_dims),
    )
    for i in range(total_dim):
        s = Statevector.from_int(i, initial_subsystem_dims)
        for key in s.to_dict().keys():
            if all(c in "01" for c in key):
                projector += s.to_operator()
                break
            else:
                continue
    return projector


def qubit_projection(unitary, subsystem_dims):
    """
    Project unitary on qubit space
    """

    proj = build_qubit_space_projector(subsystem_dims)
    new_dim = 2 ** len(subsystem_dims)
    qubitized_unitary = np.zeros((new_dim, new_dim), dtype=np.complex128)
    qubit_count1 = qubit_count2 = 0
    new_unitary = (
            proj
            @ Operator(
        unitary,
        input_dims=subsystem_dims,
        output_dims=subsystem_dims,
    )
            @ proj
    )
    for i in range(np.prod(subsystem_dims)):
        for j in range(np.prod(subsystem_dims)):
            if new_unitary.data[i, j] != 0:
                qubitized_unitary[qubit_count1, qubit_count2] = new_unitary.data[i, j]
                qubit_count2 += 1
                if qubit_count2 == new_dim:
                    qubit_count2 = 0
                    qubit_count1 += 1
                    break
    qubitized_unitary = Operator(
        qubitized_unitary,
        input_dims=(2,) * len(subsystem_dims),
        output_dims=(2,) * len(subsystem_dims),
    )
    return qubitized_unitary


def load_q_env_from_yaml_file(file_path: str):
    """
    Load Qiskit Quantum Environment from yaml file
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    low = np.array(config["ENV"]["ACTION_SPACE"]["LOW"], dtype=np.float32)
    high = np.array(config["ENV"]["ACTION_SPACE"]["HIGH"], dtype=np.float32)
    params = {
        "action_space": Box(
            low=low, high=high, shape=(config["ENV"]["N_ACTIONS"],), dtype=np.float32
        ),
        "observation_space": Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(config["ENV"]["OBSERVATION_SPACE"],),
            dtype=np.float32,
        ),
        "batch_size": config["ENV"]["BATCH_SIZE"],
        "sampling_Paulis": config["ENV"]["SAMPLING_PAULIS"],
        "n_shots": config["ENV"]["N_SHOTS"],
        "c_factor": config["ENV"]["C_FACTOR"],
        "seed": config["ENV"]["SEED"],
        "benchmark_cycle": config["ENV"]["BENCHMARK_CYCLE"],
        "target": {
            "register": config["TARGET"]["PHYSICAL_QUBITS"],
            "training_with_cal": config["ENV"]["TRAINING_WITH_CAL"],
        },
    }
    if "GATE" in config["TARGET"]:
        params["target"]["gate"] = get_standard_gate_name_mapping()[
            config["TARGET"]["GATE"].lower()
        ]
    else:
        params["target"]["dm"] = DensityMatrix.from_label(config["TARGET"]["STATE"])

    backend_params = {
        "real_backend": config["BACKEND"]["REAL_BACKEND"],
        "backend_name": config["BACKEND"]["NAME"],
        "use_dynamics": config["BACKEND"]["DYNAMICS"]["USE_DYNAMICS"],
        "physical_qubits": config["BACKEND"]["DYNAMICS"]["PHYSICAL_QUBITS"],
        "channel": config["SERVICE"]["CHANNEL"],
        "instance": config["SERVICE"]["INSTANCE"],
    }
    runtime_options = config["RUNTIME_OPTIONS"]
    check_on_exp = config["ENV"]["CHECK_ON_EXP"]
    return params, backend_params, RuntimeOptions(**runtime_options), check_on_exp


def load_agent_from_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    ppo_params = {
        "n_steps": config["AGENT"]["NUM_UPDATES"],
        "run_name": config["AGENT"]["RUN_NAME"],
        "n_updates": config["AGENT"]["NUM_UPDATES"],
        "n_epochs": config["AGENT"]["N_EPOCHS"],
        "batch_size": config["AGENT"]["MINIBATCH_SIZE"],
        "learning_rate": config["AGENT"]["LR_ACTOR"],
        # "lr_critic": config["AGENT"]["LR_CRITIC"],
        "gamma": config["AGENT"]["GAMMA"],
        "gae_lambda": config["AGENT"]["GAE_LAMBDA"],
        "ent_coef": config["AGENT"]["ENT_COEF"],
        "vf_coef": config["AGENT"]["V_COEF"],
        "max_grad_norm": config["AGENT"]["GRADIENT_CLIP"],
        "clip_range_vf": config["AGENT"]["CLIP_VALUE_LOSS"],
        "clip_range": config["AGENT"]["CLIP_RATIO"],
    }
    network_params = {
        "optimizer": config["NETWORK"]["OPTIMIZER"],
        "n_units": config["NETWORK"]["N_UNITS"],
        "activation": config["NETWORK"]["ACTIVATION"],
        "include_critic": config["NETWORK"]["INCLUDE_CRITIC"],
        "normalize_advantage": config["NETWORK"]["NORMALIZE_ADVANTAGE"],
        "checkpoint_dir": config["NETWORK"]["CHKPT_DIR"],
    }

    hpo_params = {
        "num_trials": config["HPO"]["NUM_TRIALS"],
        "n_updates": config["HPO"]["NUM_UPDATES"],
        "n_epochs": config["HPO"]["N_EPOCHS"],
        "minibatch_size": config["HPO"]["MINIBATCH_SIZE"],
        "batchsize_multiplier": config["HPO"]["BATCHSIZE_MULTIPLIER"],
        "learning_rate": config["HPO"]["LR_ACTOR"],
        "gamma": config["HPO"]["GAMMA"],
        "gae_lambda": config["HPO"]["GAE_LAMBDA"],
        "ent_coef": config["HPO"]["ENT_COEF"],
        "v_coef": config["HPO"]["V_COEF"],
        "max_grad_norm": config["HPO"]["GRADIENT_CLIP"],
        "clip_value_loss": config["HPO"]["CLIP_VALUE_LOSS"],
        "clip_value_coef": config["HPO"]["CLIP_VALUE_COEF"],
        "clip_ratio": config["HPO"]["CLIP_RATIO"],
    }

    return ppo_params, network_params, hpo_params


def create_agent_config(trial: optuna.trial.Trial, hpo_config: dict, network_config: dict, ppo_params: dict):
    agent_config = {
        'N_UPDATES': trial.suggest_int('N_UPDATES', hpo_config['n_updates'][0], hpo_config['n_updates'][1]),
        'N_EPOCHS': trial.suggest_int('N_EPOCHS', hpo_config['n_epochs'][0], hpo_config['n_epochs'][1]),
        'MINIBATCH_SIZE': trial.suggest_categorical('MINIBATCH_SIZE', hpo_config['minibatch_size']),
        'BATCHSIZE_MULTIPLIER': trial.suggest_int('BATCHSIZE_MULTIPLIER', hpo_config['batchsize_multiplier'][0], hpo_config['batchsize_multiplier'][1]),
        'LR': trial.suggest_float('LR', hpo_config['learning_rate'][0], hpo_config['learning_rate'][1], log=True),
        'GAMMA': trial.suggest_float('GAMMA', hpo_config['gamma'][0], hpo_config['gamma'][1]),
        'GAE_LAMBDA': trial.suggest_float('GAE_LAMBDA', hpo_config['gae_lambda'][0], hpo_config['gae_lambda'][1]),
        'ENT_COEF': trial.suggest_float('ENT_COEF', hpo_config['ent_coef'][0], hpo_config['ent_coef'][1]),
        'V_COEF': trial.suggest_float('V_COEF', hpo_config['v_coef'][0], hpo_config['v_coef'][1]),
        'GRADIENT_CLIP': trial.suggest_float('GRADIENT_CLIP', hpo_config['max_grad_norm'][0], hpo_config['max_grad_norm'][1]),
        'CLIP_VALUE_COEF': trial.suggest_float('CLIP_VALUE_COEF', hpo_config['clip_value_coef'][0], hpo_config['clip_value_coef'][1]),
        'CLIP_RATIO': trial.suggest_float('CLIP_RATIO', hpo_config['clip_ratio'][0], hpo_config['clip_ratio'][1]),
        }
    agent_config['BATCHSIZE'] = agent_config['MINIBATCH_SIZE'] * agent_config['BATCHSIZE_MULTIPLIER']
    # The upper hyperparameters are part of HPO scope
    hyperparams = list(agent_config.keys())

    # The following hyperparameters are NOT part of HPO scope
    agent_config['CLIP_VALUE_LOSS'] = hpo_config['clip_value_loss']

    # Add network-specific hyperparameters that are not part of HPO scope
    agent_config['OPTIMIZER'] = network_config['optimizer']
    agent_config['N_UNITS'] = network_config['n_units']
    agent_config['ACTIVATION'] = network_config['activation']
    agent_config['INCLUDE_CRITIC'] = network_config['include_critic']
    agent_config['NORMALIZE_ADVANTAGE'] = network_config['normalize_advantage']
    agent_config['RUN_NAME'] = ppo_params['run_name']

    return agent_config, hyperparams


def retrieve_backend_info(
        backend: Backend, estimator: Optional[RuntimeEstimator] = None
):
    """
    Retrieve useful Backend data to run context aware gate calibration
    """
    backend_data = BackendData(backend)
    dt = backend_data.dt if backend_data.dt is not None else 2.2222222222222221e-10
    coupling_map = CouplingMap(backend_data.coupling_map)
    if (
            coupling_map.size() == 0
            and backend_data.num_qubits > 1
            and estimator is not None
    ):
        if isinstance(estimator, RuntimeEstimator):
            coupling_map = CouplingMap(estimator.options.simulator["coupling_map"])
            if coupling_map is None:
                raise ValueError(
                    "To build a local circuit context, backend needs a coupling map"
                )

    # Check basis_gates and their respective durations of backend (for identifying timing context)
    if isinstance(backend, BackendV1):
        instruction_durations = InstructionDurations.from_backend(backend)
        basis_gates = backend.configuration().basis_gates.copy()
    elif isinstance(backend, BackendV2):
        instruction_durations = backend.instruction_durations
        basis_gates = backend.operation_names.copy()
    else:
        raise AttributeError("TorchQuantumEnvironment requires a Backend argument")
    if not instruction_durations.duration_by_name_qubits:
        raise AttributeError(
            "InstructionDurations not specified in provided Backend, required for transpilation"
        )
    return dt, coupling_map, basis_gates, instruction_durations


def retrieve_tgt_instruction_count(qc: QuantumCircuit, target: Dict):
    tgt_instruction = CircuitInstruction(
        target["gate"], [qc.qubits[i] for i in target["register"]]
    )
    return qc.data.count(tgt_instruction)


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