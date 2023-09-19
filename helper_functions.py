from itertools import permutations
from typing import Optional, Tuple, List, Union, Dict, Sequence

import numpy as np
import tensorflow as tf
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Gate, Parameter
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1, Backend, BackendV2
from qiskit.pulse.transforms import block_to_schedule
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, InstructionDurations, InstructionProperties
from qiskit_dynamics import Solver, RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend.backend_string_parser.hamiltonian_string_parser import parse_backend_hamiltonian_dict
from qiskit_dynamics.backend.dynamics_backend import _get_backend_channel_freqs, DynamicsBackend
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import BatchExperiment, BaseAnalysis, BackendData
from qiskit_experiments.library import StateTomography, ProcessTomography, RoughXSXAmplitudeCal, RoughDragCal
from qiskit_ibm_runtime import Session, IBMBackend, Estimator as Runtime_Estimator
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense

from torch_gate_calibration.basis_gate_library import EchoedCrossResonance, FixedFrequencyTransmon


def constrain_mean_value(mu_var):
    return [tf.clip_by_value(m, -1., 1.) for m in mu_var]


def constrain_std_value(std_var):
    return [tf.clip_by_value(std, 1e-3, 3) for std in std_var]

def count_gates(qc: QuantumCircuit):
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count
def remove_unused_wires(qc: QuantumCircuit):
    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if count == 0:
            qc.qubits.remove(qubit)
    return qc

def perform_standard_calibrations(backend: DynamicsBackend, calibration_files: Optional[List[str]] = None):
    """
    Generate baseline single qubit gates (X, SX, RZ, H) for all qubits using traditional calibration experiments
    :param backend: Dynamics Backend on which calibrations should be run
    :param calibration_files: Optional calibration files containing single qubit gate calibrations for provided
        DynamicsBackend instance (Qiskit Experiments does not support this feature yet)

    """

    target, qubits = backend.target, []
    for i, dim in enumerate(backend.options.subsystem_dims):
        if dim > 1:
            qubits.append(i)
    num_qubits = len(qubits)
    single_qubit_properties = {(qubit,): None for qubit in range(num_qubits)}
    single_qubit_errors = {(qubit,): 0.0 for qubit in qubits}

    control_channel_map = backend.options.control_channel_map or {(qubits[0], qubits[1]): index
                                                                  for index, qubits in
                                                                  enumerate(tuple(permutations(qubits, 2)))}
    if backend.options.control_channel_map:
        physical_control_channel_map = {(qubit_pair[0], qubit_pair[1]): backend.control_channel((qubit_pair[0],
                                                                                                 qubit_pair[1]))
                                        for qubit_pair in backend.options.control_channel_map}
    else:
        physical_control_channel_map = {(qubit_pair[0], qubit_pair[1]): [pulse.ControlChannel(index)]
                                        for index, qubit_pair in enumerate(tuple(permutations(qubits, 2)))}
    backend.set_options(control_channel_map=control_channel_map)
    coupling_map = [list(qubit_pair) for qubit_pair in control_channel_map]
    two_qubit_properties = {qubits: None for qubits in control_channel_map}
    two_qubit_errors = {qubits: 0.0 for qubits in control_channel_map}
    standard_gates: Dict[str, Gate] = get_standard_gate_name_mapping()  # standard gate library
    fixed_phase_gates, fixed_phases = ["z", "s", "sdg", "t", "tdg"], np.pi * np.array([1, 1 / 2, -1 / 2, 1 / 4, -1 / 4])
    other_gates = ["rz", "id", "h", "x", "sx", "reset"]
    single_qubit_gates = fixed_phase_gates + other_gates
    two_qubit_gates = ["ecr"]
    exp_results = {}
    existing_cals = calibration_files is not None

    phi: Parameter = standard_gates["rz"].params[0]
    if existing_cals:
        cals = Calibrations.load(files=calibration_files)
    else:
        cals = Calibrations(coupling_map=coupling_map, control_channel_map=physical_control_channel_map,
                            libraries=[FixedFrequencyTransmon(basis_gates=["x", "sx"]),
                                       EchoedCrossResonance(basis_gates=['cr45p', 'cr45m', 'ecr'])],
                            backend_name=backend.name, backend_version=backend.backend_version)
    if len(target.instruction_schedule_map().instructions) <= 1:  # Check if instructions have already been added
        for gate in single_qubit_gates:
            target.add_instruction(standard_gates[gate], properties=single_qubit_properties)
        if num_qubits > 1:
            for gate in two_qubit_gates:
                target.add_instruction(standard_gates[gate], properties=two_qubit_properties)
            target.build_coupling_map(two_q_gate=two_qubit_gates[0])

    for qubit in qubits:  # Add calibrations for each qubit
        control_channels = list(filter(lambda x: x is not None, [control_channel_map.get((i, qubit), None)
                                                                 for i in qubits]))
        # Calibration of RZ gate, virtual Z-rotation
        with pulse.build(backend, name=f"rz{qubit}") as rz_cal:
            pulse.shift_phase(-phi, pulse.DriveChannel(qubit))
            for q in control_channels:
                pulse.shift_phase(-phi, pulse.ControlChannel(q))
        # Identity gate
        id_cal = pulse.Schedule(pulse.Delay(20, pulse.DriveChannel(qubit)))  # Wait 20 cycles for identity gate

        # Update backend Target by adding calibrations for all phase gates (fixed angle virtual Z-rotations)
        target.update_instruction_properties('rz', (qubit,), InstructionProperties(calibration=rz_cal, error=0.))
        target.update_instruction_properties('id', (qubit,), InstructionProperties(calibration=id_cal, error=0.))
        target.update_instruction_properties("reset", (qubit,), InstructionProperties(calibration=id_cal, error=0.))
        for phase, gate in zip(fixed_phases, fixed_phase_gates):
            gate_cal = rz_cal.assign_parameters({phi: phase}, inplace=False)
            instruction_prop = InstructionProperties(calibration=gate_cal, error=0.)
            target.update_instruction_properties(gate, (qubit,), instruction_prop)

        # Perform calibration experiments (Rabi/Drag) for calibrating X and SX gates
        if not existing_cals:
            rabi_exp = RoughXSXAmplitudeCal([qubit], cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 100))
            drag_exp = RoughDragCal([qubit], cals, backend=backend, betas=np.linspace(-20, 20, 15))
            drag_exp.set_experiment_options(reps=[3, 5, 7])
            print(f"Starting Rabi experiment for qubit {qubit}...")
            rabi_result = rabi_exp.run().block_for_results()
            print(f"Rabi experiment for qubit {qubit} done.")
            print(f"Starting Drag experiment for qubit {qubit}...")
            drag_result = drag_exp.run().block_for_results()
            print(f"Drag experiments done for qubit {qubit} done.")
            exp_results[qubit] = [rabi_result, drag_result]

        # Build Hadamard gate schedule from following equivalence: H = S @ SX @ S

        sx_schedule = block_to_schedule(cals.get_schedule("sx", (qubit,)))
        s_schedule = block_to_schedule(target.get_calibration('s', (qubit,)))
        h_schedule = pulse.Schedule(s_schedule, sx_schedule, s_schedule, name="h")
        target.update_instruction_properties('h', (qubit,), properties=InstructionProperties(calibration=h_schedule,
                                                                                             error=0.0))

    print("All single qubit calibrations are done")
    # cals.save(file_type="csv", overwrite=True, file_prefix="Custom" + backend.name)
    error_dict = {'x': single_qubit_errors, 'sx': single_qubit_errors}
    target.update_from_instruction_schedule_map(cals.get_inst_map(), error_dict=error_dict)
    print(control_channel_map)
    # for qubit_pair in control_channel_map:
    #     print(qubit_pair)
    #     cr_ham_exp = CrossResonanceHamiltonian(physical_qubits=qubit_pair, flat_top_widths=np.linspace(0, 5000, 17),
    #                                            backend=backend)
    #     print("Calibrating CR for qubits", qubit_pair, "...")
    #     data_cr = cr_ham_exp.run().block_for_results()
    #     exp_results[qubit_pair] = data_cr

    print("Updated Instruction Schedule Map", target.instruction_schedule_map())

    return cals, exp_results
def determine_ecr_params(backend: Union[BackendV1, BackendV2], physical_qubits: List[int]):
    basis_gate = None
    basis_gates = backend.configuration().basis_gates if isinstance(backend, BackendV1) else backend.operation_names
    if "cx" in basis_gates:
        basis_gate = "cx"
        print("Basis gate Library for CX gate not yet available, will be transpiled over ECR basis gate")
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
    basis_gate_instructions = instruction_schedule_map.get(basis_gate, qubits=physical_qubits)
    instructions_array = np.array(basis_gate_instructions.instructions)[:,1]
    control_pulse = target_pulse = x_pulse = None

    if isinstance(backend, DynamicsBackend):
        x_pulse = instruction_schedule_map.get("x", q_c).instructions[0][1].pulse
        cr45p_instructions = np.array(instruction_schedule_map.get("cr45p", physical_qubits).instructions)[:,1]
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
    default_params = {("amp", q_c, "x"): x_pulse.amp,
                      ("σ", q_c, "x"): x_pulse.sigma,
                      ("β", q_c, "x"): x_pulse.beta,
                      ("duration", q_c, "x"): x_pulse.duration,
                      ("angle", q_c, "x"): x_pulse.angle}
    for sched in ["cr45p", "cr45m"]:
        default_params.update({("amp", physical_qubits, sched): control_pulse.amp,
                               ("tgt_amp", physical_qubits, sched): target_pulse.amp if hasattr(target_pulse, 'amp') else np.linalg.norm(np.max(target_pulse.samples)),
                               ("angle", physical_qubits, sched): control_pulse.angle,
                               ("tgt_angle", physical_qubits, sched): target_pulse.angle if hasattr(target_pulse, 'angle') else np.angle(np.max(target_pulse.samples)),
                               ("duration", physical_qubits, sched): control_pulse.duration,
                               ("σ", physical_qubits, sched): control_pulse.sigma,
                               ("risefall", physical_qubits, sched) : (control_pulse.duration - control_pulse.width)/(2*control_pulse.sigma)})

    return default_params, basis_gate_instructions, instructions_array

def get_schedule_dict(sched: Union[pulse.Schedule, pulse.ScheduleBlock]):
    """
    To be used for custom Qiskit Dynamics simulation with DynamicsBackend, format pulse Schedule in a Jax Pytree structure to
    speed up simulations by jitting the schedule to samples conversion

    """
    new_sched = block_to_schedule(sched) if isinstance(sched, pulse.ScheduleBlock) else sched
    assert new_sched.is_parameterized()
    instructions_array = np.array(new_sched.instructions)[:, 1]
    instructions_info = []
    for instruction in instructions_array:
        if isinstance(instruction, pulse.Play):
            if isinstance(instruction.pulse, pulse.ScalableSymbolicPulse):
                info = {"type": "Play", "channel": instruction.channel,
                                    "pulse_type": instruction.pulse.pulse_type, "parameters": instruction.pulse.parameters}
                instructions_info.append(info)
            else:
                raise QiskitError(f"{instruction.pulse} not JAX compatible")
        elif isinstance(instruction, pulse.ShiftPhase):
            info = {"type": "ShiftPhase", "channel": instruction.channel, "parameters": instruction.phase}
            instructions_info.append(info)
        elif isinstance(instruction, pulse.ShiftFrequency):
            info = {"type": "ShiftFrequency", "channel": instruction.channel, "parameters": instruction.frequency}
            instructions_info.append(info)
        elif isinstance(instruction, pulse.SetPhase):
            info = {"type": "SetPhase", "channel": instruction.channel, "parameters": instruction.phase}
            instructions_info.append(info)
        elif isinstance(instruction, pulse.SetFrequency):
            info = {"type": "SetFrequency", "channel": instruction.channel, "parameters": instruction.frequency}
            instructions_info.append(info)
        elif isinstance(instruction, pulse.instructions.RelativeBarrier):
            info = {"type": "RelativeBarrier", "channel": instruction.channels}
            instructions_info.append(info)
        elif isinstance(instruction, pulse.instructions.Acquire):
            info = {"type": "Acquire", "channel": instruction.channel, "parameters": [instruction.duration, instruction.reg_slot, instruction.mem_slot]}
            instructions_info.append(info)
        elif isinstance(instruction, pulse.instructions.Delay):
            info = {"type": "Delay", "channel": instruction.channel, "parameters": instruction.duration}
            instructions_info.append(info)
        elif isinstance(instruction, pulse.instructions.Reference):
            pass

    return instructions_info

def state_fidelity_from_state_tomography(qc_list: List[QuantumCircuit], backend: Backend,
                                         measurement_indices: Optional[Sequence[int]],
                                         analysis: Union[BaseAnalysis, None, str] = "default"):
    state_tomo = BatchExperiment([StateTomography(qc, measurement_indices=measurement_indices, analysis=analysis)
                                  for qc in qc_list], backend=backend)
    exp_data = state_tomo.run().block_for_results()
    child_data = exp_data.child_data()
    exp_results = [data.analysis_result() for data in child_data]
    fidelities = [data.analysis_result("state_fidelity").value for data in child_data]

    return fidelities, exp_results

def run_jobs(session: Session, circuits: List[QuantumCircuit], run_options = None):
    jobs = []
    runtime_inputs = {'circuits': circuits,
                      'skip_transpilation': True, **run_options}
    jobs.append(session.run('circuit_runner', inputs=runtime_inputs))

    return jobs

def gate_fidelity_from_process_tomography(qc_list: List[QuantumCircuit], backend: Backend,
                                          target_gate: Gate, physical_qubits: Optional[Sequence[int]],
                                          analysis: Union[BaseAnalysis, None, str] = "default",
                                          session: Optional[Session]=None):
    """
    Extract average gate and process fidelities from batch of Quantum Circuit for target gate
    """
    # Process tomography
    process_tomo = BatchExperiment([ProcessTomography(qc, physical_qubits=physical_qubits, analysis=analysis,
                                                      target=Operator(target_gate))for qc in qc_list],
                                   backend=backend,
                                   flatten_results=True)

    if isinstance(backend, IBMBackend):
        circuits = process_tomo._transpiled_circuits()
        jobs = run_jobs(session, circuits)
        exp_data = process_tomo._initialize_experiment_data()
        exp_data.add_jobs(jobs)
        results = process_tomo.analysis.run(exp_data).block_for_results()
    else:
        results = process_tomo.run().block_for_results()

    process_results = [results.analysis_results("process_fidelity")[i].value for i in range(len(qc_list))]
    dim, _= Operator(target_gate).dim
    avg_gate_fid = np.mean([(dim* f_pro + 1)/ (dim+1) for f_pro in process_results])
    return avg_gate_fid


def get_control_channel_map(backend: BackendV1, qubit_tgt_register: List[int]):
    """
    Get reduced control_channel_map from Backend configuration (needs to be of type BackendV1)
    :param backend: IBM Backend instance, must carry a configuration method
    :param qubit_tgt_register: Subsystem of interest from which to build control_channel_map
    """
    control_channel_map = {}
    control_channel_map_backend = {
        qubits: backend.configuration().control_channels[qubits][0].index for qubits in
        backend.configuration().control_channels}
    for qubits in control_channel_map_backend:
        if qubits[0] in qubit_tgt_register and qubits[1] in qubit_tgt_register:
            control_channel_map[qubits] = control_channel_map_backend[qubits]
    return control_channel_map


def get_solver_and_freq_from_backend(backend: BackendV1,
                                     subsystem_list: Optional[List[int]] = None,
                                     rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
                                     evaluation_mode: str = "dense",
                                     rwa_cutoff_freq: Optional[float] = None,
                                     static_dissipators: Optional[Array] = None,
                                     dissipator_operators: Optional[Array] = None,
                                     dissipator_channels: Optional[List[str]] = None, ) \
        -> Tuple[Dict[str, float], Solver]:
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
        dissipator_channels=dissipator_channels
    )

    return channel_freqs, solver

def retrieve_backend_info(backend: Backend, estimator: Optional[Runtime_Estimator]=None):
    """
    Retrieve useful Backend data to run context aware gate calibration
    """
    backend_data = BackendData(backend)
    dt = backend_data.dt if backend_data.dt is not None else 2.2222222222222221e-10
    coupling_map = CouplingMap(backend_data.coupling_map)
    if coupling_map.size() == 0 and estimator is not None:
        coupling_map = CouplingMap(estimator.options.simulator["coupling_map"])
        if coupling_map is None:
            raise ValueError("To build a local circuit context, backend needs a coupling map")
    # Check basis_gates and their respective durations of backend (for identifying timing context)
    if isinstance(backend, BackendV1):
        instruction_durations = InstructionDurations.from_backend(backend)
        basis_gates = backend.configuration().basis_gates
    elif isinstance(backend, BackendV2):
        instruction_durations = backend.instruction_durations
        basis_gates = backend.operation_names
    else:
        raise AttributeError("TorchQuantumEnvironment requires a Backend argument")
    if not instruction_durations.duration_by_name_qubits:
        raise AttributeError("InstructionDurations not specified in provided Backend, required for transpilation")
    return dt, coupling_map,  basis_gates, instruction_durations
def select_optimizer(lr: float, optimizer: str = "Adam", grad_clip: Optional[float] = None,
                     concurrent_optimization: bool = True, lr2: Optional[float] = None):
    if concurrent_optimization:
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=lr, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return tf.optimizers.SGD(learning_rate=lr, clipvalue=grad_clip)
    else:
        if optimizer == 'Adam':
            return tf.optimizers.Adam(learning_rate=lr), tf.optimizers.Adam(learning_rate=lr2, clipvalue=grad_clip)
        elif optimizer == 'SGD':
            return tf.optimizers.SGD(learning_rate=lr), tf.optimizers.SGD(learning_rate=lr2, clipvalue=grad_clip)


def generate_model(input_shape: Tuple, hidden_units: Union[List, Tuple], n_actions: int,
                   actor_critic_together: bool = True,
                   hidden_units_critic: Optional[Union[List, Tuple]] = None):
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
    Net = Dense(hidden_units[0], activation='relu', input_shape=input_shape,
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{0}")(input_layer)
    for i in range(1, len(hidden_units)):
        Net = Dense(hidden_units[i], activation='relu', kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                    bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{i}")(Net)

    mean_param = Dense(n_actions, activation='tanh', name='mean_vec')(Net)  # Mean vector output
    sigma_param = Dense(n_actions, activation="softplus", name="sigma_vec")(Net)  # Diagonal elements of cov matrix
    # output

    if actor_critic_together:
        critic_output = Dense(1, activation='linear', name="critic_output")(Net)
        return Model(inputs=input_layer, outputs=[mean_param, sigma_param, critic_output])
    else:
        assert hidden_units_critic is not None, "Network structure for critic network not provided"
        input_critic = Input(shape=input_shape)
        Critic_Net = Dense(hidden_units_critic[0], activation='relu', input_shape=input_shape,
                           kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{0}")(input_critic)
        for i in range(1, len(hidden_units)):
            Critic_Net = Dense(hidden_units[i], activation='relu',
                               kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                               bias_initializer=tf.initializers.RandomNormal(stddev=0.5), name=f"hidden_{i}")(
                Critic_Net)
            critic_output = Dense(1, activation='linear', name="critic_output")(Critic_Net)
            return Model(inputs=input_layer, outputs=[mean_param, sigma_param]), \
                Model(inputs=input_critic, outputs=critic_output)
