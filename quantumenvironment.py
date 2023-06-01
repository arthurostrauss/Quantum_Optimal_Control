"""
Class to generate a RL environment suitable for usage with TF-agents, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
"""
# Qiskit imports
from qiskit import pulse, schedule
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.transpiler import InstructionProperties
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, Gate, CircuitInstruction
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers import BackendV1, BackendV2

from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import average_gate_fidelity, state_fidelity, process_fidelity
# Qiskit dynamics for pulse simulation (benchmarking)
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit_dynamics.array import Array, wrap
from qiskit_dynamics.models import HamiltonianModel

# Qiskit Experiments for generating reliable baseline for more complex gate calibrations / state preparations
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal, RoughDragCal
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.library import ProcessTomography
from qiskit_experiments.library.tomography.basis import PauliPreparationBasis  # , Pauli6PreparationBasis
# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit.primitives import Estimator, BackendEstimator
from qiskit_ibm_runtime import Estimator as Runtime_Estimator, IBMBackend as Runtime_Backend, Options
from qiskit_aer.primitives import Estimator as Aer_Estimator
from qiskit_aer.backends.aerbackend import AerBackend

import numpy as np
from itertools import product, permutations
from typing import Dict, Union, Optional, List, Callable
from copy import deepcopy
import json
from qconfig import QiskitConfig

# QUA imports
# from qualang_tools.bakery.bakery import baking
# from qm.qua import *
# from qm.QuantumMachinesManager import QuantumMachinesManager

# Tensorflow modules
from tensorflow_probability.python.distributions import Categorical
from tf_agents.typing import types

import jax

jit = wrap(jax.jit, decorator=True)


def perform_standard_calibrations(backend: DynamicsBackend, calibration_files: Optional[List[str]] = None):
    """
    Generate baseline single qubit gates (X, SX, RZ, H) for all qubits using traditional calibration experiments
    :param backend: Dynamics Backend on which calibrations should be run
    :param calibration_files: Optional calibration files containing single qubit gate calibrations for provided
        DynamicsBackend instance (Qiskit Experiments does not support this feature yet)

    """

    target, num_qubits, qubits = backend.target, backend.num_qubits, list(range(backend.num_qubits))
    single_qubit_properties = {(qubit,): None for qubit in range(num_qubits)}
    single_qubit_errors = {(qubit,): 0.0 for qubit in qubits}

    control_channel_map = backend.options.control_channel_map or {(ctrl, tgt): None
                                                                  for ctrl, tgt in permutations(qubits, 2)}
    two_qubit_properties = {qubits: None for qubits in control_channel_map}

    # fixed_phase_gates = [(ZGate(), np.pi), (SGate(), np.pi / 2), (SdgGate(), -np.pi / 2), (TGate(), np.pi / 4),
    #                      (TdgGate(), -np.pi / 4)]
    standard_gates: Dict[str, Gate] = get_standard_gate_name_mapping()  # standard gate library
    fixed_phase_gates = ["z", "s", "sdg", "t", "tdg"]
    fixed_phases = np.pi * np.array([1, 1/2, -1/2, 1/4, -1/4])
    other_gates = ["rz", "id", "h", "x", "sx", "reset"]
    single_qubit_gates = fixed_phase_gates + other_gates
    two_qubit_gates = ["cx"]
    exp_results = {}
    existing_cals = calibration_files is not None

    phi: Parameter = standard_gates["rz"].params[0]
    if existing_cals:
        cals = Calibrations.load(files=calibration_files)
    else:
        cals = Calibrations(libraries=[FixedFrequencyTransmon(basis_gates=["x", "sx"])])
    if len(target.instruction_schedule_map().instructions) <= 1:  # Check if instructions have already been added
        for gate in single_qubit_gates:
            target.add_instruction(standard_gates[gate], properties=single_qubit_properties)
        if num_qubits > 1:
            for gate in two_qubit_gates:
                target.add_instruction(standard_gates[gate], properties=two_qubit_properties)

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
    print("All calibrations are done")
    # cals.save(file_type="csv", overwrite=True, file_prefix="Custom" + backend.name)
    error_dict = {'x': single_qubit_errors, 'sx': single_qubit_errors}
    target.update_from_instruction_schedule_map(cals.get_inst_map(), error_dict=error_dict)
    print("Updated Instruction Schedule Map", target.instruction_schedule_map())
    return cals, exp_results


def _calculate_chi_target_state(target_state: Dict, n_qubits: int):
    """
    Calculate for all P
    :param target_state: Dictionary containing info on target state (name, density matrix)
    :param n_qubits: Number of qubits
    :return: Target state supplemented with appropriate "Chi" key
    """
    assert 'dm' in target_state, 'No input data for target state, provide DensityMatrix'
    d = 2 ** n_qubits
    Pauli_basis = pauli_basis(num_qubits=n_qubits)
    target_state["Chi"] = np.array([np.trace(np.array(target_state["dm"].to_operator())
                                             @ Pauli_basis[k].to_matrix()).real
                                    for k in range(d ** 2)])
    # Real part is taken to convert it in good format,
    # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
    return target_state


def _define_target(target: Dict):

    if "register" in target:
        assert isinstance(target["register"], (List, QuantumRegister)), "Register should be of type List[int] " \
                                                                        "or Quantum Register"
    tgt_register = target.get("register", None)

    if 'gate' not in target and 'circuit' not in target and 'dm' not in target:
        raise KeyError("No target provided, need to have one of the following: 'gate' for gate calibration,"
                       " 'circuit' or 'dm' for state preparation")
    if ('gate' in target and 'circuit' in target) or ('gate' in target and 'dm' in target):
        raise KeyError("Cannot have simultaneously a gate target and a state target")
    if "circuit" in target or "dm" in target:
        target["target_type"] = "state"
        if 'circuit' in target:
            assert isinstance(target["circuit"], QuantumCircuit), "Provided circuit is not a qiskit.QuantumCircuit " \
                                                                  "object"
            target["dm"] = DensityMatrix(target["circuit"])

        assert 'dm' in target, 'no DensityMatrix or circuit argument provided to target dictionary'
        assert isinstance(target["dm"], DensityMatrix), 'Provided dm is not a DensityMatrix object'
        dm: DensityMatrix = target["dm"]
        n_qubits = dm.num_qubits

        if tgt_register is None:
            tgt_register = list(range(n_qubits))

        return _calculate_chi_target_state(target, n_qubits), "state", tgt_register, n_qubits

    elif "gate" in target:
        target["target_type"] = "gate"
        assert isinstance(target["gate"], Gate), "Provided gate is not a qiskit.circuit.Gate operation"
        gate: Gate = target["gate"]
        n_qubits = target["gate"].num_qubits
        if tgt_register is None:
            tgt_register = QuantumRegister(n_qubits)

        assert gate.num_qubits == len(tgt_register), f"Target gate number of qubits ({gate.num_qubits}) " \
                                                     f"incompatible with indicated 'register' ({len(tgt_register)})"
        if 'input_states' not in target:
            target['input_states'] = [{"circuit": PauliPreparationBasis().circuit(s).decompose()}
                                      for s in product(range(4), repeat=len(tgt_register))]

            # target['input_states'] = [{"dm": Pauli6PreparationBasis().matrix(s),
            #                            "circuit": CircuitOp(Pauli6PreparationBasis().circuit(s).decompose())}
            #                           for s in product(range(6), repeat=len(tgt_register))]

        for i, input_state in enumerate(target["input_states"]):
            if 'circuit' not in input_state:
                raise KeyError("'circuit' key missing in input_state")
            assert isinstance(input_state["circuit"], QuantumCircuit), "Provided circuit is not a" \
                                                                       "qiskit.QuantumCircuit object"

            input_circuit: QuantumCircuit = input_state['circuit']
            input_state['dm'] = DensityMatrix(input_circuit)

            if isinstance(tgt_register, QuantumRegister):
                state_target_circuit = QuantumCircuit(tgt_register)
            else:
                state_target_circuit = QuantumCircuit(len(tgt_register))

            state_target_circuit.append(input_circuit.to_instruction(), tgt_register)
            state_target_circuit.append(CircuitInstruction(gate, tgt_register))
            input_state['target_state'] = {"dm": DensityMatrix(state_target_circuit),
                                           "circuit": state_target_circuit,
                                           "target_type": "state"}
            input_state['target_state'] = _calculate_chi_target_state(input_state['target_state'], n_qubits)
        return target, "gate", tgt_register, n_qubits
    else:
        raise KeyError('target type not identified, must be either gate or state')


class QuantumEnvironment:

    def __init__(self, target: Dict, abstraction_level: str = 'circuit',
                 Qiskit_config: Optional[QiskitConfig] = None,
                 QUA_config: Optional[Dict] = None,
                 sampling_Pauli_space: int = 10, n_shots: int = 1, c_factor: float = 0.5):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param target: control target of interest (can be either a gate to be calibrated or a state to be prepared)
            Target should be a Dict containing target type ('gate' or 'state') as well as necessary fields to initiate
            the calibration (cf. example)
        :param abstraction_level: Circuit or pulse level parametrization of action space
        :param Qiskit_config: Dictionary containing all info for running Qiskit program (i.e. service, backend,
            options, parametrized_circuit)
        :param QUA_config: Dictionary containing all infor for running a QUA program
        :param sampling_Pauli_space: Number of samples to build fidelity estimator for one action
        :param n_shots: Number of shots to sample for one specific computation (action/Pauli expectation sampling)
        :param c_factor: Scaling factor for reward normalization
        """

        assert abstraction_level == 'circuit' or abstraction_level == 'pulse', 'Abstraction layer parameter can be' \
                                                                               ' either pulse or circuit'
        self.abstraction_level: str = abstraction_level
        if Qiskit_config is None and QUA_config is None:
            raise AttributeError("QuantumEnvironment requires one software configuration (can be Qiskit or QUA based)")
        if Qiskit_config is not None and QUA_config is not None:
            raise AttributeError("Cannot provide simultaneously a QUA setup and a Qiskit config ")
        elif Qiskit_config is not None:
            self._config_type = "Qiskit"
            self._config = Qiskit_config
            self.target, self.target_type, self.tgt_register, self._n_qubits = _define_target(target)

            self._d = 2 ** self.n_qubits
            self.c_factor = c_factor
            self.sampling_Pauli_space = sampling_Pauli_space
            self.n_shots = n_shots

            self.backend: Union[BackendV1, BackendV2, None] = Qiskit_config.backend
            self.parametrized_circuit_func: Callable = Qiskit_config.parametrized_circuit
            estimator_options: Union[Options, Dict] = Qiskit_config.estimator_options
            self.Pauli_ops = pauli_basis(num_qubits=self._n_qubits)

            if isinstance(self.backend, Runtime_Backend):  # Real backend, or Simulation backend from Runtime Service
                self.estimator = Runtime_Estimator(session=self.backend, options=estimator_options)

            elif self.abstraction_level == "circuit":  # Either state-vector simulation (native) or AerBackend provided
                if isinstance(self.backend, AerBackend):
                    # Estimator taking noise model into consideration, have to provide an AerBackend
                    # TODO: Extract from TranspilationOptions a dict that can go in following definition
                    self.estimator = Aer_Estimator(backend_options=self.backend.options, transpile_options=None)

                else:  # No backend specified
                    self.estimator = Estimator()  # Estimator based on state-vector simulation
            elif self.abstraction_level == 'pulse':
                self.estimator = BackendEstimator(self.backend)

                if isinstance(self.backend, DynamicsBackend):
                    calibration_files: List[str] = Qiskit_config.calibration_files
                    self.calibrations, self.exp_results = perform_standard_calibrations(self.backend, calibration_files)
                    self.pulse_backend: DynamicsBackend = deepcopy(self.backend)
                # For benchmarking the gate at each epoch, set the tools for Pulse level simulator
                self.solver: Solver = Qiskit_config.solver  # Custom Solver
                # Can describe noisy channels, if none provided, pick Solver associated to DynamicsBackend by default
                self.model_dim = self.solver.model.dim
                self.channel_freq = Qiskit_config.channel_freq
                if isinstance(self.solver.model, HamiltonianModel):
                    self.y_0 = Array(np.eye(self.model_dim))
                    self.ground_state = Array(np.array([1.0] + [0.0] * (self.model_dim - 1)))
                else:
                    self.y_0 = Array(np.eye(self.model_dim ** 2))
                    self.ground_state = Array(np.array([1.0] + [0.0] * (self.model_dim ** 2 - 1)))
        elif QUA_config is not None:
            self._config_type = "QUA"
            self._config = QUA_config
            raise AttributeError("QUA compatibility not yet implemented")
            # TODO: Add a QUA program

        # Data storage for TF-Agents or plotting
        self._step_tracker = 0
        self.action_history = []
        self.density_matrix_history = []
        self.reward_history = []
        self.qc_history = []
        if self.target_type == 'gate':
            self.process_fidelity_history = []
            self.avg_fidelity_history = []
            self.built_unitaries = []
        else:
            self.state_fidelity_history = []

    def perform_action(self, actions: types.NestedTensorOrArray, do_benchmark: bool = True):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :param do_benchmark: Indicates if actual fidelity computation should be done on top of reward computation
        :return: Reward table (reward for each run in the batch)
        """
        if isinstance(self.tgt_register, QuantumRegister):
            qc = QuantumCircuit(self.tgt_register)  # Reset the QuantumCircuit instance for next iteration
            parametrized_circ = QuantumCircuit(self.tgt_register)
        else:
            qc = QuantumCircuit(self._n_qubits)
            parametrized_circ = QuantumCircuit(self._n_qubits)
        angles, batch_size = np.array(actions), len(np.array(actions))
        self.action_history.append(angles)

        if self.target_type == 'gate':
            # Pick random input state from the list of possible input states (forming a tomographically complete set)
            index = np.random.randint(len(self.target["input_states"]))
            input_state = self.target["input_states"][index]
            target_state = input_state["target_state"]  # Ideal output state associated to input (Gate |input>=|output>)
            # Append input state circuit to full quantum circuit for gate calibration
            qc.append(input_state["circuit"].to_instruction(), self.tgt_register)

        else:  # State preparation task
            target_state = self.target
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = np.round([self.c_factor * target_state["Chi"][p] / (self._d * distribution.prob(p))
                                  for p in pauli_index], 5)

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp.from_list([(self.Pauli_ops[p].to_label(), reward_factor[i])
                                               for i, p in enumerate(pauli_index)])

        # Benchmarking block: not part of reward calculation
        # Apply parametrized quantum circuit (action), for benchmarking only

        self.parametrized_circuit_func(parametrized_circ)
        qc_list = [parametrized_circ.bind_parameters(angle_set) for angle_set in angles]
        self.qc_history.append(qc_list)
        if do_benchmark:
            self._store_benchmarks(qc_list)

        # Build full quantum circuit: concatenate input state prep and parametrized unitary
        self.parametrized_circuit_func(qc)
        job = self.estimator.run(circuits=[qc] * batch_size, observables=[observables] * batch_size,
                                 parameter_values=angles, shots=self.sampling_Pauli_space * self.n_shots,
                                 job_tags=[f"rl_qoc_step{self._step_tracker}"])
        self._step_tracker += 1
        reward_table = job.result().values
        self.reward_history.append(reward_table)
        assert len(reward_table) == batch_size
        return reward_table  # Shape [batchsize]

    def _store_benchmarks(self, qc_list: List[QuantumCircuit]):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        """

        # Circuit list for each action of the batch
        if self.abstraction_level == 'circuit':

            q_state_list = [Statevector.from_instruction(qc) for qc in qc_list]
            density_matrix = DensityMatrix(np.mean([q_state.to_operator().to_matrix() for q_state in q_state_list],
                                                   axis=0))
            self.density_matrix_history.append(density_matrix)

            if self.target_type == 'state':
                self.state_fidelity_history.append(state_fidelity(self.target["dm"], density_matrix))
            else:  # Gate calibration task
                q_process_list = [Operator(qc) for qc in qc_list]

                prc_fidelity = np.mean([process_fidelity(q_process, Operator(self.target["gate"]))
                                        for q_process in q_process_list])
                avg_fidelity = np.mean([average_gate_fidelity(q_process, Operator(self.target["gate"]))
                                        for q_process in q_process_list])
                self.built_unitaries.append(q_process_list)
                self.process_fidelity_history.append(prc_fidelity)  # Avg process fidelity over the action batch
                self.avg_fidelity_history.append(avg_fidelity)  # Avg gate fidelity over the action batch
                # for i, input_state in enumerate(self.target["input_states"]):
                #     output_states = [DensityMatrix(Operator(qc) @ input_state["dm"] @ Operator(qc).adjoint())
                #                      for qc in qc_list]
                #     self.input_output_state_fidelity_history[i].append(
                #         np.mean([state_fidelity(input_state["target_state"]["dm"],
                #                                 output_state) for output_state in output_states]))
        elif self.abstraction_level == 'pulse':
            # Pulse simulation
            schedule_list = [schedule(qc, backend=self.backend, dt=self.backend.target.dt) for qc in qc_list]
            unitaries = self._simulate_pulse_schedules(schedule_list)
            # TODO: Line below yields an error if simulation is not done over a set of qubit (fails if third level of
            # TODO: transmon is simulated), adapt the target gate operator accordingly.
            unitaries = [Operator(np.array(unitary.y[0])) for unitary in unitaries]

            if self.target_type == 'state':
                density_matrix = DensityMatrix(np.mean([Statevector.from_int(0, dims=self._d).evolve(unitary)
                                                        for unitary in unitaries]))
                self.state_fidelity_history.append(state_fidelity(self.target["dm"], density_matrix))
            else:
                self.process_fidelity_history.append(np.mean([process_fidelity(unitary, self.target["gate"])
                                                              for unitary in unitaries]))
                self.avg_fidelity_history.append(np.mean([average_gate_fidelity(unitary, self.target["gate"])
                                                          for unitary in unitaries]))
            self.built_unitaries.append(unitaries)

    def gate_fidelity_from_process_tomography(self, qc_list: List[QuantumCircuit]):
        """
        Extract average gate and process fidelities from batch of Quantum Circuit for target gate
        """
        # Process tomography
        assert self.target_type == 'gate', "Target must be of type gate"
        avg_gate_fidelity = 0.
        prc_fidelity = 0.
        batch_size = len(qc_list)
        for qc in qc_list:
            process_tomography_exp = ProcessTomography(qc, backend=self.pulse_backend,
                                                       physical_qubits=self.tgt_register)

            results = process_tomography_exp.run(self.pulse_backend).block_for_results()
            process_results = results.analysis_results(0)
            Choi_matrix = process_results.value
            avg_gate_fidelity += average_gate_fidelity(Choi_matrix, Operator(self.target["gate"]))
            prc_fidelity += process_fidelity(Choi_matrix, Operator(self.target["gate"]))

        self.process_fidelity_history.append(prc_fidelity / batch_size)
        self.avg_fidelity_history.append(avg_gate_fidelity / batch_size)
        return avg_gate_fidelity / batch_size, prc_fidelity / batch_size

    def _simulate_pulse_schedules(self, schedule_list: List[Union[pulse.Schedule, pulse.ScheduleBlock]]):
        """
        Method used to simulate pulse schedules, jit compatible
        """
        time_f = self.backend.target.dt * schedule_list[0].duration
        unitaries = self.solver.solve(
            t_span=Array([0.0, time_f]),
            y0=self.y_0,
            t_eval=[time_f],
            signals=schedule_list,
            method="jax_odeint",
        )

        return unitaries

    def clear_history(self):
        self.qc_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        if self.target_type == 'gate':
            self.avg_fidelity_history.clear()
            self.process_fidelity_history.clear()
            self.built_unitaries.clear()

        else:
            self.state_fidelity_history.clear()
            self.density_matrix_history.clear()

    def __repr__(self):
        string = f"QuantumEnvironment composed of {self._n_qubits}, \n"
        string += f"Defined target: {self.target_type} " \
                  f"({self.target.get('gate', None) if not None else self.target['dm']})\n"
        string += f"Defined backend: {self.backend},\n"
        string += f"Abstraction level: {self.abstraction_level},\n"
        return string

    @property
    def n_qubits(self):
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        assert isinstance(n_qubits, int) and n_qubits > 0, "n_qubits must be a positive integer"
        self._n_qubits = n_qubits

    @property
    def config_type(self):
        return self._config_type

    @property
    def config(self):
        return self._config

    def to_json(self):
        return json.dumps({"n_qubits": self.n_qubits, "config": self._config,
                           "abstraction_level": self.abstraction_level,
                           "sampling_Pauli_space": self.sampling_Pauli_space,
                           "n_shots": self.n_shots,
                           "target_type": self.target_type,
                           "target": self.target,
                           "c_factor": self.c_factor,
                           "reward_history": self.reward_history,
                           "action_history": self.action_history,
                           "fidelity_history": self.avg_fidelity_history if self.target_type == "gate"
                           else self.state_fidelity_history,

                           })

    @classmethod
    def from_json(cls, json_str):
        """Return a MyCustomClass instance based on the input JSON string."""

        class_info = json.loads(json_str)
        abstraction_level = class_info["abstraction_level"]
        target = class_info["target"]
        n_shots = class_info["n_shots"]
        c_factor = class_info["c_factor"]
        sampling_Pauli_space = class_info["sampling_Pauli_space"]
        config = class_info["config"]
        q_env = cls(target, abstraction_level, config, None, sampling_Pauli_space,
                    n_shots, c_factor)
        q_env.reward_history = class_info["reward_history"]
        q_env.action_history = class_info["action_history"]
        if class_info["target_type"] == "gate":
            q_env.avg_fidelity_history = class_info["fidelity_history"]
        else:
            q_env.state_fidelity_history = class_info["fidelity_history"]
        return q_env
