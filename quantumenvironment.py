"""
Class to generate a RL environment suitable for usage with TF-agents, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
"""
# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, pulse, schedule
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.transpiler import InstructionProperties
from qiskit.circuit import Parameter
from qiskit.providers import BackendV1, BackendV2
from qiskit.circuit.library import XGate, SXGate, RZGate, HGate, IGate, Reset, ZGate, SGate, SdgGate, TGate, TdgGate, \
    CXGate
from qiskit.quantum_info import DensityMatrix, Statevector, Pauli, SparsePauliOp, state_fidelity, Operator, \
    process_fidelity, average_gate_fidelity

# Qiskit dynamics for pulse simulation (benchmarking)
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit_dynamics.array import Array, wrap
from qiskit_dynamics.models import HamiltonianModel

# Qiskit Experiments for generating reliable baseline for more complex gate calibrations / state preparations
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal, RoughDragCal
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.library import ProcessTomography

# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit.primitives import Estimator, BackendEstimator
from qiskit_ibm_runtime import Estimator as Runtime_Estimator, IBMBackend as Runtime_Backend
from qiskit_aer.primitives import Estimator as Aer_Estimator
from qiskit.opflow import Zero, I, CircuitOp

import numpy as np
from itertools import product
from typing import Dict, Union, Optional, List, Callable
from copy import deepcopy
# QUA imports
# from qualang_tools.bakery.bakery import baking
# from qm.qua import *
# from qm.QuantumMachinesManager import QuantumMachinesManager

# Tensorflow modules
import tensorflow as tf
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
    phi = Parameter("phi")
    single_qubit_properties = {(qubit,): None for qubit in range(num_qubits)}
    single_qubit_errors = {(qubit,): 0.0 for qubit in qubits}

    control_channel_map = backend.options.control_channel_map or {}
    two_qubit_properties = {qubits: None for qubits in control_channel_map}
    fixed_phase_gates = [(ZGate(), np.pi), (SGate(), np.pi / 2), (SdgGate(), -np.pi / 2), (TGate(), np.pi / 4),
                         (TdgGate(), -np.pi / 4)]
    other_gates = [(RZGate(phi),), (IGate(),), (HGate(),), (XGate(),), (SXGate(),), (Reset(),)]
    single_qubit_gates = fixed_phase_gates + other_gates
    exp_results = {}
    existing_cals = calibration_files is not None

    if existing_cals:
        cals = Calibrations.load(files=calibration_files)
    else:
        # cals = Calibrations.from_backend(backend, libraries=[FixedFrequencyTransmon(basis_gates=["x", "sx"],
        #                                                                             default_values=None)
        #                                                      for _ in range(num_qubits)])
        cals = Calibrations(libraries=[FixedFrequencyTransmon(basis_gates=["x", "sx"])])
    if len(target.instruction_schedule_map().instructions) <= 1:  # Check if instructions have already been added
        for gate in single_qubit_gates:
            target.add_instruction(gate[0], properties=single_qubit_properties)
        if num_qubits > 1:
            target.add_instruction(CXGate(), properties=two_qubit_properties)
            # target.add_instruction(CXGate())
            print(target)

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
        for gate in fixed_phase_gates:
            gate_cal = rz_cal.assign_parameters({phi: gate[1]}, inplace=False)
            instruction_prop = InstructionProperties(calibration=gate_cal, error=0.)
            target.update_instruction_properties(gate[0].name, (qubit,), instruction_prop)

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


class QuantumEnvironment:

    def __init__(self, n_qubits: int, target: Dict, abstraction_level: str = 'circuit',
                 Qiskit_config: Optional[Dict] = None,
                 QUA_setup: Optional[Dict] = None,
                 sampling_Pauli_space: int = 10, n_shots: int = 1, c_factor: float = 0.5):
        """
        Class for building quantum environment for RL agent aiming to perform a state preparation task.

        :param n_qubits: Number of qubits in quantum system
        :param target: control target of interest (can be either a gate to be calibrated or a state to be prepared)
            Target should be a Dict containing target type ('gate' or 'state') as well as necessary fields to initiate
            the calibration (cf. example)
        :param abstraction_level: Circuit or pulse level parametrization of action space
        :param Qiskit_config: Dictionary containing all info for running Qiskit program (i.e. service, backend,
            options, parametrized_circuit)
        :param QUA_setup: Dictionary containing all infor for running a QUA program
        :param sampling_Pauli_space: Number of samples to build fidelity estimator for one action
        :param n_shots: Number of shots to sample for one specific computation (action/Pauli expectation sampling)
        :param c_factor: Scaling factor for reward normalization
        """

        assert abstraction_level == 'circuit' or abstraction_level == 'pulse', 'Abstraction layer parameter can be' \
                                                                               'only pulse or circuit'
        self.abstraction_level: str = abstraction_level
        if Qiskit_config is None and QUA_setup is None:
            raise AttributeError("QuantumEnvironment requires one software configuration (can be Qiskit or QUA based)")
        if Qiskit_config is not None and QUA_setup is not None:
            raise AttributeError("Cannot provide simultaneously a QUA setup and a Qiskit config ")
        elif Qiskit_config is not None:

            self.Pauli_ops = [{"name": ''.join(s), "matrix": Pauli(''.join(s)).to_matrix()}
                              for s in product(["I", "X", "Y", "Z"], repeat=n_qubits)]
            self.c_factor = c_factor
            self._n_qubits = n_qubits
            self.d = 2 ** n_qubits  # Dimension of Hilbert space
            self.sampling_Pauli_space = sampling_Pauli_space
            self.n_shots = n_shots

            self.target, self.target_type, self.tgt_register = self._define_target(target)
            self.q_register = QuantumRegister(n_qubits)

            self.backend: Union[BackendV1, BackendV2, None] = Qiskit_config["backend"]
            if self.backend is not None:
                assert n_qubits <= self.backend.num_qubits, f"n_qubits ({n_qubits} > " \
                                                            f"backend.num_qubits ({self.backend.num_qubits})"
            self.parametrized_circuit_func: Callable = Qiskit_config["parametrized_circuit"]
            self.noise_model = Qiskit_config.get("noise_model", None)
            estimator_options: Dict = Qiskit_config.get("estimator_options", None)

            if isinstance(self.backend, Runtime_Backend):
                self.estimator = Runtime_Estimator(session=self.backend, options=estimator_options)
            elif self.abstraction_level == "circuit":
                if self.noise_model is not None:
                    self.estimator = Aer_Estimator()  # Estimator taking noise model into consideration
                    # TODO implement correct options from Aer Backend
                else:
                    self.estimator = Estimator()  # Estimator based on state-vector simulation
            elif self.abstraction_level == 'pulse':
                self.estimator = BackendEstimator(self.backend)

                if not isinstance(self.backend, DynamicsBackend):
                    assert self.backend is not None, "A Backend is required for running algorithm at pulse level," \
                                                     "Must be either DynamicsBackend or IBMBackend supporting OpenPulse"
                    qubits = list(range(self._n_qubits))
                    if "qubits" in Qiskit_config:
                        assert len(qubits) >= len(self.tgt_register), f'List of qubits (len {len(qubits)} must be >= ' \
                                                                      f'than target register ' \
                                                                      f'(len {len(self.tgt_register)}' \
                                                                      'on which the target gate will act'
                        assert len(qubits) == self._n_qubits, f"List of qubits (len {len(qubits)}) does not match " \
                                                              f"indicated total number of qubits {n_qubits}"
                        qubits = Qiskit_config['qubits']

                    self.pulse_backend = DynamicsBackend.from_backend(self.backend, subsystem_list=qubits)
                else:
                    calibration_files: List[str] = Qiskit_config.get('calibration_files', None)
                    self.calibrations, self.exp_results = perform_standard_calibrations(self.backend, calibration_files)
                    self.pulse_backend: DynamicsBackend = deepcopy(self.backend)
                # For benchmarking the gate at each epoch, set the tools for Pulse level simulator
                self.solver: Solver = Qiskit_config.get('solver', self.pulse_backend.options.solver)  # Custom Solver
                # Can describe noisy channels, if none provided, pick Solver associated to DynamicsBackend by default
                self.model_dim = self.solver.model.dim
                self.channel_freq = Qiskit_config.get('channel_freq', None)
                if isinstance(self.solver.model, HamiltonianModel):
                    self.y_0 = Array(np.eye(self.model_dim))
                    self.ground_state = Array(np.array([1.0] + [0.0] * (self.model_dim - 1)))
                else:
                    self.y_0 = Array(np.eye(self.model_dim ** 2))
                    self.ground_state = Array(np.array([1.0] + [0.0] * (self.model_dim ** 2 - 1)))
        elif QUA_setup is not None:
            raise AttributeError("QUA compatibility not yet implemented")
            # TODO: Add a QUA program

        # Data storage for TF-Agents or plotting
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

    def _define_target(self, target: Dict):
        if 'register' not in target:
            target["register"] = list(range(self._n_qubits))

        tgt_register = target["register"]
        assert type(tgt_register) == QuantumRegister or type(tgt_register == List[int]), \
            'Input state shall be specified over a QuantumRegister or a List'
        assert len(tgt_register) <= self._n_qubits, \
            f"Target register has bigger size ({np.max([len(tgt_register), np.max(tgt_register)])}) " \
            f"than total number of qubits ({self._n_qubits}) characterizing the full system"
        if target.get("target_type", None) == "state" or target.get("target_type", None) is None:  # Default mode is
            # State preparation if no argument target_type is found
            if 'circuit' in target:
                gate_op = target["circuit"]

                if len(tgt_register) > self._n_qubits:
                    raise ValueError("Target register has bigger size than the total number of qubits in circuit")
                elif len(tgt_register) < self._n_qubits:
                    gate_op = target["circuit"].permute(tgt_register)
                    if gate_op.num_qubits < self._n_qubits:  # If after permutation numbers still do not match
                        gate_op ^= I ^ (self._n_qubits - gate_op.num_qubits)
                target["dm"] = DensityMatrix(gate_op @ (Zero ^ self._n_qubits))
            assert 'dm' in target, 'no DensityMatrix or circuit argument provided to target dictionary'
            assert type(target["dm"]) == DensityMatrix, 'Provided dm is not a DensityMatrix object'

            return self.calculate_chi_target_state(target), "state", tgt_register
        elif target.get("target_type", None) == "gate":
            # input_states = [self.calculate_chi_target_state(input_state) for input_state in target["input_states"]]
            assert 'input_states' in target, 'Gate calibration requires a set of input states (dict)'
            gate_op = CircuitOp(target['gate'])
            if len(tgt_register) < self._n_qubits:
                gate_op = gate_op.permute(tgt_register)
                if gate_op.num_qubits < self._n_qubits:
                    gate_op ^= I ^ (self._n_qubits - gate_op.num_qubits)
            for i, input_state in enumerate(target["input_states"]):
                assert ('circuit' or 'dm') in input_state, f'input_state {i} does not have a ' \
                                                           f'DensityMatrix or circuit description'

                input_state['target_state'] = {'target_type': 'state'}
                if 'circuit' in input_state:
                    input_circuit: CircuitOp = input_state['circuit']
                    if len(tgt_register) < self._n_qubits:
                        input_circuit = input_circuit.permute(tgt_register)
                        if input_circuit.num_qubits < self._n_qubits:
                            input_circuit ^= I ^ (self._n_qubits - input_circuit.num_qubits)

                    input_state['dm'] = DensityMatrix(input_circuit @ (Zero ^ self._n_qubits))
                    input_state['target_state']["dm"] = DensityMatrix(gate_op @ input_circuit
                                                                      @ (Zero ^ self._n_qubits))

                elif 'dm' in input_state:
                    input_state['target_state']["dm"] = Operator(gate_op) @ input_state["dm"]

                input_state['target_state'] = self.calculate_chi_target_state(
                    input_state['target_state'])
            return target, "gate", tgt_register
        else:
            raise KeyError('target type not identified, must be either gate or state')

    def calculate_chi_target_state(self, target_state: Dict):
        """
        Calculate for all P
        :param target_state: Dictionary containing info on target state (name, density matrix)
        :return: Target state supplemented with appropriate "Chi" key
        """
        assert 'dm' in target_state, 'No input data for target state, provide DensityMatrix'
        target_state["Chi"] = np.array([np.trace(np.array(target_state["dm"].to_operator())
                                                 @ self.Pauli_ops[k]["matrix"]).real for k in range(self.d ** 2)])
        # Real part is taken to convert it in good format,
        # but imaginary part is always 0. as dm is hermitian and Pauli is traceless
        return target_state

    def perform_action(self, actions: types.NestedTensorOrArray, do_benchmark: bool = True):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :param do_benchmark: Indicates if actual fidelity computation should be done on top of reward computation
        :return: Reward table (reward for each run in the batch)
        """
        qc = QuantumCircuit(self.q_register)  # Reset the QuantumCircuit instance for next iteration
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
        pauli_index, _, pauli_shots = tf.unique_with_counts(k_samples)

        reward_factor = np.round([self.c_factor * target_state["Chi"][p] / (self.d * distribution.prob(p))
                                  for p in pauli_index], 5)

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp.from_list([(self.Pauli_ops[p]["name"], reward_factor[i])
                                               for i, p in enumerate(pauli_index)])

        # Benchmarking block: not part of reward calculation
        # Apply parametrized quantum circuit (action), for benchmarking only
        parametrized_circ = QuantumCircuit(self._n_qubits)
        self.parametrized_circuit_func(parametrized_circ)
        qc_list = [parametrized_circ.bind_parameters(angle_set) for angle_set in angles]
        self.qc_history.append(qc_list)
        if do_benchmark:
            self._store_benchmarks(qc_list)

        # Build full quantum circuit: concatenate input state prep and parametrized unitary
        self.parametrized_circuit_func(qc)
        job = self.estimator.run(circuits=[qc] * batch_size, observables=[observables] * batch_size,
                                 parameter_values=angles, shots=self.sampling_Pauli_space * self.n_shots)
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
            density_matrix = DensityMatrix(np.mean([q_state for q_state in q_state_list], axis=0))
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
                density_matrix = DensityMatrix(np.mean([unitary @ Zero ^ self._n_qubits for unitary in unitaries]))
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
        # converter = InstructionToSignals(dt, self.channel_freq, channels=None)
        # signals = [converter.get_signals(sched) for sched in schedule_list]
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
