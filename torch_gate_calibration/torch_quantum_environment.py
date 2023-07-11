"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 26/06/2023
"""

# Qiskit imports
from qiskit import schedule, transpile
from qiskit.transpiler import InstructionDurations, Layout, CouplingMap
from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction, ParameterVector, Instruction
from qiskit.providers import BackendV1, BackendV2, Options
from quantumenvironment import QuantumEnvironment, _calculate_chi_target_state

from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import average_gate_fidelity, state_fidelity, process_fidelity
from qiskit_experiments.framework import BackendData
from qiskit_aer.backends import AerSimulator , UnitarySimulator, StatevectorSimulator

# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit_ibm_runtime import Estimator as Runtime_Estimator
from qiskit_aer.primitives import Estimator as AerEstimator

import numpy as np
from itertools import chain
from typing import Dict, Optional, List, Any, Tuple, TypeVar, SupportsFloat

from gymnasium import Env
from gymnasium.spaces import Space
from tensorflow_probability.python.distributions import Categorical

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def create_array(circ_trunc, batchsize, n_actions):
    arr = np.empty((circ_trunc,), dtype=object)
    for i in range(circ_trunc):
        arr[i] = np.zeros((i + 1, batchsize, n_actions))
    return arr


class TorchQuantumEnvironment(QuantumEnvironment, Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, q_env: QuantumEnvironment,
                 circuit_context: QuantumCircuit,
                 action_spec: Space,
                 observation_spec: Space,
                 batch_size: int,
                 training_steps_per_gate: int = 1500,
                 intermediate_rewards: bool = False,
                 seed=1000
                 ):
        """
             Class for building quantum environment for RL agent aiming to perform a state preparation task.

             :param q_env: Existing QuantumEnvironment object, containing a target and a configuration.
             :param circuit_context: Transpiled QuantumCircuit where target gate is located to perform context aware gate
                 calibration.
             :param action_spec: ActionSpec
             :param observation_spec: ObservationSpec
         """

        # TODO: Redeclare everything instead of reinitializing (loss of time especially for DynamicsBackend declaration)
        if q_env.config_type == "Qiskit":
            q_env.config.do_calibrations = False
            super().__init__(q_env.target, q_env.abstraction_level, q_env.config, None,
                             q_env.sampling_Pauli_space, q_env.n_shots, q_env.c_factor)
        else:
            super().__init__(q_env.target, q_env.abstraction_level, None, q_env.config,
                             q_env.sampling_Pauli_space, q_env.n_shots, q_env.c_factor)

        Env.__init__(self)
        assert self.target_type == 'gate', "This class is made for gate calibration only"
        self.backend_data = BackendData(self.backend)

        if isinstance(self.backend, BackendV1):
            instruction_durations = InstructionDurations.from_backend(self.backend)
        elif isinstance(self.backend, BackendV2):
            instruction_durations = self.backend.instruction_durations
        else:
            raise Warning("No InstructionDuration object found in backend, will not be able to run the experiment")

        self._physical_target_qubits = list(self.layout.get_physical_bits().keys())
        self._physical_neighbor_qubits = list(filter(lambda x: x not in self.physical_target_qubits,
                                                     chain(*[list(CouplingMap(self.backend_data.coupling_map).neighbors(
                                                         target_qubit))
                                                         for target_qubit in self.physical_target_qubits])))
        self.instruction_durations = instruction_durations
        self.circuit_context = transpile(circuit_context.remove_final_measurements(inplace=False), backend=self.backend,
                                         scheduling_method='asap',
                                         instruction_durations=self.instruction_durations,
                                         optimization_level=0)
        # self._benchmark_backend = UnitarySimulator()
        self.tgt_register = QuantumRegister(bits=[self.circuit_context.qubits[i]
                                                  for i in self.physical_target_qubits], name="tgt")
        self.nn_register = QuantumRegister(bits=[self.circuit_context.qubits[i]
                                                 for i in self._physical_neighbor_qubits], name='nn')
        # self.tgt_register = QuantumRegister(len(self._physical_target_qubits),
        #                                     name="tgt")
        # self.nn_register = QuantumRegister(
        #     len(self._physical_neighbor_qubits),
        #     name="nn")
        self.layout = Layout({self.tgt_register[i]: self.physical_target_qubits[i]
                              for i in range(len(self.tgt_register))} | {
                                 self.nn_register[j]: self._physical_neighbor_qubits[j]
                                 for j in range(len(self.nn_register))})

        self.estimator.set_options(initial_layout=self.layout)
        if isinstance(self.estimator, AerEstimator):
            self.estimator._transpile_options = Options(initial_layout=self.layout)

        self._d = 2 ** self.tgt_register.size
        self.target_instruction = CircuitInstruction(self.target["gate"], self.tgt_register)
        self.tgt_instruction_counts = self.circuit_context.data.count(self.target_instruction)

        self._parameters = [ParameterVector(f"a_{j}", action_spec.shape[-1]) for j in
                            range(self.tgt_instruction_counts)]
        self._param_values = create_array(self.tgt_instruction_counts, batch_size, action_spec.shape[-1])
        self._target_instruction_indices, self._target_instruction_timings = [], []

        # Store time and instruction indices where target gate is played in circuit
        for i, instruction in enumerate(self.circuit_context.data):
            if instruction == self.target_instruction:
                self._target_instruction_indices.append(i)
                self._target_instruction_timings.append(self.circuit_context.op_start_times[i])

        self.circuit_truncations, self.baseline_truncations, self.custom_gates = self._generate_circuit_truncations()
        self._batch_size = batch_size
        self._punctual_avg_fidelities = np.zeros(batch_size)
        self._punctual_circuit_fidelities = np.zeros(batch_size)
        self.circuit_fidelity_history = []
        self.action_space = action_spec
        self.observation_space = observation_spec
        self._seed = seed
        self.np_random = seed
        self._index_input_state = np.random.randint(len(self.target["input_states"]))
        self._training_steps_per_gate = training_steps_per_gate
        self._inside_trunc_tracker = 0
        self._trunc_index = 0
        self._episode_ended = False
        self._episode_tracker = 0
        self._intermediate_rewards = intermediate_rewards

    @property
    def physical_target_qubits(self):
        return self._physical_target_qubits

    @physical_target_qubits.setter
    def physical_target_qubits(self, target_qubits: List[int]):
        self._physical_target_qubits = target_qubits
        self._physical_neighbor_qubits = list(filter(lambda x: x not in self.physical_target_qubits,
                                                     chain(*[list(
                                                         CouplingMap(self.backend_data.coupling_map).neighbors(
                                                             target_qubit))
                                                         for target_qubit in self.physical_target_qubits])))

        self.tgt_register = QuantumRegister(
            bits=[self.circuit_context.qregs[0][i] for i in self._physical_target_qubits],
            name="tgt")
        self.nn_register = QuantumRegister(
            bits=[self.circuit_context.qregs[0][i] for i in self._physical_neighbor_qubits],
            name="nn")
        self.layout = Layout({self.tgt_register[i]: self.physical_target_qubits[i]
                              for i in range(len(self.tgt_register))} | {
                                 self.nn_register[j]: self._physical_neighbor_qubits[j]
                                 for j in range(len(self.nn_register))})

        self._d = 2 ** (self.tgt_register.size + self.nn_register.size)
        self.target_instruction = CircuitInstruction(self.target["gate"], self.tgt_register)
        self.tgt_instruction_counts = self.circuit_context.data.count(self.target_instruction)
        self._target_instruction_indices = []
        self._target_instruction_timings = []

        # Store time and instruction indices where target gate is played in circuit
        for i, instruction in enumerate(self.circuit_context.data):
            if instruction == self.target_instruction:
                self._target_instruction_indices.append(i)
                self._target_instruction_timings.append(self.circuit_context.op_start_times[i])
        self.circuit_truncations = self._generate_circuit_truncations()

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        try:
            assert size > 0 and isinstance(size, int)
            self._batch_size = size
        except AssertionError:
            raise ValueError('Batch size should be positive integer.')


    @property
    def done(self):
        return self._episode_ended
    @property
    def training_steps_per_gate(self):
        return self._training_steps_per_gate

    @training_steps_per_gate.setter
    def training_steps_per_gate(self, nb_of_steps: int):
        try:
            assert nb_of_steps > 0 and isinstance(nb_of_steps, int)
            self._training_steps_per_gate = nb_of_steps
        except AssertionError:
            raise ValueError('Training steps number should be positive integer.')

    def do_benchmark(self):
        # TODO: Create a saving log for specific time steps (every once in a while), should have an additional input
        if self.abstraction_level == 'circuit':
            return True
        else:
            return False

    def _retrieve_target_state(self, input_state_index: int, iteration: int) -> Dict:

        input_circuit: QuantumCircuit = self.target["input_states"][input_state_index]["circuit"]
        ref_circuit, custom_circuit = self.baseline_truncations[iteration], self.circuit_truncations[iteration]
        target_circuit = QuantumCircuit(self.tgt_register)
        custom_target_circuit = custom_circuit.copy_empty_like(name=f'qc_{input_state_index}_{ref_circuit.name[-1]}')
        custom_target_circuit.append(input_circuit.to_instruction(),
                                     self.tgt_register)
        custom_target_circuit.compose(custom_circuit, inplace=True)
        target_circuit.append(input_circuit.to_instruction(), self.tgt_register)
        for gate in ref_circuit.data:  # Append only gates that are applied on the target register
            if all([qubit in self.tgt_register for qubit in gate.qubits]):
                target_circuit.append(gate)

        return _calculate_chi_target_state({"dm": DensityMatrix(target_circuit), "circuit": custom_target_circuit,
                                            "target_type": "state"}, n_qubits=target_circuit.num_qubits)

    def _retrieve_target_unitaries(self, circuits: list[QuantumCircuit]) -> Tuple[List[Operator], List[Statevector]]:
        # unitary_backend = AerSimulator(method='unitary')
        # density_backend = AerSimulator(method='statevector')
        # circuits2 = [circuit.decompose() for circuit in circuits]
        # results = unitary_backend.run(circuits2).result()
        # q_process_list = [results.get_unitary(i) for i in range(len(circuits))]
        batchsize = len(circuits)
        q_process_list = [Operator(qc) for qc in circuits]
        batched_dm = DensityMatrix(np.sum([Statevector(qc).to_operator().to_matrix()/batchsize for qc in circuits],
                                          axis=0))
        output_states = [Statevector(qc) for qc in circuits]
        return q_process_list, output_states

    def _generate_circuit_truncations(self) -> Tuple[List[QuantumCircuit], List[QuantumCircuit], List[Instruction]]:
        custom_circuit_truncations = [QuantumCircuit(self.tgt_register, self.nn_register, name=f'c_circ_trunc_{i}')
                                      for i in range(self.tgt_instruction_counts)]
        baseline_circuit_truncations = [QuantumCircuit(self.tgt_register, self.nn_register, name=f'b_circ_trunc_{i}')
                                        for i in range(self.tgt_instruction_counts)]
        custom_gates = []
        # Build sub-circuit contexts: each circuit goes until target gate and preserves nearest neighbor operations
        for i in range(self.tgt_instruction_counts):
            counts = 0
            for start_time, instruction in zip(self.circuit_context.op_start_times, self.circuit_context.data):
                if all([qubit in self.tgt_register or qubit in self.nn_register for qubit in instruction.qubits]):
                    if counts <= i or start_time <= self._target_instruction_timings[i]:  # Append until reaching tgt i
                        baseline_circuit_truncations[i].append(instruction)

                        if instruction != self.target_instruction:
                            custom_circuit_truncations[i].append(instruction)

                        else:  # Add custom instruction in place of target gate
                            # custom_gate = Gate(name=f"{self.target['gate'].name}_{counts}",
                            #                    num_qubits=len(self.tgt_register), params=
                            #                    [param for param in self._parameters[counts]])
                            # custom_circuit_truncations[i].append(CircuitInstruction(custom_gate, self.tgt_register))
                            self.parametrized_circuit_func(custom_circuit_truncations[i], self._parameters[counts],
                                                           self.tgt_register)
                            op = custom_circuit_truncations[i].data[-1].operation
                            if op not in custom_gates:
                                custom_gates.append(op)
                            counts += 1
        return custom_circuit_truncations, baseline_circuit_truncations, custom_gates

    def _store_benchmarks(self, params: np.array):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        """
        n_actions = self.action_space.shape[-1]
        benchmark_circ = self.circuit_truncations[self._trunc_index]
        qc_list = [benchmark_circ.bind_parameters(param) for param in params]
        custom_gates_list = []
        n_custom_instructions = self._trunc_index + 1
        for i in range(n_custom_instructions):
            assigned_gate = QuantumCircuit(self.tgt_register)
            assigned_gate.append(self.custom_gates[i], self.tgt_register)
            assigned_instructions = [Operator(assigned_gate.bind_parameters(param[i*n_actions: (i+1)*n_actions]))
                                     for param in params]
            custom_gates_list.append(assigned_instructions)
        # Circuit list for each action of the batch
        if self.abstraction_level == 'circuit':
            tgt_gate = self.target["gate"]

            q_process_list, output_state_list = self._retrieve_target_unitaries(qc_list)
            batched_dm = DensityMatrix(
                np.sum([output_state.to_operator().to_matrix() / self.batch_size for output_state in output_state_list],
                       axis=0))
            batched_circuit_fidelity = state_fidelity(batched_dm,
                                                      Statevector(self.baseline_truncations[self._trunc_index]))
            # prc_fidelity = np.mean([process_fidelity(q_process, self._target_unitaries[self._trunc_index])
            #                         for q_process in q_process_list])
            avg_fidelities = np.array([[average_gate_fidelity(custom_gate, tgt_gate) for custom_gate in custom_gates]
                                       for custom_gates in custom_gates_list])
            circuit_fidelities = np.array([state_fidelity(output_state,
                                                          Statevector(self.baseline_truncations[self._trunc_index]))
                                           for output_state in output_state_list])
            avg_fidelity = np.mean(avg_fidelities, axis=1)
            self._punctual_avg_fidelities = avg_fidelities
            self._punctual_circuit_fidelities = circuit_fidelities
            self.avg_fidelity_history.append(avg_fidelity)  # Avg gate fidelity over the action batch
            self.circuit_fidelity_history.append(batched_circuit_fidelity)  # Avg gate fidelity over the action batch
            # self.built_unitaries.append(q_process_list)
            # self.process_fidelity_history.append(prc_fidelity)  # Avg process fidelity over the action batch
            # for i, input_state in enumerate(self.target["input_states"]):
            #     output_states = [DensityMatrix(Operator(qc) @ input_state["dm"] @ Operator(qc).adjoint())
            #                      for qc in qc_list]
            #     self.input_output_state_fidelity_history[i].append(
            #         np.mean([state_fidelity(input_state["target_state"]["dm"],
            #                                 output_state) for output_state in output_states]))
        else:
            # Pulse simulation
            schedule_list = [schedule(qc, backend=self.backend, dt=self.backend.target.dt) for qc in qc_list]
            unitaries = self._simulate_pulse_schedules(schedule_list)
            # TODO: Line below yields an error if simulation is not done over a set of qubit (fails if third level of
            # TODO: transmon is simulated), adapt the target gate operator accordingly.
            unitaries = [Operator(np.array(unitary.y[0])) for unitary in unitaries]
            if self.model_dim % 2 != 0:
                dms = [DensityMatrix.from_int(0, self.model_dim).evolve(unitary) for unitary in unitaries]

            qubitized_unitaries = [np.zeros((self._d, self._d)) for _ in range(len(unitaries))]
            for u in range(len(unitaries)):
                for i in range(self._d):
                    for j in range(self._d):
                        qubitized_unitaries[u][i, j] = unitaries[u]
            if self.target_type == 'state':
                density_matrix = DensityMatrix(np.mean([Statevector.from_int(0, dims=self._d).evolve(unitary)
                                                        for unitary in unitaries]))
                self.state_fidelity_history.append(state_fidelity(self.target["dm"], density_matrix))
            else:
                self.process_fidelity_history.append(
                    np.mean([process_fidelity(unitary, Operator(self.baseline_truncations[self._trunc_index]))
                             for unitary in unitaries]))
                self.avg_fidelity_history.append(
                    np.mean([average_gate_fidelity(unitary, Operator(self.baseline_truncations[self._trunc_index]))
                             for unitary in unitaries]))
            self.built_unitaries.append(unitaries)

    def _select_trunc_index(self):
        if self._intermediate_rewards:
            return self.step_tracker % self.tgt_instruction_counts
        else:
            return np.min([self._step_tracker // self.training_steps_per_gate, self.tgt_instruction_counts - 1])

    def episode_length(self, global_step: int):
        assert global_step == self.step_tracker, "Given step not synchronized with internal environment step counter"
        return self._select_trunc_index() + 1

    def close(self) -> None:
        if isinstance(self.estimator, Runtime_Estimator):
            self.estimator.session.close()

    def _get_info(self) -> Any:
        step = self._episode_tracker
        if self._episode_ended:
            if self.do_benchmark():
                info = {
                    "reset_stage": False,
                    "step": step,
                    "average gate fidelity": self.avg_fidelity_history[-1],
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "circuit fidelity": self.circuit_fidelity_history[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "max circuit fidelity": np.max(self.circuit_fidelity_history),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "arg max circuit fidelity": np.argmax(self.circuit_fidelity_history),
                    "truncation_index": self._trunc_index,
                    "input_state": self.target["input_states"][self._index_input_state]["circuit"].name
                }
            else:
                info = {
                    "reset_stage": False,
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "truncation_index": self._trunc_index,
                    "input_state": self.target["input_states"][self._index_input_state]["circuit"].name
                }
        else:
            info = {
                    "reset_stage": self._inside_trunc_tracker == 0,
                    "step": step,
                    "gate_index": self._inside_trunc_tracker,
                    "input_state": self.target["input_states"][self._index_input_state]["circuit"].name,
                    "truncation_index": self._trunc_index,
            }
        return info

    def perform_action(self, actions, do_benchmark: bool = False):
        raise NotImplementedError("This method shall not be used in this class, use QuantumEnvironment instead")

    def compute_reward(self, fidelity_access=True):
        trunc_index = self._trunc_index
        target_state = self._retrieve_target_state(self._index_input_state, trunc_index)
        training_circ: QuantumCircuit = target_state["circuit"]
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)

        k_samples = distribution.sample(self.sampling_Pauli_space)

        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = np.round([self.c_factor * target_state["Chi"][p] / (self._d * np.exp(distribution.log_prob(p)))
                                  for p in pauli_index], 5)

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        # observables = SparsePauliOp.from_list([(pauli_basis(training_circ.num_qubits)[p].to_label(), reward_factor[i])
        #                                        for i, p in enumerate(pauli_index)])
        observables = SparsePauliOp.from_list([("I" * len(self.nn_register) +
                                                pauli_basis(len(self.tgt_register))[p].to_label(),
                                                reward_factor[i]) for i, p in enumerate(pauli_index)])
        # self.parametrized_circuit_func(training_circ, self._parameters[:iteration])

        reshaped_params = np.reshape(np.vstack([param_set for param_set in self._param_values[trunc_index]]),
                                     (self.batch_size, (trunc_index + 1) * self.action_space.shape[-1]))

        # qc_list = [benchmark_circ.bind_parameters(param) for param in reshaped_params]
        # self.qc_history.append(qc_list)
        if self.do_benchmark():
            self._store_benchmarks(reshaped_params)
        if fidelity_access:
            reward_table = self._punctual_circuit_fidelities
            self.reward_history.append(reward_table)
            return reward_table
            # Build full quantum circuit: concatenate input state prep and parametrized unitary

        job = self.estimator.run(circuits=[training_circ] * self.batch_size,
                                 observables=[observables] * self.batch_size,
                                 parameter_values=reshaped_params,
                                 shots=self.sampling_Pauli_space * self.n_shots,
                                 job_tags=[f"rl_qoc_step{self._step_tracker}"])

        reward_table = job.result().values
        self.reward_history.append(reward_table)
        assert len(reward_table) == self.batch_size

        return reward_table

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self._param_values = create_array(self.tgt_instruction_counts, self.batch_size, self.action_space.shape[0])
        self._inside_trunc_tracker = 0
        self._episode_tracker += 1
        self._episode_ended = False
        self._index_input_state = np.random.randint(len(self.target["input_states"]))

        return np.array([self._index_input_state, self._inside_trunc_tracker]), self._get_info()

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        # trunc_index tells us which circuit truncation should be trained
        # Dependent on global_step and method select_trunc_index
        self._trunc_index = self._select_trunc_index()
        trunc_index = self._trunc_index
        # Figure out if in middle of param loading or should compute the final reward (step_status < trunc_index or ==)
        step_status = self._inside_trunc_tracker
        self._step_tracker += 1

        if self._episode_ended:
            terminated = True
            return self.reset()[0], np.zeros(self.batch_size), terminated, False, self._get_info()

        if trunc_index >= self.tgt_instruction_counts:
            # raise IndexError(f"Circuit does contain only {self.tgt_instruction_counts} target gates and step"
            #                  f" function tries to access gate nb {trunc_index} ")
            truncated = True
            return self.reset()[0], np.zeros(self.batch_size), False, truncated, self._get_info()

        params, batch_size = np.array(action), len(np.array(action))
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        self._param_values[trunc_index][step_status] = params

        if step_status < trunc_index:  # Intermediate step within the same circuit
            self._inside_trunc_tracker += 1
            terminated = False

            if self._intermediate_rewards:
                reward_table = self.compute_reward()
                obs = reward_table  # Set observation to obtained reward (might not be the smartest choice here)
                return obs, reward_table, terminated, False, self._get_info()
            else:
                obs = np.array([self._index_input_state, self._inside_trunc_tracker])
                return obs, np.zeros(batch_size), terminated, False, self._get_info()

        else:
            terminated = True
            self._episode_ended = terminated
            reward_table = self.compute_reward()
            if self._intermediate_rewards:
                obs = reward_table
            else:
                obs = np.array([self._index_input_state, self._inside_trunc_tracker])

            return obs, reward_table, terminated, False, self._get_info()
