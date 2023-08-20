"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 26/06/2023
"""

from itertools import chain
from typing import Dict, Optional, List, Any, Tuple, TypeVar, SupportsFloat

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Space
# Qiskit imports
from qiskit import schedule, transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction, ParameterVector, Instruction
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.providers import BackendV1, BackendV2, Options as Aer_Options
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import average_gate_fidelity, state_fidelity, process_fidelity
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.transpiler import InstructionDurations, Layout, CouplingMap
from qiskit_aer.backends import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_experiments.framework import BackendData
# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit_ibm_runtime import Estimator as Runtime_Estimator, Sampler
from tensorflow_probability.python.distributions import Categorical

from quantumenvironment import QuantumEnvironment, _calculate_chi_target_state

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
                 benchmark_cycle: int = 100,
                 intermediate_rewards: bool = False,
                 seed: Optional[int] = None
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
            basis_gates = self.backend.configuration().basis_gates
        elif isinstance(self.backend, BackendV2):
            instruction_durations = self.backend.instruction_durations
            basis_gates = self.backend.operation_names
        else:
            raise AttributeError("TorchQuantumEnvironment requires a Backend argument")
        self._physical_target_qubits = list(self.layout.get_physical_bits().keys())
        coupling_map = CouplingMap(self.backend_data.coupling_map)
        dt = self.backend_data.dt
        if dt is None:
            dt = 2.2222222222222221e-10

        if coupling_map.size() == 0:
            coupling_map = CouplingMap(self.estimator.options.simulator["coupling_map"])
        self._physical_neighbor_qubits = list(filter(lambda x: x not in self.physical_target_qubits,
                                                     chain(*[list(coupling_map.neighbors(target_qubit))
                                                         for target_qubit in self.physical_target_qubits])))

        ibm_basis_gates = ['id', 'rz', 'sx', 'x', 'ecr', 'reset', 'measure', 'delay']
        if isinstance(self.estimator, Runtime_Estimator):
            if "basis_gates" in self.estimator.options.simulator:
                ibm_basis_gates = self.estimator.options.simulator["basis_gates"]

        self.circuit_context = transpile(circuit_context.remove_final_measurements(inplace=False),
                                         backend=self.backend,
                                         scheduling_method='asap',
                                         basis_gates=ibm_basis_gates,
                                         coupling_map=coupling_map,
                                         instruction_durations=instruction_durations,
                                         optimization_level=0, dt=dt)
        # self._benchmark_backend = UnitarySimulator()
        self.tgt_register = QuantumRegister(bits=[self.circuit_context.qubits[i]
                                                  for i in self.physical_target_qubits], name="tgt")
        self.nn_register = QuantumRegister(bits=[self.circuit_context.qubits[i]
                                                 for i in self._physical_neighbor_qubits], name='nn')

        self.layout = Layout({self.tgt_register[i]: self.physical_target_qubits[i]
                              for i in range(len(self.tgt_register))} | {
                                 self.nn_register[j]: self._physical_neighbor_qubits[j]
                                 for j in range(len(self.nn_register))})

        if isinstance(self.estimator, Runtime_Estimator):
            # self.estimator.options.simulator["skip_transpilation"]=True
            # TODO: Could change resilience level
            self.estimator.set_options(optimization_level = 0, resilience_level=0)
            # TODO: Use line below when Runtime Options supports Layout class
            self.estimator.options.simulator["initial_layout"]=self.physical_target_qubits+self.physical_neighbor_qubits
            self._sampler = Sampler(session=self.estimator.session, options=dict(self.estimator.options))

        elif isinstance(self.estimator, AerEstimator):
            self.estimator._transpile_options = Aer_Options(initial_layout=self.layout, optimization_level=0)
            self._sampler = AerSampler(backend_options=self.backend.options,
                                       transpile_options={'initial_layout':self.layout, 'optimization_level':0},
                                       skip_transpilation=False)
        elif isinstance(self.estimator, BackendEstimator):
            self.estimator.set_transpile_options(initial_layout=self.layout, optimization_level=0)
            self._sampler = BackendSampler(self.backend, self.estimator.options, skip_transpilation=False)
            self._sampler.set_transpile_options(initial_layout=self.layout, optimization_level=0)

        else:
            raise TypeError("Estimator primitive not recognized (must be either BackendEstimator, Aer or Runtime")

        self._d = 2 ** self.tgt_register.size
        self.target_instruction = CircuitInstruction(self.target["gate"], self.tgt_register)
        self.tgt_instruction_counts = self.circuit_context.data.count(self.target_instruction)

        self._parameters = [ParameterVector(f'a_{j}', action_spec.shape[-1]) for j in
                            range(self.tgt_instruction_counts)]
        self._param_values = create_array(self.tgt_instruction_counts, batch_size, action_spec.shape[-1])


        # Store time and instruction indices where target gate is played in circuit
        self._target_instruction_indices, self._target_instruction_timings = [], []
        for i, instruction in enumerate(self.circuit_context.data):
            if instruction == self.target_instruction:
                self._target_instruction_indices.append(i)
                self._target_instruction_timings.append(self.circuit_context.op_start_times[i])

        self.circuit_truncations, self.baseline_truncations, self.custom_gates = self._generate_circuit_truncations()
        self._batch_size = batch_size
        self._punctual_avg_fidelities = np.zeros(batch_size)
        self._punctual_circuit_fidelities = np.zeros(batch_size)
        self._punctual_observable: SparsePauliOp = SparsePauliOp("X")
        self.circuit_fidelity_history = []
        self.action_space = action_spec
        self.observation_space = observation_spec
        self._seed = seed
        if isinstance(seed, int):
            self.np_random = seed
        self._index_input_state = np.random.randint(len(self.target["input_states"]))
        self._training_steps_per_gate = training_steps_per_gate
        self._inside_trunc_tracker = 0
        self._trunc_index = 0
        self._episode_ended = False
        self._episode_tracker = 0
        self._benchmark_cycle = benchmark_cycle
        self._intermediate_rewards = intermediate_rewards
        self._max_return = 0
        self._best_action = np.zeros((batch_size, action_spec.shape[-1]))

    @property
    def physical_target_qubits(self):
        return self._physical_target_qubits
    @property
    def physical_neighbor_qubits(self):
        return self._physical_neighbor_qubits
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

    @property
    def benchmark_cycle(self):
        return self._benchmark_cycle
    @benchmark_cycle.setter
    def benchmark_cycle(self, step):
        assert step >=1, 'Cycle needs to be a positive integer'
        self._benchmark_cycle = step
    def do_benchmark(self):
        # TODO: Create a saving log for specific time steps (every once in a while), should have an additional input
        if isinstance(self.estimator, Runtime_Estimator) or self.abstraction_level == 'circuit':
            return self._episode_tracker % self.benchmark_cycle == 0 and self._episode_tracker > 0
        else:
            return False

    def _retrieve_target_state(self, input_state_index: int, iteration: int) -> Dict:
        """
        Calculate anticipated target state for fidelity estimation based on input state preparation
        and truncation of the circuit
        :param input_state_index: Integer index indicating the PauliPreparationBasis vector
        :param iteration: Truncation index of the transpiled contextual circuit
        """
        input_circuit: QuantumCircuit = self.target["input_states"][input_state_index]["circuit"]
        ref_circuit, custom_circuit = self.baseline_truncations[iteration], self.circuit_truncations[iteration]
        target_circuit = QuantumCircuit(self.tgt_register)
        custom_target_circuit = custom_circuit.copy_empty_like(name=f'qc_{input_state_index}_{ref_circuit.name[-1]}')
        custom_target_circuit.append(input_circuit.to_instruction(), self.tgt_register)
        custom_target_circuit.compose(custom_circuit, inplace=True)
        target_circuit.append(input_circuit.to_instruction(), self.tgt_register)
        for gate in ref_circuit.data:  # Append only gates that are applied on the target register
            if all([qubit in self.tgt_register for qubit in gate.qubits]):
                target_circuit.append(gate)

        return _calculate_chi_target_state({"dm": DensityMatrix(target_circuit), "circuit": custom_target_circuit,
                                            "target_type": "state", "theoretical_circ": target_circuit},
                                           n_qubits=target_circuit.num_qubits)

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
        custom_gates_list = []
        n_custom_instructions = self._trunc_index + 1  # Count custom instructions present in the current truncation
        benchmark_circ = self.circuit_truncations[self._trunc_index].copy(name='b_circ')
        baseline_circ = self.baseline_truncations[self._trunc_index]

        if isinstance(self.estimator, (Runtime_Estimator, AerEstimator)):

            # Assess circuit fidelity with ComputeUncompute algo
            fidelity_checker = ComputeUncompute(self._sampler)
            job = fidelity_checker.run([benchmark_circ]*len(params), [baseline_circ]*len(params), values_1=params)
            circuit_fidelities = job.result().fidelities
            self._punctual_circuit_fidelities = circuit_fidelities
            self.circuit_fidelity_history.append(np.mean(circuit_fidelities))

            # TODO: Implement a valid gate characterization subroutine compatible with Runtime primitives
            # Calculate average gate fidelities for each gate instance within circuit truncation

            unitary_backend = AerSimulator(method="unitary")
            for i in range(n_custom_instructions):
                assigned_gate = QuantumCircuit(self.tgt_register)  # Assign parameter to each custom gate
                # to check its individual gate fidelity (have to detour by a QuantumCircuit)
                assigned_gate.append(self.custom_gates[i], self.tgt_register)
                assigned_gate.save_unitary()
                gates_result = unitary_backend.run(assigned_gate.decompose(),
                                                   parameter_binds=[{self._parameters[i][j]:
                                                                         params[:, i * n_actions + j]
                                                                     for j in range(n_actions)
                                                                     }]).result()
                assigned_instructions = [gates_result.get_unitary(i) for i in range(self.batch_size)]
                custom_gates_list.append(assigned_instructions)

            tgt_gate = self.target["gate"]
            avg_fidelities = np.array([[average_gate_fidelity(custom_gate, tgt_gate) for custom_gate in custom_gates]
                                       for custom_gates in custom_gates_list])
            self._punctual_avg_fidelities = avg_fidelities
            self.avg_fidelity_history.append(np.mean(avg_fidelities, axis=1))

        else:
            if self.abstraction_level == 'circuit':
                # Calculate circuit fidelity with statevector simulation
                benchmark_circ.save_statevector()
                state_backend = AerSimulator(method='statevector')
                states_result = state_backend.run(benchmark_circ.decompose(),
                                                  parameter_binds=[{self._parameters[i][j]: params[:, i * n_actions + j]
                                                                    for i in range(n_custom_instructions)
                                                                    for j in range(n_actions)
                                                                    }]).result()
                output_state_list = [states_result.get_statevector(i) for i in range(self.batch_size)]
                batched_dm = DensityMatrix(np.mean([state.to_operator().to_matrix() for state in output_state_list], axis=0))
                batched_circuit_fidelity = state_fidelity(batched_dm, Statevector(baseline_circ))
                self.circuit_fidelity_history.append(batched_circuit_fidelity)  # Circuit fidelity over the action batch

                circuit_fidelities = [state_fidelity(state, Statevector(baseline_circ)) for state in output_state_list]
                self._punctual_circuit_fidelities = np.array(circuit_fidelities)


                # Calculate average gate fidelities for each gate instance within circuit truncation
                unitary_backend = AerSimulator(method="unitary")
                for i in range(n_custom_instructions):
                    assigned_gate = QuantumCircuit(self.tgt_register)  # Assign parameter to each custom gate
                    # to check its individual gate fidelity (have to detour by a QuantumCircuit)
                    assigned_gate.append(self.custom_gates[i], self.tgt_register)
                    assigned_gate.save_unitary()
                    gates_result = unitary_backend.run(assigned_gate.decompose(),
                                                       parameter_binds=[{self._parameters[i][j]:
                                                                             params[:, i * n_actions + j]
                                                                         for j in range(n_actions)
                                                                         }]).result()
                    assigned_instructions = [gates_result.get_unitary(i) for i in range(self.batch_size)]
                    custom_gates_list.append(assigned_instructions)

                tgt_gate = self.target["gate"]
                avg_fidelities = np.array([[average_gate_fidelity(custom_gate, tgt_gate) for custom_gate in custom_gates]
                                           for custom_gates in custom_gates_list])
                self._punctual_avg_fidelities = avg_fidelities
                self.avg_fidelity_history.append(np.mean(avg_fidelities, axis=1))  # Avg gate fidelity over action batch

            else:  # Pulse simulation
                qc_list = [benchmark_circ.bind_parameters(param) for param in
                           params]  # Bind each action of the batch to a circ
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
                    "max avg gate fidelity": [np.max(np.array(self.avg_fidelity_history)[:, i])
                                              for i in range(len(self.avg_fidelity_history[0]))],
                    "arg max avg gate fidelity": [np.argmax(np.array(self.avg_fidelity_history)[:, i])
                                                  for i in range(len(self.avg_fidelity_history[0]))],
                    "arg max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "arg max circuit fidelity": np.argmax(self.circuit_fidelity_history),
                    "best action": self._best_action,
                    "truncation_index": self._trunc_index,
                    "input_state": self.target["input_states"][self._index_input_state]["circuit"].name,
                    "observable": self._punctual_observable
                }
            else:
                info = {
                    "reset_stage": False,
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "best action": self._best_action,
                    "truncation_index": self._trunc_index,
                    "input_state": self.target["input_states"][self._index_input_state]["circuit"].name,
                    "observable": self._punctual_observable
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "input_state": self.target["input_states"][self._index_input_state]["circuit"].name,
                "truncation_index": self._trunc_index,
                "observable": self._punctual_observable
            }
        return info

    def _get_obs(self):
        obs = np.array([self._index_input_state/len(self.target["input_states"]), self._inside_trunc_tracker])
        # obs = np.array([0., 0])
        return obs
    def perform_action(self, actions, do_benchmark: bool = False):
        raise NotImplementedError("This method shall not be used in this class, use QuantumEnvironment instead")

    def compute_reward(self, fidelity_access=False):
        trunc_index = self._inside_trunc_tracker
        target_state = self._retrieve_target_state(self._index_input_state, trunc_index)
        training_circ: QuantumCircuit = target_state["circuit"]
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)

        k_samples = distribution.sample(self.sampling_Pauli_space)

        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = np.round([self.c_factor * target_state["Chi"][p] / (self._d * distribution.prob(p))
                                  for p in pauli_index], 5)

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        # observables = SparsePauliOp.from_list([(pauli_basis(training_circ.num_qubits)[p].to_label(), reward_factor[i])
        #                                        for i, p in enumerate(pauli_index)])
        observables = SparsePauliOp.from_list([("I" * len(self.nn_register) +
                                                pauli_basis(len(self.tgt_register))[p].to_label(),
                                                reward_factor[i]) for i, p in enumerate(pauli_index)])
        self._punctual_observable = observables
        # self.parametrized_circuit_func(training_circ, self._parameters[:iteration])

        reshaped_params = np.reshape(np.vstack([param_set for param_set in self._param_values[trunc_index]]),
                                     (self.batch_size, (trunc_index + 1) * self.action_space.shape[-1]))

        # qc_list = [benchmark_circ.bind_parameters(param) for param in reshaped_params]
        # self.qc_history.append(qc_list)
        if self.do_benchmark():
            # print("Starting benchmarking...")
            self._store_benchmarks(reshaped_params)
            # print("Finished benchmarking")
        if fidelity_access:
            reward_table = self._punctual_circuit_fidelities
        else:

            print("Sending job...")
            job = self.estimator.run(circuits=[training_circ] * self.batch_size,
                                     observables=[observables] * self.batch_size,
                                     parameter_values=reshaped_params,
                                     shots=self.sampling_Pauli_space * self.n_shots)

            reward_table = job.result().values
            scaling_reward_factor = len(observables) / 4**len(self.tgt_register)
            reward_table /= scaling_reward_factor
            print('Job done')

        if np.mean(reward_table) > self._max_return:
            self._max_return = np.mean(reward_table)
            self._best_action = np.mean(reshaped_params, axis=0)
        self.reward_history.append(reward_table)
        assert len(reward_table) == self.batch_size

        return reward_table

    def clear_history(self):
        self.step_tracker = 0
        self._episode_tracker = 0
        self.qc_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.avg_fidelity_history.clear()
        self.process_fidelity_history.clear()
        self.built_unitaries.clear()
        self.circuit_fidelity_history.clear()

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
        if isinstance(self.estimator, Runtime_Estimator):
            self.estimator.options.environment["job_tags"]=[f"rl_qoc_step{self._step_tracker}"]
        # TODO: Remove line below when it works for gate fidelity as well
        #self._index_input_state = 0
        return self._get_obs(), self._get_info()

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        # trunc_index tells us which circuit truncation should be trained
        # Dependent on global_step and method select_trunc_index
        trunc_index = self._trunc_index = self._select_trunc_index()

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
            print("Batchsize changed")
            self.batch_size = batch_size
        self._param_values[trunc_index][step_status] = params

        if step_status < trunc_index:  # Intermediate step within the circuit truncation
            self._inside_trunc_tracker += 1
            terminated = False

            if self._intermediate_rewards:
                reward_table = self.compute_reward()
                obs = reward_table  # Set observation to obtained reward (might not be the smartest choice here)
                return obs, reward_table, terminated, False, self._get_info()
            else:

                return self._get_obs(), np.zeros(batch_size), terminated, False, self._get_info()

        else:
            terminated = True
            self._episode_ended = terminated
            reward_table = self.compute_reward()
            if self._intermediate_rewards:
                obs = reward_table
            else:
                obs = self._get_obs()

            return obs, reward_table, terminated, False, self._get_info()

    def __repr__(self):
        string = QuantumEnvironment.__repr__(self)
        string += f'Batchsize: {self.batch_size}, \n'
        string += f'Number of target gates in circuit context: {self.tgt_instruction_counts}\n'
        return string