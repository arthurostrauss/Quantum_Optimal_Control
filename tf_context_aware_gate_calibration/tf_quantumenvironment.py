"""
Class to generate a RL environment suitable for usage with TF-agents, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 26/05/2023
"""

# Qiskit imports
from qiskit import pulse, schedule, transpile
from qiskit.transpiler import InstructionDurations, Layout, CouplingMap
from qiskit.circuit import (
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Gate,
    CircuitInstruction,
    ParameterVector,
)
from qiskit.providers import BackendV1, BackendV2
from quantumenvironment import QuantumEnvironment, _calculate_chi_target_state

from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info.operators import SparsePauliOp, Operator, pauli_basis
from qiskit.quantum_info.operators.measures import (
    average_gate_fidelity,
    state_fidelity,
    process_fidelity,
)
from qiskit_experiments.framework import BackendData

# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit.primitives import Estimator, BackendEstimator
from qiskit_ibm_runtime import (
    Estimator as Runtime_Estimator,
    IBMBackend as Runtime_Backend,
)
from qiskit_aer.primitives import Estimator as Aer_Estimator
from qiskit_aer.backends.aerbackend import AerBackend

import numpy as np
from itertools import product, chain
from typing import Dict, Optional, List, Any, Tuple

# Tensorflow modules
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.specs import ArraySpec, TensorSpec, BoundedArraySpec
from tf_agents.environments import PyEnvironment


def create_array(circ_trunc, batchsize, n_actions):
    arr = np.empty((circ_trunc,), dtype=object)
    for i in range(circ_trunc):
        arr[i] = np.zeros((i + 1, batchsize, n_actions))
    return arr


class TFQuantumEnvironment(QuantumEnvironment, PyEnvironment):
    def __init__(
        self,
        q_env: QuantumEnvironment,
        circuit_context: QuantumCircuit,
        action_spec: types.Spec,
        observation_spec: types.Spec,
        batch_size: int,
        training_steps_per_gate: int = 1500,
        intermediate_rewards: bool = False,
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
            super().__init__(
                q_env.target,
                q_env.abstraction_level,
                q_env.config,
                None,
                q_env.sampling_Pauli_space,
                q_env.n_shots,
                q_env.c_factor,
            )
        else:
            super().__init__(
                q_env.target,
                q_env.abstraction_level,
                None,
                q_env.config,
                q_env.sampling_Pauli_space,
                q_env.n_shots,
                q_env.c_factor,
            )

        PyEnvironment.__init__(self, handle_auto_reset=True)
        assert (
            self.target_type == "gate"
        ), "This class is made for gate calibration only"
        self.backend_data = BackendData(self.backend)

        if isinstance(self.backend, BackendV1):
            instruction_durations = InstructionDurations.from_backend(self.backend)
        elif isinstance(self.backend, BackendV2):
            instruction_durations = self.backend.instruction_durations
        else:
            raise Warning(
                "No InstructionDuration object found in backend, will not be able to run the experiment"
            )

        self._physical_target_qubits = list(self.layout.get_physical_bits().keys())
        self._physical_neighbor_qubits = list(
            filter(
                lambda x: x not in self.physical_target_qubits,
                chain(
                    *[
                        list(
                            CouplingMap(self.backend_data.coupling_map).neighbors(
                                target_qubit
                            )
                        )
                        for target_qubit in self.physical_target_qubits
                    ]
                ),
            )
        )

        # if self.tgt_register in circuit_context.qregs:
        #     qreg = self.tgt_register
        #     ancilla_reg = [reg for reg in circuit_context.qregs if reg != qreg]
        #     for reg in ancilla_reg:
        #         self.layout.add_register(reg)
        # elif len(circuit_context.qregs) > 1:
        #     raise ValueError("Target physical qubits not assignable at the moment when circuit is composed with more"
        #                      "than one QuantumRegister if target register is not in circuit"
        #                      " (mapping not straightforward)")
        # else:
        #     qreg = QuantumRegister(bits=[circuit_context.qregs[0][i] for i in self.physical_qubits], name="tgt_reg")
        #     ancilla_reg = QuantumRegister(
        #         bits=[qubit for j, qubit in enumerate(circuit_context.qregs[0]) if j not in self.physical_qubits],
        #         name='ancilla_reg')
        #     self.layout = Layout({qreg[i]: self.physical_qubits[i] for i in range(len(qreg))})
        #     self.layout.add_register(ancilla_reg)

        self.circuit_context = transpile(
            circuit_context.remove_final_measurements(inplace=False),
            backend=self.backend,
            scheduling_method="asap",
            instruction_durations=instruction_durations,
            optimization_level=0,
        )

        self.estimator.set_options(skip_transpilation=True, initial_layout=None)

        self.tgt_register = QuantumRegister(
            bits=[
                self.circuit_context.qregs[0][i] for i in self.physical_target_qubits
            ],
            name="tgt",
        )
        self.nn_register = QuantumRegister(
            bits=[
                self.circuit_context.qregs[0][i] for i in self._physical_neighbor_qubits
            ],
            name="nn",
        )
        self._d = 2 ** (self.tgt_register.size + self.nn_register.size)
        self.target_instruction = CircuitInstruction(
            self.target["gate"], self.tgt_register
        )
        self.tgt_instruction_counts = self.circuit_context.data.count(
            self.target_instruction
        )

        self._parameters = [
            ParameterVector(f"a_{j}", action_spec.shape[-1])
            for j in range(self.tgt_instruction_counts)
        ]
        self._param_values = create_array(
            self.tgt_instruction_counts, batch_size, action_spec.shape[-1]
        )
        self._target_instruction_indices, self._target_instruction_timings = [], []

        # Store time and instruction indices where target gate is played in circuit
        for i, instruction in enumerate(self.circuit_context.data):
            if instruction == self.target_instruction:
                self._target_instruction_indices.append(i)
                self._target_instruction_timings.append(
                    self.circuit_context.op_start_times[i]
                )

        (
            self.circuit_truncations,
            self.baseline_truncations,
        ) = self._generate_circuit_truncations()

        self._batch_size = batch_size
        self._action_spec = action_spec
        self._observation_spec = observation_spec
        self._index_input_state = np.random.randint(len(self.target["input_states"]))
        self._training_steps_per_gate = training_steps_per_gate
        self._inside_trunc_tracker = 0
        self._episode_ended = False
        self._intermediate_rewards = intermediate_rewards

    def _generate_circuit_truncations(
        self,
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        custom_circuit_truncations = [
            QuantumCircuit(
                self.tgt_register, self.nn_register, name=f"c_circ_trunc_{i}"
            )
            for i in range(self.tgt_instruction_counts)
        ]
        baseline_circuit_truncations = [
            QuantumCircuit(
                self.tgt_register, self.nn_register, name=f"b_circ_trunc_{i}"
            )
            for i in range(self.tgt_instruction_counts)
        ]
        # Build sub-circuit contexts: each circuit goes until target gate and preserves nearest neighbor operations
        for i in range(self.tgt_instruction_counts):
            counts = 0
            for start_time, instruction in zip(
                self.circuit_context.op_start_times, self.circuit_context.data
            ):
                if all(
                    [
                        qubit in self.tgt_register or qubit in self.nn_register
                        for qubit in instruction.qubits
                    ]
                ):
                    if (
                        counts <= i or start_time <= self._target_instruction_timings[i]
                    ):  # Append until reaching tgt i
                        baseline_circuit_truncations[i].append(instruction)

                        if instruction != self.target_instruction:
                            custom_circuit_truncations[i].append(instruction)

                        else:  # Add custom instruction in place of target gate
                            # custom_gate = Gate(name=f"{self.target['gate'].name}_{counts}",
                            #                    num_qubits=len(self.tgt_register), params=
                            #                    [param for param in self._parameters[counts]])
                            # custom_circuit_truncations[i].append(CircuitInstruction(custom_gate, self.tgt_register))
                            self.parametrized_circuit_func(
                                custom_circuit_truncations[i],
                                self._parameters[counts],
                                self.tgt_register,
                            )
                            counts += 1
        return custom_circuit_truncations, baseline_circuit_truncations

    @property
    def physical_target_qubits(self):
        return self._physical_target_qubits

    @physical_target_qubits.setter
    def physical_target_qubits(self, target_qubits: List[int]):
        self._physical_target_qubits = target_qubits
        self._physical_neighbor_qubits = list(
            filter(
                lambda x: x not in self.physical_target_qubits,
                chain(
                    *[
                        list(
                            CouplingMap(self.backend_data.coupling_map).neighbors(
                                target_qubit
                            )
                        )
                        for target_qubit in self.physical_target_qubits
                    ]
                ),
            )
        )
        self.tgt_register = QuantumRegister(
            bits=[
                self.circuit_context.qregs[0][i] for i in self._physical_target_qubits
            ],
            name="tgt",
        )
        self.nn_register = QuantumRegister(
            bits=[
                self.circuit_context.qregs[0][i] for i in self._physical_neighbor_qubits
            ],
            name="nn",
        )
        self._d = 2 ** (self.tgt_register.size + self.nn_register.size)
        self.target_instruction = CircuitInstruction(
            self.target["gate"], self.tgt_register
        )
        self.tgt_instruction_counts = self.circuit_context.data.count(
            self.target_instruction
        )
        self._target_instruction_indices = []
        self._target_instruction_timings = []

        # Store time and instruction indices where target gate is played in circuit
        for i, instruction in enumerate(self.circuit_context.data):
            if instruction == self.target_instruction:
                self._target_instruction_indices.append(i)
                self._target_instruction_timings.append(
                    self.circuit_context.op_start_times[i]
                )
        self.circuit_truncations = self._generate_circuit_truncations()

    def do_benchmark(self, step_tracker, current_time_step):
        # TODO: Create a saving log for specific time steps (every once in a while), should have an additional input
        return False

    def _retrieve_target_state(self, input_state_index: int, iteration: int) -> Dict:
        input_circuit: QuantumCircuit = self.target["input_states"][input_state_index][
            "circuit"
        ]
        ref_circuit, custom_circuit = (
            self.baseline_truncations[iteration],
            self.circuit_truncations[iteration],
        )
        target_circuit = ref_circuit.copy_empty_like()
        custom_target_circuit = custom_circuit.copy_empty_like(
            name=f"qc_{input_state_index}_{ref_circuit.name[-1]}"
        )

        custom_target_circuit.append(input_circuit.to_instruction(), self.tgt_register)
        custom_target_circuit.compose(custom_circuit, inplace=True)

        target_circuit.append(input_circuit.to_instruction(), self.tgt_register)
        target_circuit.compose(ref_circuit, inplace=True)

        return _calculate_chi_target_state(
            {
                "dm": DensityMatrix(target_circuit),
                "circuit": custom_target_circuit,
                "target_type": "state",
            },
            n_qubits=target_circuit.num_qubits,
        )

    def store_benchmarks(self, qc_list: List[QuantumCircuit]):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        """

        # Circuit list for each action of the batch
        if self.abstraction_level == "circuit":
            q_state_list = [Statevector.from_instruction(qc) for qc in qc_list]
            density_matrix = DensityMatrix(
                np.mean([q_state.to_operator() for q_state in q_state_list], axis=0)
            )
            self.density_matrix_history.append(density_matrix)

            if self.target_type == "state":
                self.state_fidelity_history.append(
                    state_fidelity(self.target["dm"], density_matrix)
                )
            else:  # Gate calibration task
                q_process_list = [Operator(qc) for qc in qc_list]

                prc_fidelity = np.mean(
                    [
                        process_fidelity(q_process, Operator(self.target["gate"]))
                        for q_process in q_process_list
                    ]
                )
                avg_fidelity = np.mean(
                    [
                        average_gate_fidelity(q_process, Operator(self.target["gate"]))
                        for q_process in q_process_list
                    ]
                )
                self.built_unitaries.append(q_process_list)
                self.process_fidelity_history.append(
                    prc_fidelity
                )  # Avg process fidelity over the action batch
                self.avg_fidelity_history.append(
                    avg_fidelity
                )  # Avg gate fidelity over the action batch
                # for i, input_state in enumerate(self.target["input_states"]):
                #     output_states = [DensityMatrix(Operator(qc) @ input_state["dm"] @ Operator(qc).adjoint())
                #                      for qc in qc_list]
                #     self.input_output_state_fidelity_history[i].append(
                #         np.mean([state_fidelity(input_state["target_state"]["dm"],
                #                                 output_state) for output_state in output_states]))
        else:
            # Pulse simulation
            schedule_list = [
                schedule(qc, backend=self.backend, dt=self.backend.target.dt)
                for qc in qc_list
            ]
            unitaries = self._simulate_pulse_schedules(schedule_list)
            # TODO: Line below yields an error if simulation is not done over a set of qubit (fails if third level of
            # TODO: transmon is simulated), adapt the target gate operator accordingly.
            unitaries = [Operator(np.array(unitary.y[0])) for unitary in unitaries]
            if self.model_dim % 2 != 0:
                dms = [
                    DensityMatrix.from_int(0, self.model_dim).evolve(unitary)
                    for unitary in unitaries
                ]

            qubitized_unitaries = [
                np.zeros((self.d, self.d)) for _ in range(len(unitaries))
            ]
            for u in range(len(unitaries)):
                for i in range(self.d):
                    for j in range(self.d):
                        qubitized_unitaries[u][i, j] = unitaries[u]
            if self.target_type == "state":
                density_matrix = DensityMatrix(
                    np.mean(
                        [
                            Statevector.from_int(0, dims=self.d).evolve(unitary)
                            for unitary in unitaries
                        ]
                    )
                )
                self.state_fidelity_history.append(
                    state_fidelity(self.target["dm"], density_matrix)
                )
            else:
                self.process_fidelity_history.append(
                    np.mean(
                        [
                            process_fidelity(unitary, self.target["gate"])
                            for unitary in unitaries
                        ]
                    )
                )
                self.avg_fidelity_history.append(
                    np.mean(
                        [
                            average_gate_fidelity(unitary, self.target["gate"])
                            for unitary in unitaries
                        ]
                    )
                )
            self.built_unitaries.append(unitaries)

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def reward_spec(self) -> types.NestedArraySpec:
        # return TensorSpec(shape=(self.batch_size,), dtype=float)
        return ArraySpec(shape=(self.batch_size,), dtype=float, name="reward")

    # def discount_spec(self) -> types.NestedArraySpec:
    #     return BoundedArraySpec(shape=(self.batch_size,), dtype=float, minimum=0., maximum=1., name='discount')
    #
    #
    # def time_step_spec(self) -> ts.TimeStep:
    #     return ts.TimeStep(
    #         observation=self.observation_spec(),
    #         step_type=ArraySpec(shape=(self.batch_size,), dtype=int, name="step_type"),
    #         reward=self.reward_spec(),
    #         discount=self.discount_spec()
    #     )

    def batched(self) -> bool:
        if self.config == "Qiskit":
            return True
        else:
            return False

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        try:
            assert size > 0 and isinstance(size, int)
            self._batch_size = size
        except AssertionError:
            raise ValueError("Batch size should be positive integer.")

    @property
    def training_steps_per_gate(self):
        return self._training_steps_per_gate

    @training_steps_per_gate.setter
    def training_steps_per_gate(self, nb_of_steps: int):
        try:
            assert nb_of_steps > 0 and isinstance(nb_of_steps, int)
            self._training_steps_per_gate = nb_of_steps
        except AssertionError:
            raise ValueError("Training steps number should be positive integer.")

    def _reset(self) -> ts.TimeStep:
        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_spec().shape[0]
        )
        self._inside_trunc_tracker = 0
        self._episode_ended = False
        self._current_time_step = ts.restart(
            observation=np.reshape(
                np.tile(
                    [self._index_input_state, self._inside_trunc_tracker],
                    self.batch_size,
                ),
                (20, 2),
            ),
            reward_spec=ArraySpec(shape=(self.batch_size,), dtype=float, name="reward"),
        )
        return self.current_time_step()

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._episode_ended:
            return self.reset()
        trunc_index = self._select_trunc_index()

        if trunc_index >= self.tgt_instruction_counts:
            raise IndexError(
                f"Circuit does contain only {self.tgt_instruction_counts} target gates and step"
                f" function tries to access gate nb {trunc_index} "
            )
        self._step_tracker += 1
        # Figure out if in middle of param loading or should compute the reward
        step_status = self._inside_trunc_tracker
        params, batch_size = np.array(action), len(np.array(action))
        # self.batch_size = batch_size
        self._param_values[trunc_index][step_status] = params

        if step_status < trunc_index:
            self._inside_trunc_tracker += 1
            if self._intermediate_rewards:
                reward_table = self.compute_reward(trunc_index)
                obs = reward_table  # Set observation to obtained reward (might not be the smartest choice here)
                self._current_time_step = ts.transition(
                    observation=obs,
                    reward=reward_table,
                    discount=1.0,
                )
            else:
                obs = np.reshape(
                    np.tile(
                        [self._index_input_state, self._inside_trunc_tracker],
                        self.batch_size,
                    ),
                    (20, 2),
                )
                self._current_time_step = ts.transition(
                    observation=obs,
                    reward=np.zeros((batch_size,)),
                    discount=1.0,
                )

        else:
            reward_table = self.compute_reward(trunc_index)
            if self._intermediate_rewards:
                obs = reward_table
            else:
                obs = np.array([self._index_input_state, self._inside_trunc_tracker])

            self._current_time_step = ts.termination(
                observation=obs, reward=reward_table
            )
            self._episode_ended = True
        return self.current_time_step()

    def compute_reward(self, trunc_index: int):
        self._index_input_state = np.random.randint(len(self.target["input_states"]))

        target_state = self._retrieve_target_state(self._index_input_state, trunc_index)
        training_circ: QuantumCircuit = target_state["circuit"]
        # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
        distribution = Categorical(probs=target_state["Chi"] ** 2)
        k_samples = distribution.sample(self.sampling_Pauli_space)
        pauli_index, pauli_shots = np.unique(k_samples, return_counts=True)
        reward_factor = np.round(
            [
                self.c_factor
                * target_state["Chi"][p]
                / (self._d * distribution.prob(p))
                for p in pauli_index
            ],
            5,
        )

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables = SparsePauliOp.from_list(
            [
                (pauli_basis(training_circ.num_qubits)[p].to_label(), reward_factor[i])
                for i, p in enumerate(pauli_index)
            ]
        )

        # self.parametrized_circuit_func(training_circ, self._parameters[:iteration])

        # Benchmarking block: not part of reward calculation
        # Apply parametrized quantum circuit (action), for benchmarking only
        #
        benchmark_circ = training_circ.copy(name="benchmark_circ")

        reshaped_params = np.reshape(
            np.vstack([param_set for param_set in self._param_values[trunc_index]]),
            (self.batch_size, (trunc_index + 1) * self.action_spec().shape[-1]),
        )
        qc_list = [benchmark_circ.bind_parameters(param) for param in reshaped_params]
        self.qc_history.append(qc_list)
        if self.do_benchmark(self._step_tracker, self.current_time_step()):
            self.store_benchmarks(qc_list)

            # Build full quantum circuit: concatenate input state prep and parametrized unitary
        job = self.estimator.run(
            circuits=[training_circ] * self.batch_size,
            observables=[observables] * self.batch_size,
            parameter_values=reshaped_params,
            shots=self.sampling_Pauli_space * self.n_shots,
            job_tags=[f"rl_qoc_step{self._step_tracker}"],
        )

        reward_table = job.result().values
        self.reward_history.append(reward_table)
        assert len(reward_table) == self.batch_size
        return reward_table

    def close(self) -> None:
        if isinstance(self.estimator, Runtime_Estimator):
            self.estimator.session.close()

    def get_info(self) -> Any:
        return self._index_input_state

    def perform_action(
        self, actions: types.NestedTensorOrArray, do_benchmark: bool = True
    ):
        raise NotImplementedError(
            "This method shall not be used in this class, use QuantumEnvironment instead"
        )

    def _select_trunc_index(self):
        if self._intermediate_rewards:
            return self.step_tracker % self.tgt_instruction_counts
        else:
            return self._step_tracker // self.training_steps_per_gate
