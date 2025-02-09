"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 26/06/2023
"""

from __future__ import annotations

import sys
from typing import (
    Dict,
    Optional,
    List,
    Any,
    TypeVar,
    SupportsFloat,
    Union,
)

import numpy as np
from gymnasium.spaces import Box
from qiskit import schedule

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ParameterVector,
    CircuitInstruction,
    Gate,
    Instruction,
)
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import BackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import (
    Layout,
    InstructionProperties,
    PassManager,
)

from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.library import ProcessTomography

from ..helpers import (
    simulate_pulse_input,
    MomentAnalysisPass,
    CustomGateReplacementPass,
)
from ..helpers.circuit_utils import get_instruction_timings, get_gate
from .configuration.qconfig import QEnvConfig, GateTargetConfig
from .base_q_env import (
    GateTarget,
    BaseQuantumEnvironment,
    QiskitBackendInfo,
)

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def create_array(circ_trunc, batchsize, n_actions):
    arr = np.empty((circ_trunc,), dtype=object)
    for i in range(circ_trunc):
        arr[i] = np.zeros((i + 1, batchsize, n_actions))
    return arr


def target_instruction_timings(
    circuit_context: QuantumCircuit, target_instruction: CircuitInstruction
) -> tuple[List[int], List[int]]:
    """
    Return the timings of the target instructions in the circuit context

    Args:
        circuit_context: The circuit context containing the target instruction
        target_instruction: The target instruction to be found in the circuit context
    """
    try:
        op_start_times = circuit_context.op_start_times
    except AttributeError:
        op_start_times = get_instruction_timings(circuit_context)

    target_instruction_timings = []
    for i, instruction in enumerate(circuit_context.data):
        if instruction == target_instruction:
            target_instruction_timings.append(op_start_times[i])
    return op_start_times, target_instruction_timings


class ContextAwareQuantumEnvironment(BaseQuantumEnvironment):

    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: QuantumCircuit,
        training_steps_per_gate: Union[List[int], int] = 1500,
        intermediate_rewards: bool = False,
        **context_kwargs,
    ):
        """
        Class for wrapping a quantum environment enabling the calibration of a gate in a context-aware manner, that
        is with respect to an input circuit context. The input circuit context is assumed to have been transpiled and
        scheduled, and the target gate to be calibrated is assumed to be present in this circuit context. The class will
        look for the locations of the target gates in the circuit context and will enable the writing of new parametrized
        circuits, where each gate instance is replaced by a custom gate defined by the Callable parametrized_circuit_func.

        Args:
            training_config: The configuration of the training environment
            circuit_context: The circuit context containing the target gate to be calibrated
            training_steps_per_gate: The number of training steps per gate instance
            intermediate_rewards: Whether to provide intermediate rewards during the training
            context_kwargs: Additional keyword arguments to be passed to the context-aware environment (e.g.
            backend containing circuit dependent noise model)
        """

        super().__init__(training_config)
        self._optimal_actions = None
        self._param_values = None
        self.circ_tgt_register = None
        self.target_instruction = None
        self._circuit_context = None
        self._training_steps_per_gate = training_steps_per_gate
        self._intermediate_rewards = intermediate_rewards
        self.custom_instructions: List[Instruction] = []
        self.new_gates: List[Gate] = []

        self.observation_space = Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32
        )
        self.set_unbound_circuit_context(circuit_context, **context_kwargs)

    def initial_reward_fit(self, params: np.array):
        """
        Method to fit the initial reward function to the first set of actions in the environment
        with respect to the number of repetitions of the cycle circuit
        """
        qc = self.circuits[self.trunc_index].copy()
        for n in range(1, max(self.config.n_reps)):
            pubs, shots = self._reward_methods[self.config.reward_method](qc, params)

    def define_target_and_circuits(self):
        """
        Define target gate and circuits for calibration
        """
        if self.circuit_context.parameters:
            raise ValueError("Circuit context still contains unassigned parameters")
        assert isinstance(
            self.config.target, GateTargetConfig
        ), "Target should be a gate"

        if self.backend_info.coupling_map.size() == 0 and self.backend is None:
            # Build a fully connected coupling map if no backend is provided
            self.backend_info.num_qubits = self.circuit_context.num_qubits
        tgt_instruction_counts = self.tgt_instruction_counts
        moment_pass = MomentAnalysisPass()
        pm = PassManager(moment_pass)
        _ = pm.run(self.circuit_context)
        moments = pm.property_set["moments"]  # type: dict[int, list[DAGOpNode]]
        layouts = [
            Layout({qubit: q for q, qubit in enumerate(self.circuit_context.qubits)})
            for _ in range(tgt_instruction_counts)
        ]
        context_dag = circuit_to_dag(self.circuit_context)
        baseline_dags = [
            context_dag.copy_empty_like() for _ in range(tgt_instruction_counts)
        ]

        target_op_nodes = context_dag.named_nodes(
            self.target_instruction.operation.name
        )
        target_op_nodes: List[DAGOpNode] = list(
            filter(
                lambda node: node.qargs == self.target_instruction.qubits
                and node.cargs == self.target_instruction.clbits,
                target_op_nodes,
            )
        )

        if isinstance(self.backend, BackendV2):
            operations_mapping = {
                op.name: op for op in self.backend.operations if hasattr(op, "name")
            }
        else:
            operations_mapping = {}

        # Case 1: Each truncation is a different layer of the full circuit, with only one ref to target gate
        counts = 0
        for moment in moments.values():
            switch_truncation = False
            for op in moment:
                baseline_dags[counts].apply_operation_back(op.op, op.qargs, op.cargs)
                bit_indices = {
                    q: context_dag.find_bit(q).index for q in context_dag.qubits
                }
                if (
                    target_op_nodes[0].op == op.op
                    and target_op_nodes[0].qargs == op.qargs
                    and target_op_nodes[0].cargs == op.cargs
                ):
                    # if DAGOpNode.semantic_eq(
                    #     target_op_nodes[0], op, bit_indices, bit_indices
                    # ):
                    switch_truncation = True
            if switch_truncation:
                counts += 1
            if counts == tgt_instruction_counts:
                break

        baseline_circuits = [dag_to_circuit(dag) for dag in baseline_dags]

        custom_gate_pass = [
            CustomGateReplacementPass(
                [self.target_instruction],
                [self.parametrized_circuit_func],
                [self.parameters[i]],
                [self._func_args],
            )
            for i in range(tgt_instruction_counts)
        ]

        pms = [PassManager(pass_) for pass_ in custom_gate_pass]
        custom_circuits = [pm.run(circ) for pm, circ in zip(pms, baseline_circuits)]

        # Case 2: each truncation concatenates the previous ones:
        # for i in range(tgt_instruction_counts, 0, -1):
        #     ref_dag = baseline_dags[0].copy_empty_like()
        #     custom_circ = custom_circuits[0].copy_empty_like()
        #     for j in range(i):
        #         ref_dag.compose(baseline_dags[j], inplace=True)
        #         custom_circ.compose(custom_circuits[j], inplace=True)
        #     baseline_dags[i - 1] = ref_dag
        #     custom_circuits[i - 1] = custom_circ

        baseline_circuits = [dag_to_circuit(dag) for dag in baseline_dags]

        for custom_circ, baseline_circ in zip(custom_circuits, baseline_circuits):
            custom_circ.metadata["baseline_circuit"] = baseline_circ.copy()

        input_states_choice = getattr(
            self.config.reward_config.reward_args, "input_states_choice", "pauli4"
        )
        if isinstance(self.backend, BackendV2):
            for op in self.backend.operations:
                if hasattr(op, "name") and op.name not in operations_mapping:
                    # If new custom instruction was added, store it and update operations mapping
                    self.custom_instructions.append(op)
                    operations_mapping[op.name] = op

        target = [
            GateTarget(
                self.config.target.gate,
                self.physical_target_qubits,
                baseline_circuit,
                self.circ_tgt_register,
                layout,
                input_states_choice=input_states_choice,
            )
            for baseline_circuit, layout in zip(baseline_circuits, layouts)
        ]
        return target, custom_circuits, baseline_circuits

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the Environment, chooses a new input state"""
        super().reset(seed=seed)

        new_obs = self._get_obs()
        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_space.shape[0]
        )
        self._inside_trunc_tracker = 0
        return new_obs, self._get_info()

    def modify_environment_params(self):
        # self.n_reps = int(np.random.randint(4, 5))
        print(f"\n Number of repetitions: {self.n_reps}")

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # trunc_index tells us which circuit truncation should be trained
        # Dependent on global_step and method select_trunc_index
        trunc_index = self.trunc_index
        # Figure out if in middle of param loading or should compute the final reward (step_status < trunc_index or ==)
        step_status = self._inside_trunc_tracker
        self._step_tracker += 1

        if self._episode_ended:
            terminated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                terminated,
                False,
                self._get_info(),
            )

        if trunc_index >= self.tgt_instruction_counts:
            # raise IndexError(f"Circuit does contain only {self.tgt_instruction_counts} target gates and step"
            #                  f" function tries to access gate nb {trunc_index} ")
            truncated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                False,
                truncated,
                self._get_info(),
            )

        params, batch_size = np.array(action), len(np.array(action))
        if batch_size != self.batch_size:
            raise ValueError(
                f"Action batch size {batch_size} does not match environment batch size {self.batch_size}"
            )
        self._param_values[trunc_index][step_status] = params
        params = np.reshape(
            np.vstack([param_set for param_set in self._param_values[trunc_index]]),
            (self.batch_size, (trunc_index + 1) * self.action_space.shape[-1]),
        )
        if step_status < trunc_index:  # Intermediate step within the circuit truncation
            self._inside_trunc_tracker += 1
            terminated = False

            if self._intermediate_rewards:
                reward = self.perform_action(params)
                obs = reward  # Set observation to obtained reward (might not be the smartest choice here)
                return obs, reward, terminated, False, self._get_info()
            else:
                return (
                    self._get_obs(),
                    np.zeros(batch_size),
                    terminated,
                    False,
                    self._get_info(),
                )

        else:
            terminated = self._episode_ended = True
            reward = self.perform_action(params)
            if self._intermediate_rewards:
                obs = reward
            else:
                obs = self._get_obs()

            # Using Negative Log Error as the Reward
            if np.mean(reward) > self._max_return:
                self._max_return = np.mean(reward)
                self._optimal_actions[self.trunc_index] = self.mean_action
            self.reward_history.append(reward)
            assert (
                len(reward) == self.batch_size
            ), f"Reward table size mismatch {len(reward)} != {self.batch_size} "
            assert not np.any(np.isinf(reward)) and not np.any(
                np.isnan(reward)
            ), "Reward table contains NaN or Inf values"
            optimal_error_precision = 1e-6
            max_fidelity = 1.0 - optimal_error_precision
            reward = np.clip(reward, a_min=0.0, a_max=max_fidelity)
            reward = -np.log(1.0 - reward)
            reward /= self.n_reps

            return obs, reward, terminated, False, self._get_info()

    def _get_obs(self):
        if isinstance(self.target, GateTarget) and self.config.reward_method == "state":
            return np.array(
                [
                    0.0,
                    self.trunc_index,
                ]
                + list(self._observable_to_observation())
            )
        else:
            return np.array([0, self.trunc_index])

    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: Batch of actions
        """

        if (
            self.config.check_on_exp
        ):  # Perform real experiments to retrieve from measurement data fidelities
            # Assess circuit fidelity with ComputeUncompute algo
            # try:
            #     print("Starting Direct Fidelity Estimation...")
            #     input_state = np.random.choice(self.target.input_states)
            #     observables, shots = retrieve_observables(
            #         input_state.target_state,
            #         self.config.benchmark_config.dfe_precision,
            #         sampling_paulis=self.sampling_pauli_space,
            #         c_factor=self.c_factor
            #     )
            #     observables = extend_observables(observables, qc, self.target)
            #     if self.abstraction_level == "circuit":
            #         qc = self.backend_info.custom_transpile(
            #             qc,
            #             initial_layout=self.layout,
            #             scheduling=False,
            #         )
            #     pubs = [
            #         (
            #             qc,
            #             obs.apply_layout(qc.layout),
            #             [self.mean_action],
            #             1 / np.sqrt(shot),
            #         )
            #         for obs, shot in zip(
            #             observables.group_commuting(qubit_wise=True), shots
            #         )
            #     ]
            #     if isinstance(self.estimator, EstimatorV2):
            #         self.estimator.options.update(
            #             job_tags=[f"DFE_step{self._step_tracker}"]
            #         )
            #     job = self.estimator.run(pubs=pubs)
            #     results = job.result()
            #     circuit_fidelities = np.sum(
            #         [result.data.evs for result in results], axis=0
            #     ) / len(observables)
            #     print("Finished DFE")
            #     return circuit_fidelities
            # except Exception as exc:
            #     self.close()
            #     raise exc
            raise NotImplementedError(
                "Direct Fidelity Estimation is not yet supported in the context-aware environment"
            )

        else:  # Perform simulation at circuit or pulse level
            print("Starting simulation benchmark...")
            if not self.config.reward_method == "fidelity":
                params = np.array(
                    [self.mean_action]
                )  # Benchmark policy only through mean action
            if self.abstraction_level == "circuit":
                fids = self.simulate_circuit(qc, params)
            else:  # Pulse simulation
                fids = self.simulate_pulse_circuit(qc, params)
            print("Finished simulation benchmark \n")
            print("Fidelities: ", np.mean(fids))
            return fids

    @property
    def parameters(self) -> List[ParameterVector]:
        return self._parameters

    @property
    def tgt_instruction_counts(self) -> int:
        """
        Return number of target instructions present in circuit context
        """
        return self.circuit_context.data.count(self.target_instruction)

    @property
    def target(self) -> GateTarget:
        """
        Return current target to be calibrated
        """
        return self._target[self.trunc_index]

    def get_target(self, trunc_index: Optional[int] = None):
        """
        Return target to be calibrated at given truncation index.
        If no index is provided, return list of all targets.

        Args:
            trunc_index: Index of truncation to return target for.
        """
        if trunc_index is None:
            return self._target
        else:
            return self._target[trunc_index]

    @property
    def trunc_index(self) -> int:
        if self._intermediate_rewards:
            return self.step_tracker % self.tgt_instruction_counts
        else:
            return np.min(
                [
                    self._step_tracker // self.training_steps_per_gate,
                    self.tgt_instruction_counts - 1,
                ]
            )

    @property
    def training_steps_per_gate(self) -> int:
        return self._training_steps_per_gate

    @property
    def optimal_action(self):
        """
        Return optimal action for current gate instance
        """
        return self._optimal_actions[self.trunc_index]

    def optimal_actions(self, indices: Optional[int | List[int]] = None):
        """
        Return optimal actions for selected circuit truncation index.
        If no indices are provided, return list of all optimal actions.

        """
        if isinstance(indices, int):
            return self._optimal_actions[indices]
        elif isinstance(indices, List):
            return [self._optimal_actions[index] for index in indices]
        else:
            return self._optimal_actions

    @training_steps_per_gate.setter
    def training_steps_per_gate(self, nb_of_steps: int):
        try:
            assert nb_of_steps > 0 and isinstance(nb_of_steps, int)
            self._training_steps_per_gate = nb_of_steps
        except AssertionError:
            raise ValueError("Training steps number should be positive integer.")

    def episode_length(self, global_step: int) -> int:
        # assert (
        #         global_step == self.step_tracker
        # ), "Given step not synchronized with internal environment step counter"
        return 1 + self.trunc_index

    def clear_history(self) -> None:
        """Reset all counters related to training"""
        super().clear_history()

    @property
    def unbound_circuit_context(self) -> QuantumCircuit:
        """
        Return the unbound circuit context (relevant when circuit context is parameterized)
        """
        return self._unbound_circuit_context

    def set_unbound_circuit_context(self, new_context: QuantumCircuit, **kwargs):
        """
        Update the unbound circuit context
        Keyword arguments can be used to assign values to parameters in the new context for immediate target calculation

        """
        self._unbound_circuit_context = new_context
        if not new_context.parameters:
            self.set_circuit_context(new_context.copy(), **kwargs)

    @property
    def circuit_context(self) -> QuantumCircuit:
        """
        Return the current circuit context
        """
        return self._circuit_context

    def set_circuit_context(
        self,
        new_context: Optional[QuantumCircuit] = None,
        backend: Optional[BackendV2] = None,
        **kwargs,
    ):
        """
        Update the circuit context and all relevant attributes
        """
        if new_context is not None:  # Update circuit context from scratch
            self._circuit_context = new_context
            self.custom_circuit_context = self.circuit_context.copy_empty_like(
                "custom_context"
            )
            # Define target register and nearest neighbor register for truncated circuits
            self.circ_tgt_register = QuantumRegister(
                bits=[
                    self._circuit_context.qubits[i] for i in self.physical_target_qubits
                ],
                name="tgt",
            )

            # Adjust target register to match it with circuit context
            self.target_instruction = CircuitInstruction(
                get_gate(self.config.target.gate),
                (qubit for qubit in self.circ_tgt_register),
            )
            tgt_instruction_counts = self.tgt_instruction_counts
            if tgt_instruction_counts == 0:
                raise ValueError("Target gate not found in circuit context")

            self._parameters = [
                ParameterVector(f"a_{j}", self.n_actions)
                for j in range(tgt_instruction_counts)
            ]

            self._param_values = create_array(
                tgt_instruction_counts,
                self.batch_size,
                self.action_space.shape[-1],
            )

            self._optimal_actions = [
                np.zeros(self.config.n_actions) for _ in range(tgt_instruction_counts)
            ]

        else:
            for param in kwargs:
                if self._circuit_context.has_parameter(param):
                    self._circuit_context.assign_parameters(
                        {param: kwargs[param]}, inplace=True
                    )
                else:
                    raise ValueError(f"Parameter {param} not found in circuit context")

        if backend is not None:  # Update backend and backend info if provided
            self.backend = backend
            self._backend_info = QiskitBackendInfo(
                backend,
                self.config.backend_config.instruction_durations,
                self.pass_manager,
                self.config.backend_config.skip_transpilation,
            )

        self._target, self.circuits, self.baseline_circuits = (
            self.define_target_and_circuits()
        )

    def update_gate_calibration(self, gate_names: Optional[List[str]] = None):
        """
        Update gate calibrations with optimal actions

        Args:
            gate_names: Names of the custom gates to be created
        """
        if self.abstraction_level == "pulse":
            if gate_names is not None and len(gate_names) != len(
                self.custom_instructions
            ):
                raise ValueError(
                    "Number of gate names does not match number of custom instructions"
                )
            else:
                gate_names = [
                    f"{gate.name}_{i}_opt"
                    for i, gate in enumerate(self.custom_instructions)
                ]

            value_dicts = [{} for _ in range(self.tgt_instruction_counts)]
            for i, custom_circ in enumerate(self.circuits):
                for param in custom_circ.parameters:
                    if (
                        isinstance(param, ParameterVectorElement)
                        and param.vector in self.parameters
                    ):
                        vector_index = self.parameters.index(param.vector)
                        param_index = param.index
                        value_dicts[i][param] = self.optimal_actions(vector_index)[
                            param_index
                        ]

                    elif param.name.startswith("a_"):
                        vector_index = int(param.name.split("_")[1])
                        param_index = int(param.name.split("_")[2])
                        value_dicts[i][param] = self.optimal_actions(vector_index)[
                            param_index
                        ]

            # context_qc = self.custom_circuit_context.assign_parameters(
            #     value_dicts[-1], inplace=False)
            contextual_schedules = schedule(self.circuits, self.backend)

            gate_qc = [
                QuantumCircuit(self.circ_tgt_register)
                for _ in range(self.tgt_instruction_counts)
            ]
            schedules, durations = [], []
            for i, gate in enumerate(self.custom_instructions):
                baseline_circ = self.baseline_circuits[i]
                custom_circ = self.circuits[i].assign_parameters(
                    value_dicts[i], inplace=False
                )

                gate_qc[i].append(gate, self.circ_tgt_register)
                gate_qc[i].assign_parameters(self.optimal_actions(i), inplace=True)

                def _define(self2):
                    qc = QuantumCircuit(len(self.physical_target_qubits))
                    # Sort the qubits to ensure the gate is applied on the correct qubits ordering
                    sorted_indices = sorted(self.physical_target_qubits)
                    index_map = {value: i for i, value in enumerate(sorted_indices)}
                    new_indices = [
                        index_map[value] for value in self.physical_target_qubits
                    ]
                    qc.append(self.target.gate, new_indices)
                    self2._definition = qc

                def array(self2, dtype=None):
                    return self.target.gate.to_matrix()

                def new_init(self2):
                    Gate.__init__(self2, gate_names[i], self.target.gate.num_qubits, [])

                new_gate_methods = {
                    "_define": _define,
                    "__array__": array,
                    "__init__": new_init,
                }
                new_gate_cls = type(
                    f"{gate.name.capitalize()}_{i}", (Gate,), new_gate_methods
                )
                new_gate = new_gate_cls()
                self.new_gates.append(new_gate)
                cal_sched = schedule(gate_qc[i], self.backend)
                duration = cal_sched.duration
                schedules.append(cal_sched)
                durations.append(duration)
                contextual_schedules[i].assign_parameters(value_dicts[i], inplace=True)
                if isinstance(self.backend, DynamicsBackend):
                    sim_result = simulate_pulse_input(
                        self.backend,
                        contextual_schedules[i],
                        target=Operator(self.baseline_circuits[i]),
                    )
                    error = 1.0 - sim_result["gate_fidelity"]["raw"]

                else:
                    exp = ProcessTomography(
                        custom_circ,
                        self.backend,
                        self.involved_qubits_list[i],
                        target=Operator(baseline_circ),
                    )
                    exp_data = exp.run(shots=10000).block_for_results()
                    process_matrix = exp_data.analysis_results("state").value
                    process_fid = exp_data.analysis_results("process_fidelity").value
                    dim, _ = process_matrix.dim
                    avg_gate_fid = (dim * process_fid + 1) / (dim + 1)
                    error = 1.0 - avg_gate_fid

                instruction_prop = InstructionProperties(duration, error, cal_sched)

                self.backend.target.add_instruction(
                    new_gate, {tuple(self.physical_target_qubits): instruction_prop}
                )

    @property
    def involved_qubits_list(self):
        """
        Returns a list of lists of physical qubits involved in each circuit truncation
        """
        involved_qubits = []
        for target in self._target:
            involved_qubits.extend(list(target.layout.get_physical_bits().keys()))
        return involved_qubits
