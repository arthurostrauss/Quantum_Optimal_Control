"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate/
execute quantum circuits. The environment is designed to be context-aware, meaning it can focus on a specific
target gate within a larger quantum circuit. This workflow could be extended to multiple target gates in the future.

Author: Arthur Strauss
Created on 26/06/2023
Last modified on 28/04/2025
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
    Sequence,
)

import numpy as np
from gymnasium.spaces import Box

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ParameterVector,
    CircuitInstruction,
    Gate,
    Instruction,
    Parameter,
)
from qiskit.circuit.library import StatePreparation
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import BackendV2
from qiskit.quantum_info import Operator, DensityMatrix
from qiskit.transpiler import (
    Layout,
    InstructionProperties,
    PassManager,
)

from qiskit_experiments.library import ProcessTomography

from ..helpers import (
    MomentAnalysisPass,
    CustomGateReplacementPass,
    retrieve_primitives,
)
from ..helpers.circuit_utils import get_instruction_timings
from .configuration.qconfig import QEnvConfig
from .base_q_env import (
    GateTarget,
    StateTarget,
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
            circuit_context: The circuit context containing the target gate to be calibrated. If None, a new circuit
                context will be created with the target gate appended to it on the target qubits. Can also be a list of
                circuits, in which case training will be performed by switching between the circuits at random at each
                training step.
            virtual_target_qubits: The virtual target qubits to be used in the circuit context. If None, will be set to
                the first qubits of the circuit context.
            training_steps_per_gate: The number of training steps per gate instance
            intermediate_rewards: Whether to provide intermediate rewards during the training
            context_kwargs: Additional keyword arguments to be passed to the context-aware environment (e.g.
            backend containing circuit dependent noise model)
        """

        super().__init__(training_config)

        self.circ_tgt_register = None
        self.custom_instructions: List[Instruction] = []
        self.new_gates: List[Gate] = []

        self.observation_space = Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self._parameters = [
            [Parameter(f"a_{j}_{i}") for i in range(self.n_actions)]
            for j in range(len(self.target.circuits))
        ]

        self._optimal_actions = [
            np.zeros(self.config.n_actions) for _ in range(len(self.target.circuits))
        ]
        self.circuits = self.define_circuits()

    def define_circuits(self) -> list[QuantumCircuit]:
        """
        Define the circuits to be used in the environment.
        This method should be overridden by the subclass.

        Returns:
            list[QuantumCircuit]: A list of QuantumCircuit objects.
        """
        circuits = []
        for i, circ in enumerate(self.target.circuits):

            pm = PassManager(
                CustomGateReplacementPass(
                    self.target.target_instructions[i],
                    [self.parametrized_circuit_func],
                    [self.parameters[i]],
                    [self._func_args],
                )
            )
            custom_circ = pm.run(circ)
            custom_circ.metadata["baseline_circuit"] = circ.copy(f"baseline_circ_{circ.name}")
            circuits.append(custom_circ)

        return circuits

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # trunc_index tells us which circuit truncation should be trained
        # Dependent on global_step and method select_trunc_index
        # Figure out if in middle of param loading or should compute the final reward (step_status < trunc_index or ==)
        step_status = self._inside_trunc_tracker

        if self._episode_ended:
            terminated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                terminated,
                False,
                self._get_info(),
            )

        params, batch_size = np.array(action), len(np.array(action))
        if batch_size != self.batch_size:
            raise ValueError(
                f"Action batch size {batch_size} does not match environment batch size {self.batch_size}"
            )
        terminated = self._episode_ended = True
        reward = self.perform_action(params)

        obs = self._get_obs()

        # Using Negative Log Error as the Reward
        if np.mean(reward) > self._max_return:
            self._max_return = np.mean(reward)
            self._optimal_actions[self.circuit_choice] = self.mean_action

        assert (
            len(reward) == self.batch_size
        ), f"Reward table size mismatch {len(reward)} != {self.batch_size} "
        assert not np.any(np.isinf(reward)) and not np.any(
            np.isnan(reward)
        ), "Reward table contains NaN or Inf values"
        optimal_error_precision = 1e-6
        max_fidelity = 1.0 - optimal_error_precision
        if self._fit_function is not None:
            reward = self._fit_function(reward, self.n_reps)
        reward = np.clip(reward, a_min=0.0, a_max=max_fidelity)
        self.reward_history.append(reward)
        reward = -np.log10(1.0 - reward)

        return obs, reward, terminated, False, self._get_info()

    def _get_obs(self) -> ObsType:
        if isinstance(self.target, GateTarget) and self.config.reward_method == "state":
            return np.array(
                [
                    0.0,
                    self.circuit_choice,
                ]
                + list(self._observable_to_observation()),
                dtype=np.float32,
            )
        else:
            return np.array([0, self.circuit_choice])

    def compute_benchmarks(
        self, qc: QuantumCircuit, params: np.ndarray, update_env_history=True
    ) -> np.ndarray:
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
                params = np.array([self.mean_action])  # Benchmark policy only through mean action
            if self.abstraction_level == "circuit":
                fids = self.simulate_circuit(qc, params, update_env_history)
            else:  # Pulse simulation
                fids = self.simulate_pulse_circuit(qc, params, update_env_history)
            print("Finished simulation benchmark \n")

            fidelity_type = "Gate" if self.target.causal_cone_size <= 3 else "State"
            if len(fids) == 1:
                print(f"{fidelity_type} Fidelity (per Cycle): ", fids[0])
            else:
                print(f"{fidelity_type} Fidelities (per Cycle): ", np.mean(fids))
            return fids

    @property
    def parameters(self) -> List[ParameterVector] | List[List[Parameter]]:
        return self._parameters

    @property
    def target(self) -> GateTarget | StateTarget:
        """
        Return current target to be calibrated
        """
        return self.config.target

    @property
    def virtual_target_qubits(self) -> List[Qubit] | QuantumRegister:
        """
        Return the virtual target qubits used in the circuit context
        """
        if isinstance(self.config.target, GateTarget):
            return self.config.target.virtual_target_qubits[self.circuit_choice]
        else:
            return self.config.target.tgt_register

    @property
    def optimal_action(self):
        """
        Return optimal action for current gate instance
        """
        return self._optimal_actions[self.circuit_choice]

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

    def episode_length(self, global_step: int) -> int:
        return 1 + self.circuit_choice

    def clear_history(self) -> None:
        """Reset all counters related to training"""
        super().clear_history()

    @property
    def circuit_context(self) -> QuantumCircuit:
        """
        Return the current circuit context
        """
        return self.target.circuit

    @property
    def circuit_choice(self) -> int:
        """
        Return the index of the circuit choice in the context-aware environment.
        """
        return super().circuit_choice

    @circuit_choice.setter
    def circuit_choice(self, value: int):
        """
        Set the index of the circuit choice in the context-aware environment.
        """
        self.config.target.circuit_choice = value

    @property
    def has_context(self) -> bool:
        """
        Check if the environment has a circuit context (i.e., different from circuit containing only the target gate).
        :return: Boolean indicating if the environment has a circuit context
        """
        return self.target.has_context

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
            self.custom_circuit_context = self.circuit_context.copy_empty_like("custom_context")

        else:
            self._circuit_context = self._unbound_circuit_context.assign_parameters(kwargs)

        if backend is not None:  # Update backend and backend info if provided
            self.backend = backend
            self.backend_info = QiskitBackendInfo(
                backend,
                self.config.backend_config.instruction_durations,
                self.pass_manager,
                self.config.backend_config.skip_transpilation,
            )

        self._target, self.circuits, self.baseline_circuits = self.define_target_and_circuits()

    def update_gate_calibration(self, gate_names: Optional[List[str]] = None):
        """
        Update gate calibrations with optimal actions

        Args:
            gate_names: Names of the custom gates to be created
        """
        try:
            from qiskit import schedule, pulse
            from qiskit_dynamics import DynamicsBackend
            from ..helpers.pulse_utils import (
                simulate_pulse_input,
                get_optimal_z_rotation,
            )
        except ImportError as e:
            raise ImportError(
                "Pulse calibration requires Qiskit Pulse, Qiskit Dynamics and Qiskit Experiments below 0.10."
                "Please set your Qiskit version to 1.x to use this feature."
            )
        if self.abstraction_level == "pulse":
            if gate_names is not None and len(gate_names) != len(self.custom_instructions):
                raise ValueError(
                    "Number of gate names does not match number of custom instructions"
                )
            else:
                gate_names = [
                    f"{gate.name}_{i}_opt" for i, gate in enumerate(self.custom_instructions)
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
                        value_dicts[i][param] = self.optimal_actions(vector_index)[param_index]

                    elif param.name.startswith("a_"):
                        vector_index = int(param.name.split("_")[1])
                        param_index = int(param.name.split("_")[2])
                        value_dicts[i][param] = self.optimal_actions(vector_index)[param_index]

            # context_qc = self.custom_circuit_context.assign_parameters(
            #     value_dicts[-1], inplace=False)
            contextual_schedules = schedule(self.circuits, self.backend)

            gate_qc = [
                QuantumCircuit(self.circ_tgt_register) for _ in range(self.tgt_instruction_counts)
            ]
            schedules, durations = [], []
            for i, gate in enumerate(self.custom_instructions):
                baseline_circ = self.baseline_circuits[i]
                custom_circ = self.circuits[i].assign_parameters(value_dicts[i], inplace=False)

                gate_qc[i].append(gate, self.circ_tgt_register)
                gate_qc[i].assign_parameters(self.optimal_actions(i), inplace=True)

                def _define(self2):
                    qc = QuantumCircuit(len(self.physical_target_qubits))
                    # Sort the qubits to ensure the gate is applied on the correct qubits ordering
                    sorted_indices = sorted(self.physical_target_qubits)
                    index_map = {value: i for i, value in enumerate(sorted_indices)}
                    new_indices = [index_map[value] for value in self.physical_target_qubits]
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
                new_gate_cls = type(f"{gate.name.capitalize()}_{i}", (Gate,), new_gate_methods)
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

    # def define_target_and_circuits(self):
    #     """
    #     Define target gates and circuits for calibration
    #     """
    #     if self.circuit_context.parameters:
    #         raise ValueError("Circuit context still contains unassigned parameters")
    #     # assert isinstance(self.config.target, GateTargetConfig), "Target should be a gate"

    #     if self.backend_info.coupling_map.size() == 0 and self.backend is None:
    #         # Build a fully connected coupling map if no backend is provided
    #         self.backend_info.num_qubits = self.circuit_context.num_qubits
    #     tgt_instruction_counts = self.tgt_instruction_counts
    #     moment_pass = MomentAnalysisPass()
    #     pm = PassManager(moment_pass)
    #     _ = pm.run(self.circuit_context)
    #     moments = pm.property_set["moments"]  # type: dict[int, list[DAGOpNode]]
    #     layouts = [
    #         Layout(
    #             {
    #                 qubit: q
    #                 for q, qubit in zip(self.physical_target_qubits, self.circuit_context.qubits)
    #             }
    #         )
    #         for _ in range(tgt_instruction_counts)
    #     ]
    #     context_dag = circuit_to_dag(self.circuit_context)
    #     baseline_dags = [context_dag.copy_empty_like() for _ in range(tgt_instruction_counts)]

    #     target_op_nodes = context_dag.named_nodes(self.target_instruction.operation.name)
    #     target_op_nodes: List[DAGOpNode] = list(
    #         filter(
    #             lambda node: node.qargs == self.target_instruction.qubits
    #             and node.cargs == self.target_instruction.clbits,
    #             target_op_nodes,
    #         )
    #     )

    #     if isinstance(self.backend, BackendV2):
    #         operations_mapping = {
    #             op.name: op for op in self.backend.operations if hasattr(op, "name")
    #         }
    #     else:
    #         operations_mapping = {}

    #     # Case 1: Each truncation is a different layer of the full circuit, with only one ref to target gate
    #     counts = 0
    #     for moment in moments.values():
    #         switch_truncation = False
    #         for op in moment:
    #             baseline_dags[counts].apply_operation_back(op.op, op.qargs, op.cargs)
    #             if (
    #                 target_op_nodes[0].op == op.op
    #                 and target_op_nodes[0].qargs == op.qargs
    #                 and target_op_nodes[0].cargs == op.cargs
    #             ):
    #                 # if DAGOpNode.semantic_eq(
    #                 #     target_op_nodes[0], op, bit_indices, bit_indices
    #                 # ):
    #                 switch_truncation = True
    #         if switch_truncation:
    #             counts += 1
    #         if counts == tgt_instruction_counts:
    #             break

    #     baseline_circuits = [dag_to_circuit(dag) for dag in baseline_dags]

    #     custom_gate_pass = [
    #         CustomGateReplacementPass(
    #             [self.target_instruction],
    #             [self.parametrized_circuit_func],
    #             [self.parameters[i]],
    #             [self._func_args],
    #         )
    #         for i in range(tgt_instruction_counts)
    #     ]

    #     pms = [PassManager(pass_) for pass_ in custom_gate_pass]
    #     custom_circuits = [
    #         pm.run(circ).copy(f"custom_circ_{i}")
    #         for i, (pm, circ) in enumerate(zip(pms, baseline_circuits))
    #     ]

    #     # Case 2: each truncation concatenates the previous ones:
    #     # for i in range(tgt_instruction_counts, 0, -1):
    #     #     ref_dag = baseline_dags[0].copy_empty_like()
    #     #     custom_circ = custom_circuits[0].copy_empty_like()
    #     #     for j in range(i):
    #     #         ref_dag.compose(baseline_dags[j], inplace=True)
    #     #         custom_circ.compose(custom_circuits[j], inplace=True)
    #     #     baseline_dags[i - 1] = ref_dag
    #     #     custom_circuits[i - 1] = custom_circ

    #     for i in range(tgt_instruction_counts):
    #         custom_circuits[i].metadata["baseline_circuit"] = baseline_circuits[i].copy(
    #             f"baseline_circ_{i}"
    #         )

    #     input_states_choice = getattr(
    #         self.config.reward.reward_args, "input_states_choice", "pauli4"
    #     )
    #     if isinstance(self.backend, BackendV2):
    #         for op in self.backend.operations:
    #             if hasattr(op, "name") and op.name not in operations_mapping:
    #                 # If a new custom instruction was added, store it and update operation mapping
    #                 self.custom_instructions.append(op)
    #                 operations_mapping[op.name] = op

    #     target = (
    #         [
    #             GateTarget(
    #                 self.config.target.gate,
    #                 self.physical_target_qubits,
    #                 baseline_circuit,
    #                 self.virtual_target_qubits,
    #                 self.circ_tgt_register,
    #                 layout,
    #                 input_states_choice=input_states_choice,
    #             )
    #             for baseline_circuit, layout in zip(baseline_circuits, layouts)
    #         ]
    #         if isinstance(self.config.target, GateTargetConfig)
    #         else [
    #             StateTarget(
    #                 self.config.target.state,
    #                 baseline_circuit,
    #                 self.physical_target_qubits,
    #                 self.circ_tgt_register,
    #                 layout,
    #             )
    #             for baseline_circuit, layout in zip(baseline_circuits, layouts)
    #         ]
    #     )
    #     return target, custom_circuits, baseline_circuits