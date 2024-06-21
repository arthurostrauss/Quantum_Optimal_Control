"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 26/06/2023
"""

from itertools import product
import sys
from typing import Dict, Optional, List, Any, TypeVar, SupportsFloat, Union

import numpy as np
from gymnasium.spaces import Box

# Qiskit imports
from qiskit import transpile
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ParameterVector,
    CircuitInstruction,
)
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.transpiler import Layout
from qiskit_aer.backends import AerSimulator
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_aer.noise import NoiseModel
from qiskit_dynamics import DynamicsBackend
from qiskit_ibm_runtime import EstimatorV2

from rl_qoc.helper_functions import (
    projected_statevector,
    get_instruction_timings,
)
from rl_qoc.qconfig import QEnvConfig
from rl_qoc.base_q_env import (
    GateTarget,
    BaseQuantumEnvironment,
)
from rl_qoc.custom_jax_sim import JaxSolver

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


class ContextAwareQuantumEnvironment(BaseQuantumEnvironment):

    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: QuantumCircuit,
        training_steps_per_gate: Union[List[int], int] = 1500,
        intermediate_rewards: bool = False,
    ):
        """
        Class for wrapping a quantum environment enabling the calibration of a gate in a context-aware manner, that
        is with respect to an input circuit context. The input circuit context is assumed to have been transpiled and
        scheduled, and the target gate to be calibrated is assumed to be present in this circuit context. The class will
        look for the locations of the target gates in the circuit context and will enable the writing of new parametrized
        circuits, where each gate instance is replaced by a custom gate defined by the Callable parametrized_circuit_func.

        Args:
            training_config:
            circuit_context:
            training_steps_per_gate:
            intermediate_rewards:
        """
        self._training_steps_per_gate = training_steps_per_gate
        self._intermediate_rewards = intermediate_rewards
        self.circuit_fidelity_history = []
        self.circuit_context = circuit_context
        # Define target register and nearest neighbor register for truncated circuits
        self.circ_tgt_register = QuantumRegister(
            bits=[
                self.circuit_context.qubits[i] for i in training_config.physical_qubits
            ],
            name="tgt",
        )

        # Adjust target register to match it with circuit context
        self.target_instruction = CircuitInstruction(
            training_config.target["gate"], (qubit for qubit in self.circ_tgt_register)
        )
        self._tgt_instruction_counts = self.circuit_context.data.count(
            self.target_instruction
        )
        if self.tgt_instruction_counts == 0:
            raise ValueError("Target gate not found in circuit context")

        self._parameters = [
            ParameterVector(f"a_{j}", training_config.n_actions)
            for j in range(self.tgt_instruction_counts)
        ]

        # Store time and instruction indices where target gate is played in circuit
        try:
            self._op_start_times = self.circuit_context.op_start_times
        except AttributeError:
            self._op_start_times = get_instruction_timings(self.circuit_context)

        self._target_instruction_timings = []
        for i, instruction in enumerate(self.circuit_context.data):
            if instruction == self.target_instruction:
                self._target_instruction_timings.append(self._op_start_times[i])

        super().__init__(training_config)

        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_space.shape[-1]
        )
        self.observation_space = Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32
        )

    def define_target_and_circuits(self):
        """
        Define target gate and circuits for calibration
        """

        assert "gate" in self.config.target, "Target should be a gate"

        # Build registers for all relevant qubits
        circ_nn_register, circ_anc_register = (
            QuantumRegister(
                bits=[self.circuit_context.qubits[i] for i in qubits],
                name=reg_name,
            )
            for reg_name, qubits in zip(
                ["nn", "anc"],
                [self.physical_neighbor_qubits, self.physical_next_neighbor_qubits],
            )
        )
        tgt_register = QuantumRegister(len(self.physical_target_qubits), name="tgt")
        layouts = [
            Layout(
                {
                    tgt_register[i]: self.physical_target_qubits[i]
                    for i in range(tgt_register.size)
                }
            )
            for _ in range(self.tgt_instruction_counts)
        ]
        nn_registers, anc_registers = [
            [QuantumRegister(1, name=f"{name}_{i}") for i in range(reg.size)]
            for name, reg in zip(["nn", "anc"], [circ_nn_register, circ_anc_register])
        ]
        # Create mapping between circuit context qubits and custom circuit associated single qubit registers
        mapping = {
            circ_reg[i]: reg[i]
            for circ_reg, reg in zip(
                [circ_nn_register, self.circ_tgt_register, circ_anc_register],
                [nn_registers, tgt_register, anc_registers],
            )
            for i in range(circ_reg.size)
        }

        # Initialize custom and baseline circuits for each target gate (by default only contains target qubits)
        custom_circuits, baseline_circuits = [
            [
                QuantumCircuit(tgt_register, name=name + str(i))
                for i in range(self.tgt_instruction_counts)
            ]
            for name in ["c_circ_trunc_", "b_circ_trunc_"]
        ]
        # Build sub-circuit contexts: each circuit goes until target gate and preserves nearest neighbor operations
        for i in range(self.tgt_instruction_counts):  # Loop over target gates
            counts = 0
            for start_time, instruction in zip(
                self._op_start_times, self.circuit_context.data
            ):  # Loop over instructions in circuit context

                # Check if instruction involves target or nearest neighbor qubits
                involves_target_qubits = any(
                    [
                        qubit in reg
                        for reg in [self.circ_tgt_register, circ_nn_register]
                        for qubit in instruction.qubits
                    ]
                )
                if involves_target_qubits:
                    involved_qubits = [
                        qubit
                        for qubit in instruction.qubits
                        if qubit not in self.circ_tgt_register
                    ]
                else:
                    involved_qubits = []

                # If instruction involves target or nn qubits and happens before target gate, add it to custom circuit

                if (
                    counts <= i or start_time <= self._target_instruction_timings[i]
                ) and involves_target_qubits:
                    for qubit in involved_qubits:
                        if (
                            mapping[qubit] not in custom_circuits[i].qregs
                        ):  # Add register if not already added
                            baseline_circuits[i].add_register(mapping[qubit])
                            custom_circuits[i].add_register(mapping[qubit])
                            if (
                                self.circuit_context.layout.final_layout is not None
                            ):  # Update physical layout
                                layouts[i].add(
                                    mapping[qubit][0],
                                    self.circuit_context.layout.final_layout[qubit],
                                )
                            else:
                                layouts[i].add(
                                    mapping[qubit][0],
                                    self.circuit_context.qubits.index(qubit),
                                )

                    baseline_circuits[i].append(
                        instruction.operation,
                        (
                            (
                                mapping[q][0]
                                if q not in self.circ_tgt_register
                                else mapping[q]
                            )
                            for q in instruction.qubits
                        ),
                    )
                    if instruction != self.target_instruction:
                        custom_circuits[i].append(
                            instruction.operation,
                            (
                                (
                                    mapping[q][0]
                                    if q not in self.circ_tgt_register
                                    else mapping[q]
                                )
                                for q in instruction.qubits
                            ),
                        )
                    else:  # Add custom instruction in place of target gate
                        try:
                            self.parametrized_circuit_func(
                                custom_circuits[i],
                                self.parameters[counts],
                                tgt_register,
                                **self._func_args,
                            )
                        except TypeError:
                            raise TypeError("Failed to call parametrized_circuit_func")
                        counts += 1
            # custom_circuits[i] = remove_unused_wires(custom_circuits[i])
            # baseline_circuits[i] = remove_unused_wires(baseline_circuits[i])

        target = GateTarget(
            self.config.target["gate"],
            self.physical_target_qubits,
            self.config.n_reps,
            baseline_circuits,
            tgt_register,
            layouts,
        )
        return target, custom_circuits, baseline_circuits

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the Environment, chooses a new input state"""
        super().reset(seed=seed)

        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_space.shape[0]
        )
        self._inside_trunc_tracker = 0
        return self._get_obs(), self._get_info()

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
                self._optimal_action = self.mean_action
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

            return obs, reward, terminated, False, self._get_info()

    def _get_obs(self):
        if isinstance(self.target, GateTarget) and self.config.reward_method == "state":
            return np.array(
                [
                    self._index_input_state
                    / len(self.target.input_states[self.trunc_index]),
                    self._target_instruction_timings[self._inside_trunc_tracker],
                ]
                + list(self._observable_to_observation())
            )
        else:
            return np.array(
                [0, self._target_instruction_timings[self._inside_trunc_tracker]]
            )

    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: Batch of actions
        """
        new_qc = qc.copy()
        n_actions = self.action_space.shape[-1]
        n_custom_instructions = (
            self.trunc_index + 1
        )  # Count custom instructions present in the current truncation
        baseline_circ = self.baseline_circuits[self.trunc_index]
        target = Statevector(baseline_circ)

        if (
            self.config.check_on_exp
        ):  # Perform real experiments to retrieve from measurement data fidelities
            # Assess circuit fidelity with ComputeUncompute algo
            try:
                # job = self.fidelity_checker.run(
                #     [qc] * len(params),
                #     [baseline_circ] * len(params),
                #     values_1=params,
                # )
                # circuit_fidelities = job.result().fidelities
                angle_sets = np.clip(
                    np.random.normal(
                        self.mean_action,
                        self.std_action,
                        size=(self.config.benchmark_batch_size, n_actions),
                    ),
                    self.action_space.low,
                    self.action_space.high,
                )

                print("Starting Direct Fidelity Estimation...")
                observables, shots = self.retrieve_observables(
                    self._input_state.target_state,
                    self.circuits[self.trunc_index],
                    self.config.benchmark_config.dfe_precision,
                )
                if self.abstraction_level == "circuit":
                    new_qc = self.backend_info.custom_transpile(
                        new_qc,
                        initial_layout=self.layout[self.trunc_index],
                        scheduling=False,
                    )
                pubs = [
                    (
                        new_qc,
                        obs.apply_layout(new_qc.layout),
                        angle_sets,
                        1 / np.sqrt(shot),
                    )
                    for obs, shot in zip(
                        observables.group_commuting(qubit_wise=True), shots
                    )
                ]
                if isinstance(self.estimator, EstimatorV2):
                    self.estimator.options.update(
                        job_tags=[f"DFE_step{self._step_tracker}"]
                    )
                job = self.estimator.run(pubs=pubs)
                results = job.result()
                circuit_fidelities = np.sum(
                    [result.data.evs for result in results], axis=0
                ) / len(observables)
                print("Finished DFE")
            except Exception as exc:
                self.close()
                raise exc

        else:  # Perform ideal simulation at circuit or pulse level
            if self.abstraction_level == "circuit":
                # Calculate circuit fidelity with statevector simulation
                if isinstance(self.backend, AerBackend):
                    backend = self.backend
                elif self.backend is None:
                    backend = AerSimulator(method="statevector")

                else:
                    noise_model = NoiseModel.from_backend(self.backend)
                    backend = AerSimulator(
                        noise_model=noise_model, method="density_matrix"
                    )
                new_qc.save_density_matrix()
                circ = transpile(new_qc, backend=backend, optimization_level=0)

                states_result = backend.run(
                    circ,
                    parameter_binds=[
                        {
                            self._parameters[i][j]: params[:, i * n_actions + j]
                            for i in range(n_custom_instructions)
                            for j in range(n_actions)
                        }
                    ],
                ).result()
                output_states = [
                    states_result.data(i)["density_matrix"]
                    for i in range(self.batch_size)
                ]

            else:  # Pulse simulation
                # Calculate circuit fidelity with pulse simulation
                if isinstance(self.backend, DynamicsBackend) and isinstance(
                    self.backend.options.solver, JaxSolver
                ):
                    # Jax compatible pulse simulation

                    output_states = np.array(self.backend.options.solver.batched_sims)[
                        :, 1, :
                    ]

                    output_states = [
                        projected_statevector(s, self.backend.options.subsystem_dims)
                        for s in output_states
                    ]

                else:
                    raise NotImplementedError(
                        "Pulse simulation not yet implemented for this backend"
                    )
            circuit_fidelities = [
                state_fidelity(state, Statevector(baseline_circ))
                for state in output_states
            ]
            # circuit_fidelities = [state_fidelity(partial_trace(state,
            #                                                    list(range(state.num_qubits))[target.num_qubits:]),
            #                                      partial_trace(Statevector(baseline_circ),
            #                                                    list(range(state.num_qubits))[target.num_qubits:]))
            #                       for state in output_states]
        self.circuit_fidelity_history.append(np.mean(circuit_fidelities))
        print("Fidelity stored", self.circuit_fidelity_history[-1])
        return circuit_fidelities

    @property
    def parameters(self) -> List[ParameterVector]:
        return self._parameters

    @property
    def fidelity_history(self):
        return self.circuit_fidelity_history

    @property
    def tgt_instruction_counts(self) -> int:
        return self._tgt_instruction_counts

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
        self.circuit_fidelity_history.clear()
