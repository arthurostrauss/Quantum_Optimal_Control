"""
Class to generate a RL environment suitable for usage with Gym and PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 26/06/2023
"""

from itertools import chain, product
from typing import Dict, Optional, List, Any, Tuple, TypeVar, SupportsFloat, Union

import numpy as np

# Qiskit imports
from qiskit import transpile, schedule
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ParameterVector,
)
from qiskit.circuit.library import ECRGate
from qiskit.quantum_info.operators import SparsePauliOp, pauli_basis, Operator
from qiskit.quantum_info.states import partial_trace
from qiskit.quantum_info.operators.measures import (
    average_gate_fidelity,
    process_fidelity,
    state_fidelity,
)
from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit.transpiler import Layout, InstructionProperties, Target
from qiskit_aer.backends import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import BackendData
from qiskit_experiments.library.tomography.basis import PauliPreparationBasis
from qiskit_ibm_provider import IBMBackend

# Qiskit Primitive: for computing Pauli expectation value sampling easily
from qiskit_ibm_runtime import (
    Estimator as RuntimeEstimator,
    Sampler,
    IBMRuntimeError,
    Session,
)
from tensorflow_probability.python.distributions import Categorical

from basis_gate_library import EchoedCrossResonance, FixedFrequencyTransmon
from helper_functions import (
    gate_fidelity_from_process_tomography,
    retrieve_backend_info,
    get_ecr_params,
    set_primitives_transpile_options,
    handle_session,
)
from qconfig import QEnvConfig
from quantumenvironment import QuantumEnvironment, _calculate_chi_target_state
from custom_jax_sim import (
    DynamicsBackendEstimator,
    JaxSolver,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def create_array(circ_trunc, batchsize, n_actions):
    arr = np.empty((circ_trunc,), dtype=object)
    for i in range(circ_trunc):
        arr[i] = np.zeros((i + 1, batchsize, n_actions))
    return arr


class ContextAwareQuantumEnvironment(QuantumEnvironment):
    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: QuantumCircuit,
        training_steps_per_gate: Union[List[int], int] = 1500,
        intermediate_rewards: bool = False,
    ):
        """
        Class for wrapping a quantum environment in a Gym environment able to tackle
        context-aware gate calibration. Target gate should be pre-defined through the declaration of a
        QuantumEnvironment, and a circuit in which this gate is used should be provided.
        This class will look for the locations of the target gates in the circuit context and will enable the
        writing of new parametrized circuits, where each gate instance is replaced by a custom gate defined by the Callable parametrized_circuit_func in QuantumEnvironment.

        :param training_config: Training configuration containing all hyperparameters characterizing the environment
            for the RL training
        :param circuit_context: QuantumCircuit where target gate is located to perform context aware gate
            calibration.
        :param training_steps_per_gate: Number of calibration steps per gate instance in circuit context
           (for sequential calibration mode)
        :param intermediate_rewards: Specify if calibration within circuit should be sequential (False)
         or across circuit context (True). For now, sequential mode is preferred.
        """

        super().__init__(training_config)

        assert (
            self.target_type == "gate"
        ), "This class is made for gate calibration only"

        # Retrieve information on the backend for building circuit context workflow
        self.backend_data = BackendData(self.backend)
        dt, coupling_map, basis_gates, instruction_durations = retrieve_backend_info(
            self.backend, self.estimator
        )
        if "cx" in basis_gates:
            print(
                "Basis gate Library for CX gate not yet available, will be transpiled over ECR basis gate"
            )
        # Retrieve qubits forming the local circuit context (target qubits + nearest neighbor qubits on the chip)

        self._physical_neighbor_qubits = list(
            filter(
                lambda x: x not in self.physical_target_qubits,
                chain(
                    *[
                        list(coupling_map.neighbors(target_qubit))
                        for target_qubit in self.physical_target_qubits
                    ]
                ),
            )
        )
        if self.abstraction_level == "pulse":
            if "ecr" not in basis_gates and self.backend_data.num_qubits > 1:
                target: Target = self.backend.target
                target.add_instruction(
                    ECRGate(),
                    properties={qubits: None for qubits in coupling_map.get_edges()},
                )
                cals = Calibrations.from_backend(
                    self.backend,
                    [
                        FixedFrequencyTransmon(["x", "sx"]),
                        EchoedCrossResonance(["cr45p", "cr45m", "ecr"]),
                    ],
                    add_parameter_defaults=True,
                )

                for qubit_pair in coupling_map.get_edges():
                    if target.has_calibration("cx", qubit_pair):
                        default_params, _, _ = get_ecr_params(self.backend, qubit_pair)
                        error = self.backend.target["cx"][qubit_pair].error
                        target.update_instruction_properties(
                            "ecr",
                            qubit_pair,
                            InstructionProperties(
                                error=error,
                                calibration=cals.get_schedule(
                                    "ecr", qubit_pair, default_params
                                ),
                            ),
                        )
                basis_gates.append("ecr")
                for i, gate in enumerate(basis_gates):
                    if gate == "cx":
                        basis_gates.pop(i)
                # raise ValueError("Backend must carry 'ecr' as basis_gate for transpilation, will change in the future")

        self.circuit_context = transpile(
            circuit_context.remove_final_measurements(inplace=False),
            backend=self.backend,
            scheduling_method="asap",
            basis_gates=basis_gates,
            coupling_map=coupling_map if coupling_map.size() != 0 else None,
            instruction_durations=instruction_durations,
            optimization_level=0,
            dt=dt,
        )

        self.tgt_register = QuantumRegister(
            bits=[self.circuit_context.qubits[i] for i in self.physical_target_qubits],
            name="tgt",
        )
        self.nn_register = QuantumRegister(
            bits=[
                self.circuit_context.qubits[i] for i in self._physical_neighbor_qubits
            ],
            name="nn",
        )

        self.layout = Layout(
            {
                self.tgt_register[i]: self.physical_target_qubits[i]
                for i in range(len(self.tgt_register))
            }
            | {
                self.nn_register[j]: self._physical_neighbor_qubits[j]
                for j in range(len(self.nn_register))
            }
        )
        self._input_circuits = [
            PauliPreparationBasis().circuit(s).decompose()
            for s in product(
                range(4), repeat=self.tgt_register.size + self.nn_register.size
            )
        ]
        skip_transpilation = False

        set_primitives_transpile_options(
            self.estimator,
            self.sampler,
            self.layout,
            skip_transpilation,
            self.physical_target_qubits + self.physical_neighbor_qubits,
        )

        self.fidelity_checker._sampler = self.sampler
        self._d = 2**self.tgt_register.size

        # Adjust target register to match it with circuit context
        self.target_instruction.qubits = tuple(self.tgt_register)

        self._tgt_instruction_counts = self.circuit_context.data.count(
            self.target_instruction
        )

        self._parameters = [
            ParameterVector(f"a_{j}", self.action_space.shape[-1])
            for j in range(self.tgt_instruction_counts)
        ]
        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_space.shape[-1]
        )

        # Store time and instruction indices where target gate is played in circuit

        self._target_instruction_timings = []
        if instruction_durations is not None:
            for i, instruction in enumerate(self.circuit_context.data):
                if instruction == self.target_instruction:
                    self._target_instruction_timings.append(
                        self.circuit_context.op_start_times[i]
                    )

        (
            self.circuit_truncations,
            self.baseline_truncations,
            self.custom_gates,
        ) = self._generate_circuit_truncations()
        self._punctual_avg_fidelities = np.zeros(self.batch_size)
        self._punctual_circuit_fidelities = np.zeros(self.batch_size)
        self._punctual_observable: SparsePauliOp = SparsePauliOp("X")
        self.circuit_fidelity_history = []
        self._index_input_state = np.random.randint(len(self.target["input_states"]))
        self._training_steps_per_gate = training_steps_per_gate
        self._inside_trunc_tracker = 0
        self._trunc_index = 0

        self._intermediate_rewards = intermediate_rewards
        self._max_return = 0
        self._best_action = np.zeros((self.batch_size, self.action_space.shape[-1]))

    @property
    def physical_neighbor_qubits(self):
        return self._physical_neighbor_qubits

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
            raise ValueError("Training steps number should be positive integer.")

    def _retrieve_target_state(self, input_state_index: int, iteration: int) -> Dict:
        """
        Calculate anticipated target state for fidelity estimation based on input state preparation
        and truncation of the circuit
        :param input_state_index: Integer index indicating the PauliPreparationBasis vector
        :param iteration: Truncation index of the transpiled contextual circuit
        """
        # input_circuit: QuantumCircuit = self.target["input_states"][input_state_index]["circuit"]
        input_circuit = self._input_circuits[input_state_index]
        ref_circuit, custom_circuit = (
            self.baseline_truncations[iteration],
            self.circuit_truncations[iteration],
        )
        target_circuit = ref_circuit.compose(input_circuit, inplace=False, front=True)

        # custom_target_circuit.append(input_circuit.to_instruction(), self.tgt_register)
        # target_circuit.append(input_circuit.to_instruction(), self.tgt_register)

        # for gate in ref_circuit.data:  # Append only gates that are applied on the target register
        #     if all([qubit in self.tgt_register for qubit in gate.qubits]):
        #         target_circuit.append(gate)

        return _calculate_chi_target_state(
            {
                "dm": DensityMatrix(target_circuit)
                if len(self.physical_neighbor_qubits) == 0
                else partial_trace(
                    Statevector(target_circuit), self.physical_neighbor_qubits
                ),
                "circuit": custom_circuit,
                "target_type": "state",
                "input_state_circ": input_circuit,
            },
            n_qubits=len(self.tgt_register),
        )

    def _generate_circuit_truncations(
        self,
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit], List[QuantumCircuit]]:
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
        custom_gates, custom_gate_circ = [], []
        # Build sub-circuit contexts: each circuit goes until target gate and preserves nearest neighbor operations
        for i in range(self.tgt_instruction_counts):
            counts = 0
            for start_time, instruction in zip(
                self.circuit_context.op_start_times, self.circuit_context.data
            ):
                qubits_in_vicinity = np.array(
                    [
                        (qubit, qubit in self.tgt_register or qubit in self.nn_register)
                        for qubit in instruction.qubits
                    ]
                )
                other_qubits = [
                    qubit[0]
                    for qubit in qubits_in_vicinity
                    if not qubit[1]
                    and qubit[0] not in custom_circuit_truncations[i].qubits
                ]

                # if all([qubit in self.tgt_register or qubit in self.nn_register for qubit in instruction.qubits]):
                if any(qubits_in_vicinity[:, 1]):
                    if (
                        counts <= i or start_time <= self._target_instruction_timings[i]
                    ):  # Append until reaching tgt i
                        if other_qubits:
                            last_reg_name = (
                                baseline_circuit_truncations[i].qregs[-1].name
                            )
                            if last_reg_name != "nn":
                                new_reg = QuantumRegister(
                                    name=last_reg_name
                                    + str(int(last_reg_name[-1]) + 1),
                                    bits=other_qubits,
                                )
                            else:
                                new_reg = QuantumRegister(
                                    name="anc_0", bits=other_qubits
                                )
                            baseline_circuit_truncations[i].add_register(new_reg)
                            custom_circuit_truncations[i].add_register(new_reg)
                            # baseline_circuit_truncations[i].add_bits(other_qubits)
                            # custom_circuit_truncations[i].add_bits(other_qubits)
                        baseline_circuit_truncations[i].append(instruction)

                        if instruction != self.target_instruction:
                            custom_circuit_truncations[i].append(instruction)

                        else:  # Add custom instruction in place of target gate
                            if self.parametrized_circuit_func is not None:
                                self.parametrized_circuit_func(
                                    custom_circuit_truncations[i],
                                    self._parameters[counts],
                                    self.tgt_register,
                                )
                            else:
                                custom_circuit_truncations[i].append(instruction)
                            op = custom_circuit_truncations[i].data[-1]
                            if op not in custom_gates:
                                custom_gates.append(op)
                                gate_circ = QuantumCircuit(self.tgt_register)
                                if self.parametrized_circuit_func is not None:
                                    self.parametrized_circuit_func(
                                        gate_circ,
                                        self._parameters[counts],
                                        self.tgt_register,
                                    )
                                else:
                                    gate_circ.append(self.target_instruction)
                                custom_gate_circ.append(gate_circ)
                            counts += 1

        return (
            custom_circuit_truncations,
            baseline_circuit_truncations,
            custom_gate_circ,
        )

    def store_benchmarks(self, params: np.array):
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: Batch of actions
        :param check_on_exp: If True, run on backend a set of fidelity characterization experiments (ComputeUncompute
            for circuit fidelity, ProcessTomography for gate fidelities), if False, simulate StateVector
        """
        n_actions = self.action_space.shape[-1]

        n_custom_instructions = (
            self._trunc_index + 1
        )  # Count custom instructions present in the current truncation
        benchmark_circ = self.circuit_truncations[self._trunc_index].copy(name="b_circ")
        baseline_circ = self.baseline_truncations[self._trunc_index]

        custom_gate_circuits = []
        for i in range(n_custom_instructions):
            assigned_gate = QuantumCircuit(
                self.tgt_register
            )  # Assign parameter to each custom gate
            # to check its individual gate fidelity (have to detour by a QuantumCircuit)
            self.parametrized_circuit_func(
                assigned_gate, self._parameters[i], self.tgt_register
            )
            custom_gate_circuits.append(assigned_gate)

        if (
            self.check_on_exp
        ):  # Perform real experiments to retrieve from measurement data fidelities
            # Assess circuit fidelity with ComputeUncompute algo
            try:
                job = self.fidelity_checker.run(
                    [benchmark_circ] * len(params),
                    [baseline_circ] * len(params),
                    values_1=params,
                )
                circuit_fidelities = job.result().fidelities
            except Exception as exc:
                self.close()
                raise exc

            self._punctual_circuit_fidelities = circuit_fidelities
            self.circuit_fidelity_history.append(np.mean(circuit_fidelities))

        else:  # Perform ideal simulation at circuit or pulse level
            if self.abstraction_level == "circuit":
                # Calculate circuit fidelity with statevector simulation

                benchmark_circ.save_statevector()
                state_backend = AerSimulator(method="statevector")
                states_result = state_backend.run(
                    benchmark_circ.decompose(),
                    parameter_binds=[
                        {
                            self._parameters[i][j]: params[:, i * n_actions + j]
                            for i in range(n_custom_instructions)
                            for j in range(n_actions)
                        }
                    ],
                ).result()
                output_state_list = [
                    states_result.get_statevector(i) for i in range(self.batch_size)
                ]
                batched_dm = DensityMatrix(
                    np.mean(
                        [
                            state.to_operator().to_matrix()
                            for state in output_state_list
                        ],
                        axis=0,
                    )
                )
                batched_circuit_fidelity = state_fidelity(
                    batched_dm, Statevector(baseline_circ)
                )
                self.circuit_fidelity_history.append(
                    batched_circuit_fidelity
                )  # Circuit fidelity over the action batch

                circuit_fidelities = [
                    state_fidelity(state, Statevector(baseline_circ))
                    for state in output_state_list
                ]
                self._punctual_circuit_fidelities = np.array(circuit_fidelities)

                # Calculate average gate fidelities for each gate instance within circuit truncation
                custom_gates_list = []
                unitary_backend = AerSimulator(method="unitary")
                for i, gate in enumerate(self.custom_gates):
                    gate.save_unitary()
                    gates_result = unitary_backend.run(
                        gate.decompose(),
                        parameter_binds=[
                            {
                                self._parameters[i][j]: params[:, i * n_actions + j]
                                for j in range(n_actions)
                            }
                        ],
                    ).result()
                    assigned_instructions = [
                        gates_result.get_unitary(i) for i in range(self.batch_size)
                    ]
                    custom_gates_list.append(assigned_instructions)

                tgt_gate = self.target["gate"]
                avg_fidelities = np.array(
                    [
                        [
                            average_gate_fidelity(custom_gate, tgt_gate)
                            for custom_gate in custom_gates
                        ]
                        for custom_gates in custom_gates_list
                    ]
                )
                self._punctual_avg_fidelities = avg_fidelities
                self.avg_fidelity_history.append(
                    np.mean(avg_fidelities, axis=1)
                )  # Avg gate fidelity over action batch

            else:  # Pulse simulation
                assert isinstance(self.backend, DynamicsBackend)

                model_dim = self.backend.options.solver.model.dim % 2
                qc_list = [
                    benchmark_circ.assign_parameters(param) for param in params
                ]  # Bind each action of the batch to a circ
                schedule_list = [
                    schedule(qc, backend=self.backend, dt=self.backend_data.dt)
                    for qc in qc_list
                ]
                unitaries = self._simulate_pulse_schedules(schedule_list)

                unitaries = [Operator(np.array(unitary.y[0])) for unitary in unitaries]
                if model_dim != 0:
                    # TODO: Line below yields an error if simulation is not done over a set of qubit
                    #  (fails if third level of
                    # TODO: transmon is simulated), project unitary in qubit subspace
                    dms = [
                        DensityMatrix.from_int(0, model_dim).evolve(unitary)
                        for unitary in unitaries
                    ]
                    qubitized_unitaries = [
                        np.zeros((self._d, self._d)) for _ in range(len(unitaries))
                    ]
                    for u in range(len(unitaries)):
                        for i in range(self._d):
                            for j in range(self._d):
                                qubitized_unitaries[u][i, j] = unitaries[u]

                if self.target_type == "state":
                    density_matrix = DensityMatrix(
                        np.mean(
                            [
                                Statevector.from_int(0, dims=self._d).evolve(unitary)
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
                                process_fidelity(unitary, Operator(baseline_circ))
                                for unitary in unitaries
                            ]
                        )
                    )
                    self.avg_fidelity_history.append(
                        np.mean(
                            [
                                average_gate_fidelity(unitary, Operator(baseline_circ))
                                for unitary in unitaries
                            ]
                        )
                    )
                self.built_unitaries.append(unitaries)

    def _select_trunc_index(self):
        if self._intermediate_rewards:
            return self.step_tracker % self.tgt_instruction_counts
        else:
            return np.min(
                [
                    self._step_tracker // self.training_steps_per_gate,
                    self.tgt_instruction_counts - 1,
                ]
            )

    def episode_length(self, global_step: int):
        assert (
            global_step == self.step_tracker
        ), "Given step not synchronized with internal environment step counter"
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
                    "max avg gate fidelity": [
                        np.max(np.array(self.avg_fidelity_history)[:, i])
                        for i in range(len(self.avg_fidelity_history[0]))
                    ],
                    "arg max avg gate fidelity": [
                        np.argmax(np.array(self.avg_fidelity_history)[:, i])
                        for i in range(len(self.avg_fidelity_history[0]))
                    ],
                    "arg max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "arg max circuit fidelity": np.argmax(
                        self.circuit_fidelity_history
                    ),
                    "best action": self._best_action,
                    "truncation_index": self._trunc_index,
                    "input_state": self._input_circuits[self._index_input_state].name,
                    "observable": self._punctual_observable,
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
                    "input_state": self._input_circuits[self._index_input_state].name,
                    "observable": self._punctual_observable,
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "input_state": self._input_circuits[self._index_input_state].name,
                "truncation_index": self._trunc_index,
                "observable": self._punctual_observable,
            }
        return info

    def _get_obs(self):
        obs = np.array(
            [
                self._index_input_state / len(self.target["input_states"]),
                self._inside_trunc_tracker,
            ]
        )
        # obs = np.array([0., 0])
        return obs

    def perform_action(self, actions, do_benchmark: bool = False):
        raise NotImplementedError(
            "This method shall not be used in this class, use QuantumEnvironment instead"
        )

    def compute_reward(self, fidelity_access=False) -> np.ndarray:
        trunc_index = self._inside_trunc_tracker
        target_state = self._retrieve_target_state(self._index_input_state, trunc_index)
        qc: QuantumCircuit = self.circuit_truncations[trunc_index]

        # Retrieve Pauli observables to sample, and build a weighted sum to feed the Estimator primitive
        observables, pauli_shots = self.retrieve_observables(target_state, qc)

        self._punctual_observable = observables
        # self.parametrized_circuit_func(training_circ, self._parameters[:iteration])

        reshaped_params = np.reshape(
            np.vstack([param_set for param_set in self._param_values[trunc_index]]),
            (self.batch_size, (trunc_index + 1) * self.action_space.shape[-1]),
        )

        if self.do_benchmark():
            print("Starting benchmarking...")
            self.store_benchmarks(reshaped_params)
            print("Finished benchmarking")
        if fidelity_access:
            reward_table = self._punctual_circuit_fidelities
        else:
            print("Sending job...")
            try:
                handle_session(
                    qc,
                    target_state["input_circuit"],
                    self.estimator,
                    self.backend,
                    self._session_counts,
                )

                # Append input state prep circuit to the custom circuit with front composition
                full_circ = qc.compose(
                    target_state["input_circuit"], inplace=False, front=True
                )
                print(observables)

                job = self.estimator.run(
                    circuits=[full_circ] * self.batch_size,
                    observables=[observables] * self.batch_size,
                    parameter_values=reshaped_params,
                    shots=int(np.max(pauli_shots) * self.n_shots),
                )

                reward_table = job.result().values
            except Exception as exc:
                self.close()
                raise exc
            scaling_reward_factor = len(observables) / 4 ** len(self.tgt_register)
            reward_table *= scaling_reward_factor
            reward_table = -np.log10(1 - reward_table)
            print("Job done")

        if np.mean(reward_table) > self._max_return:
            self._max_return = np.mean(reward_table)
            self._best_action = np.mean(reshaped_params, axis=0)
        self.reward_history.append(reward_table)
        assert (
            len(reward_table) == self.batch_size
        ), f"Reward table size mismatch {len(reward_table)} != {self.batch_size} "

        return reward_table

    def clear_history(self) -> None:
        """Reset all counters related to training"""
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
        """Reset the Environment, chooses a new input state"""
        super().reset(seed=seed)

        self._param_values = create_array(
            self.tgt_instruction_counts, self.batch_size, self.action_space.shape[0]
        )
        self._inside_trunc_tracker = 0
        self._episode_tracker += 1
        self._episode_ended = False
        self._index_input_state = np.random.randint(len(self._input_circuits))
        if isinstance(self.estimator, RuntimeEstimator):
            self.estimator.options.environment["job_tags"] = [
                f"rl_qoc_step{self._step_tracker}"
            ]

        # TODO: Remove line below when it works for gate fidelity as well
        # self._index_input_state = 0
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

        if step_status < trunc_index:  # Intermediate step within the circuit truncation
            self._inside_trunc_tracker += 1
            terminated = False

            if self._intermediate_rewards:
                reward_table = self.compute_reward()
                obs = reward_table  # Set observation to obtained reward (might not be the smartest choice here)
                return obs, reward_table, terminated, False, self._get_info()
            else:
                return (
                    self._get_obs(),
                    np.zeros(batch_size),
                    terminated,
                    False,
                    self._get_info(),
                )

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
        string += f"Number of target gates in circuit context: {self.tgt_instruction_counts}\n"
        return string
