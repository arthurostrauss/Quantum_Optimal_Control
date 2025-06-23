from __future__ import annotations

import numpy as np
from oqc import CompilationResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives.containers.bit_array import BitArray
from qm import QuantumMachine, Program
from qm.qua._expressions import QuaArrayVariable
from quam.utils.qua_types import QuaVariableInt

from ..environment import (
    ContextAwareQuantumEnvironment,
    QEnvConfig,
)

from .qua_utils import binary, get_gaussian_sampling_input, clip_qua
from .qm_config import QMConfig
from qm.qua import *
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union
from qiskit_qm_provider import (
    Parameter as QuaParameter,
    ParameterTable,
    Direction,
    InputType,
    ParameterPool,
    QMBackend,
)
from qualang_tools.callable_from_qua import callable_from_qua, patch_qua_program_addons
from ..rewards import CAFERewardDataList, ChannelRewardDataList, StateRewardDataList

ALL_MEASUREMENTS_TAG = "measurements"
"""The tag to save all measurements results to."""
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"
RewardDataType = Union[CAFERewardDataList, ChannelRewardDataList, StateRewardDataList]

# @callable_from_qua
# def qua_print(*args):
#     text = ""
#     for i in range(0, len(args)-1, 2):
#         text += f"{args[i]}= {args[i+1]} | "
#     if len(args) % 2 == 1:
#         text += f"{args[-1]} | "
#     print(text)


def _get_state_int(qc: QuantumCircuit, result: CompilationResult, state_int: QuaVariableInt):
    for c, clbit in enumerate(qc.clbits):
        bit = qc.find_bit(clbit)
        if len(bit.registers) == 0:
            bit_output = result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{c}"]
        else:
            creg, creg_index = bit.registers[0]
            bit_output = result.result_program[creg.name][creg_index]
        assign(
            state_int,
            state_int + (1 << c) * Cast.to_int(bit_output),
        )
    return state_int


class QMEnvironment(ContextAwareQuantumEnvironment):

    def __init__(
        self,
        training_config: QEnvConfig,
        job: Optional[RunningQmJob] = None,
    ):
        ParameterPool.reset()
        super().__init__(training_config)
        if not isinstance(self.config.backend_config, QMConfig) or not isinstance(
            self.backend, QMBackend
        ):
            raise ValueError(
                "The backend should be a QMBackend object and the config should be a QMConfig object"
            )

        if not self.config.reward_method in ["state", "channel", "cafe"]:
            raise ValueError(
                "The reward method should be one of the following: state, channel, cafe"
            )
        mu = QuaParameter(
            "mu",
            [0.0] * self.n_actions,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        sigma = QuaParameter(
            "sigma",
            [1.0] * self.n_actions,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        self.policy = ParameterTable([mu, sigma], name="policy")
        self.reward = QuaParameter(
            "reward",
            [0] * 2**self.n_qubits,
            input_type=self.input_type,
            direction=Direction.INCOMING,
        )

        self.real_time_circuit = self.config.reward.get_real_time_circuit(
            self.circuits,
            self.target,
            self.config,
            skip_transpilation=True,
        )
        self.input_state_vars = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
            filter_function=lambda x: "input" in x.name,
            name="input_state_vars",
        )
        self.observable_vars: Optional[ParameterTable] = (
            ParameterTable.from_qiskit(
                self.real_time_circuit,
                input_type=self.input_type,
                filter_function=lambda x: "observable" in x.name,
                name="observable_vars",
            )
            if self.config.dfe
            else None
        )

        self.n_reps_var: Optional[QuaParameter] = (
            QuaParameter(
                "n_reps",
                self.n_reps,
                input_type=self.input_type,
                direction=Direction.OUTGOING,
            )
            if self.real_time_circuit.has_var("n_reps")
            else None
        )

        self.real_time_circuit_parameters = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=None,
            filter_function=lambda x: isinstance(x, Parameter),
            name="real_time_circuit_parameters",
        )
        self.circuit_choice_var: Optional[QuaParameter] = (
            QuaParameter(
                "circuit_choice",
                0,
                input_type=self.input_type,
                direction=Direction.OUTGOING,
            )
            if self.real_time_circuit.has_var("circuit_choice")
            else None
        )

        self.pauli_shots = QuaParameter(
            "pauli_shots",
            self.n_shots,
            input_type=self.input_type if self.config.dfe else None,
            direction=Direction.OUTGOING if self.config.dfe else None,
        )

        self.max_input_state = QuaParameter(
            "max_input_state",
            0,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        self.max_observables = (
            QuaParameter(
                "max_observables",
                0,
                input_type=self.input_type,
                direction=Direction.OUTGOING,
            )
            if self.config.dfe
            else None
        )

        if self.input_type == InputType.DGX:
            ParameterPool.patch_opnic_wrapper(self.qm_backend_config.opnic_dev_path)

        self._qm_job: Optional[RunningQmJob] = job
        self._qm: Optional[QuantumMachine] = None
        self.backend.update_compiler_from_target(self.input_type)
        if hasattr(self.real_time_circuit, "calibrations") and self.real_time_circuit.calibrations:
            self.backend.update_calibrations(qc=self.real_time_circuit, input_type=self.input_type)
        self._step_indices = {}
        self._total_data_points = 0

    def step(self, action):
        """
        Perform the action on the quantum environment
        """
        dim = 2**self.n_qubits
        if self._qm_job is None:
            raise RuntimeError(
                "The QUA program has not been started yet. Call start_program() first."
            )

        push_args = {
            "job": self.qm_job,
            "qm": self.qm,
            "verbosity": self.qm_backend_config.verbosity,
        }
        mean_val = self.mean_action.tolist()
        std_val = self.std_action.tolist()

        additional_input = (
            self.config.execution_config.dfe_precision if self.config.dfe else self.baseline_circuit
        )
        reward_data = self.config.reward.get_reward_data(
            self.circuit,
            np.zeros((1, self.n_actions)),
            self.target,
            self.config,
            additional_input,
        )
        self._reward_data = reward_data

        # Push policy parameters to trigger real-time action sampling
        self.policy.push_to_opx({"mu": mean_val, "sigma": std_val}, **push_args)
        print("Just pushed policy parameters to OPX:", mean_val, std_val)
        if self.circuit_choice_var is not None:
            self.circuit_choice_var.push_to_opx(self.circuit_choice, **push_args)

        if self.n_reps_var is not None:
            self.n_reps_var.push_to_opx(self.n_reps, **push_args)

        # Push the data to compute reward to the OPX
        input_state_indices = reward_data.input_indices
        max_input_state = len(input_state_indices)
        num_obs_per_input_state = tuple(
            len(reward_data.observables_indices[i]) if self.config.dfe else 1
            for i in range(max_input_state)
        )
        cumulative_datapoints = np.cumsum(num_obs_per_input_state).tolist()
        step_data_points = int(cumulative_datapoints[-1])
        self._step_indices[self.step_tracker] = (
            self._total_data_points,
            self._total_data_points + step_data_points,
        )
        self._total_data_points += step_data_points

        self.max_input_state.push_to_opx(max_input_state, **push_args)
        reward = np.zeros(shape=(self.batch_size,))

        for i, input_state in enumerate(input_state_indices):
            input_state_dict = {
                input_var: input_state_val
                for input_var, input_state_val in zip(self.input_state_vars.parameters, input_state)
            }
            self.input_state_vars.push_to_opx(input_state_dict, **push_args)
            if self.config.dfe:
                observables = reward_data.observables_indices[i]
                num_obs = len(observables)
                self.max_observables.push_to_opx(num_obs, **push_args)
                self.pauli_shots.push_to_opx(reward_data.shots[i], **push_args)
                for j, observable in enumerate(observables):
                    observable_dict = {
                        observable_var: observable_val
                        for observable_var, observable_val in zip(
                            self.observable_vars.parameters, observable
                        )
                    }
                    self.observable_vars.push_to_opx(observable_dict, **push_args)
        fetching_index, finishing_index = self._step_indices[self.step_tracker]
        fetching_size = finishing_index - fetching_index
        collected_counts = self.reward.fetch_from_opx(
            self.qm_job,
            fetching_index=fetching_index,
            fetching_size=fetching_size,
            verbosity=self.qm_backend_config.verbosity,
            time_out=self.backend.options.timeout,
        )

        # The counts are a flattened list of counts from the initial shape (max_input_state, max_observables, batch_size)
        # Reshape the counts to the original shape
        if self.config.dfe:
            counts = []
            # collected_counts is an array containing a flat dimension of size np.cumsum(num_obs_per_input_state)
            # each item of this flat array is an array of shape (batch_size, dim)
            # We need to reshape it in such a way that the batch_size is the first dimension,
            # the input state is the second dimension, and the observables per input are the third dimension
            formatted_counts = []
            count_idx = 0
            for i_idx in range(max_input_state):
                formatted_counts.append([])
                num_obs = num_obs_per_input_state[i_idx]
                for o_idx in range(num_obs):
                    formatted_counts[i_idx].append([])
                    counts_array = np.array(collected_counts[count_idx], dtype=int)
                    formatted_counts[i_idx][o_idx] = counts_array

            # Now reshape the formatted_counts to put batch_size dimension as the first dimension
            for batch_idx in range(self.batch_size):
                counts.append([])
                for i_idx in range(max_input_state):
                    counts[batch_idx].append([])
                    for o_idx in range(num_obs_per_input_state[i_idx]):
                        counts[batch_idx][i_idx].append(formatted_counts[i_idx][o_idx][batch_idx])

        else:
            shape = (max_input_state, self.batch_size, dim)
            transpose_shape = (1, 0, 2)
            counts = np.transpose(np.array(collected_counts).reshape(shape), transpose_shape)

        # Convert the counts to a dictionary of counts

        for batch_idx in range(self.batch_size):
            if self.config.dfe:
                exp_value = reward_data.id_coeff
                for i_idx in range(max_input_state):
                    for o_idx in range(num_obs_per_input_state[i_idx]):
                        counts_dict = {
                            binary(i, self.n_qubits): counts[batch_idx][i_idx][o_idx][i]
                            for i in range(dim)
                        }
                        obs = reward_data[i_idx].hamiltonian.group_commuting(True)[o_idx]
                        # Build a null SparsePauliOp (coeff set to 0.)
                        diag_obs = SparsePauliOp("I" * self.n_qubits, 0.0)
                        for obs_, coeff in zip(obs.paulis, obs.coeffs):
                            diag_obs_label = ""
                            for char in obs_.to_label():
                                # Convert each non-trivial Pauli term to Z term
                                diag_obs_label += char if char == "I" else "Z"
                            diag_obs += SparsePauliOp(diag_obs_label, coeff)
                        bit_array = BitArray.from_counts(counts_dict, num_bits=self.n_qubits)
                        exp_value += bit_array.expectation_values(diag_obs.simplify())
                exp_value /= reward_data.pauli_sampling
                reward[batch_idx] = exp_value
            else:
                survival_probability = np.zeros(shape=(max_input_state,))
                for i_idx in range(max_input_state):
                    counts_dict = {
                        binary(i, self.n_qubits): counts[batch_idx][i_idx][i] for i in range(dim)
                    }
                    bit_array = BitArray.from_counts(counts_dict, num_bits=self.n_qubits)
                    survival_probability[i_idx] = (
                        bit_array.get_int_counts().get(0, 0) / self.n_shots
                    )
                reward[batch_idx] = np.mean(survival_probability)

        if np.mean(reward) > self._max_return:
            self._max_return = np.mean(reward)
            self._optimal_actions[self.circuit_choice] = self.mean_action

        reward = np.clip(reward, 0.0, 1.0 - 1e-6)
        self.reward_history.append(reward)
        reward = -np.log10(1.0 - reward)  # Convert to negative log10 scale

        return self._get_obs(), reward, True, False, self._get_info()

    def rl_qoc_training_qua_prog(self, num_updates: int = 1000) -> Program:
        """
        Generate a QUA program tailor-made for the RL-based calibration project
        """
        self.reset_parameters()

        qc = self.real_time_circuit
        dim = int(2**self.n_qubits)
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) >= 2:
                raise ValueError("Overlapping classical registers are not supported")

        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)

            self.real_time_circuit_parameters.declare_variables()
            self.max_input_state.declare_variable()
            self.input_state_vars.declare_variables()
            self.policy.declare_variables()
            self.reward.declare_variable()
            self.reward.declare_stream()
            for var in [self.circuit_choice_var, self.n_reps_var, self.max_observables]:
                if var is not None:
                    var.declare_variable()
            if self.observable_vars is not None:
                self.observable_vars.declare_variables()
                obs_idx = declare(int)

            # Number of shots for each observable (or total number of shots if no observables)
            self.pauli_shots.declare_variable()
            n_u = declare(int)
            input_state_count = declare(int)
            state_int = declare(int, value=0)
            batch_r = Random(self.seed)

            if self.backend.init_macro is not None:
                self.backend.init_macro()
            # Infinite loop to run the training
            with for_(n_u, 0, n_u < num_updates, n_u + 1):
                self.policy.load_input_values()  # Load µ and σ

                for var in [self.circuit_choice_var, self.n_reps_var, self.max_input_state]:
                    if var is not None:
                        var.load_input_value()

                with for_(
                    input_state_count,
                    0,
                    input_state_count < self.max_input_state.var,
                    input_state_count + 1,
                ):
                    # Load info about input states to prepare (single qubit indices)
                    self.input_state_vars.load_input_values()

                    if self.config.dfe:
                        self.max_observables.load_input_value()
                        self.pauli_shots.load_input_value()  # TODO: RELAX THIS TO PUT IT PER OBSERVABLE WITHIN LOOP BELOW)
                        with for_(obs_idx, 0, obs_idx < self.max_observables.var, obs_idx + 1):
                            # Load info about observable to measure (single qubit indices)
                            self.observable_vars.load_input_values()

                            self._rl_macro(
                                batch_r,
                                state_int,
                                n_u,
                            )

                    else:
                        self._rl_macro(
                            batch_r,
                            state_int,
                            n_u,
                        )

            with stream_processing():
                buffer = (self.batch_size, dim)
                self.reward.stream_processing(buffer=buffer)

        return rl_qoc_training_prog

    def _rl_macro(
        self,
        batch_r: Random,
        state_int: QuaVariableInt,
        n_u: QuaVariableInt,
    ):

        # Declare variables for efficient Gaussian random sampling
        b = declare(int)
        j = declare(int)
        n_lookup, cos_array, ln_array = get_gaussian_sampling_input()
        uniform_r = declare(fixed)
        u1, u2 = declare(int), declare(int)
        tmp1 = declare(fixed, size=self.n_actions)
        tmp2 = declare(fixed, size=self.n_actions)
        lower_bound = declare(fixed, value=self.action_space.low)
        upper_bound = declare(fixed, value=self.action_space.high)
        mu = self.policy.get_variable("mu")
        sigma = self.policy.get_variable("sigma")
        batch_r.set_seed(self.seed + n_u)
        with for_(b, 0, b < self.batch_size, b + 2):
            # Sample from a multivariate Gaussian distribution (Muller-Box method)
            with for_(j, 0, j < self.n_actions, j + 1):
                assign(uniform_r, batch_r.rand_fixed())
                assign(u1, Cast.unsafe_cast_int(uniform_r >> 19))
                assign(u2, Cast.unsafe_cast_int(uniform_r) & ((1 << 19) - 1))
                assign(
                    tmp1[j],
                    mu[j] + sigma[j] * ln_array[u1] * cos_array[u2 & (n_lookup - 1)],
                )
                assign(
                    tmp2[j],
                    mu[j]
                    + sigma[j] * ln_array[u1] * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)],
                )
                clip_qua(tmp1[j], lower_bound[j], upper_bound[j])
                clip_qua(tmp2[j], lower_bound[j], upper_bound[j])

            with for_(j, 0, j < 2, j + 1):
                # Assign the sampled actions to the action batch
                for i, parameter in enumerate(self.real_time_circuit_parameters.parameters):
                    parameter.assign(tmp1[i], condition=(j == 0), value_cond=tmp2[i])

                counts, state_int = self._run_circuit(state_int, self.reward.var)
                # qua_print("counts", counts, "state_int", state_int)
                self.reward.stream_back()

    def _run_circuit(self, state_int: QuaVariableInt, counts: QuaArrayVariable):
        """
        Run the circuit and get the counts
        """
        qc = self.real_time_transpiled_circuit
        n_shots = declare(int)

        with for_(state_int, 0, state_int < 2**self.n_qubits, state_int + 1):
            assign(counts[state_int], 0)
        assign(state_int, 0)  # Reset state_int for the next shot
        with for_(n_shots, 0, n_shots < self.pauli_shots.var, n_shots + 1):
            param_inputs = [
                self.input_state_vars,
                self.real_time_circuit_parameters,
                self.circuit_choice_var,
                self.n_reps_var,
                self.observable_vars,
            ]
            param_inputs = [param for param in param_inputs if param is not None]

            result = self.backend.quantum_circuit_to_qua(qc, param_inputs)
            state_int = _get_state_int(qc, result, state_int)

            assign(counts[state_int], counts[state_int] + 1)
            assign(state_int, 0)  # Reset state_int for the next shot

        return counts, state_int

    def reset_parameters(self):
        self.policy.reset()
        self.reward.reset()
        self.real_time_circuit_parameters.reset()
        self.input_state_vars.reset()
        self.max_input_state.reset()
        self.pauli_shots.reset()
        if self.observable_vars is not None:
            self.observable_vars.reset()
            self.pauli_shots.reset()
            self.max_observables.reset()
        if self.n_reps_var is not None:
            self.n_reps_var.reset()
        if self.circuit_choice_var is not None:
            self.circuit_choice_var.reset()

    @property
    def qm_backend_config(self) -> QMConfig:
        """
        Get the QM backend configuration
        """
        return self.config.backend_config

    @property
    def backend(self) -> QMBackend:
        return super().backend

    @property
    def real_time_transpiled_circuit(self) -> QuantumCircuit:
        """
        Get the real-time circuit transpiled for QUA execution
        """
        return self.backend_info.custom_transpile(
            self.real_time_circuit,
            optimization_level=1,
            initial_layout=self.layout,
            remove_final_measurements=False,
            scheduling=False,
        )

    def start_program(
        self,
    ) -> RunningQmJob:
        """
        Start the QUA program

        Returns:
            RunningQmJob: The running Qmjob
        """
        if self.input_type == InputType.DGX:
            ParameterPool.configure_stream()
        if hasattr(self.real_time_circuit, "calibrations") and self.real_time_circuit.calibrations:
            self.backend.update_calibrations(qc=self.real_time_circuit, input_type=self.input_type)
        self.backend.update_compiler_from_target()
        prog = self.rl_qoc_training_qua_prog(num_updates=self.qm_backend_config.num_updates)
        self._qm_job = self.qm.execute(
            prog, compiler_options=self.qm_backend_config.compiler_options
        )
        return self._qm_job

    def close(self) -> bool:
        """
        Close the environment (stop the running QUA program)
        Returns:

        """
        if self.input_type == InputType.DGX:
            ParameterPool.close_streams()
        finish = self.qm_job.halt()
        if not finish:
            print("Failed to halt the job")
        print("Job status: ", self.qm_job.status)
        self._qm_job = None
        return finish

    @property
    def input_type(self) -> InputType:
        """
        Get the input type for streaming to OPX
        """
        return self.qm_backend_config.input_type

    @property
    def qm_job(self) -> RunningQmJob:
        """
        Get the running QM job
        """
        return self._qm_job

    @property
    def qm(self) -> QuantumMachine:
        """
        Get the QM object
        """
        return self.backend.qm
