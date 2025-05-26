from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.result import QuasiDistribution, sampled_expectation_value
from qiskit.primitives.containers.bit_array import BitArray
from qm import QuantumMachine, QuantumMachinesManager, Program, CompilerOptionArguments

from ..environment import (
    ContextAwareQuantumEnvironment,
    QEnvConfig,
)

from qiskit_qm_provider import QMBackend
from .qua_utils import binary, get_gaussian_sampling_input
from .qm_config import QMConfig, DGXConfig
from qm.qua import *
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union
from qiskit_qm_provider import (
    Parameter as QuaParameter,
    ParameterTable,
    Direction,
    InputType,
    ParameterPool,
)
from ..rewards import CAFERewardDataList, ChannelRewardDataList, StateRewardDataList

ALL_MEASUREMENTS_TAG = "measurements"
"""The tag to save all measurements results to."""
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"
RewardDataType = Union[CAFERewardDataList, ChannelRewardDataList, StateRewardDataList]


class QMEnvironment(ContextAwareQuantumEnvironment):

    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: Optional[QuantumCircuit] = None,
        job: Optional[RunningQmJob] = None,
    ):
        ParameterPool.reset()
        super().__init__(training_config, circuit_context)
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
            self.get_target(),
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
            if self.real_time_circuit.get_var("n_reps", None) is not None
            else None
        )

        self.real_time_circuit_parameters = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
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
            if self.real_time_circuit.get_var("circuit_choice", None) is not None
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
        self.backend.update_calibrations(self.real_time_circuit, input_type=self.input_type)
        self._qm_job: Optional[RunningQmJob] = job
        self._qm: Optional[QuantumMachine] = None

    def step(self, action):
        """
        Perform the action on the quantum environment
        """

        dim = 2**self.n_qubits
        if self._qm_job is None:
            raise RuntimeError(
                "The QUA program has not been started yet. Call start_program() first."
            )

        verbosity = (
            self.qm_backend_config.verbosity if isinstance(self.qm_backend_config, DGXConfig) else 2
        )
        push_args = {
            "job": self.qm_job,
            "qm": self.qm,
            "verbosity": verbosity,
        }
        mean_val = self.mean_action.tolist()
        std_val = self.std_action.tolist()

        additional_input = (
            self.config.execution_config.dfe_precision if self.config.dfe else self.baseline_circuit
        )
        reward_data: RewardDataType = self.config.reward.get_reward_data(
            self.circuit,
            np.zeros((1, self.n_actions)),
            self.target,
            self.config,
            additional_input,
        )

        # Push policy parameters to trigger real-time action sampling
        self.policy.push_to_opx({"mu": mean_val, "sigma": std_val}, **push_args)
        if self.circuit_choice_var is not None:
            self.circuit_choice_var.push_to_opx(self.trunc_index, **push_args)

        if self.n_reps_var is not None:
            self.n_reps_var.push_to_opx(self.n_reps, **push_args)

        # Push the reward data to the OPX
        input_state_indices = reward_data.input_indices
        max_input_state = len(input_state_indices)

        self.max_input_state.push_to_opx(max_input_state, **push_args)
        if self.config.dfe:
            self.max_observables.push_to_opx(len(reward_data.observables_indices), **push_args)
            max_observables = len(reward_data.observables_indices)

        reward = np.zeros(shape=(self.batch_size,))
        for i, input_state in enumerate(input_state_indices):
            input_state_dict = {
                input_var: input_state_val
                for input_var, input_state_val in zip(self.input_state_vars.parameters, input_state)
            }
            self.input_state_vars.push_to_opx(input_state_dict, **push_args)
            if self.config.dfe:
                for j, observable in enumerate(reward_data.observables_indices):
                    observable_dict = {
                        observable_var: list(observable_val)
                        for observable_var, observable_val in zip(
                            self.observable_vars.parameters, observable
                        )
                    }
                    self.observable_vars.push_to_opx(observable_dict, **push_args)
                    self.pauli_shots.push_to_opx(reward_data.shots[j], **push_args)

        collected_counts = self.reward.fetch_from_opx(
            **push_args,
            fetching_index=self.step_tracker,
            fetching_size=self.batch_size
            * max_input_state
            * (max_observables if self.config.dfe else 1),
        )

        # The counts are a flattened list of counts from the initial shape (max_input_state, max_observables, batch_size)
        # Reshape the counts to the original shape
        if self.config.dfe:
            counts = np.array(collected_counts).reshape(
                (max_input_state, max_observables, self.batch_size, dim)
            )
        else:
            counts = np.array(collected_counts).reshape((max_input_state, self.batch_size, dim))

        # In what follows, we will want to put the batch_size dimension first, so we transpose the counts
        if self.config.dfe:
            counts = np.transpose(counts, (2, 0, 1, 3))
        else:
            counts = np.transpose(counts, (1, 0, 2))
        # Convert the counts to a dictionary of counts

        for batch_idx in range(self.batch_size):
            if self.config.dfe:
                exp_value = reward_data.id_coeff
                for i_idx in range(max_input_state):
                    for o_idx in range(max_observables):
                        counts_dict = {
                            binary(i, self.n_qubits): counts[batch_idx, i_idx, o_idx, i]
                            for i in range(dim)
                        }
                        obs = reward_data[i_idx].hamiltonian.group_commuting(True)[o_idx]
                        diag_obs = SparsePauliOp("I" * self.n_qubits, 0.0)
                        for obs_, coeff in zip((obs.paulis, obs.coeffs)):
                            diag_obs_label = ""
                            for char in obs_.to_label():
                                diag_obs_label += char if char == "I" else "Z"
                            diag_obs += SparsePauliOp(diag_obs_label, coeff)
                        bit_array = BitArray.from_counts(counts_dict, num_bits=self.n_qubits)
                        exp_value += bit_array.expectation_values(diag_obs)
                exp_value /= reward_data.pauli_sampling
                reward[batch_idx] = exp_value
            else:
                survival_probability = np.zeros(shape=(max_input_state,))
                for i_idx in range(max_input_state):
                    counts_dict = {
                        binary(i, self.n_qubits): counts[batch_idx, i_idx, i] for i in range(dim)
                    }
                    bit_array = BitArray.from_counts(counts_dict, num_bits=self.n_qubits)
                    survival_probability[i_idx] = (
                        bit_array.get_int_counts().get(0, 0) / self.n_shots
                    )
                reward[batch_idx] = np.mean(survival_probability)

        return self._get_obs(), reward, True, False, self._get_info()

    def rl_qoc_training_qua_prog(self, num_updates: int = 1000) -> Program:
        """
        Generate a QUA program tailor-made for the RL-based calibration project
        """
        self.reset_parameters()

        qc = self.real_time_circuit
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) >= 2:
                raise ValueError("Overlapping classical registers are not supported")

        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)

            self.real_time_circuit_parameters.declare_variables(declare_streams=False)
            self.max_input_state.declare_variable()
            self.input_state_vars.declare_variables()
            self.policy.declare_variables()
            self.reward.declare_variable()
            if self.circuit_choice_var is not None:
                self.circuit_choice_var.declare_variable()
            if self.n_reps_var is not None:
                self.n_reps_var.declare_variable()
            if self.observable_vars is not None:
                self.observable_vars.declare_variables()
                observable_count = declare(int)
                self.max_observables.declare_variable()

            # Number of shots for each observable (or total number of shots if no observables)
            self.pauli_shots.declare_variable()
            n_updates = declare(int)
            input_state_count = declare(int)

            if self.backend.init_macro is not None:
                self.backend.init_macro()
            # Infinite loop to run the training
            with for_(
                n_updates,
                0,
                n_updates < num_updates,
                n_updates + 1,
            ):
                self.policy.load_input_values()  # Load µ and σ

                if self.circuit_choice_var is not None:  # Switch between circuit contexts
                    self.circuit_choice_var.load_input_value()

                if self.n_reps_var is not None:  # Variable number of repetitions
                    # Load number of repetitions of cycle circuit (can vary at each iteration)
                    self.n_reps_var.load_input_value()

                self.max_input_state.load_input_value()  # Load number of input states to prepare
                if self.observable_vars is not None:  # DFE reward only
                    self.max_observables.load_input_value()

                assign(input_state_count, 0)

                with while_(input_state_count < self.max_input_state.var):
                    assign(input_state_count, input_state_count + 1)
                    # Load info about input states to prepare (single qubit indices)
                    self.input_state_vars.load_input_values()

                    if self.config.dfe:
                        assign(observable_count, 0)
                        with while_(observable_count < self.max_observables.var):
                            assign(observable_count, observable_count + 1)
                            # Load info about observable to measure (single qubit indices)
                            self.observable_vars.load_input_values()
                            self.pauli_shots.load_input_value()
                            self._rl_qua_macro()
                    else:
                        self._rl_qua_macro()

            with stream_processing():
                self.reward.stream_processing()

        self.reset_parameters()
        return rl_qoc_training_prog

    def reset_parameters(self):
        self.policy.reset()
        self.reward.reset()
        self.real_time_circuit_parameters.reset()
        self.input_state_vars.reset()
        self.max_input_state.reset()
        if self.observable_vars is not None:
            self.observable_vars.reset()
            self.pauli_shots.reset()
            self.max_observables.reset()
        if self.n_reps_var is not None:
            self.n_reps_var.reset()
        if self.circuit_choice_var is not None:
            self.circuit_choice_var.reset()

    def _rl_qua_macro(self):
        b = declare(int)
        j = declare(int)
        lower_bound = declare(fixed, value=self.action_space.low)
        upper_bound = declare(fixed, value=self.action_space.high)
        μ = self.policy.get_variable("mu")
        σ = self.policy.get_variable("sigma")

        with for_(b, 0, b < self.batch_size // 2, b + 1):
            temp_action, temp_action2 = self._action_sampling(μ, σ)

            with for_(j, 0, j < 2, j + 1):
                for i, parameter in enumerate(self.real_time_circuit_parameters.parameters):
                    parameter.assign_value(
                        temp_action[i],
                        condition=(j == 0),
                        value_cond=temp_action2[i],
                    )
                    parameter.clip(lower_bound[i], upper_bound[i])

                counts, state_int = self._run_circuit()

                self.reward.assign_value(counts, is_qua_array=True)
                if self.input_type != InputType.INPUT_STREAM:
                    self.reward.save_to_stream()

                self.reward.stream_back()

    def _action_sampling(self, mu, sigma):
        """
        Sample actions from a multivariate Gaussian distribution using the Muller-Box method
        Args:
            µ: Mean of the Gaussian distribution (as a QUA array variable)
            σ: Standard deviation of the Gaussian distribution (as a QUA array variable)

        Returns:

        """
        # Declare variables for efficient Gaussian random sampling
        j = declare(int)
        n_lookup, cos_array, ln_array = get_gaussian_sampling_input()
        uniform_r = declare(fixed)
        u1, u2 = declare(int), declare(int)
        batch_r = Random(self.seed)
        temp_action = declare(fixed, size=self.n_actions)
        temp_action2 = declare(fixed, size=self.n_actions)

        # Sample actions from multivariate Gaussian distribution (Muller-Box method)
        with for_(j, 0, j < self.n_actions, j + 1):
            assign(uniform_r, batch_r.rand_fixed())
            assign(u1, Cast.unsafe_cast_int(uniform_r >> 19))
            assign(
                u2,
                Cast.unsafe_cast_int(uniform_r) & ((1 << 19) - 1),
            )
            assign(
                temp_action[j],
                mu[j] + sigma[j] * ln_array[u1] * cos_array[u2 & (n_lookup - 1)],
            )
            assign(
                temp_action2[j],
                mu[j] + sigma[j] * ln_array[u1] * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)],
            )

            return temp_action, temp_action2

    def _run_circuit(self):
        """
        Run the circuit and get the counts
        """
        qc = self.real_time_circuit
        n_shots = declare(int)
        state_int = declare(int, value=0)
        counts = declare(int, value=[0 for _ in range(2**self.n_qubits)])

        with for_(
            n_shots,
            0,
            n_shots < self.pauli_shots.var,
            n_shots + 1,
        ):
            param_inputs = [
                self.input_state_vars,
                self.real_time_circuit_parameters,
            ]
            if self.circuit_choice_var is not None:
                param_inputs.append(self.circuit_choice_var)
            if self.n_reps_var is not None:
                param_inputs.append(self.n_reps_var)

            if self.config.dfe:
                param_inputs.append(self.observable_vars)

            result = self.backend.quantum_circuit_to_qua(
                self.real_time_transpiled_circuit,
                param_inputs,
            )
            for c, clbit in enumerate(qc.clbits):
                bit = qc.find_bit(clbit)
                if len(bit.registers) == 0:
                    bit_output = result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{c}"]
                else:
                    creg, creg_index = bit.registers[0]
                    bit_output = result.result_program[creg.name][creg_index]
                assign(
                    state_int,
                    state_int + 1 << c * Cast.to_int(bit_output),
                )

            assign(counts[state_int], counts[state_int] + 1)

        return counts, state_int

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
            remove_final_measurement=False,
            scheduling=False,
        )

    def start_program(
        self,
        num_updates: int = 1000,
        compiler_options: Optional[CompilerOptionArguments] = None,
    ) -> RunningQmJob:
        """
        Start the QUA program

        Returns:
            RunningQmJob: The running Qmjob
        """
        if self.input_type == InputType.DGX:
            ParameterPool.configure_stream()
        self.backend.update_calibrations(self.real_time_circuit, self.input_type)
        prog = self.rl_qoc_training_qua_prog(num_updates=num_updates)
        qmm: QuantumMachinesManager = self.backend.machine.connect()
        config = self.backend.machine.generate_config()
        self._qm = qmm.open_qm(config)
        return self.qm.execute(prog, compiler_options=compiler_options)

    def close(self) -> None:
        """
        Close the environment (stop the running QUA program)
        Returns:

        """
        if self.input_type == InputType.DGX:
            ParameterPool.close_streams()
        finish = self.qm_job.halt()
        if not finish:
            print("Failed to halt the job")
        self._qm_job = None

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
        return self._qm
