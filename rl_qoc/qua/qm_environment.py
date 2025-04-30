from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import QuasiDistribution, sampled_expectation_value
from qiskit.primitives.containers.bit_array import BitArray
from qm import QuantumMachine, QuantumMachinesManager

from ..environment import (
    ContextAwareQuantumEnvironment,
    QEnvConfig,
)
from ..rewards.real_time import get_real_time_reward_circuit
from .qm_backend import QMBackend
from .qua_utils import *
from .qm_config import QMConfig, DGXConfig
from qm.qua import *
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Literal
from .parameter_table import (
    Parameter as QuaParameter,
    ParameterTable,
    Direction,
    InputType,
    DGXParameterPool,
)

ALL_MEASUREMENTS_TAG = "measurements"
"""The tag to save all measurements results to."""
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"


class QMEnvironment(ContextAwareQuantumEnvironment):

    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: Optional[QuantumCircuit] = None,
    ):

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
        self.policy = ParameterTable([mu, sigma])
        self.reward = QuaParameter(
            "reward",
            [0] * 2**self.n_qubits,
            input_type=self.input_type,
            direction=Direction.INCOMING,
        )

        self.real_time_circuit = get_real_time_reward_circuit(
            self.circuits,
            self.get_target(),
            self.config,
        )
        self.input_state_vars = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
            filter_function=lambda x: "input" in x.name,
        )
        self.observable_vars: Optional[ParameterTable] = (
            ParameterTable.from_qiskit(
                self.real_time_circuit,
                input_type=self.input_type,
                filter_function=lambda x: "observable" in x.name,
            )
            if self.config.dfe
            else None
        )

        self.n_reps_var: Optional[ParameterTable] = (
            ParameterTable.from_qiskit(
                self.real_time_circuit,
                input_type=self.input_type,
                filter_function=lambda x: "n_reps" in x.name,
            )
            if self.real_time_circuit.get_var("n_reps", None) is not None
            else None
        )

        self.real_time_circuit_parameters = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
            filter_function=lambda x: isinstance(x, Parameter),
        )
        self.circuit_choice_var: Optional[ParameterTable] = (
            ParameterTable(
                [
                    QuaParameter(
                        "circuit_choice",
                        0,
                        input_type=self.input_type,
                        direction=Direction.OUTGOING,
                    )
                ]
            )
            if len(self.circuits) > 1
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
            DGXParameterPool.patch_opnic_wrapper(self.qm_backend_config.opnic_dev_path)

        self._qm_job: Optional[RunningQmJob] = None
        self._qm: Optional[QuantumMachine] = None

    def rl_qoc_training_qua_prog(self):
        """
        Generate a QUA program tailor-made for the RL-based calibration project
        """

        qc = self.real_time_circuit
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) >= 2:
                raise ValueError("Overlapping classical registers are not supported")

        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)

            self.real_time_circuit_parameters.declare_variables(
                pause_program=False, declare_streams=False
            )
            self.max_input_state.declare_variable(pause_program=False)
            self.input_state_vars.declare_variables(pause_program=False)
            self.policy.declare_variables(pause_program=False)
            self.reward.declare_variable(pause_program=False)
            if isinstance(self.circuit_choice_var, ParameterTable):
                self.circuit_choice_var.declare_variables(pause_program=False)
            if self.n_reps_var is not None:
                self.n_reps_var.declare_variables(pause_program=False)
            if self.observable_vars is not None:
                self.observable_vars.declare_variables(pause_program=False)
                observable_count = declare(int)
                self.max_observables.declare_variable(pause_program=False)

            # Number of shots for each observable (or total number of shots if no observables)
            self.pauli_shots.declare_variable(pause_program=False)
            num_updates = declare(int)
            input_state_count = declare(int)

            # Infinite loop to run the training
            with for_(
                num_updates,
                0,
                num_updates < self.qm_backend_config.num_updates,
                num_updates + 1,
            ):
                self.policy.load_input_values()  # Load µ and σ

                if (
                    self.circuit_choice_var is not None
                ):  # Load circuit choice (switch over circuit contexts)
                    self.circuit_choice_var.load_input_values()

                if self.n_reps_var is not None:
                    # Load number of repetitions of cycle circuit (can vary at each iteration)
                    self.n_reps_var.load_input_values()

                self.max_input_state.load_input_value()  # Load number of input states to prepare
                if (
                    self.observable_vars is not None
                ):  # DFE: Load number of qubit-wise non-commuting observables
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
                self.reward.stream.buffer(self.batch_size, self.reward.length).save_all(
                    self.reward.name
                )

        return rl_qoc_training_prog

    def _rl_qua_macro(self):
        b = declare(int)
        j = declare(int)
        lower_bound = declare(fixed, value=self.action_space.low)
        upper_bound = declare(fixed, value=self.action_space.high)

        with for_(b, 0, b < self.batch_size // 2, b + 1):
            temp_action, temp_action2 = self._action_sampling()

            with for_(j, 0, j < 2, j + 1):
                for i, parameter in enumerate(
                    self.real_time_circuit_parameters.parameters
                ):
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

                self.reward.send_to_python()

    def _action_sampling(self):
        # Declare variables for efficient Gaussian random sampling
        j = declare(int)
        n_lookup, cos_array, ln_array = get_gaussian_sampling_input()
        uniform_r = declare(fixed)
        u1, u2 = declare(int), declare(int)
        batch_r = Random(self.seed)
        temp_action = declare(fixed, size=self.n_actions)
        temp_action2 = declare(fixed, size=self.n_actions)

        μ = self.policy.get_variable("mu")
        σ = self.policy.get_variable("sigma")

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
                μ[j] + σ[j] * ln_array[u1] * cos_array[u2 & (n_lookup - 1)],
            )
            assign(
                temp_action2[j],
                μ[j]
                + σ[j]
                * ln_array[u1]
                * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)],
            )

            return temp_action, temp_action2

    def _run_circuit(self):
        """
        Run the circuit and get the counts
        """
        qc = self.real_time_circuit
        n_shots = declare(int)
        state_int = declare(int, value=0)
        counts = declare(int, value=[0] * 2**self.n_qubits)

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

            compilation_result = self.backend.quantum_circuit_to_qua(
                self.real_time_circuit,
                param_inputs,
            )
            for c, clbit in enumerate(qc.clbits):
                bit = qc.find_bit(clbit)

                if len(bit.registers) == 0:
                    bit_output = compilation_result.result_program[
                        f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{c}"
                    ]
                else:
                    creg, creg_index = bit.registers[0]
                    bit_output = compilation_result.result_program[creg.name][
                        creg_index
                    ]
                assign(
                    state_int,
                    state_int + 2**c * Cast.to_int(bit_output),
                )

            assign(counts[state_int], counts[state_int] + 1)

        return counts, state_int

    def step(self, action: np.array):
        """
        Perform the action on the quantum environment
        """

        dim = 2**self.n_qubits
        if self._qm_job is None:
            self._qm_job = self.start_program()

        verbosity = (
            self.qm_backend_config.verbosity
            if isinstance(self.qm_backend_config, DGXConfig)
            else 2
        )
        push_args = {
            "job": self.qm_job,
            "qm": self.qm,
            "verbosity": verbosity,
        }
        mean_val = self.mean_action.tolist()
        std_val = self.std_action.tolist()

        # Push policy parameters to trigger real-time action sampling
        self.policy.push_to_opx({"mu": mean_val, "sigma": std_val}, **push_args)
        if self.circuit_choice_var is not None:
            self.circuit_choice_var.push_to_opx(
                {"circuit_choice": self.trunc_index},
                **push_args,
            )

        if self.n_reps_var is not None:
            self.n_reps_var.push_to_opx(
                {"n_reps": self.n_reps},
                **push_args,
            )

        additional_input = (
            self.config.execution_config.dfe_precision
            if self.config.dfe
            else self.baseline_circuit
        )
        reward_data = self.config.reward.get_reward_data(
            self.circuit,
            np.zeros((1, self.n_actions)),
            self.target,
            self.config,
            additional_input,
        )
        # Push the reward data to the OPX
        input_state_indices = reward_data.input_indices

        self.max_input_state.push_to_opx(len(input_state_indices), **push_args)
        if hasattr(reward_data, "observable_indices"):
            self.max_observables.push_to_opx(
                len(reward_data.observable_indices), **push_args
            )

        reward = []
        for i, input_state in enumerate(input_state_indices):
            self.input_state_vars.push_to_opx(
                {
                    f"input_state_{j}": input_state[j]
                    for j in range(len(self.input_state_vars))
                },
                **push_args,
            )
            if self.config.dfe:
                for j, observable in enumerate(reward_data.observable_indices):
                    self.observable_vars.push_to_opx(
                        {
                            f"observable_{k}": observable[k]
                            for k in range(len(self.observable_vars))
                        },
                        **push_args,
                    )
                    self.pauli_shots.push_to_opx(reward_data.shots[j], **push_args)

                    for b in range(self.batch_size):
                        # Collect the counts from the OPX
                        counts = self.reward.fetch_from_opx(**push_args)
                        counts_dict = {
                            binary(i, self.n_qubits): counts[i] for i in range(dim)
                        }
                        # Get the expectation value from the reward data
                        obs = reward_data[j].hamiltonian
                        # Create an appropriate diagonal observable
                        new_op = SparsePauliOp("I" * self.n_qubits, 0.0)
                        for obs_, coeff in zip((obs.paulis, obs.coeffs)):
                            diag_obs_label = ""
                            for char in obs_.to_label():
                                diag_obs_label += char if char == "I" else "Z"
                            new_op += SparsePauliOp(diag_obs_label, coeff)

                        bit_array = BitArray.from_counts(
                            counts_dict, num_bits=self.n_qubits
                        )
                        exp_value = bit_array.expectation_values(new_op)
                        reward.append(exp_value)

        # Reward: BitArray.from_counts()

    @property
    def qm_backend_config(self) -> QMConfig | DGXConfig:
        """
        Get the QM backend configuration
        """
        return self.config.backend_config

    @property
    def backend(self) -> QMBackend:
        return super().backend

    def start_program(self) -> RunningQmJob:
        """
        Start the QUA program

        Returns:
            RunningQmJob: The running Qmjob
        """
        if self.input_type == InputType.DGX:
            DGXParameterPool.configure_stream()
        self.backend.update_calibrations(self.real_time_circuit)
        prog = self.rl_qoc_training_qua_prog()
        qmm: QuantumMachinesManager = self.backend.machine.connect()
        qmm.close_all_quantum_machines()
        config = self.backend.machine.generate_config()
        self._qm = qmm.open_qm(config)
        job = self.qm.execute(prog)
        return job

    def close(self) -> None:
        """
        Close the environment (stop the running QUA program)
        Returns:

        """
        if self.input_type == InputType.DGX:
            DGXParameterPool.close_streams()
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
