from __future__ import annotations

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
        self.observable_vars = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
            filter_function=lambda x: "observable" in x.name,
        )
        if self.real_time_circuit.get_var("n_reps", None) is not None:
            self.n_reps_var = ParameterTable.from_qiskit(
                self.real_time_circuit,
                input_type=self.input_type,
                filter_function=lambda x: "n_reps" in x.name,
            )
        else:
            self.n_reps_var = None

        self.real_time_circuit_parameters = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
            filter_function=lambda x: isinstance(x, Parameter),
        )
        if len(self.circuits) > 1:
            self.circuit_choice_var = ParameterTable(
                [
                    QuaParameter(
                        "circuit_choice",
                        0,
                        input_type=self.input_type,
                        direction=Direction.OUTGOING,
                    )
                ]
            )
        else:
            self.circuit_choice_var = None

        if self.input_type == InputType.DGX:
            DGXParameterPool.patch_opnic_wrapper(self.qm_backend_config.opnic_dev_path)

        self._qm_job: Optional[RunningQmJob] = None
        self._qm: Optional[QuantumMachine] = None

    def rl_qoc_training_qua_prog(self):
        """
        Generate a QUA program tailor-made for the RL-based calibration project
        """
        max_input_state = QuaParameter(
            "max_input_state",
            0,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        max_observables = QuaParameter(
            "max_observables",
            0,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        pauli_shots = QuaParameter(
            "pauli_shots", 0, input_type=self.input_type, direction=Direction.OUTGOING
        )

        program_params = ParameterTable(
            [
                max_input_state,
                max_observables,
                pauli_shots,
            ]
        )

        dim = 2**self.n_qubits
        qc = self.real_time_circuit
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) >= 2:
                raise ValueError("Overlapping classical registers are not supported")

        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)
            program_params.declare_variables(pause_program=False, declare_streams=False)
            self.real_time_circuit_parameters.declare_variables(
                pause_program=False, declare_streams=False
            )
            self.input_state_vars.declare_variables(pause_program=False)
            self.observable_vars.declare_variables(pause_program=False)
            self.policy.declare_variables(pause_program=False)
            self.reward.declare_variable(pause_program=False)
            if isinstance(self.circuit_choice_var, ParameterTable):
                self.circuit_choice_var.declare_variables(pause_program=False)
            if self.n_reps_var is not None:
                self.n_reps_var.declare_variables(pause_program=False)
            μ = self.policy.get_variable("mu")
            σ = self.policy.get_variable("sigma")
            counts = declare(int, value=[0] * dim)
            b = declare(int)
            j = declare(int)
            n_shots = declare(int)
            num_updates = declare(int)
            state_int = declare(int, value=0)
            input_state_count = declare(int)
            observable_count = declare(int)

            batch_r = Random(self.seed)
            temp_action = declare(fixed, size=self.n_actions)
            temp_action2 = declare(fixed, size=self.n_actions)
            lower_bound = declare(fixed, value=self.action_space.low)
            upper_bound = declare(fixed, value=self.action_space.high)

            # Declare variables for efficient Gaussian random sampling
            n_lookup, cos_array, ln_array = get_gaussian_sampling_input()
            uniform_r = declare(fixed)
            u1, u2 = declare(int), declare(int)
            # Infinite loop to run the training
            with for_(num_updates, 0, num_updates < self.qm_backend_config.num_updates):
                self.policy.load_input_values()
                max_input_state.load_input_value()
                max_observables.load_input_value()

                assign(input_state_count, 0)
                assign(observable_count, 0)

                if isinstance(self.circuit_choice_var, ParameterTable):
                    self.circuit_choice_var.load_input_values()

                if isinstance(self.n_reps_var, ParameterTable):
                    self.n_reps_var.load_input_values()

                with while_(input_state_count < max_input_state.var):
                    assign(input_state_count, input_state_count + 1)
                    # Load info about input states to prepare
                    self.input_state_vars.load_input_values()
                    with while_(observable_count < max_observables.var):
                        assign(observable_count, observable_count + 1)
                        self.observable_vars.load_input_values()
                        pauli_shots.load_input_value()

                        with for_(b, 0, b < self.batch_size // 2, b + 1):
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
                                    μ[j]
                                    + σ[j]
                                    * ln_array[u1]
                                    * cos_array[u2 & (n_lookup - 1)],
                                )
                                assign(
                                    temp_action2[j],
                                    μ[j]
                                    + σ[j]
                                    * ln_array[u1]
                                    * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)],
                                )

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

                                with for_(
                                    n_shots,
                                    0,
                                    n_shots < pauli_shots.var,
                                    n_shots + 1,
                                ):
                                    param_inputs = [
                                        self.input_state_vars,
                                        self.observable_vars,
                                        self.real_time_circuit_parameters,
                                    ]
                                    if isinstance(
                                        self.circuit_choice_var, ParameterTable
                                    ):
                                        param_inputs.append(self.circuit_choice_var)
                                    if isinstance(self.n_reps_var, ParameterTable):
                                        param_inputs.append(self.n_reps_var)

                                    compilation_result = (
                                        self.backend.quantum_circuit_to_qua(
                                            self.real_time_circuit,
                                            param_inputs,
                                        )
                                    )
                                    for c, clbit in enumerate(qc.clbits):
                                        bit = qc.find_bit(clbit)

                                        if len(bit.registers) == 0:
                                            bit_output = (
                                                compilation_result.result_program[
                                                    f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{i}"
                                                ]
                                            )
                                        else:
                                            creg, creg_index = bit.registers[0]
                                            bit_output = (
                                                compilation_result.result_program[
                                                    creg.name
                                                ][creg_index]
                                            )
                                        assign(
                                            state_int,
                                            state_int + 2**c * Cast.to_int(bit_output),
                                        )

                                    assign(counts[state_int], counts[state_int] + 1)

                                self.reward.assign_value(counts, is_qua_array=True)
                                if self.input_type != InputType.INPUT_STREAM:
                                    self.reward.save_to_stream()

                                self.reward.send_to_python()

            with stream_processing():
                self.reward.stream.buffer(self.batch_size, self.reward.length).save_all(
                    self.reward.name
                )

        return rl_qoc_training_prog

    def step(self, action: np.array):
        """
        Perform the action on the quantum environment
        """

        if self._qm_job is None:
            self._qm_job = self.start_program()

        mean_val = self.mean_action.tolist()
        std_val = self.std_action.tolist()

        # Push policy parameters to trigger real-time action sampling
        self.policy.push_to_opx(
            {"mu": mean_val, "sigma": std_val},
            self.qm_job,
            self.qm,
            (
                self.qm_backend_config.verbosity
                if isinstance(self.qm_backend_config, DGXConfig)
                else 2
            ),
        )

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
