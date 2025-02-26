from __future__ import annotations

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
from .parameter_table import Parameter as QuaParameter, ParameterTable


class QMEnvironment(ContextAwareQuantumEnvironment):

    @property
    def backend(self) -> QMBackend:
        return self.backend

    def __init__(self, training_config: QEnvConfig, circuit_context: QuantumCircuit):

        super().__init__(training_config, circuit_context)
        if not isinstance(self.config.backend_config, QMConfig) or not isinstance(
            self.backend, QMBackend
        ):
            raise ValueError(
                "The backend should be a QMBackend object and the config should be a QMConfig object"
            )

        self.µ = QuaParameter("µ", [0.0] * self.n_actions, input_type=self.input_type)
        self.σ = QuaParameter("σ", [1.0] * self.n_actions, input_type=self.input_type)
        self.policy = ParameterTable([self.µ, self.σ])
        self._qm_job: Optional[RunningQmJob] = None
        self.real_time_circuit = get_real_time_reward_circuit(
            self.circuits,
            self.get_target(),
            self.backend_info,
            self.config.execution_config,
            self.config.reward_method,
        )
        self.real_time_circuit_parameters = ParameterTable.from_quantum_circuit(
            self.real_time_circuit
        )

    def rl_qoc_training_qua_prog(self):
        """
        Generate a QUA program tailor-made for the RL based calibration project
        """
        # TODO: Set IO2 and IO1 to be the number of input states and the number of observables respectively
        max_input_state = QuaParameter("max_input_state", 0, input_type=self.input_type)
        max_observables = QuaParameter("max_observables", 0, input_type=self.input_type)
        pauli_shots = QuaParameter("pauli_shots", 0, input_type=self.input_type)
        observable_indices = ParameterTable(
            [
                QuaParameter(f"observable_{i}", 0, input_type=self.input_type)
                for i in range(self.n_qubits)
            ]
        )
        input_state_indices = ParameterTable(
            [
                QuaParameter(f"input_state_{i}", 0, input_type=self.input_type)
                for i in range(self.n_qubits)
            ]
        )
        program_params = ParameterTable(
            [
                max_input_state,
                max_observables,
                pauli_shots,
            ]
        )
        action = QuaParameter("action", [0.0] * self.n_actions)

        dim = 2**self.n_qubits

        with program() as rl_qoc_training_prog:
            # Declare necessary variables (all are counters variables to loop over the corresponding hyperparameters)
            program_params.declare_variables(pause_program=False, declare_streams=False)
            action.declare_variable(pause_program=False, declare_stream=False)
            self.real_time_circuit_parameters.declare_variables(
                pause_program=False, declare_streams=False
            )

            counts = declare(int, value=[0] * dim)
            b = declare(int)
            j = declare(int)
            n_shots = declare(int)
            state_int = declare(int, value=0)
            input_state_count = declare(int, value=0)
            observable_count = declare(int, value=0)

            batch_r = Random(self.seed)
            counts_st = [declare_stream() for _ in range(dim)]

            temp_action = declare(fixed, size=self.n_actions)
            temp_action2 = declare(fixed, size=self.n_actions)
            lower_bound = declare(fixed, value=self.action_space.low)
            upper_bound = declare(fixed, value=self.action_space.high)

            # Declare variables for efficient Gaussian random sampling
            n_lookup, cos_array, ln_array = get_gaussian_sampling_input()
            uniform_r = declare(fixed)
            u1, u2 = declare(int), declare(int)
            # Infinite loop to run the training
            with infinite_loop_():
                self.policy.load_input_values()
                max_input_state.load_input_value()
                max_observables.load_input_value()

                with for_(b, 0, b < self.batch_size // 2, b + 1):
                    # Sample actions from multivariate Gaussian distribution (Muller-Box method)
                    with for_(j, 0, j < self.n_actions, j + 1):
                        assign(uniform_r, batch_r.rand_fixed())
                        assign(u1, Cast.unsafe_cast_int(uniform_r >> 19))
                        assign(u2, Cast.unsafe_cast_int(uniform_r) & ((1 << 19) - 1))
                        assign(
                            temp_action[j],
                            self.μ.var[j]
                            + self.σ.var[j]
                            * ln_array[u1]
                            * cos_array[u2 & (n_lookup - 1)],
                        )
                        assign(
                            temp_action2[j],
                            self.μ.var[j]
                            + self.σ.var[j]
                            * ln_array[u1]
                            * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)],
                        )

                    for current_action in [temp_action, temp_action2]:
                        action.assign_value(current_action, is_qua_array=True)
                        action.clip(lower_bound, upper_bound, True)

                        with while_(input_state_count < max_input_state.var):
                            assign(input_state_count, input_state_count + 1)
                            # Load info about input states to prepare
                            input_state_indices.load_input_values()
                        with while_(observable_count < max_observables.var):
                            assign(observable_count, observable_count + 1)
                            observable_indices.load_input_values()
                            pauli_shots.load_input_value()

                            with for_(
                                n_shots,
                                0,
                                n_shots < pauli_shots.var * self.n_shots,
                                n_shots + 1,
                            ):
                                inputs = {}
                                compilation_result = (
                                    self.backend.quantum_circuit_to_qua(
                                        self.real_time_circuit
                                    )
                                )

                                with switch_(state_int):
                                    for i in range(dim):  # Bitstring  conversion
                                        with case_(i):
                                            assign(
                                                counts[i], counts[i] + 1
                                            )  # counts for 00, 01, 10 and 11
                                assign(state_int, 0)  # Resetting the state
                            for i in range(dim):  # Resetting Bitstring collection
                                save(counts[i], counts_st[i])
                                assign(counts[i], 0)

                        assign(observable_count, 0)

                    assign(input_state_count, 0)

            with stream_processing():
                for i in range(dim):
                    counts_st[i].buffer(self.batch_size).save(binary(i, self.n_qubits))

        return rl_qoc_training_prog

    def step(self, action: np.array):
        """
        Perform the action on the quantum environment
        """
        if self._qm_job is None:
            self.start_program()

        mean_val = self.mean_action.tolist()
        std_val = self.std_action.tolist()

        # Push policy parameters to trigger real-time action sampling
        for parameter, val, dummy_packet in zip(
            self.policy.parameters, [mean_val, std_val], [False, True]
        ):
            parameter.push_to_opx(
                val,
                job=self.qm_job,
                dgx_lib=self.qm_backend_config.dgx_lib,
                dgx_stream=self.qm_backend_config.dgx_stream,
                start_with_dummy_packet=dummy_packet,
            )

    @property
    def qm_backend_config(self) -> QMConfig | DGXConfig:
        """
        Get the QM backend configuration
        """
        return self.config.backend_config

    def start_program(self) -> RunningQmJob:
        """
        Start the QUA program
        """
        prog = self.rl_qoc_training_qua_prog()
        config = self.backend.machine.generate_config()
        qmm = self.backend.connect()
        qm = qmm.open_qm(config)
        self._qm_job = qm.execute(prog)

        return self.qm_job

    def sample_actions(self, mean_action, std_action, batch_r, action_stream):
        """
        (QUA macro)
        Sample actions from a Gaussian distribution and clip them to the action space
        Args:
            mean_action: QUA fixed variable representing the mean of the Gaussian distribution
            std_action: QUA fixed variable representing the standard deviation of the Gaussian distribution
            batch_r: QUA Random object
            action_stream: Stream object to save the actions (for returning them to agent)

        Returns:
            z1, z2: The sampled (clipped) actions

        """
        z1, z2 = declare(fixed, size=self.n_actions), declare(
            fixed, size=self.n_actions
        )

        for i in range(self.n_actions):
            z1[i], z2[i] = rand_gauss_moller_box(
                z1[i], z2[i], mean_action[i], std_action[i], batch_r
            )
            save(z1[i], action_stream)
            save(z2[i], action_stream)
            clip_qua_var(z1[i], self.action_space.low[i], self.action_space.high[i])
            clip_qua_var(z2[i], self.action_space.low[i], self.action_space.high[i])
            # assign(parameter.var, z1[i])
            # save(parameter.var, action_stream)
            # clip_qua_var(
            #     parameter.var, self.action_space.low[i], self.action_space.high[i]
            # )

        return z1, z2

    def close(self) -> None:
        """
        Close the environment (stop the running QUA program)
        Returns:

        """
        self.job.halt()

    @property
    def input_type(self) -> Literal["dgx", "input_stream", "IO1", "IO2"]:
        """
        Get the input type for streaming to OPX
        """
        return self.config.backend_config.input_type

    @property
    def qm_job(self) -> RunningQmJob:
        """
        Get the running QM job
        """
        return self._qm_job
