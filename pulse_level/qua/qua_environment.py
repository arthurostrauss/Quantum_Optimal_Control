import numpy as np
from qiskit.circuit import ParameterVector
from qm.jobs.running_qm_job import RunningQmJob

from base_q_env import BaseQuantumEnvironment, BaseTarget
from qconfig import QEnvConfig, QuaConfig
from qiskit import QuantumCircuit, schedule as build_schedule
from qualang_tools.video_mode import ParameterTable
from qm.qua import *
from qua_backend import QMBackend, schedule_to_qua_instructions
from qua_utils import *


class QUAEnvironment(BaseQuantumEnvironment):
    def episode_length(self, global_step: int) -> int:
        pass

    def define_target_and_circuits(
        self,
    ) -> tuple[BaseTarget, List[QuantumCircuit], List[QuantumCircuit]]:
        pass

    def _get_obs(self):
        pass

    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        pass

    @property
    def tgt_instruction_counts(self) -> int:
        pass

    @property
    def parameters(self) -> List[ParameterVector] | ParameterVector:
        pass

    @property
    def trunc_index(self) -> int:
        pass

    @property
    def backend(self) -> QMBackend:
        return self.backend

    def __init__(self, training_config: QEnvConfig):
        super().__init__(training_config)
        if not isinstance(self.config.backend_config, QuaConfig) or not isinstance(
            self.backend, QMBackend
        ):
            raise ValueError("The backend should be a QMBackend object")

        self.parameter_table = [
            ParameterTable(
                {
                    self.parameters[i][j].name: 0.0
                    for j in range(len(self.parameters[i]))
                }
            )
            for i in range(self.circuits)
        ]
        self.progs = [self.rl_qoc_qua_prog(qc) for qc in self.circuits]
        self.job = self.start_program(self.circuits[0])

    def rl_qoc_qua_prog(self, qc: QuantumCircuit):
        """
        Generate a QUA program tailor-made for the RL based calibration project
        """
        # TODO: Set IO2 and IO1 to be the number of input states and the number of observables respectively
        trunc_index = self.trunc_index
        n_actions = self.action_space.shape[-1]
        real_time_parameters = self.parameter_table[trunc_index]
        sched_qc = build_schedule(qc, self.backend)
        qubits = [
            list(self.backend.quam.qubits.values())[i] for i in self.layout[trunc_index]
        ]
        dim = 2**self.n_qubits

        with program() as rl_qoc:
            # Declare necessary variables
            batchsize, n_shots, n_reps = declare(int), declare(int), declare(int)
            counts = declare(int, value=[0] * dim)
            state_int = declare(int, value=0)
            input_state_count, observable_count = declare(int, value=0), declare(
                int, value=0
            )
            max_input_state, max_observable = declare_input_stream(
                int, name="max_input_state"
            ), declare_input_stream(int, name="max_observable")
            input_state_indices = declare_input_stream(
                int, name="input_state_indices", size=self.n_qubits
            )
            observables_indices = declare_input_stream(
                int, name="observables_indices", size=self.n_qubits
            )
            pauli_shots = declare_input_stream(
                int, name="pauli_shots"
            )  # Number of shots for each observable
            mean_action = declare_input_stream(
                fixed, name="mean_action", size=n_actions
            )
            std_action = declare_input_stream(fixed, name="std_action", size=n_actions)
            # param_vars is a Python list of QUA fixed variables
            param_vars = (
                real_time_parameters.declare_variables()
            )  # TODO: Pause is currently happening here
            batch_r = Random()
            batch_r.set_seed(self.seed)
            action_stream = declare_stream()
            counts_st = [declare_stream() for _ in range(dim)]
            # Infinite loop to run the training

            with infinite_loop_():
                advance_input_stream(mean_action)
                advance_input_stream(std_action)
                advance_input_stream(max_input_state)
                advance_input_stream(max_observable)

                with for_(batchsize, 0, batchsize < self.batch_size, batchsize + 1):
                    self.sample_actions(mean_action, std_action, batch_r, action_stream)
                    with while_(input_state_count < max_input_state):
                        advance_input_stream(
                            input_state_indices
                        )  # Load info about the input states to prepare
                        with while_(observable_count < max_observable):
                            advance_input_stream(observables_indices)
                            advance_input_stream(pauli_shots)
                            with for_(n_shots, 0, n_shots < pauli_shots, n_shots + 1):
                                # Prepare the input states
                                prepare_input_state(input_state_indices, qubits)
                                # Run the circuit
                                with for_(n_reps, 0, n_reps < self.n_reps, n_reps + 1):
                                    schedule_to_qua_instructions(
                                        sched_qc, self.backend, real_time_parameters
                                    )

                                # Measure the observables
                                state_int = measure_observable(
                                    state_int, observables_indices, qubits
                                )
                                with switch_(state_int):
                                    for i in range(dim):  # Bitstring conversion
                                        with case_(i):
                                            assign(
                                                counts[i], counts[i] + 1
                                            )  # counts for 00, 01, 10 and 11
                                assign(state_int, 0)  # Resetting the state
                            for i in range(dim):  # Resetting Bitstring collection
                                save(counts[i], counts_st[i])
                                assign(counts[i], 0)

                            assign(observable_count, observable_count + 1)
                        assign(observable_count, 0)
                        assign(input_state_count, input_state_count + 1)
                    assign(input_state_count, 0)

            with stream_processing():
                action_stream.buffer(self.batch_size).save_all("actions")
                for i in range(dim):
                    counts_st[i].buffer(self.batch_size).save(binary(i, self.n_qubits))

        return rl_qoc

    def perform_action(self, actions: np.array):
        """
        Perform the actions on the quantum environment
        """
        trunc_index = self._inside_trunc_tracker
        qc = self.circuits[trunc_index].copy()
        self.backend.qm.set_io1_value()
        self.backend.qm.set_io2_value(self._index_input_state)

    def start_program(self, qc: QuantumCircuit) -> RunningQmJob:
        """
        Start the QUA program
        """
        prog = self.rl_qoc_qua_prog(qc)
        job = self.backend.qm.execute(prog)
        self.job = job
        return job

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

        """
        n_actions = len(self.parameter_table[self.trunc_index])
        z1, z2 = declare(fixed, size=n_actions), declare(fixed, size=n_actions)

        for i, parameter in enumerate(
            self.parameter_table[self.trunc_index].table.values()
        ):
            z1[i], z2[i] = rand_gauss_moller_box(
                z1[i], z2[i], mean_action[i], std_action[i], batch_r
            )
            assign(parameter.var, z1[i])
            save(parameter.var, action_stream)
            clip_qua_var(
                parameter.var, self.action_space.low[i], self.action_space.high[i]
            )
        return self.parameter_table[self.trunc_index].variables

    def close(self) -> None:
        """
        Close the environment (stop the running QUA program)
        Returns:

        """
        self.job.halt()
