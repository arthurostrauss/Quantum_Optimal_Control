from qm.jobs.running_qm_job import RunningQmJob

from ..environment import ContextAwareQuantumEnvironment
from ..environment.qconfig import QEnvConfig
from qiskit import QuantumCircuit
from qua_backend import QMBackend
from qua_utils import *


class QUAEnvironment(ContextAwareQuantumEnvironment):

    @property
    def backend(self) -> QMBackend:
        return self.backend

    def __init__(self, training_config: QEnvConfig, circuit_context: QuantumCircuit):
        super().__init__(training_config, circuit_context)
        # if not isinstance(self.config.backend_config, QuaConfig) or not isinstance(
        #     self.backend, QMBackend
        # ):
        #     raise ValueError("The backend should be a QMBackend object")
        self.parameter_tables = [
            ParameterTable(
                {
                    param.name: [0.0 for _ in range(self.n_actions)]
                    for param in parameters
                }
            )
            for parameters in self.parameters
        ]

        self.job = self.start_program(self.circuits[0])

    def rl_qoc_training_qua_prog(self):
        """
        Generate a QUA program tailor-made for the RL based calibration project
        """
        # TODO: Set IO2 and IO1 to be the number of input states and the number of observables respectively
        trunc_index = self._inside_trunc_tracker
        n_actions = self.n_actions
        real_time_parameters = self.parameter_tables[trunc_index]
        input_stream_template = (0, int, "input_stream")
        counter_template = (0, int)
        indices_template = ([0 for _ in range(self.n_qubits)], int, "input_stream")
        program_params = ParameterTable(
            {
                "max_input_state": input_stream_template,
                "max_observable": input_stream_template,
                "input_state_indices": indices_template,
                "observables_indices": indices_template,
                "pauli_shots": input_stream_template,
                "batchsize": counter_template,
                "n_shots": counter_template,
                "n_reps": counter_template,
                "input_state_count": counter_template,
                "observable_count": counter_template,
            }
        )

        qubits = [
            list(self.backend.machine.qubits.values())[i]
            for i in self.layout[trunc_index]
        ]
        dim = 2**self.n_qubits

        with program() as rl_qoc_training_prog:
            # Declare necessary variables (all are counters variables to loop over the corresponding hyperparameters)
            program_params.declare_variables(pause_program=False)
            for param_table in self.parameter_tables:
                param_table.declare_variables(pause_program=False)
            batchsize = program_params["batchsize"]
            n_shots = program_params["n_shots"]
            n_reps = program_params["n_reps"]  # Number of repetitions of the circuit
            input_state_count = program_params["input_state_count"]
            observable_count = program_params["observable_count"]
            max_input_state = program_params["max_input_state"]
            max_observable = program_params["max_observable"]
            input_state_indices = program_params["input_state_indices"]
            observables_indices = program_params["observables_indices"]
            pauli_shots = program_params["pauli_shots"]
            mean_actions = [
                declare_input_stream(
                    fixed, size=self.n_actions, name=f"mean_action_{j}"
                )
                for j in range(len(self.circuits))
            ]
            std_actions = [
                declare_input_stream(fixed, size=self.n_actions, name=f"std_action_{j}")
                for j in range(len(self.circuits))
            ]

            counts = declare(int, value=[0] * dim)
            state_int = declare(int, value=0)

            batch_r = Random()
            batch_r.set_seed(self.seed)
            action_stream = declare_stream()
            counts_st = [declare_stream() for _ in range(dim)]
            # Infinite loop to run the training

            with infinite_loop_():
                advance_input_stream(max_input_state)
                advance_input_stream(max_observable)
                for i in range(trunc_index + 1):
                    advance_input_stream(mean_actions[i])
                    advance_input_stream(std_actions[i])
                with for_(batchsize, 0, batchsize < self.batch_size, batchsize + 1):
                    # Sample actions from multivariate Gaussian distribution (Muller-Box method)
                    for i in range(trunc_index + 1):
                        param_table = self.parameter_tables[i]
                        z1, z2 = self.sample_actions(
                            mean_actions[i], std_actions[i], batch_r, action_stream
                        )
                        param_table.assign_parameters({self.parameters[i].name: z1})

                    with while_(input_state_count < max_input_state):
                        assign(input_state_count, input_state_count + 1)
                        # Load info about input states to prepare
                        advance_input_stream(input_state_indices)
                        with while_(observable_count < max_observable):
                            assign(observable_count, observable_count + 1)
                            advance_input_stream(observables_indices)
                            advance_input_stream(pauli_shots)
                            with for_(n_shots, 0, n_shots < pauli_shots, n_shots + 1):
                                # Prepare the input states
                                prepare_input_state_pauli6(input_state_indices, qubits)
                                # Run the circuit
                                with for_(n_reps, 0, n_reps < self.n_reps, n_reps + 1):
                                    if len(self.circuits) > 1:
                                        with switch_(trunc_index):
                                            for i in range(len(self.circuits)):
                                                with case_(i):
                                                    current_table = (
                                                        self.parameter_tables[0]
                                                    )
                                                    for table in self.parameter_tables[
                                                        1:i
                                                    ]:
                                                        current_table = (
                                                            current_table.add_table(
                                                                table
                                                            )
                                                        )
                                                    # TODO: Enable Result correctly with OQ3
                                                    result = self.backend.quantum_circuit_to_qua(
                                                        self.circuits[i], current_table
                                                    )
                                                    self.circuits[i].run()
                                    else:  # TODO: Enable Result correctly with OQ3
                                        self.circuits[0].run()

                                # Measure the observables
                                state_int = measure_observable(
                                    state_int, observables_indices, qubits
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
                action_stream.buffer((self.batch_size, self.n_actions)).save_all(
                    "actions"
                )
                for i in range(dim):
                    counts_st[i].buffer(self.batch_size).save(binary(i, self.n_qubits))

        return rl_qoc_training_prog

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
        prog = self.rl_qoc_training_qua_prog()
        config = self.backend.machine.generate_config()
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
    def parameter_table(self):
        return self.parameter_tables[self.trunc_index]
