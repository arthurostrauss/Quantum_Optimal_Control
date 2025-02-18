"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 03/11/2024
"""

from __future__ import annotations

import time

# For compatibility for options formatting between Estimators.
import json
import signal
from typing import Callable, Any

from gymnasium import Env
from gymnasium.core import ObsType
from qiskit import transpile

# Qiskit imports
from qiskit.circuit import (
    ParameterVector,
    Parameter,
)

# Qiskit Estimator Primitives: for computing Pauli expectation value sampling easily
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
)

from qiskit.quantum_info import partial_trace

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info.states import DensityMatrix, Statevector

from qiskit.transpiler import (
    Layout,
    PassManager,
)
from qiskit.providers import BackendV2
from qiskit_aer.noise import NoiseModel

# Qiskit dynamics for pulse simulation (& benchmarking)
from qiskit_dynamics import DynamicsBackend

# Qiskit Experiments for generating reliable baseline for complex gate calibrations / state preparations
from qiskit_ibm_runtime import (
    EstimatorV2 as RuntimeEstimatorV2,
)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from .backend_info import QiskitBackendInfo
from ..custom_jax_sim import PulseEstimatorV2, simulate_pulse_level
from ..helpers.helper_functions import (
    retrieve_primitives,
    handle_session,
    get_hardware_runtime_single_circuit,
    has_noise_model,
)
from ..helpers.circuit_utils import substitute_target_gate, retrieve_neighbor_qubits
from ..helpers.pulse_utils import (
    handle_virtual_rotations,
    projected_state,
    qubit_projection,
    rotate_frame,
)
from .reward_methods import *
from .configuration.qconfig import (
    QEnvConfig,
)


class BaseQuantumEnvironment(ABC, Env):

    def __init__(self, training_config: QEnvConfig):
        """
        Initialize the quantum environment
        Args:
            training_config: QEnvConfig object containing the training configuration
        """
        self._training_config = training_config
        self.parametrized_circuit_func: Callable = training_config.parametrized_circuit
        self._func_args = training_config.parametrized_circuit_kwargs
        if isinstance(self.backend, BackendV2) or self.backend is None:
            self._backend_info = QiskitBackendInfo(
                self.backend,
                training_config.backend_config.instruction_durations,
                pass_manager=training_config.backend_config.pass_manager,
                skip_transpilation=training_config.backend_config.skip_transpilation,
            )
        elif training_config.backend_config.config_type == "qibo":
            from ..qibo.qibo_config import QiboBackendInfo

            self._backend_info = QiboBackendInfo(
                training_config.backend_config.n_qubits,
                training_config.backend_config.coupling_map,
            )
        else:
            raise ValueError("Backend should be a BackendV2 object or a string")
        self._physical_target_qubits = training_config.target.get(
            "physical_qubits", None
        )

        self._estimator, self._sampler = retrieve_primitives(
            self.backend,
            self.config.backend_config,
            training_config.backend_config.as_dict().get("primitive_options", None),
        )

        self._target, self.circuits, self.baseline_circuits = [], [], []

        self._mean_action = np.zeros(self.action_space.shape[-1])
        self._std_action = np.ones(self.action_space.shape[-1])
        # Data storage
        self._optimal_action = np.zeros(self.action_space.shape[-1])
        self._session_counts = 0
        self._step_tracker = 0
        self._inside_trunc_tracker = 0
        self._total_shots = []
        self._hardware_runtime = []
        self._max_return = 0
        self._episode_ended = False
        self._episode_tracker = 0
        self.action_history = []
        self.reward_history = []
        self._pubs, self._ideal_pubs = [], []
        self._calibration_pubs: List[
            CalibrationEstimatorPub | CalibrationSamplerPub
        ] = []
        self._observables, self._pauli_shots = None, None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.process_fidelity_history = []
        self.avg_fidelity_history = []
        self.circuit_fidelity_history = []
        self.circuit_fidelity_history_nreps = []
        self.avg_fidelity_history_nreps = []
        self._fit_function: Optional[Callable] = None
        self._fit_params: Optional[np.array] = None

        # Call reset of Env class to set seed
        self._seed = training_config.seed
        super().reset(seed=self._seed)
        self._n_reps_rng = np.random.default_rng(self.np_random.integers(2**32))

    @abstractmethod
    def define_target_and_circuits(
        self,
    ) -> tuple[
        GateTarget | StateTarget | List[GateTarget],
        List[QuantumCircuit],
        List[QuantumCircuit],
    ]:
        """
        Define the target gate or state and the circuits to be executed on the quantum system.
        This method should be implemented by the user and called at construction of the environment.
        It can typically be called after the initialization of this base class.
        """
        raise NotImplementedError("Define target method not implemented")

    @abstractmethod
    def episode_length(self, global_step: int) -> int:
        """
        Args:
            global_step: Step in the training loop

        Returns: Episode length

        """
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Benchmark through tomography or through simulation the policy
        Args:
            qc: Quantum circuit to benchmark
            params:

        Returns: Fidelity metric or array of fidelities for all actions in the batch

        """

    def initial_reward_fit(
        self,
        params: np.array,
        execution_config: Optional[ExecutionConfig] = None,
        fit_function: Optional[Callable] = None,
        inverse_fit_function: Optional[Callable] = None,
        update_fit_params: bool = True,
        reward_method: Literal[
            "cafe", "channel", "orbit", "state", "xeb", "fidelity"
        ] = "cafe",
    ) -> plt.Figure:
        """
        Method to fit the initial reward function to the first set of actions in the environment
        with respect to the number of repetitions of the cycle circuit
        """

        initial_execution_config = self.config.execution_config
        initial_reward_method = self.config.reward_method
        if execution_config is not None:
            self.config.execution_config = execution_config
        self.config.reward_method = reward_method
        reward_data = []
        for i in range(len(self.config.execution_config.n_reps)):
            self.config.execution_config.n_reps_index = i
            reward = self.perform_action(params, update_env_history=False)
            reward_data.append(np.mean(reward))
        if fit_function is None or inverse_fit_function is None:

            def fit_function(n, spam, eps_lin, eps_quad):
                return 1 - spam - eps_lin * n - eps_quad * n**2

            def inverse_fit_function(reward, n, spam, eps_lin, eps_quad):
                return reward + eps_lin * (n - 1) + eps_quad * (n**2 - 1)

        p0 = [0.0, 0.0, 0.0]  # Initial guess for the parameters
        lower_bounds = [0.0, 0.0, 0.0]
        upper_bounds = [1.0, np.inf, np.inf]

        popt, pcov = curve_fit(
            fit_function,
            self.config.execution_config.n_reps,
            reward_data,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
        )

        # Create a figure and return it to the user
        fig, ax = plt.subplots()
        ax.plot(
            self.config.execution_config.n_reps, reward_data, label="Data", marker="o"
        )
        ax.plot(
            self.config.execution_config.n_reps,
            [fit_function(n, *popt) for n in self.config.execution_config.n_reps],
            label="Fit",
        )
        ax.set_xlabel("Number of repetitions")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.set_title("Initial reward fit (for varying number of repetitions)")
        # Print found parameters
        print("Found parameters:", popt)

        self.config.execution_config = initial_execution_config
        self.config.reward_method = initial_reward_method
        if update_fit_params:
            self._fit_function = lambda reward, n: inverse_fit_function(
                reward, n, *popt
            )
            self._fit_params = popt
        return fig

    def perform_action(self, actions: np.array, update_env_history: bool = True):
        """
        Send the action batch to the quantum system and retrieve reward
        :param actions: action vectors to execute on quantum system
        :param update_env_history: Boolean to update the environment history
        :return: Reward table (reward for each action in the batch)
        """
        qc = self.circuits[self.trunc_index].copy()
        params, batch_size = np.array(actions), actions.shape[0]
        if len(params.shape) == 1:
            params = np.expand_dims(params, axis=0)
        # if batch_size != self.batch_size:
        #     raise ValueError(f"Batch size mismatch: {batch_size} != {self.batch_size} ")

        # Get the reward method from the configuration
        reward_method = self.config.reward_method
        if self.do_benchmark():  # Benchmarking or fidelity access
            fids = self.compute_benchmarks(qc, params)
            if reward_method == "fidelity":
                if update_env_history:
                    self._total_shots.append(0)
                    self._hardware_runtime.append(0.0)
                return fids

        # Check if the reward method exists in the dictionary
        last_input = (
            self.config.execution_config.dfe_precision
            if self.config.dfe
            else self.baseline_circuits[self.trunc_index]
        )
        self._pubs = self.config.reward_config.get_reward_pubs(
            qc,
            params,
            self.target,
            self.backend_info,
            self.config.execution_config,
            last_input,
        )
        total_shots = self.config.reward_config.total_shots
        if update_env_history:
            self._total_shots.append(total_shots)
            if self.backend_info.instruction_durations is not None:
                self._hardware_runtime.append(
                    get_hardware_runtime_single_circuit(
                        qc,
                        self.backend_info.instruction_durations.duration_by_name_qubits,
                    )
                    * total_shots
                )
                print(
                    "Hardware runtime taken:",
                    np.round(sum(self.hardware_runtime) / 3600, 4),
                    "hours ",
                    np.round(sum(self.hardware_runtime) / 60, 4),
                    "min ",
                    np.round(sum(self.hardware_runtime) % 60, 4),
                    "seconds",
                )

        counts = (
            self._session_counts
            if isinstance(self.estimator, RuntimeEstimatorV2)
            else self.trunc_index
        )
        self.estimator = handle_session(self.estimator, self.backend, counts)
        primitive_type = (
            "Estimator" if isinstance(self.primitive, BaseEstimatorV2) else "Sampler"
        )
        print(f"Sending {primitive_type} job...")
        start = time.time()

        job = self.primitive.run(pubs=self._pubs)
        pub_results = job.result()
        print("Time for running", time.time() - start, "seconds")

        if self.config.dfe:
            reward_table = np.sum(
                [pub_result.data.evs for pub_result in pub_results], axis=0
            ) / len(self.config.reward_config.observables)
            if self.config.reward_method == "channel":
                dim = 2**self.target.causal_cone_size
                reward_table = (dim * reward_table + 1) / (dim + 1)

        else:
            if self.config.reward_method == "xeb":
                # TODO: Implement XEB reward computation using Sampler
                raise NotImplementedError("XEB reward computation not implemented yet")
            else:
                pub_data = [
                    [
                        pub_result.data.meas[i].postselect(
                            self.target.causal_cone_qubits_indices,
                            [0] * self.target.causal_cone_size,
                        )
                        for i in range(batch_size)
                    ]
                    for pub_result in pub_results
                ]
                survival_probability = [
                    [bit_array.num_shots / self.n_shots for bit_array in bit_arrays]
                    for bit_arrays in pub_data
                ]

                reward_table = np.mean(survival_probability, axis=0)

        print(f"Finished {primitive_type} job")
        print("Reward (avg):", np.mean(reward_table), "Std:", np.std(reward_table))

        return reward_table  # Shape [batch size]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment to its initial state
        :param seed: Seed for random number generator
        :param options: Options for the reset
        :return: Initial observation
        """
        super().reset(seed=seed)
        self._episode_tracker += 1
        self._episode_ended = False
        self.modify_environment_params()
        self.config.execution_config.n_reps_index = self._n_reps_rng.integers(
            0, len(self.config.n_reps)
        )

        if isinstance(self.estimator, RuntimeEstimatorV2):
            self.estimator.options.environment.job_tags = [
                f"rl_qoc_step{self._step_tracker}"
            ]

        return self._get_obs(), self._get_info()

    def simulate_circuit(
        self, qc: QuantumCircuit, params: np.array, update_env_history: bool = True
    ) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        This method should be called only when the abstraction level is "circuit"
        :param qc: QuantumCircuit to execute on quantum system
        :param params: List of Action vectors to execute on quantum system
        :param update_env_history: Boolean to update the environment history
        :return: Fidelity metric or array of fidelities for all actions in the batch
        """

        if self.abstraction_level != "circuit":
            raise ValueError(
                "This method should only be called when the abstraction level is 'circuit'"
            )

        qc_channel = qc.copy(name="qc_channel")
        qc_state = qc.copy(name="qc_state")
        qc_channel_nreps = qc.repeat(self.n_reps).copy(name="qc_channel_nreps")
        qc_state_nreps = qc.repeat(self.n_reps).copy(name="qc_state_nreps")

        returned_fidelity_type = (
            "gate"
            if isinstance(self.target, GateTarget) and qc.num_qubits <= 3
            else "state"
        )
        returned_fidelities = []
        backend = AerSimulator()
        if self.backend is None or (
            isinstance(self.backend, AerSimulator) and not has_noise_model(self.backend)
        ):  # Ideal simulation

            noise_model = None
            qc_channel.save_unitary()
            qc_channel_nreps.save_unitary()
            qc_state.save_statevector()
            qc_state_nreps.save_statevector()
            channel_output = "unitary"
            state_output = "statevector"

        else:  # Noisy simulation
            if isinstance(self.backend, AerSimulator):
                noise_model = self.backend.options.noise_model
            else:
                noise_model = NoiseModel.from_backend(self.backend)
            qc_channel.save_superop()
            qc_channel_nreps.save_superop()
            qc_state.save_density_matrix()
            qc_state_nreps.save_density_matrix()
            channel_output = "superop"
            state_output = "density_matrix"

        basis_gates = backend.operation_names
        if noise_model is not None:
            basis_gates += noise_model.basis_gates
        qc_channel, qc_channel_nreps, qc_state, qc_state_nreps = transpile(
            [qc_channel, qc_channel_nreps, qc_state, qc_state_nreps],
            backend=backend,
            optimization_level=0,
            basis_gates=basis_gates,
        )
        if isinstance(self.parameters, ParameterVector):
            parameters = [self.parameters]
            n_custom_instructions = 1
        else:  # List of ParameterVectors
            parameters = self.parameters
            n_custom_instructions = self.trunc_index + 1

        parameter_binds = [
            {
                parameters[i][j]: params[:, i * self.n_actions + j]
                for i in range(n_custom_instructions)
                for j in range(self.n_actions)
            }
        ]
        data_length = len(params)

        if isinstance(self.target, GateTarget):
            circuits = [qc_channel, qc_channel_nreps, qc_state, qc_state_nreps]
            methods = [channel_output] * 2 + [state_output] * 2
            fid_arrays = [
                self.avg_fidelity_history,
                self.avg_fidelity_history_nreps,
                self.circuit_fidelity_history,
                self.circuit_fidelity_history_nreps,
            ]
        else:
            circuits = [qc_state, qc_state_nreps]
            fid_arrays = [
                self.circuit_fidelity_history,
                self.circuit_fidelity_history_nreps,
            ]
            methods = [state_output] * 2
        for circ, method, fid_array in zip(circuits, methods, fid_arrays):
            # Avoid channel simulation for more than 3 qubits
            if (method == "superop" or method == "unitary") and circ.num_qubits > 3:
                fidelities = [0.0] * data_length
                n_reps = 1
            else:
                result = backend.run(
                    circ,
                    parameter_binds=parameter_binds,
                    method=method,
                    noise_model=noise_model,
                ).result()
                outputs = [result.data(i)[method] for i in range(data_length)]
                n_reps = self.n_reps if "nreps" in circ.name else 1
                fidelities = [
                    self.target.fidelity(output, n_reps) for output in outputs
                ]
            if (
                (method == "superop" or method == "unitary")
                and returned_fidelity_type == "gate"
                and n_reps == 1
            ):
                returned_fidelities = fidelities
            elif (
                (method == "density_matrix" or method == "statevector")
                and returned_fidelity_type == "state"
                and n_reps == 1
            ):
                returned_fidelities = fidelities
            if update_env_history:
                fid_array.append(np.mean(fidelities))

        return returned_fidelities

    def _observable_to_observation(self):
        """
        Convert the observable to an observation to be given to the agent
        """
        if self.config.reward_method == "state":
            # n_qubits = self.observables.num_qubits
            # d = 2**n_qubits
            # pauli_to_index = {pauli: i for i, pauli in enumerate(pauli_basis(n_qubits))}
            # array_obs = np.zeros(d**2)
            # for pauli in self.observables:
            #     array_obs[pauli_to_index[pauli.paulis[0]]] = pauli.coeffs[0]

            array_obs = []
            return array_obs
        else:
            raise NotImplementedError("Channel estimator not yet implemented")

    def simulate_pulse_circuit(
        self,
        qc: QuantumCircuit,
        params: Optional[np.array] = None,
        update_env_history: bool = True,
    ) -> List[float]:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        This method should be called only when the abstraction level is "pulse"

        :param qc: QuantumCircuit to execute on quantum system
        :param params: List of Action vectors to execute on quantum system
        :param update_env_history: Boolean to update the environment history
        """
        if self.abstraction_level != "pulse":
            raise ValueError(
                "This method should only be called when the abstraction level is 'pulse'"
            )
        if not isinstance(self.backend, DynamicsBackend):
            raise ValueError(
                f"Pulse simulation requires a DynamicsBackend; got {self.backend}"
            )
        returned_fidelity_type = (
            "gate"
            if isinstance(self.target, GateTarget) and qc.num_qubits <= 3
            else "state"
        )  # Fidelity type to return (gate or state fidelity metric)
        returned_fidelities = []
        subsystem_dims = list(
            filter(lambda x: x > 1, self.backend.options.subsystem_dims)
        )  # Filter out qubits with dimension 1 (trivial dimension stated for DynamicsBackend)
        n_benchmarks = 1  # Number of benchmarks to run (1 if no n_reps, 2 if n_reps > 1, to benchmark both qc and qc_nreps)
        qc_nreps = None
        if self.n_reps > 1 and isinstance(
            self.target, GateTarget
        ):  # No need to benchmark n_reps for state targets
            qc_nreps = qc.copy("qc_nreps")
            for _ in range(self.n_reps - 1):
                qc_nreps.compose(qc, inplace=True)
            n_benchmarks *= 2

        y0_gate = Operator(
            np.eye(np.prod(subsystem_dims)),
            input_dims=tuple(subsystem_dims),
            output_dims=tuple(subsystem_dims),
        )
        y0_state = Statevector.from_int(0, dims=subsystem_dims)
        if params is None or isinstance(self.estimator, PulseEstimatorV2):
            circuits = [qc]
            circuits_n_reps = [qc_nreps] if qc_nreps is not None else []
            data_length = 1 if params is None else len(params)
        else:
            if not isinstance(params, np.ndarray):
                params = np.array(params)
            if len(params.shape) == 1:
                params = np.expand_dims(params, axis=0)
            circuits = [qc.assign_parameters(p) for p in params]
            circuits_n_reps = (
                [qc_nreps.assign_parameters(p) for p in params]
                if qc_nreps is not None
                else []
            )
            data_length = len(params)
        circuits_list = circuits + circuits_n_reps

        if isinstance(self.estimator, PulseEstimatorV2):
            # TODO: Handle this case
            sampler_pubs = [(circ, params) for circ in circuits_list]
            y0_list = [y0_state]
            if qc.num_qubits < 3 and isinstance(
                self.target, GateTarget
            ):  # Benchmark channel only for 1-2 qubits
                y0_list += [y0_gate]
                for circ in circuits_list:
                    sampler_pubs.append((circ, params))
                n_benchmarks *= 2

            output_data = []
            for y0, pub in zip(y0_list, sampler_pubs):
                results = simulate_pulse_level(pub, self.backend, y0)
                for result in results:
                    yf = result.y[-1]
                    tf = result.t[-1]
                    rotate_frame(yf, tf, self.backend)
                    output_data.append(yf)

            # Reshape data to isolate benchmarks (Output type can be either State or Channel, and for both qc and qc_nreps)

            output_data = [
                output_data[i * data_length : (i + 1) * data_length]
                for i in range(n_benchmarks)
            ]
            # Reorder data to match the order of the circuits
            qc_data_mapping = {"qc_state": output_data[0], "qc_channel": output_data[1]}
            if qc_nreps is not None:
                qc_data_mapping["qc_state_nreps"] = output_data[2]
                qc_data_mapping["qc_channel_nreps"] = output_data[3]

            circuit_order = [
                "qc_state",
                "qc_state_nreps",
                "qc_channel",
                "qc_channel_nreps",
            ]
            new_output_data = [
                qc_data_mapping.get(name, None)
                for name in circuit_order
                if name in qc_data_mapping
            ]
            output_data = new_output_data

        else:  # Standard Dynamics simulation

            y0_list = (
                [y0_state] * n_benchmarks * data_length
            )  # Initial state for each benchmark

            if qc.num_qubits < 3 and isinstance(
                self.target, GateTarget
            ):  # Benchmark channel only for 1-2 qubits
                y0_list += [y0_gate] * n_benchmarks * data_length
                circuits_list += circuits + circuits_n_reps
                n_benchmarks *= (
                    2  # Double the number of benchmarks to include channel fidelity
                )
            # Simulate all circuits
            output_data = []
            results = self.backend.solve(circuits_list, y0=y0_list)
            for solver_result in results:
                yf = solver_result.y[-1]
                tf = solver_result.t[-1]
                yf = rotate_frame(yf, tf, self.backend)

                output_data.append(yf)

                # Reshape data to isolate benchmarks (Output type can be either State or Channel, and for both qc and qc_nreps)
            output_data = [
                output_data[i * data_length : (i + 1) * data_length]
                for i in range(n_benchmarks)
            ]

        if self.n_reps > 1:  # Benchmark both qc and qc_nreps
            circ_list = [qc, qc_nreps, qc, qc_nreps]
            fid_arrays = [
                self.circuit_fidelity_history,
                self.avg_fidelity_history,
                self.circuit_fidelity_history_nreps,
                self.avg_fidelity_history_nreps,
            ]
        else:  # Benchmark only qc
            circ_list = [qc] * 2
            fid_arrays = [self.circuit_fidelity_history, self.avg_fidelity_history]

        for circ, data, fid_array in zip(circ_list, output_data, fid_arrays):
            n_reps = 1 if "nreps" not in circ.name else self.n_reps

            if isinstance(data[0], (Statevector, DensityMatrix)):
                data = [projected_state(state, subsystem_dims) for state in data]
                if self.target.n_qubits != len(
                    subsystem_dims
                ):  # If state has less qubits than the backend, trace out the rest
                    data = [
                        partial_trace(
                            state,
                            [
                                qubit
                                for qubit in range(state.num_qubits)
                                if qubit not in self.target.physical_qubits
                            ],
                        )
                        for state in data
                    ]
            elif isinstance(data[0], Operator):  # Project channel to qubit subspace
                data = [qubit_projection(op, subsystem_dims) for op in data]

            # Compute fidelities (type of input automatically detected and handled -> state -> state fidelity, channel -> gate fidelity)
            fidelities = [
                (
                    self.target.fidelity(output, n_reps, validate=False)
                    if n_reps > 1
                    else self.target.fidelity(output, validate=False)
                )
                for output in data
            ]

            if (
                returned_fidelity_type == "gate"
            ):  # Optimize gate fidelity by finding optimal Z-rotations before and after gate
                fidelities = handle_virtual_rotations(
                    data, fidelities, subsystem_dims, n_reps, self.target
                )
                if n_reps == 1:
                    returned_fidelities = fidelities
            elif returned_fidelity_type == "state" and n_reps == 1:
                returned_fidelities = fidelities
            if update_env_history:
                fid_array.append(np.mean(fidelities))

        return returned_fidelities

    def update_gate_calibration(self, gate_name: Optional[str] = None):
        """
        Update gate calibration parameters

        :param gate_name: Name of custom gate to add to target (if None,
         use target gate and update its attached calibration)
        """
        raise NotImplementedError(
            "Gate calibration not implemented for this environment"
        )

    def modify_environment_params(self, **kwargs):
        """
        Modify environment parameters (can be overridden by subclasses to modify specific parameters)
        """
        pass

    @property
    def config(self) -> QEnvConfig:
        return self._training_config

    @property
    def backend(self) -> Optional[BackendV2]:
        return self._training_config.backend

    @property
    def estimator(self) -> BaseEstimatorV2:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimatorV2):
        self._estimator = estimator

    @property
    def sampler(self) -> BaseSamplerV2:
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSamplerV2):
        self._sampler = sampler

    @property
    def primitive(self) -> BaseEstimatorV2 | BaseSamplerV2:
        """
        Return the primitive to use for the environment (estimator or sampler)
        """
        return self.estimator if self.config.reward_config.dfe else self.sampler

    @property
    def physical_target_qubits(self):
        return self._physical_target_qubits

    @property
    def physical_neighbor_qubits(self):
        return retrieve_neighbor_qubits(
            self.backend_info.coupling_map, self.physical_target_qubits
        )

    @property
    def physical_next_neighbor_qubits(self):
        return retrieve_neighbor_qubits(
            self.backend_info.coupling_map,
            self.physical_target_qubits + self.physical_neighbor_qubits,
        )

    @property
    @abstractmethod
    def tgt_instruction_counts(self) -> int:
        """
        Number of occurrences of the target instruction in the circuit
        """
        raise NotImplementedError("Target instruction counts not implemented")

    @property
    def fidelity_history(self):
        return (
            self.avg_fidelity_history
            if self.target.target_type == "gate"
            and self.target.target_circuit.num_qubits <= 3
            else self.circuit_fidelity_history
        )

    @property
    def step_tracker(self):
        return self._step_tracker

    @property
    def abstraction_level(self):
        """
        Return the abstraction level of the environment (can be 'circuit' or 'pulse')
        """
        return "pulse" if self.circuits[0].calibrations else "circuit"

    @step_tracker.setter
    def step_tracker(self, step: int):
        assert step >= 0, "step must be positive integer"
        self._step_tracker = step

    def signal_handler(self, signum, frame):
        """Signal handler for SIGTERM and SIGINT signals."""
        print(f"Received signal {signum}, closing environment...")
        self.close()

    def close(self) -> None:
        if hasattr(self.estimator, "session"):
            self.estimator.session.close()

    def clear_history(self):
        """
        Clear all stored data to start new training.
        """
        self._step_tracker = 0
        self._episode_tracker = 0
        self.action_history.clear()
        self.reward_history.clear()
        self._total_shots.clear()
        self._hardware_runtime.clear()
        self.avg_fidelity_history.clear()
        self.process_fidelity_history.clear()
        self.circuit_fidelity_history.clear()

    @property
    def benchmark_cycle(self) -> int:
        """
        Cycle at which fidelity benchmarking is performed
        :return:
        """
        return self.config.benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, step: int) -> None:
        """
        Set cycle at which fidelity benchmarking is performed
        :param step:
        :return:
        """
        assert step >= 0, "Cycle needs to be a positive integer"
        self.config.benchmark_cycle = step

    def do_benchmark(self) -> bool:
        """
        Check if benchmarking should be performed at current step
        :return:
        """
        if self.config.reward_method == "fidelity":
            return True
        elif self.benchmark_cycle == 0:
            return False
        else:
            return self._episode_tracker % self.benchmark_cycle == 0

    def _get_info(self) -> Any:
        step = self._episode_tracker
        if self._episode_ended:
            if self.do_benchmark():
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "circuit fidelity": self.fidelity_history[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "max circuit fidelity": np.max(self.fidelity_history),
                    "arg max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "arg max circuit fidelity": np.argmax(self.fidelity_history),
                    "optimal action": self.optimal_action,
                }
            else:
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "optimal action": self.optimal_action,
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "truncation_index": self.trunc_index,
            }
        return info

    def _ident_str(self):
        """This is a one-line description of the environment with some key parameters."""
        if isinstance(self.target, GateTarget):
            ident_str = f"gate_calibration_{self.target.gate.name}-gate_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        elif isinstance(self.target, StateTarget):
            ident_str = f"state_preparation_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        else:
            raise ValueError("Target type not recognized")
        return ident_str

    def __repr__(self):
        string = f"QuantumEnvironment composed of {self.n_qubits} qubits, \n"
        string += (
            f"Defined target: {self.target.target_type} "
            f"({self.target.gate if isinstance(self.target, GateTarget) else self.target.dm})\n"
        )
        string += f"Physical qubits: {self.target.physical_qubits}\n"
        string += f"Backend: {self.backend},\n"
        string += f"Abstraction level: {self.abstraction_level},\n"
        string += f"Run options: N_shots ({self.n_shots}), Sampling_Pauli_space ({self.sampling_paulis}), \n"
        string += f"Batch size: {self.batch_size}, \n"
        return string

    # Properties

    @property
    def seed(self):
        return self._seed

    @property
    def c_factor(self):
        return self.config.c_factor

    @property
    def action_space(self):
        return self.config.action_space

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        self.config.batch_size = size

    @property
    def n_reps(self) -> int:
        return self.config.execution_config.current_n_reps

    @property
    def n_shots(self) -> int:
        return self.config.n_shots

    @n_shots.setter
    def n_shots(self, n_shots: int):
        self.config.n_shots = n_shots

    @property
    def sampling_paulis(self) -> int:
        return self.config.sampling_paulis

    @sampling_paulis.setter
    def sampling_paulis(self, sampling_paulis: int):
        self.config.sampling_paulis = sampling_paulis

    @property
    def target(self) -> GateTarget | StateTarget:
        """
        Return the target object (GateTarget | StateTarget) of the environment
        """
        return self._target

    @property
    def n_qubits(self):
        return self.target.n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        assert (
            isinstance(n_qubits, int) and n_qubits > 0
        ), "n_qubits must be a positive integer"
        self.target.n_qubits = n_qubits

    @property
    def layout(self):
        return self.target.layout

    @layout.setter
    def layout(self, layout: Layout):
        if not isinstance(layout, Layout):
            raise ValueError("Layout should be a Qiskit Layout object")
        self.target.layout = layout

    @property
    @abstractmethod
    def parameters(
        self,
    ) -> List[ParameterVector | List[Parameter]] | ParameterVector | List[Parameter]:
        """
        Return the Qiskit Parameter(s) instance(s) defining the abstract actions applied on the environment
        """
        raise NotImplementedError("Parameters not implemented")

    @property
    def involved_qubits(self):
        """
        Return the qubits involved in the calibration task
        """
        return list(self.layout.get_physical_bits().keys())

    @property
    def backend_info(self) -> BackendInfo:
        """
        Return the backend information object
        """
        return self._backend_info

    @property
    def pass_manager(self) -> Optional[PassManager]:
        """
        Return the custom pass manager for transpilation (if specified)
        """
        return self.backend_info.pass_manager

    @property
    def observables(self) -> SparsePauliOp:
        """
        Return set of observables sampled for current epoch of training (relevant only for
        direct fidelity estimation methods, e.g. 'channel' or 'state')
        """
        if self.config.reward_config.dfe:
            return self.config.reward_config.observables
        else:
            raise ValueError(
                f"Observables not defined for reward method {self.config.reward_method}"
            )

    @property
    def total_shots(self):
        """
        Return the total number of shots executed on the quantum system
        (as a list of shots executed for each step of the training)
        """
        return self._total_shots

    @property
    def pubs(self) -> List[EstimatorPub | SamplerPub]:
        """
        Return the current PUBs used in the environment
        """
        return self._pubs

    @property
    def hardware_runtime(self):
        """
        Return the total hardware runtime for the quantum system
        (as a list of runtimes for each step of the training)
        """
        return self._hardware_runtime

    @property
    def n_actions(self):
        return self.action_space.shape[-1]

    @property
    def optimal_action(self):
        return self._optimal_action

    @property
    def mean_action(self):
        return self._mean_action

    @mean_action.setter
    def mean_action(self, value):
        self._mean_action = np.array(value)

    @property
    def std_action(self):
        return self._std_action

    @std_action.setter
    def std_action(self, value):
        self._std_action = np.array(value)

    @property
    def ident_str(self):
        return self._ident_str()

    @property
    @abstractmethod
    def trunc_index(self) -> int:
        """
        Index of the truncation to be applied
        """
        raise NotImplementedError("Truncation index not implemented")

    @property
    def metadata(self):
        """
        Return metadata of the environment
        """
        return self.config.env_metadata

    def to_json(self):
        return json.dumps(
            {
                "n_qubits": self.n_qubits,
                "config": self.config.as_dict(),
                "abstraction_level": self.abstraction_level,
                "sampling_Pauli_space": self.sampling_paulis,
                "n_shots": self.n_shots,
                "target_type": self.target.target_type,
                "target": self.target,
                "c_factor": self.c_factor,
                "reward_history": self.reward_history,
                "action_history": self.action_history,
                "fidelity_history": (
                    self.avg_fidelity_history
                    if self.target.target_type == "gate"
                    else self.circuit_fidelity_history
                ),
            }
        )
