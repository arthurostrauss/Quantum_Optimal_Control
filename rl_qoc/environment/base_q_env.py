"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
a quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 30/04/2025
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import json
import signal
from typing import Callable, Any, Optional, List, Literal, Sequence, TYPE_CHECKING
import numpy as np
from gymnasium import Env
from gymnasium.core import ObsType

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
    Parameter,
)
from qiskit.transpiler import CouplingMap

# Qiskit Estimator Primitives: for computing Pauli expectation value sampling easily
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    Operator,
    SparsePauliOp,
    partial_trace,
)

from qiskit.transpiler import (
    Layout,
    PassManager,
)
from qiskit.providers import BackendV2
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator

# Qiskit Experiments for generating reliable baseline for complex gate calibrations / state preparations
from qiskit_ibm_runtime import (
    EstimatorV2 as RuntimeEstimatorV2,
)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from .configuration.qconfig import QEnvConfig, ExecutionConfig
from .target import GateTarget, StateTarget

from ..helpers.helper_functions import (
    retrieve_primitives,
    get_hardware_runtime_single_circuit,
    has_noise_model,
)
from ..helpers.circuit_utils import retrieve_neighbor_qubits

if TYPE_CHECKING:
    from ..rewards import Reward, REWARD_STRINGS
    from ..rewards.reward_data import RewardDataList


class BaseQuantumEnvironment(ABC, Env):
    """
    An abstract base class for quantum reinforcement learning environments.

    This class provides a common interface for quantum environments, including methods for defining circuits,
    performing actions, and computing benchmarks. It is designed to be subclassed by specific quantum environments.
    """

    def __init__(self, training_config: QEnvConfig):
        """
        Initializes the BaseQuantumEnvironment.

        Args:
            training_config: The configuration for the training environment.
        """
        self._env_config = training_config
        self.parametrized_circuit_func: Callable = training_config.parametrized_circuit
        self._func_args = training_config.parametrized_circuit_kwargs
        self._physical_target_qubits = training_config.target.physical_qubits

        self._estimator, self._sampler = retrieve_primitives(self.config.backend_config)

        self.circuits = []

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
        self._reward_data = None
        self._observables, self._pauli_shots = None, None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.process_fidelity_history = []
        self.avg_fidelity_history = []
        self.circuit_fidelity_history = []
        self.circuit_fidelity_history_nreps = []
        self.avg_fidelity_history_nreps = []
        self._fit_function: Optional[Callable] = None
        self._action_to_cycle_reward_function: Optional[Callable] = None
        self._fit_params: Optional[np.ndarray] = None
        self._total_updates = None

        # Call reset of Env class to set seed
        self._seed = training_config.seed
        super().reset(seed=self._seed)
        self._n_reps_rng = np.random.default_rng(self.np_random.integers(2**32))
        self.config.reward.set_reward_seed(self.np_random.integers(2**32))

    @abstractmethod
    def define_circuits(
        self,
    ) -> list[QuantumCircuit]:
        """
        Defines the circuits to be used in the environment.

        This method should be implemented by subclasses to provide the specific quantum circuits for the environment.

        Returns:
            A list of QuantumCircuit objects.
        """
        raise NotImplementedError("Define target method not implemented")

    @abstractmethod
    def episode_length(self, global_step: int) -> int:
        """
        Determines the length of an episode.

        Args:
            global_step: The current step in the training loop.

        Returns:
            The length of the episode.
        """
        pass

    @abstractmethod
    def _get_obs(self):
        """
        Returns the observation of the environment.

        This method should be implemented by subclasses to provide the specific observation for the environment.

        Returns:
            The observation of the environment.
        """
        pass

    @abstractmethod
    def compute_benchmarks(
        self, qc: QuantumCircuit, params: np.ndarray, update_env_history=True
    ) -> np.ndarray:
        """
        Computes benchmarks for the given circuit and parameters.

        Args:
            qc: The quantum circuit to benchmark.
            params: The parameters for the circuit.
            update_env_history: Whether to update the environment's history.

        Returns:
            An array of fidelity values.
        """

    def initial_reward_fit(
        self,
        params: np.ndarray,
        execution_config: Optional[ExecutionConfig] = None,
        reward_method: Optional[Sequence[REWARD_STRINGS | Reward]] = None,
        fit_function: Optional[Callable] = None,
        inverse_fit_function: Optional[Callable] = None,
        update_fit_params: Optional[REWARD_STRINGS | Reward] = None,
    ) -> List[np.ndarray]:
        """
        Fits the initial reward function to the first set of actions.

        This method is used to calibrate the reward function based on the initial performance of the agent.

        Args:
            params: The action parameters to fit the initial reward function.
            execution_config: The execution configuration to use for the fit.
            reward_method: The reward method(s) to plot.
            fit_function: The function to fit the data points.
            inverse_fit_function: The function to compute the inverse of the fit function.
            update_fit_params: The method from which to update the fit parameters.

        Returns:
            A list of reward data for each method.
        """
        from ..rewards import Reward, REWARD_STRINGS

        fig, ax = plt.subplots()
        ax.set_title("Initial reward fit (for varying number of repetitions)")
        ax.set_xlabel("Number of repetitions")
        ax.set_ylabel("Reward")
        initial_execution_config = self.config.execution_config
        initial_reward = self.config.reward
        if execution_config is not None:
            self.config.execution_config = execution_config
        if reward_method is not None:
            if isinstance(reward_method, (str, Reward)):
                reward_method = [reward_method]

        else:
            reward_method = [self.config.reward_method]
        reward_data = []
        for m, method in enumerate(reward_method):
            reward_data.append([])
            if isinstance(method, Reward):
                self.config.reward = method
            elif isinstance(method, str):
                self.config.reward_method = method
            for i in range(len(self.config.execution_config.n_reps)):
                self.config.execution_config.n_reps_index = i
                print("Number of repetitions:", self.n_reps)
                reward = self.perform_action(params, update_env_history=False)
                reward_data[m].append(np.mean(reward))
            if fit_function is None or inverse_fit_function is None:

                def fit_function(n, spam, eps_lin, eps_quad):
                    return 1 - spam - eps_lin * n - eps_quad * n**2

                def inverse_fit_function(reward, n, spam, eps_lin, eps_quad):
                    return reward + eps_lin * (n - 1) + eps_quad * (n**2 - 1)

            p0 = [0.0, 0.0, 0.0]  # Initial guess for the parameters
            lower_bounds = [0.0, 0.0, 0.0]
            upper_bounds = [0.1, 0.2, 0.1]

            popt, pcov = curve_fit(
                fit_function,
                self.config.execution_config.n_reps,
                reward_data[m],
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
            )

            ax.plot(
                self.config.execution_config.n_reps,
                reward_data[m],
                label=f"Data ({method})",
                marker="o",
            )
            ax.plot(
                self.config.execution_config.n_reps,
                [fit_function(n, *popt) for n in self.config.execution_config.n_reps],
                label=f"Fit ({method})",
            )
            ax.legend()

            # Print found parameters
            print(f"Found parameters ({method} method):", popt)
            if update_fit_params == method:
                self._fit_function = lambda reward, n: inverse_fit_function(reward, n, *popt)
                self._fit_params = popt

        if execution_config is not None:
            self.config.execution_config = initial_execution_config

        self.config.reward = initial_reward
        return reward_data

    def perform_action(self, actions: np.ndarray, update_env_history: bool = True):
        """
        Performs an action in the environment.

        Args:
            actions: The actions to be performed.
            update_env_history: Whether to update the environment's history.

        Returns:
            The reward for the action.
        """
        if not actions.shape[-1] == self.n_actions:
            raise ValueError(f"Action shape mismatch: {actions.shape[-1]} != {self.n_actions}")
        qc = self.circuit.copy()
        params, batch_size = np.array(actions), actions.shape[0]
        if len(params.shape) == 1:
            params = np.expand_dims(params, axis=0)

        rewarder = self.config.reward

        if self.do_benchmark():  # Benchmarking or fidelity access
            fids = self.compute_benchmarks(qc, params, update_env_history)

        additional_input = (
            self.config.execution_config.dfe_precision if self.config.dfe else self.baseline_circuit
        )
        if self.config.execution_config.n_reps_mode == "sequential":
            reward_data = rewarder.get_reward_data(
                qc,
                params,
                self.target,
                self.config,
                additional_input,
            )
            total_shots = reward_data.total_shots
            if update_env_history:
                self.update_env_history(qc, total_shots)
            self._pubs = reward_data.pubs
            self._reward_data = reward_data
            reward = rewarder.get_reward_with_primitive(reward_data, self.primitive)

            print("Reward (avg):", np.mean(reward), "Std:", np.std(reward))

            return reward
        else:
            raise NotImplementedError("Only sequential mode is supported for now")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to its initial state.

        Args:
            seed: The seed for the random number generator.
            options: Additional options for the reset.

        Returns:
            A tuple containing the initial observation and additional information.
        """
        super().reset(seed=seed)
        self._episode_tracker += 1
        self._episode_ended = False
        options = options or {}
        self.set_env_params(**options)
        if len(self.config.execution_config.n_reps) > 1:
            self.config.execution_config.n_reps_index = self._n_reps_rng.integers(
                0, len(self.config.execution_config.n_reps)
            )

        if isinstance(self.estimator, RuntimeEstimatorV2):
            self.estimator.options.environment.job_tags = [f"rl_qoc_step{self._step_tracker}"]

        return self._get_obs(), self._get_info()

    def simulate_circuit(
        self,
        qc: QuantumCircuit,
        params: np.ndarray,
        update_env_history: bool = True,
        output_fidelity: Literal["cycle", "nreps"] = "cycle",
    ) -> np.ndarray:
        """
        Simulates the given quantum circuit.

        Args:
            qc: The quantum circuit to simulate.
            params: The parameters for the circuit.
            update_env_history: Whether to update the environment's history.
            output_fidelity: The type of fidelity to return.

        Returns:
            An array of fidelity values.
        """

        if self.abstraction_level != "circuit":
            raise ValueError(
                "This method should only be called when the abstraction level is 'circuit'"
            )

        qc_channel = qc.copy(name="qc_channel")
        qc_state = qc.copy(name="qc_state")
        qc_channel_nreps = qc.repeat(self.n_reps).decompose()
        qc_state_nreps = qc.repeat(self.n_reps).decompose()
        names = ["qc_channel", "qc_state", "qc_channel_nreps", "qc_state_nreps"]

        qc_channel, qc_state, qc_channel_nreps, qc_state_nreps = (
            self.config.backend_config.custom_transpile(
                [qc_channel, qc_state, qc_channel_nreps, qc_state_nreps],
                optimization_level=0,
                initial_layout=self.target.layout,
                scheduling=False,
                remove_final_measurements=False,
            )
        )
        for circ, name in zip([qc_channel, qc_state, qc_channel_nreps, qc_state_nreps], names):
            circ.name = name

        returned_fidelity_type = (
            "gate"
            if isinstance(self.target, GateTarget) and self.target.causal_cone_size <= 3
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

        if isinstance(self.parameters, ParameterVector) or all(
            isinstance(param, Parameter) for param in self.parameters
        ):
            parameters = [self.parameters]
            n_custom_instructions = 1
        else:  # List of ParameterVectors
            parameters = self.parameters
            n_custom_instructions = self.circuit_choice + 1

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
            if (method == "superop" or method == "unitary") and self.target.causal_cone_size > 3:
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
                fidelities = [self.target.fidelity(output, n_reps) for output in outputs]
            if (method == "superop" or method == "unitary") and returned_fidelity_type == "gate":
                if output_fidelity == "cycle" and n_reps == 1:
                    returned_fidelities = fidelities
                elif output_fidelity == "nreps" and n_reps > 1:
                    returned_fidelities = fidelities
            elif (
                method == "density_matrix" or method == "statevector"
            ) and returned_fidelity_type == "state":
                if output_fidelity == "cycle" and n_reps == 1:
                    returned_fidelities = fidelities
                elif output_fidelity == "nreps" and n_reps > 1:
                    returned_fidelities = fidelities
            if update_env_history:
                fid_array.append(np.mean(fidelities))

        return returned_fidelities

    def _observable_to_observation(self):
        """
        Converts the observable to an observation.

        Returns:
            The observation.
        """
        if self.config.reward_method == "state":
            array_obs = []
            return array_obs
        else:
            raise NotImplementedError("Channel estimator not yet implemented")

    def simulate_pulse_circuit(
        self,
        qc: QuantumCircuit,
        params: Optional[np.ndarray] = None,
        update_env_history: bool = True,
    ) -> List[float]:
        """
        Simulates the given pulse circuit.

        Args:
            qc: The quantum circuit to simulate.
            params: The parameters for the circuit.
            update_env_history: Whether to update the environment's history.

        Returns:
            A list of fidelity values.
        """
        try:
            from qiskit_dynamics import DynamicsBackend
            from ..custom_jax_sim import PulseEstimatorV2, simulate_pulse_level
            from ..helpers.pulse_utils import (
                handle_virtual_rotations,
                projected_state,
                qubit_projection,
                rotate_frame,
            )
        except ImportError:
            raise ImportError(
                "Qiskit Dynamics is required for pulse simulation, as well as "
                "Qiskit version below 2.0.0 for Qiskit Pulse. "
            )
        if self.abstraction_level != "pulse":
            raise ValueError(
                "This method should only be called when the abstraction level is 'pulse'"
            )
        if not isinstance(self.backend, DynamicsBackend):
            raise ValueError(f"Pulse simulation requires a DynamicsBackend; got {self.backend}")
        returned_fidelity_type = (
            "gate" if isinstance(self.target, GateTarget) and qc.num_qubits <= 3 else "state"
        )
        returned_fidelities = []
        subsystem_dims = list(
            filter(lambda x: x > 1, self.backend.options.subsystem_dims)
        )
        n_benchmarks = 1
        qc_nreps = None
        if self.n_reps > 1 and isinstance(
            self.target, GateTarget
        ):
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
                [qc_nreps.assign_parameters(p) for p in params] if qc_nreps is not None else []
            )
            data_length = len(params)
        circuits_list = circuits + circuits_n_reps

        if isinstance(self.estimator, PulseEstimatorV2):
            sampler_pubs = [(circ, params) for circ in circuits_list]
            y0_list = [y0_state]
            if qc.num_qubits < 3 and isinstance(
                self.target, GateTarget
            ):
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

            output_data = [
                output_data[i * data_length : (i + 1) * data_length] for i in range(n_benchmarks)
            ]
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
                qc_data_mapping.get(name, None) for name in circuit_order if name in qc_data_mapping
            ]
            output_data = new_output_data

        else:

            y0_list = [y0_state] * n_benchmarks * data_length

            if qc.num_qubits < 3 and isinstance(
                self.target, GateTarget
            ):
                y0_list += [y0_gate] * n_benchmarks * data_length
                circuits_list += circuits + circuits_n_reps
                n_benchmarks *= 2
            output_data = []
            results = self.backend.solve(circuits_list, y0=y0_list)
            for solver_result in results:
                yf = solver_result.y[-1]
                tf = solver_result.t[-1]
                yf = rotate_frame(yf, tf, self.backend)

                output_data.append(yf)

            output_data = [
                output_data[i * data_length : (i + 1) * data_length] for i in range(n_benchmarks)
            ]

        if self.n_reps > 1:
            circ_list = [qc, qc_nreps, qc, qc_nreps]
            fid_arrays = [
                self.circuit_fidelity_history,
                self.avg_fidelity_history,
                self.circuit_fidelity_history_nreps,
                self.avg_fidelity_history_nreps,
            ]
        else:
            circ_list = [qc] * 2
            fid_arrays = [self.circuit_fidelity_history, self.avg_fidelity_history]

        for circ, data, fid_array in zip(circ_list, output_data, fid_arrays):
            n_reps = 1 if "nreps" not in circ.name else self.n_reps

            if isinstance(data[0], (Statevector, DensityMatrix)):
                data = [projected_state(state, subsystem_dims) for state in data]
                if self.target.n_qubits != len(
                    subsystem_dims
                ):
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
            elif isinstance(data[0], Operator):
                data = [qubit_projection(op, subsystem_dims) for op in data]

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
            ):
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
        Updates the gate calibration with the optimal actions.

        Args:
            gate_name: The name of the custom gate to be created.
        """
        raise NotImplementedError("Gate calibration not implemented for this environment")

    def set_env_params(self, **kwargs):
        """
        Sets the environment parameters.

        Args:
            **kwargs: The keyword arguments to set.
        """
        for key, value in kwargs.items():
            if key == "backend":
                self.backend = value
            else:
                try:
                    setattr(self.config, key, value)
                except AttributeError:
                    raise AttributeError(f"Invalid attribute {key} for environment config")
                except Exception as e:
                    raise ValueError(f"Error setting {key} to {value}: {e}")

    @property
    def config(self) -> QEnvConfig:
        """The configuration of the environment."""
        return self._env_config

    @property
    def backend(self) -> Optional[BackendV2]:
        """The backend of the environment."""
        return self.config.backend

    @backend.setter
    def backend(self, backend: BackendV2):
        self.config.backend = backend
        self._estimator, self._sampler = retrieve_primitives(self.config.backend_config)

    @property
    def estimator(self) -> BaseEstimatorV2:
        """The estimator of the environment."""
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimatorV2):
        self._estimator = estimator

    @property
    def sampler(self) -> BaseSamplerV2:
        """The sampler of the environment."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSamplerV2):
        self._sampler = sampler

    @property
    def primitive(self) -> BaseEstimatorV2 | BaseSamplerV2:
        """The primitive of the environment."""
        return self.estimator if self.config.reward.dfe else self.sampler

    @property
    def physical_target_qubits(self):
        """The physical target qubits of the environment."""
        return self._physical_target_qubits

    @property
    def physical_neighbor_qubits(self):
        """The physical neighbor qubits of the environment."""
        if (
            self.backend is not None
            and hasattr(self.backend, "coupling_map")
            and isinstance(self.backend.coupling_map, CouplingMap)
        ):
            return retrieve_neighbor_qubits(self.backend.coupling_map, self.physical_target_qubits)
        else:
            return

    @property
    def physical_next_neighbor_qubits(self):
        """The physical next neighbor qubits of the environment."""
        if (
            self.backend is not None
            and hasattr(self.backend, "coupling_map")
            and isinstance(self.backend.coupling_map, CouplingMap)
        ):
            return retrieve_neighbor_qubits(
                self.backend.coupling_map,
                self.physical_target_qubits + self.physical_neighbor_qubits,
            )
        else:
            return

    @property
    def transpiled_circuits(self) -> Optional[List[QuantumCircuit]]:
        """The transpiled circuits of the environment."""
        if self.circuits:
            return self.config.backend_config.custom_transpile(
                self.circuits,
                initial_layout=self.target.layout,
                optimization_level=0,
                remove_final_measurements=False,
            )
        return None

    @property
    def baseline_circuits(self) -> List[QuantumCircuit]:
        """The baseline circuits of the environment."""
        return self.config.target.circuits

    @property
    def fidelity_history(self):
        """The fidelity history of the environment."""
        return (
            self.avg_fidelity_history
            if self.target.target_type == "gate" and self.target.circuit.num_qubits <= 3
            else self.circuit_fidelity_history
        )

    @property
    def step_tracker(self):
        """The step tracker of the environment."""
        return self._step_tracker

    @property
    def abstraction_level(self):
        """The abstraction level of the environment."""
        return (
            "pulse"
            if hasattr(self.circuit, "calibrations") and self.circuit.calibrations
            else "circuit"
        )

    @step_tracker.setter
    def step_tracker(self, step: int):
        assert step >= 0, "step must be positive integer"
        self._step_tracker = step

    def signal_handler(self, signum, frame):
        """
        Handles signals.
        """
        print(f"Received signal {signum}, closing environment...")
        self.close()

    def close(self) -> None:
        """Closes the environment."""
        if hasattr(self.estimator, "session"):
            self.estimator.session.close()

    def clear_history(self):
        """
        Clears the history of the environment.
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
        """The benchmark cycle of the environment."""
        return self.config.benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, step: int) -> None:
        assert step >= 0, "Cycle needs to be a positive integer"
        self.config.benchmark_cycle = step

    def do_benchmark(self) -> bool:
        """
        Checks if a benchmark should be performed.

        Returns:
            True if a benchmark should be performed, False otherwise.
        """
        if self.benchmark_cycle == 0:
            return False
        else:
            return self._episode_tracker % self.benchmark_cycle == 0

    @property
    def total_updates(self):
        """The total number of updates."""
        return self._total_updates

    @total_updates.setter
    def total_updates(self, total_updates: int):
        self._total_updates = total_updates

    def _get_info(self) -> Any:
        """
        Returns the info of the environment.

        Returns:
            The info of the environment.
        """
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
                    "n_reps": self.n_reps,
                    "n_shots": self.n_shots,
                    "sampling_paulis": self.sampling_paulis,
                    "batch_size": self.batch_size,
                    "c_factor": self.c_factor,
                    "reward_method": self.config.reward_method,
                    "circuit_choice": self.circuit_choice,
                }
            else:
                info = {
                    "step": step,
                    "average return": np.mean(self.reward_history, axis=1)[-1],
                    "max return": np.max(np.mean(self.reward_history, axis=1)),
                    "arg_max return": np.argmax(np.mean(self.reward_history, axis=1)),
                    "optimal action": self.optimal_action,
                    "circuit_choice": self.circuit_choice,
                    "n_reps": self.n_reps,
                    "n_shots": self.n_shots,
                    "sampling_paulis": self.sampling_paulis,
                    "batch_size": self.batch_size,
                    "c_factor": self.c_factor,
                    "reward_method": self.config.reward_method,
                }
        else:
            info = {
                "reset_stage": self._inside_trunc_tracker == 0,
                "step": step,
                "gate_index": self._inside_trunc_tracker,
                "truncation_index": self.circuit_choice,
            }
        return info

    def __str__(self):
        """
        Returns a string representation of the environment.

        Returns:
            A string representation of the environment.
        """
        if isinstance(self.target, GateTarget):
            ident_str = f"gate_calibration_{self.target.gate.name}-gate_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        elif isinstance(self.target, StateTarget):
            ident_str = f"state_preparation_physical_qubits_{'-'.join(map(str, self.target.physical_qubits))}"
        else:
            raise ValueError("Target type not recognized")
        return ident_str

    def __repr__(self):
        """
        Returns a string representation of the environment.

        Returns:
            A string representation of the environment.
        """
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

    @property
    def seed(self):
        """The seed of the environment."""
        return self._seed

    @property
    def c_factor(self):
        """The c_factor of the environment."""
        return self.config.c_factor

    @property
    def action_space(self):
        """The action space of the environment."""
        return self.config.action_space

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def batch_size(self) -> int:
        """The batch size of the environment."""
        return self.config.batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        self.config.batch_size = size

    @property
    def n_reps(self) -> int:
        """The number of repetitions of the environment."""
        return self.config.execution_config.current_n_reps

    @n_reps.setter
    def n_reps(self, n_reps: int):
        if n_reps < 1:
            raise ValueError("n_reps must be at least 1")
        n_reps_list = list(self.config.execution_config.n_reps)
        if n_reps in n_reps_list:
            self.config.execution_config.n_reps_index = n_reps_list.index(n_reps)
        else:
            n_reps_list.append(n_reps)
            self.config.execution_config.n_reps = n_reps_list
            self.config.execution_config.n_reps_index = len(n_reps_list) - 1

    @property
    def n_shots(self) -> int:
        """The number of shots of the environment."""
        return self.config.n_shots

    @n_shots.setter
    def n_shots(self, n_shots: int):
        self.config.n_shots = n_shots

    @property
    def sampling_paulis(self) -> int:
        """The sampling paulis of the environment."""
        return self.config.sampling_paulis

    @sampling_paulis.setter
    def sampling_paulis(self, sampling_paulis: int):
        self.config.sampling_paulis = sampling_paulis

    @property
    def target(self) -> GateTarget | StateTarget:
        """The target of the environment."""
        return self.config.target

    @property
    def circuit_choice(self) -> int:
        """The circuit choice of the environment."""
        return (
            self.config.target.circuit_choice
            if hasattr(self.config.target, "circuit_choice")
            else 0
        )

    @property
    def n_qubits(self):
        """The number of qubits of the environment."""
        return self.target.n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits):
        assert isinstance(n_qubits, int) and n_qubits > 0, "n_qubits must be a positive integer"
        self.target.n_qubits = n_qubits

    @property
    def layout(self):
        """The layout of the environment."""
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
        """The parameters of the environment."""
        raise NotImplementedError("Parameters not implemented")

    @property
    def involved_qubits(self):
        """The involved qubits of the environment."""
        return list(self.layout.get_physical_bits().keys())

    @property
    def pass_manager(self) -> Optional[PassManager]:
        """The pass manager of the environment."""
        return self.config.backend_config.pass_manager

    @property
    def observables(self) -> SparsePauliOp:
        """The observables of the environment."""
        if self.config.reward.dfe:
            return self.config.reward.observables
        else:
            raise ValueError(
                f"Observables not defined for reward method {self.config.reward_method}"
            )

    @property
    def total_shots(self):
        """The total shots of the environment."""
        return self._total_shots

    @property
    def circuit(self):
        """The circuit of the environment."""
        return self.circuits[self.circuit_choice]

    @property
    def baseline_circuit(self):
        """The baseline circuit of the environment."""
        return self.baseline_circuits[self.circuit_choice]

    @property
    def pubs(self) -> List[EstimatorPub | SamplerPub]:
        """The pubs of the environment."""
        return self._pubs

    @property
    def reward_data(self) -> RewardDataList:
        """The reward data of the environment."""
        return self._reward_data

    @property
    def hardware_runtime(self):
        """The hardware runtime of the environment."""
        return self._hardware_runtime

    @property
    def n_actions(self):
        """The number of actions of the environment."""
        return self.action_space.shape[-1]

    @property
    def optimal_action(self):
        """The optimal action of the environment."""
        return self._optimal_action

    @property
    def mean_action(self):
        """The mean action of the environment."""
        return self._mean_action

    @mean_action.setter
    def mean_action(self, value):
        self._mean_action = np.array(value)

    @property
    def std_action(self):
        """The std action of the environment."""
        return self._std_action

    @std_action.setter
    def std_action(self, value):
        self._std_action = np.array(value)

    @property
    def ident_str(self):
        """The ident str of the environment."""
        return self.__str__()

    @property
    def metadata(self):
        """The metadata of the environment."""
        return self.config.env_metadata

    def to_json(self):
        """
        Returns a json representation of the environment.

        Returns:
            A json representation of the environment.
        """
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

    def update_env_history(self, qc, total_shots, hardware_runtime=None):
        """
        Updates the environment history.

        Args:
            qc: The quantum circuit.
            total_shots: The total shots.
            hardware_runtime: The hardware runtime.
        """
        self._total_shots.append(total_shots)
        if hardware_runtime is not None:
            self._hardware_runtime.append(hardware_runtime)
        elif self.config.backend_config.instruction_durations is not None:
            self._hardware_runtime.append(
                get_hardware_runtime_single_circuit(
                    qc,
                    self.config.backend_config.instruction_durations.duration_by_name_qubits,
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
