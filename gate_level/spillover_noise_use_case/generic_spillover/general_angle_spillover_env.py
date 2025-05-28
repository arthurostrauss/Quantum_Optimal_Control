import numpy as np
from typing import Optional, Dict, Any, SupportsFloat
from qiskit import QuantumCircuit
from rl_qoc.environment.context_aware_quantum_environment import (
    ObsType,
    ActType,
    ContextAwareQuantumEnvironment,
)
from gymnasium.spaces import Box
from rl_qoc import ContextAwareQuantumEnvironment, QEnvConfig, GateTarget
from spillover_effect_on_subsystem import (
    noisy_backend,
    circuit_context,
    LocalSpilloverNoiseAerPass,
    numpy_to_hashable,
)
from rl_qoc.helpers import causal_cone_circuit
import itertools


class GeneralAngleSpilloverEnv(ContextAwareQuantumEnvironment):
    """
    Quantum environment with spillover noise on a subsystem where each epoch will have a random set of angles.
    The input circuit context of this environment is expected to have symbolic Parameters for all rotation axes.
    It also should have the form of one layer of single qubit rotation gates (rx, ry, rz) and a layer of two-qubit gates.
    Binding of those parameters will be done automatically at the reset of the environment.
    """

    def __init__(
        self,
        q_env_config: QEnvConfig,
        unbound_circuit_context: QuantumCircuit,
        gamma_matrix: np.ndarray,
        num_params: int,
        discrete_history_length: int,
        applied_qubits: list,  # list of qubits which we will include the effect of
        circuit_param_distribution: Optional[str] = "uniform",
        optimal_error_precision: Optional[float] = 1e-4,
        obs_bounds: Optional[tuple] = (-np.pi, np.pi),
    ):
        """
        Initialize the environment
        """
        self.num_params = num_params
        self.circuit_param_distribution = circuit_param_distribution
        self.optimal_error_precision = optimal_error_precision
        self.discrete_history_length = discrete_history_length
        self.gamma_matrix = gamma_matrix
        self.circuit_parameters = unbound_circuit_context.parameters
        self.num_qubits = len(applied_qubits)  # Number of qubits

        super().__init__(q_env_config, unbound_circuit_context)
        self._rotation_angles_rng = np.random.default_rng(
            self.np_random.integers(2**32)
        )
        self.observation_space = Box(
            low=np.array([-1.0] * len(applied_qubits)),
            high=np.array([1.0] * len(applied_qubits)),
            dtype=np.float64,
        )
        self.obs_bounds = obs_bounds

        # Define discrete observation values for a single qubit
        self.single_qubit_discrete_obs_vals_raw = np.linspace(
            self.observation_space.low[0],
            self.observation_space.high[0],
            self.num_params,
        )
        # Generate all possible combinations for N qubits
        self.discrete_obs_vals_raw = np.array(
            list(
                itertools.product(
                    self.single_qubit_discrete_obs_vals_raw, repeat=self.num_qubits
                )
            )
        )
        self.discrete_obs_vals_angles = self.obs_raw_to_angles(
            self.discrete_obs_vals_raw
        )
        print(f"Observation Vals for Agent: {self.discrete_obs_vals_raw}")
        print(f"Observation Vals for Env: {self.discrete_obs_vals_angles}")

        # Initialize reward history and probability weights for all possible combinations
        self.discrete_reward_history = np.ones(
            (discrete_history_length, len(self.discrete_obs_vals_raw))
        )
        self.prob_weights = np.ones(len(self.discrete_obs_vals_raw))
        self.prob_weights /= len(self.prob_weights)

        self.obs_angles = np.zeros(self.observation_space.shape)
        self.applied_qubits = applied_qubits
        self.obs_raw = np.zeros(self.observation_space.shape)

    def obs_raw_to_angles(self, obs_raw):
        # Obs raw between -1. and 1.
        # Angles between self.obs_bounds[0] and self.obs_bounds[1]
        scale = (self.obs_bounds[1] - self.obs_bounds[0]) / 2
        mean = (self.obs_bounds[0] + self.obs_bounds[1]) / 2
        obs_scaled = obs_raw * scale
        angles = obs_scaled + mean
        return angles

    def angles_to_obs_raw(self, angles):
        # Angles between self.obs_bounds[0] and self.obs_bounds[1]
        # Obs raw between -1. and 1.
        scale = (self.obs_bounds[1] - self.obs_bounds[0]) / 2
        mean = (self.obs_bounds[0] + self.obs_bounds[1]) / 2
        obs_scaled = angles - mean
        obs_raw = obs_scaled / scale
        return obs_raw

    def reset(
        self,
        debug_obs: Optional[np.ndarray] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment
        """
        old_angles, info = super().reset(seed=seed, options=options)

        if debug_obs is not None:
            self.obs_angles = debug_obs
            self.obs_raw = self.angles_to_obs_raw(self.obs_angles)
        else:
            self.obs_raw = self._get_new_obs_raw()
            self.obs_angles = self.obs_raw_to_angles(self.obs_raw)

        self.phi_vec = np.zeros(self.unbound_circuit_context.num_qubits)
        self.phi_vec[self.applied_qubits] = self.obs_angles

        param_dict = {
            self.circuit_parameters[i].name: self.phi_vec[i]
            for i in range(len(self.phi_vec))
        }
        circuit = self.unbound_circuit_context.assign_parameters(param_dict)
        backend = noisy_backend(
            circuit, self.gamma_matrix, self.config.env_metadata["target_subsystem"]
        )
        self.set_circuit_context(None, backend=backend, **param_dict)
        env_obs = self._get_obs()
        print("Sampled angles: ", self.phi_vec)
        print(f"Environment Observation: {env_obs}")
        return env_obs, {}

    def _get_new_obs_raw(self):
        if self.circuit_param_distribution == "uniform":
            obs_raw = np.random.uniform(
                self.observation_space.low,
                self.observation_space.high,
                self.observation_space.shape,
            )
        elif self.circuit_param_distribution == "simple_discrete":
            # Sample each qubit's observation independently
            obs_raw = np.random.choice(
                a=self.single_qubit_discrete_obs_vals_raw,
                size=self.observation_space.shape[0],
                replace=True,
            )
        elif self.circuit_param_distribution == "moving_discrete":
            # Update probability weights based on reward history
            self.prob_weights = (
                np.mean(self.discrete_reward_history, axis=0)
                + np.log10(self.optimal_error_precision)
            ) ** 2
            self.prob_weights /= np.sum(self.prob_weights)
            # Sample an index from all possible combinations
            idx = np.random.choice(
                len(self.discrete_obs_vals_raw),
                p=self.prob_weights,
            )
            obs_raw = self.discrete_obs_vals_raw[idx]
        return obs_raw

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        print(f"batch action shape: {action.shape}")
        obs, reward, terminated, truncated, info = super().step(action)
        print(f"obs: {obs}")
        print(f"reward: {reward}")

        if self.circuit_param_distribution == "moving_discrete":
            self.update_discrete_history(reward, obs)

        return obs, reward, terminated, truncated, info

    def update_discrete_history(self, reward, obs):
        # Find the index of the closest discrete observation value
        obs_ind = np.argmin(np.sum((self.discrete_obs_vals_raw - obs) ** 2, axis=1))
        self.discrete_reward_history[:, obs_ind] = np.append(
            [np.mean(reward)], self.discrete_reward_history[:-1, obs_ind], axis=0
        )

    def clear_history(self):
        super().clear_history()
        self.discrete_reward_history = np.ones_like(self.discrete_reward_history)

    def _get_obs(self):
        """
        Get the observation
        """
        return self.obs_raw

    def _get_info(self) -> Any:
        return {}
