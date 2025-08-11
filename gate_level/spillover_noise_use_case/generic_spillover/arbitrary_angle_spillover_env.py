from typing import Optional, Dict, Any, List, Sequence, Literal

from gymnasium.spaces import Box, Dict as DictSpace
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout

from rl_qoc import ContextAwareQuantumEnvironment, QEnvConfig, GateTarget
import numpy as np
from spillover_effect_on_subsystem import (
    noisy_backend,
    circuit_context,
    LocalSpilloverNoiseAerPass,
    numpy_to_hashable,
)
from rl_qoc.environment.context_aware_quantum_environment import ObsType
from rl_qoc.helpers import causal_cone_circuit


class ArbitraryAngleSpilloverEnv(ContextAwareQuantumEnvironment):
    """
    Quantum environment with spillover noise on a subsystem where each epoch will have a random set of angles.
    The input circuit context of this environment is expected to have symbolic Parameters for all rotation axes.
    It also should have the form of one layer of single qubit rotation gates (rx, ry, rz) and a layer of two-qubit gates.
    Binding of those parameters will be done automatically at the reset of the environment.
    """

    def __init__(
        self,
        q_env_config: QEnvConfig,
        gamma_matrix: np.ndarray,
    ):
        """
        Initialize the environment
        """
        self.gamma_matrix = gamma_matrix

        super().__init__(q_env_config)
        if not isinstance(self.target, GateTarget):
            raise ValueError("Target must be a GateTarget")
        if len(self.target.circuits) > 1:
            raise ValueError("Target must have only one circuit")
        self.circuit_parameters = self.target.circuit.parameters
        self._rotation_angles_rng = np.random.default_rng(self.np_random.integers(2**32))
        self.observation_space = DictSpace({p.name: Box(low=-np.pi, high=np.pi, shape=(1,)) for p in self.circuit_parameters})
        

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment
        :param seed: Seed for the environment
        :param options: Options for the environment
        :return: Initial observation and info
        """
        # Reset the environment
        obs, info = super().reset(seed=seed, options=options)

        self.backend = noisy_backend(
            self.target.circuit, self.gamma_matrix, self.config.env_metadata["target_subsystem"]
        )
        # Generate the initial observation

        # Return the initial observation and info
        return obs, {}

    def _get_obs(self):
        """
        Get the observation
        :return: Observation
        """
        return {p.name: self._rotation_angles_rng.uniform(0, 2 * np.pi) for p in self.circuit_parameters}

    def _get_info(self) -> Any:
        return {}
