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
    ):
        """
        Initialize the environment
        """

        super().__init__(q_env_config)
        if not isinstance(self.target, GateTarget):
            raise ValueError("Target must be a GateTarget")
        if len(self.target.circuits) > 1:
            raise ValueError("Target must have only one circuit")
        self.circuit_parameters = self.target.circuit.parameters
        self._rotation_angles_rng = np.random.default_rng(self.np_random.integers(2**32))
        self.observation_space = DictSpace(
            {p.name: Box(low=0.0, high=np.pi, shape=(1,)) for p in self.circuit_parameters}
        )

    def _get_obs(self):
        """
        Get the observation
        :return: Observation
        """
        return {p.name: val for p, val in self.target.context_parameters.items()}

    def _get_info(self) -> Any:
        return {}
