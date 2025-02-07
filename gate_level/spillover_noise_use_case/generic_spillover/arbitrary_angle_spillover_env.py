from typing import Optional, Dict, Any

from rl_qoc import ContextAwareQuantumEnvironment, QEnvConfig
import numpy as np
from .spillover_effect_on_subsystem import (
    noisy_backend,
    circuit_context,
    LocalSpilloverNoiseAerPass,
    numpy_to_hashable,
)
from rl_qoc.environment.context_aware_quantum_environment import ObsType
from typing import List, Literal
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.circuit import ParameterVector


class ArbitraryAngleSpilloverEnv(ContextAwareQuantumEnvironment):

    def __init__(
        self,
        q_env_config: QEnvConfig,
        rotation_axes: List[Literal["rx", "ry", "rz"]],
        coupling_map: CouplingMap,
        gamma_matrix: np.ndarray,
    ):
        """
        Initialize the environment
        """
        self.rotation_axes = rotation_axes
        self.gamma_matrix = gamma_matrix
        self.circuit_parameters = ParameterVector("θ", len(rotation_axes))

        super().__init__(
            q_env_config,
            circuit_context(
                len(rotation_axes), rotation_axes, self.circuit_parameters, coupling_map
            ),
        )

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
        super().reset(seed=seed, options=options)

        phi = np.random.uniform(0, 2 * np.pi, len(self.rotation_axes))

        param_dict = {self.circuit_parameters[i]: phi[i] for i in range(len(phi))}
        circuit = self.circuit_context.assign_parameters(param_dict)
        backend = noisy_backend(
            circuit, self.gamma_matrix, self.config.env_metadata["target_subsystem"]
        )
        pm = PassManager(
            [
                LocalSpilloverNoiseAerPass(
                    numpy_to_hashable(self.gamma_matrix),
                    self.config.env_metadata["target_subsystem"],
                )
            ]
        )
        circuit = pm.run(circuit)
        # Generate the initial observation
        self.set_circuit_context(circuit, backend)
        obs = phi
        # Return the initial observation and info
        return obs, {}
