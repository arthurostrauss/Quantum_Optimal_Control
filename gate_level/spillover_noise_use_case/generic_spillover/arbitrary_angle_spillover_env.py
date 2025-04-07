from typing import Optional, Dict, Any

from gymnasium.spaces import Box
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
        unbound_circuit_context: QuantumCircuit,
        gamma_matrix: np.ndarray,
    ):
        """
        Initialize the environment
        """
        self.gamma_matrix = gamma_matrix
        self.circuit_parameters = unbound_circuit_context.parameters

        super().__init__(q_env_config, unbound_circuit_context)
        self._rotation_angles_rng = np.random.default_rng(
            self.np_random.integers(2**32)
        )
        self.observation_space = Box(
            low=np.array([0.0] * len(self.circuit_parameters)),
            high=np.array([2 * np.pi] * len(self.circuit_parameters)),
            dtype=np.float32,
        )

    # def define_target_and_circuits(self):
    #     """
    #     Define the target gate and the circuits to be executed
    #     """
    #     # original_context = self.circuit_context
    #     # circuit_context = causal_cone_circuit(
    #     #     self.circuit_context, list(self.config.env_metadata["target_subsystem"])
    #     # )[0]
    #     # self._physical_target_qubits = list(range(circuit_context.num_qubits))
    #     # self._circuit_context = circuit_context
    #     _, custom_circuits, baseline_circuits = (
    #         super().define_target_and_circuits()
    #     )
    #     layouts = [Layout({self.circuit_context.qubits[q]: i for i, q in enumerate(self.config.env_metadata["target_subsystem"])})
    #                for _ in range(len(baseline_circuits))]
    #     input_states_choice = getattr(
    #         self.config.reward.reward_args, "input_states_choice", "pauli4"
    #     )
    #     target = [
    #         GateTarget(
    #             self.config.target.gate,
    #             [0, 1],
    #             causal_cone_circuit(baseline_circuit, list(self.config.env_metadata["target_subsystem"]))[0],
    #             self.circ_tgt_register,
    #             layout,
    #             input_states_choice=input_states_choice,
    #         )
    #         for baseline_circuit, layout in zip(baseline_circuits, layouts)
    #     ]
    #
    #     return target, custom_circuits, baseline_circuits

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
        phi, info = super().reset(seed=seed, options=options)

        # phi = np.random.uniform(0, 2 * np.pi, self.unbound_circuit_context.num_qubits)

        param_dict = {self.circuit_parameters[i].name: phi[i] for i in range(len(phi))}
        circuit = self.unbound_circuit_context.assign_parameters(param_dict)
        backend = noisy_backend(
            circuit, self.gamma_matrix, self.config.env_metadata["target_subsystem"]
        )
        # Generate the initial observation
        self.set_circuit_context(None, backend=backend, **param_dict)
        obs = phi
        print("Sampled angles: ", obs)
        # Return the initial observation and info
        return obs, {}

    def _get_obs(self):
        """
        Get the observation
        :return: Observation
        """
        phi = self._rotation_angles_rng.uniform(
            0,
            2 * np.pi,
            self.unbound_circuit_context.num_qubits * self.tgt_instruction_counts,
        )
        return phi

    def _get_info(self) -> Any:
        return {}
