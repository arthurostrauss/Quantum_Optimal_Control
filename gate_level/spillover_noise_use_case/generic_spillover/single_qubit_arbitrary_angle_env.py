from typing import Optional, Dict, Any

from gymnasium.spaces import Box
from qiskit import QuantumCircuit

from rl_qoc import ContextAwareQuantumEnvironment, QEnvConfig
import numpy as np

from rl_qoc.environment.context_aware_quantum_environment import ObsType

from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, coherent_unitary_error
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

gate_map = get_standard_gate_name_mapping()


def noisy_backend(circuit: QuantumCircuit, γ: float):
    rotation_angle = 0.0
    rotation_axis = None
    for inst in circuit.data:
        if inst.operation.name in ["rx", "ry", "rz"]:
            rotation_angle = inst.operation.params[0]
            rotation_axis = inst.operation.name
            break
    noisy_unitary = type(gate_map[rotation_axis])(rotation_angle * γ).to_matrix()
    noise_model = NoiseModel(
        basis_gates=["h", "rx", "rz", "t", "s", "sdg", "tdg", "u", "x", "z"]
    )
    noise_model.add_all_qubit_quantum_error(
        coherent_unitary_error(noisy_unitary), rotation_axis
    )
    config = AerBackendConfiguration(
        backend_name="overrotation_backend",
        backend_version="2",
        n_qubits=circuit.num_qubits,
        basis_gates=["h", "rx", "rz", "t", "s", "sdg", "tdg", "u", "x", "z"],
        custom_instructions=AerSimulator._CUSTOM_INSTR,
        description="",
        gates=[],
        max_shots=int(1e7),
        coupling_map=[],
    )

    return AerSimulator(configuration=config, noise_model=noise_model)


class ArbitraryAngleCoherentEnv(ContextAwareQuantumEnvironment):
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
        γ: float,
    ):
        """
        Initialize the environment
        """
        self.γ = γ
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
        backend = noisy_backend(circuit, self.γ)
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
