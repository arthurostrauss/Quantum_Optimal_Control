from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .base_reward import Reward, Pub, Primitive, Target, GateTarget
from ..environment.configuration.qconfig import QEnvConfig
from ..environment.backend_info import BackendInfo
from ..helpers import has_noise_model, handle_n_reps
from qiskit.primitives.containers.sampler_pub import SamplerPub


@dataclass
class FidelityReward(Reward):
    """
    Configuration for computing the reward based on fidelity estimation
    """

    _ideal_sim_methods = ("statevector", "unitary")
    _noisy_sim_methods = ("density_matrix", "superop")

    @property
    def reward_method(self):
        return "fidelity"

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: Target,
        env_config: QEnvConfig,
        *args,
    ) -> List[Pub]:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
        """
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info
        new_qc = handle_n_reps(
            qc,
            execution_config.current_n_reps,
            backend_info.backend,
            execution_config.control_flow_enabled,
        )
        new_qc = backend_info.custom_transpile(
            new_qc,
            optimization_level=0,
            initial_layout=target.layout,
            scheduling=False,
            remove_final_measurements=False,
        )
        new_qc.metadata["n_reps"] = execution_config.current_n_reps

        return [SamplerPub.coerce((new_qc, params, execution_config.n_shots))]

    def get_reward_with_primitive(
        self,
        pubs: List[Pub],
        primitive: Primitive,
        target: Target,
        env_config: QEnvConfig,
        **kwargs,
    ) -> np.array:
        """
        Compute the reward based on the primitive and the pubs

        Args:
            pubs: List of pubs related to the reward method
            primitive: Primitive to compute the reward
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
        """
        backend_info = env_config.backend_info

        fidelities = []
        for pub in pubs:
            qc = pub.circuit
            params = pub.parameter_values.as_array()
            try:
                n_reps = qc.metadata["n_reps"]
            except KeyError:
                raise ValueError(
                    "Number of repetitions is required for computing the reward"
                )
            if qc.calibrations:
                fidelities.append(
                    self.get_pulse_fidelity(
                        qc, params, target, backend_info, n_reps=n_reps
                    )
                )
            else:
                fidelities.append(
                    self.get_fidelity(qc, params, target, backend_info, n_reps=n_reps)
                )

        return np.mean(fidelities, axis=0)

    def get_fidelity(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: Target,
        backend_info: BackendInfo,
        n_reps: Optional[int] = 1,
    ):
        """
        Compute the fidelity (or fidelities) between the circuit (binded with the parameters) and the target.
        If the specified target is a state, the state fidelity with respect to the target is computed.
        If the specified target is a gate, the average gate fidelity with respect to the target is computed.

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend info containing the noise model, and transpiler passes
            n_reps: Number of repetitions to compute the fidelity
        """
        qc_copy = qc.copy()
        backend = backend_info.backend
        is_aer_sim = isinstance(backend, AerSimulator)
        if isinstance(target, GateTarget) and target.causal_cone_size <= 3:
            outputs = ["unitary", "superop"]
        else:
            outputs = ["statevector", "density_matrix"]

        if backend is None or (is_aer_sim and not has_noise_model(backend)):
            # Ideal simulation
            noise_model = None
            # Isolate the output that is present in ideal_sim_methods from outputs
            output = next(
                output for output in outputs if output in self._ideal_sim_methods
            )

        else:
            # Noisy simulation
            noise_model = (
                backend.options.noise_model
                if is_aer_sim
                else NoiseModel.from_backend(backend)
            )
            output = next(
                output for output in outputs if output in self._noisy_sim_methods
            )

        getattr(qc_copy, f"save_{output}")()  # Aer Save method

        circuit = backend_info.custom_transpile(
            qc_copy,
            optimization_level=0,
            initial_layout=target.layout,
            scheduling=False,
            remove_final_measurements=False,
        )
        parameters = circuit.parameters
        parameter_binds = [
            {parameters[j]: params[:, j] for j in range(len(parameters))}
        ]
        if backend is None:
            backend = AerSimulator()
        job = backend.run(
            circuit,
            parameter_binds=parameter_binds,
            method=output,
            noise_model=noise_model,
        )
        result = job.result()
        outputs = [result.data(i)[output] for i in range(len(params))]
        fidelities = [target.fidelity(exp, n_reps) for exp in outputs]

        return fidelities[0] if len(fidelities) == 1 else fidelities

    def get_pulse_fidelity(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: Target,
        backend_info: BackendInfo,
        n_reps: Optional[int] = 1,
    ):
        """
        Compute the fidelity (or fidelities) between the circuit (binded with the parameters) and the target.
        If the specified target is a state, the state fidelity with respect to the target is computed.
        If the specified target is a gate, the average gate fidelity with respect to the target is computed.

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend info containing the noise model, and transpiler passes
            n_reps: Number of repetitions to compute the fidelity
        """
        from qiskit_dynamics.backend import DynamicsBackend
        from ..custom_jax_sim import PulseEstimatorV2

        backend = backend_info.backend
        if not isinstance(backend, DynamicsBackend):
            raise ValueError(
                "Pulse fidelity can only be computed for a DynamicsBackend"
            )
        target_type = target.target_type
        # Filter out qubits with dimension 1 (trivial dimension stated for DynamicsBackend)
        subsystem_dims = list(filter(lambda x: x > 1, backend.options.subsystem_dims))
        qc_copy = handle_n_reps(qc, n_reps, backend, False)

        if target_type == "state":
            y0 = Statevector.from_int(0, dims=subsystem_dims)
        else:
            y0 = Operator(
                np.eye(int(np.prod(subsystem_dims))),
                input_dims=tuple(subsystem_dims),
                output_dims=tuple(subsystem_dims),
            )
