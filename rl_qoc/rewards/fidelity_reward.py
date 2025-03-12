from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .base_reward import Reward, Pub, Primitive, Target, GateTarget
from .. import ExecutionConfig
from ..helpers import has_noise_model, handle_n_reps
from ..environment.backend_info import BackendInfo
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
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
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
        new_qc = handle_n_reps(
            qc,
            execution_config.n_reps,
            backend_info.backend,
            execution_config.control_flow_enabled,
        )
        return [SamplerPub.coerce((new_qc, params, execution_config.n_shots))]

    def get_reward_with_primitive(
        self, pubs: List[Pub], primitive: Primitive, target: Target, **kwargs
    ) -> np.array:
        """
        Compute the reward based on the primitive and the pubs

        Args:
            pubs: List of pubs related to the reward method
            primitive: Primitive to compute the reward
            target: Target gate or state to prepare
        """
        backend_info: BackendInfo = kwargs.get("backend_info")
        if backend_info is None:
            raise ValueError("Backend information is required for computing the reward")
        if hasattr(primitive, "backend"):
            backend = primitive.backend
        elif hasattr(primitive, "_backend"):
            backend = primitive._backend
        else:
            backend = AerSimulator()

        fidelities = []
        for pub in pubs:
            qc = pub.circuit
            params = pub.parameter_values.as_array()
            fidelities.append(self.get_fidelity(qc, params, target, backend))

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
        qc_copy = handle_n_reps(qc, n_reps, backend_info.backend, False)

        backend = backend_info.backend
        if isinstance(target, GateTarget) and target.causal_cone_size <= 3:
            outputs = ["unitary", "superop"]
        else:
            outputs = ["statevector", "density_matrix"]

        if backend is None or (
            isinstance(backend, AerSimulator) and not has_noise_model(backend)
        ):
            # Ideal simulation
            noise_model = None
            # Isolate the output that is present in ideal_sim_methods from outputs
            output = [
                output for output in outputs if output in self._ideal_sim_methods
            ][0]
            getattr(qc_copy, f"save_{output}")
        else:
            # Noisy simulation
            if isinstance(backend, AerSimulator):
                noise_model = backend.options.noise_model
            else:
                noise_model = NoiseModel.from_backend(backend)

            output = [
                output for output in outputs if output in self._noisy_sim_methods
            ][0]
            getattr(qc_copy, f"save_{output}")

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
