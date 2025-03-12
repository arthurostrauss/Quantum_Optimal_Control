from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .base_reward import Reward, Pub, Primitive, Target, GateTarget
from ..helpers import has_noise_model
from ..environment.backend_info import BackendInfo


@dataclass
class FidelityReward(Reward):
    """
    Configuration for computing the reward based on fidelity estimation
    """

    @property
    def reward_method(self):
        return "fidelity"

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
        if n_reps > 1:
            qc_copy = qc_copy.repeat(n_reps).decompose()
        backend = backend_info.backend
        ideal_sim_methods = ["statevector", "unitary"]
        noisy_sim_methods = ["density_matrix", "superop"]
        outputs = ["statevector", "unitary", "density_matrix", "superop"]
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
            output = [output for output in outputs if output in ideal_sim_methods][0]
            getattr(qc_copy, f"save_{output}")
        else:
            # Noisy simulation
            if isinstance(backend, AerSimulator):
                noise_model = backend.options.noise_model
            else:
                noise_model = NoiseModel.from_backend(backend)

            output = [output for output in outputs if output in noisy_sim_methods][0]
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
