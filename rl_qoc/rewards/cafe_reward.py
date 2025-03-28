from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub
from ..environment.target import GateTarget
from ..environment.configuration.qconfig import QEnvConfig
from .base_reward import Reward
from ..helpers.circuit_utils import handle_n_reps, causal_cone_circuit
from qiskit_aer import AerSimulator


@dataclass
class CAFEReward(Reward):
    """
    Configuration for computing the reward based on Context-Aware Fidelity Estimation (CAFE)
    """

    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"

    def __post_init__(self):
        super().__post_init__()
        self._ideal_pubs: Optional[List[SamplerPub]] = None

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}

    @property
    def reward_method(self):
        return "cafe"

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        env_config: QEnvConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> List[SamplerPub]:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
            baseline_circuit: Ideal circuit that qc should implement
        """
        if not isinstance(target, GateTarget):
            raise ValueError("CAFE reward can only be computed for a target gate")
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info
        pubs, ideal_pubs, total_shots = [], [], 0
        if baseline_circuit is not None:
            circuit_ref = baseline_circuit.copy()
        else:
            circuit_ref = qc.metadata["baseline_circuit"].copy()
        layout = target.layout
        batch_size = params.shape[0]

        # samples, shots = np.unique(
        #     np.random.choice(len(input_circuits), self.sampling_Pauli_space),
        #     return_counts=True,
        # )
        # for sample, shot in zip(samples, shots):
        for input_state in target.input_states:
            run_qc = QuantumCircuit.copy_empty_like(
                qc, name="cafe_circ"
            )  # Circuit with custom target gate
            ref_qc = QuantumCircuit.copy_empty_like(
                circuit_ref, name="cafe_ref_circ"
            )  # Circuit with reference gate

            for circuit, context, control_flow in zip(
                [run_qc, ref_qc],
                [qc, circuit_ref],
                [execution_config.control_flow_enabled, False],
            ):
                # Bind input states to the circuits
                circuit.compose(
                    input_state.circuit,
                    qubits=target.causal_cone_qubits,
                    inplace=True,
                )
                circuit.barrier()
                cycle_circuit = handle_n_reps(
                    context,
                    execution_config.current_n_reps,
                    backend_info.backend,
                    control_flow=control_flow,
                )
                circuit.compose(cycle_circuit, inplace=True)

            # Compute inverse unitary for reference circuit
            sim_qc = causal_cone_circuit(ref_qc.decompose(), target.causal_cone_qubits)[
                0
            ]
            sim_qc.save_unitary()
            backend = (
                backend_info.backend
                if isinstance(backend_info.backend, AerSimulator)
                else AerSimulator()
            )
            sim_unitary = (
                backend.run(sim_qc, noise_model=None, method="unitary")
                .result()
                .get_unitary()
            )

            reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
            reverse_unitary_qc.unitary(
                sim_unitary.adjoint(),  # Inverse unitary
                target.causal_cone_qubits,
                label="U_inv",
            )
            reverse_unitary_qc = backend_info.custom_transpile(
                reverse_unitary_qc,
                initial_layout=layout,
                scheduling=False,
                optimization_level=3,  # Find smallest circuit implementing inverse unitary
                remove_final_measurements=False,
            )

            # Bind inverse unitary + measurement to run circuit
            for circ, pubs_ in zip([run_qc, ref_qc], [pubs, ideal_pubs]):
                transpiled_circuit = backend_info.custom_transpile(
                    circ, initial_layout=layout, scheduling=False
                )
                transpiled_circuit.barrier()
                # Add the inverse unitary + measurement to the circuit
                transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                transpiled_circuit.measure_all()
                pubs_.append((transpiled_circuit, params, execution_config.n_shots))
            total_shots += batch_size * execution_config.n_shots

        self._total_shots = total_shots
        self._ideal_pubs = ideal_pubs

        return [SamplerPub.coerce(pub) for pub in pubs]

    def get_reward_with_primitive(
        self,
        pubs: List[SamplerPub],
        primitive: BaseSamplerV2,
        target: GateTarget,
        env_config: QEnvConfig,
        **kwargs
    ) -> np.array:
        """
        Compute the reward based on the input pubs
        """
        job = primitive.run(pubs)
        pub_results = job.result()
        batch_size = pubs[0].parameter_values.shape[0]
        n_shots = pubs[0].shots
        assert all(
            [pub.shots == n_shots for pub in pubs]
        ), "All pubs should have the same number of shots"
        assert all(
            [pub.parameter_values.shape[0] == batch_size for pub in pubs]
        ), "All pubs should have the same batch size"
        num_bits = pub_results[0].data.meas[0].num_bits
        if num_bits == target.causal_cone_size:
            # No post-selection based on causal cone
            pub_data = [
                [pub_result.data.meas[i] for i in range(batch_size)]
                for pub_result in pub_results
            ]
            survival_probability = [
                [
                    bit_array.get_int_counts().get(0, 0) / n_shots
                    for bit_array in bit_arrays
                ]
                for bit_arrays in pub_data
            ]
        else:
            # Post-select based on causal cone qubits
            pub_data = [
                [
                    pub_result.data.meas[i].postselect(
                        target.causal_cone_qubits_indices,
                        [0] * target.causal_cone_size,
                    )
                    for i in range(batch_size)
                ]
                for pub_result in pub_results
            ]
            survival_probability = [
                [bit_array.num_shots / n_shots for bit_array in bit_arrays]
                for bit_arrays in pub_data
            ]
        reward = np.mean(survival_probability, axis=0)
        return reward

    def get_shot_budget(self, pubs: List[SamplerPub]) -> int:
        """
        Retrieve number of shots associated to the input pub list
        """
        return sum([pub.shots * pub.parameter_values.shape[0] for pub in pubs])
