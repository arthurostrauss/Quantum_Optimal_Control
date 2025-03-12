from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2

from .base_reward import Reward
from ..environment.backend_info import BackendInfo
from ..environment.target import GateTarget
from ..environment.configuration.execution_config import ExecutionConfig
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.quantum_info import random_clifford, Operator

from ..helpers import causal_cone_circuit


@dataclass
class ORBITReward(Reward):
    """
    Configuration for computing the reward based on ORBIT
    """

    use_interleaved: bool = False
    clifford_seed: int = 2000

    def __post_init__(self):
        super().__post_init__()
        self._ideal_pubs: [List[SamplerPub]] = []
        self.clifford_rng = np.random.default_rng(self.clifford_seed)

    @property
    def reward_method(self):
        return "orbit"

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> List[SamplerPub]:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            baseline_circuit: Ideal circuit that qc should implement

        Returns:
            List of pubs related to the reward method
        """
        if not isinstance(target, GateTarget):
            raise ValueError("ORBIT reward can only be computed for a target gate")

        layout = target.layout
        if baseline_circuit is not None:
            circuit_ref = baseline_circuit
        else:
            circuit_ref = qc.metadata["baseline_circuit"]
        pubs, ideal_pubs, total_shots = [], [], 0
        batch_size = params.shape[0]

        if self.use_interleaved:
            # try:
            #     Clifford(circuit_ref)
            # except QiskitError as e:
            #     raise ValueError(
            #         "Circuit should be a Clifford circuit for using interleaved RB directly"
            #     ) from e
            # ref_element = circuit_ref.to_gate(label="ref_circ")
            # custom_element = qc.to_gate(label="custom_circ")
            # exp = InterleavedRB(
            #     ref_element,
            #     target.causal_cone_qubits,
            #     [execution_config.current_n_reps],
            #     backend_info.backend,
            #     execution_config.sampling_paulis,
            #     execution_config.seed,
            #     circuit_order="RRRIII",
            # )
            # # ref_circuits = exp.circuits()[0: self.n_reps]
            # interleaved_circuits = exp.circuits()[execution_config.current_n_reps:]
            # run_circuits = [
            #     substitute_target_gate(circ, ref_element, custom_element)
            #     for circ in interleaved_circuits
            # ]
            # run_circuits = backend_info.custom_transpile(
            #     run_circuits,
            #     initial_layout=layout,
            #     scheduling=False,
            #     remove_final_measurements=False,
            # )
            # pubs = [(circ, params, execution_config.n_shots) for circ in run_circuits]
            # total_shots += batch_size * execution_config.n_shots * len(pubs)
            # self._ideal_pubs = [
            #     (circ, params, execution_config.n_shots) for circ in interleaved_circuits
            # ]
            pass
        else:
            for seq in range(execution_config.sampling_paulis):
                ref_qc = QuantumCircuit.copy_empty_like(
                    circuit_ref,
                    name="orbit_ref_circ",
                )
                run_qc = QuantumCircuit.copy_empty_like(qc, name="orbit_run_circ")
                for l in range(execution_config.current_n_reps):
                    r_cliff = random_clifford(qc.num_qubits, self.clifford_rng)
                    for circ, context in zip([run_qc, ref_qc], [qc, circuit_ref]):
                        circ.compose(r_cliff.to_circuit(), inplace=True)
                        circ.barrier()
                        circ.compose(context, inplace=True)
                        circ.barrier()

                sim_qc = causal_cone_circuit(
                    ref_qc.decompose(), target.causal_cone_qubits
                )[0]
                reverse_unitary = Operator(sim_qc).adjoint()
                reverse_unitary_qc = QuantumCircuit.copy_empty_like(run_qc)
                reverse_unitary_qc.unitary(
                    reverse_unitary, target.causal_cone_qubits, label="U_inv"
                )

                reverse_unitary_qc = backend_info.custom_transpile(
                    reverse_unitary_qc,
                    initial_layout=layout,
                    scheduling=False,
                    optimization_level=3,
                    remove_final_measurements=False,
                )  # Try to get the smallest possible circuit for the reverse unitary

                for circ, pubs_ in zip([run_qc, ref_qc], [pubs, self._ideal_pubs]):
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

        return [SamplerPub.coerce(pub) for pub in pubs]

    def get_reward_with_primitive(
        self,
        pubs: List[SamplerPub],
        primitive: BaseSamplerV2,
        target: GateTarget,
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
