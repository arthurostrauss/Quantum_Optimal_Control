from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from qiskit import ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classical.types import Uint
from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.quantum_info import random_clifford, Operator

from ..base_reward import Reward
from .orbit_reward_data import ORBITRewardDataList, ORBITRewardData
from ...environment.target import GateTarget
from ...environment.configuration.qconfig import QEnvConfig
from ...helpers import causal_cone_circuit, validate_circuit_and_target


@dataclass
class ORBITReward(Reward):
    """
    Configuration for computing the reward based on ORBIT
    """

    use_interleaved: bool = False
    clifford_seed: int = 2000

    def __post_init__(self):
        self._ideal_pubs: [List[SamplerPub]] = []
        self.clifford_rng = np.random.default_rng(self.clifford_seed)

    @property
    def reward_method(self):
        return "orbit"

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.ndarray,
        target: GateTarget,
        env_config: QEnvConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> ORBITRewardDataList:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
            baseline_circuit: Ideal circuit that qc should implement

        Returns:
            List of pubs related to the reward method
        """
        if not isinstance(target, GateTarget):
            raise ValueError("ORBIT reward can only be computed for a target gate")
        execution_config = env_config.execution_config
        backend_info = env_config.backend_config
        layout = target.layout
        if baseline_circuit is not None:
            circuit_ref = baseline_circuit
        else:
            circuit_ref = qc.metadata["baseline_circuit"]
        reward_data = []
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
                ref_qc = circuit_ref.copy_empty_like(name="orbit_ref_circ")
                run_qc = qc.copy_empty_like(name="orbit_run_circ")
                for l in range(execution_config.current_n_reps):
                    r_cliff = random_clifford(target.causal_cone_size, self.clifford_rng)
                    for circ, context in zip([run_qc, ref_qc], [qc, circuit_ref]):
                        circ.compose(
                            r_cliff.to_circuit(),
                            inplace=True,
                            qubits=target.causal_cone_qubits,
                        )
                        circ.barrier()
                        circ.compose(context, inplace=True)
                        circ.barrier()

                sim_qc = causal_cone_circuit(ref_qc.decompose(), target.causal_cone_qubits)[0]
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

                transpiled_circuit = backend_info.custom_transpile(
                    run_qc, initial_layout=layout, scheduling=False
                )
                transpiled_circuit.barrier()
                transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                transpiled_circuit.measure_all()
                pub = SamplerPub.coerce((transpiled_circuit, params, execution_config.n_shots))
                reward_data.append(
                    ORBITRewardData(
                        pub,
                        target.causal_cone_qubits_indices,
                        reverse_unitary_qc,
                        execution_config.current_n_reps,
                    )
                )
                # for circ, pubs_ in zip([run_qc, ref_qc], [pubs, self._ideal_pubs]):
                #     transpiled_circuit = backend_info.custom_transpile(
                #         circ, initial_layout=layout, scheduling=False
                #     )
                #     transpiled_circuit.barrier()
                #     # Add the inverse unitary + measurement to the circuit
                #     transpiled_circuit.compose(reverse_unitary_qc, inplace=True)
                #     transpiled_circuit.measure_all()
                #     pubs_.append((transpiled_circuit, params, execution_config.n_shots))

        return ORBITRewardDataList(reward_data)

    def get_reward_with_primitive(
        self,
        reward_data: ORBITRewardDataList,
        primitive: BaseSamplerV2,
    ) -> np.ndarray:
        """
        Compute the reward based on the input pubs
        """
        pubs = reward_data.pubs
        job = primitive.run(pubs)
        pub_results = job.result()
        batch_size = pubs[0].parameter_values.shape[0]
        n_shots = pubs[0].shots
        causal_cone_qubits_indices = reward_data.causal_cone_qubits_indices
        causal_cone_size = len(causal_cone_qubits_indices)

        assert all(
            [pub.shots == n_shots for pub in pubs]
        ), "All pubs should have the same number of shots"
        assert all(
            [pub.parameter_values.shape[0] == batch_size for pub in pubs]
        ), "All pubs should have the same batch size"

        num_bits = pub_results[0].data.meas[0].num_bits
        if num_bits == causal_cone_size:
            pub_data = [
                [pub_result.data.meas[i] for i in range(batch_size)] for pub_result in pub_results
            ]
            survival_probability = [
                [bit_array.get_int_counts().get(0, 0) / n_shots for bit_array in bit_arrays]
                for bit_arrays in pub_data
            ]
        else:
            pub_data = [
                [
                    pub_result.data.meas[i].postselect(
                        causal_cone_qubits_indices,
                        [0] * causal_cone_size,
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

    def get_real_time_circuit(
        self,
        circuits: QuantumCircuit | List[QuantumCircuit],
        target: GateTarget | List[GateTarget],
        env_config: QEnvConfig,
        *args,
    ) -> QuantumCircuit:
        """
        Get the real time circuit for the given target
        """
        l = env_config.current_n_reps
        all_n_reps = env_config.n_reps

        prep_circuits = [circuits] if isinstance(circuits, QuantumCircuit) else circuits
        targets = [target] if isinstance(target, GateTarget) else target
        validate_circuit_and_target(prep_circuits, targets)
        ref_target = targets[0]
        qubits = [qc.qubits for qc in prep_circuits]
        if not all(qc.qubits == qubits[0] for qc in prep_circuits):
            raise ValueError("All circuits should have the same qubits")

        qc = prep_circuits[0].copy_empty_like(name="real_time_orbit_qc")
        num_qubits = qc.num_qubits

        if num_qubits != 1:
            raise ValueError("ORBIT reward can only be computed for a single qubit")
        if ref_target.causal_cone_size != num_qubits:
            raise ValueError("ORBIT reward can only be computed for a single qubit causal cone")

        n_reps_var = qc.add_input("n_reps", Uint(8)) if len(all_n_reps) > 1 else l

        if not qc.clbits:
            meas = ClassicalRegister(ref_target.causal_cone_size, name="meas")
            qc.add_register(meas)
        else:
            meas = qc.cregs[0]
            if meas.size != ref_target.causal_cone_size:
                raise ValueError(
                    "Classical register size should be equal to the number of qubits in the causal cone"
                )
        random_clifford()
