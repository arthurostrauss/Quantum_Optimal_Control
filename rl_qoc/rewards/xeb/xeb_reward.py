from dataclasses import dataclass, field
from typing import Literal, Union, Dict, Optional, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import UnitaryGate, RYGate, SXGate, TGate
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from ..base_reward import Reward
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import GateTarget
from .xeb_reward_data import XEBRewardDataList, XEBRewardData
from ...helpers import causal_cone_circuit


@dataclass
class XEBReward(Reward):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    xeb_fidelity_type: Literal["log", "linear"] = "linear"
    gate_set_choice: Union[Literal["sw", "t"], Dict[int, Gate]] = "sw"
    gate_set_seed: int = 2000
    gate_set_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.gate_set_rng = np.random.default_rng(self.gate_set_seed)
        if isinstance(self.gate_set_choice, str):
            sy = RYGate(np.pi / 2)
            if self.gate_set_choice == "sw":
                sw = UnitaryGate(
                    np.array([[1, -np.sqrt(1j)], [np.sqrt(-1j), 1]]) / np.sqrt(2), "sw"
                )
                self.gate_set_choice = {0: SXGate(), 1: sy, 2: sw}
            elif self.gate_set_choice == "t":
                self.gate_set_choice = {0: SXGate(), 1: sy, 2: TGate()}

            else:
                raise ValueError("Invalid gate set choice")
        if not self.xeb_fidelity_type in ["log", "linear"]:
            raise ValueError("Invalid fidelity type, choose between 'log' and 'linear'")

    @property
    def reward_args(self):
        return {
            "gate_set_choice": self.gate_set_choice,
            "gate_set_seed": self.gate_set_seed,
        }

    def set_reward_seed(self, seed: int):
        """
        Set the seed for the random number generator
        """
        self.gate_set_seed = seed + 908
        self.gate_set_rng = np.random.default_rng(self.gate_set_seed)

    @property
    def reward_method(self):
        return "xeb"

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        env_config: QEnvConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> XEBRewardDataList:
        """
        Compute pubs related to the reward method
        """
        depth = env_config.current_n_reps
        seqs = env_config.sampling_paulis
        n_shots = env_config.n_shots
        num_qubits = qc.num_qubits

        if baseline_circuit is not None:
            circuit_ref = baseline_circuit
        else:
            try:
                circuit_ref = qc.metadata["baseline_circuit"]
            except KeyError:
                raise ValueError("No baseline circuit defined")

        reward_data = []
        gate_choice = len(self.gate_set_choice)
        sq_gates = np.zeros((seqs, depth, num_qubits), dtype=int)

        for s in range(seqs):
            for q in range(num_qubits):
                sq_gates[s, 0, q] = self.gate_set_rng.integers(gate_choice)
                for d in range(1, depth):
                    choices = list(set(range(gate_choice)) - {sq_gates[s, d - 1, q]})
                    sq_gates[s, d, q] = self.gate_set_rng.choice(choices)

        for s in range(seqs):
            run_qc = qc.copy_empty_like(name=f"xeb_circ_{s}")
            ref_qc = qc.copy_empty_like(name=f"xeb_ref_circ_{s}")
            for d in range(depth):
                for circ, context in zip([run_qc, ref_qc], [qc, circuit_ref]):
                    for q, qubit in enumerate(qc.qubits):
                        circ.append(self.gate_set_choice[sq_gates[s, d, q]], [qubit])
                    circ.barrier()
                    circ.compose(context, inplace=True)
                    circ.barrier()

            sim_qc = causal_cone_circuit(ref_qc, target.causal_cone_qubits)[0]
            statevector = Statevector(sim_qc)
            run_qc.measure_all()
            transpiled_circuit = env_config.backend_info.custom_transpile(
                run_qc,
                initial_layout=target.layout,
                scheduling=False,
                remove_final_measurements=False,
            )
            reward_data.append(
                XEBRewardData(
                    (transpiled_circuit, params, n_shots),
                    statevector,
                    target.causal_cone_qubits_indices,
                )
            )

        return XEBRewardDataList(reward_data)

    def get_reward_with_primitive(
        self,
        reward_data: XEBRewardDataList,
        primitive: BaseSamplerV2,
    ) -> np.array:
        pubs = reward_data.pubs
        job = primitive.run(pubs)

        results = job.result()
        batch_size = pubs[0].parameter_values.shape[0]
        n_shots = pubs[0].shots
        causal_cone_qubits_indices = reward_data.causal_cone_qubits_indices
        causal_cone_size = len(causal_cone_qubits_indices)
        num_bits = results[0].data.meas[0].num_bits
        if num_bits == causal_cone_size:
            # No post-selection based on causal cone
            pub_data = [
                [pub_result.data.meas[i] for i in range(batch_size)] for pub_result in results
            ]
            experimental_probabilities = [
                [
                    {key: val / n_shots for key, val in bit_array.get_counts().items()}
                    for bit_array in bit_arrays
                ]
                for bit_arrays in pub_data
            ]
            # Complete the potential missing keys of the counts dictionaries (in case a bitstring was not sampled)
            for i in range(len(pub_data)):
                for j in range(len(pub_data[i])):
                    for key in [bin(k)[2:].zfill(num_bits) for k in range(2**num_bits)]:
                        if key not in experimental_probabilities[i][j]:
                            experimental_probabilities[i][j][key] = 0

            experimental_probabilities = [
                [
                    np.array(
                        [
                            experimental_probabilities[i][j][bin(k)[2:].zfill(num_bits)]
                            for k in range(2**num_bits)
                        ]
                    )
                    for j in range(len(pub_data[i]))
                ]
                for i in range(len(pub_data))
            ]
            xeb_fidelities = []
            for i in range(len(pub_data)):
                xeb_fidelities.append(
                    [
                        self.compute_xeb_fidelity(experimental_probs, reward_data[i].state)
                        for experimental_probs in experimental_probabilities[i]
                    ]
                )

            return np.mean(xeb_fidelities, 0)

    def compute_xeb_fidelity(self, experimental_probs, state: Statevector):
        """
        Compute XEB fidelity estimator based on fidelity type
        """
        incoherent_dist = np.ones(state.dim) / state.dim
        expected_probs = state.probabilities(decimals=4)
        if self.xeb_fidelity_type == "log":
            return compute_log_fidelity(incoherent_dist, expected_probs, experimental_probs)
        elif self.xeb_fidelity_type == "linear":
            e_u = np.sum(expected_probs**2)
            u_u = np.sum(expected_probs) / state.dim
            m_u = np.sum(expected_probs * experimental_probs)
            x = e_u - u_u
            y = m_u - u_u
            return x * y / x**2


def cross_entropy(p, q, epsilon=1e-15):
    """
    Calculate cross entropy between two probability distributions.

    Parameters:
    - p: numpy array, the true probability distribution
    - q: numpy array, the predicted probability distribution
    - epsilon: small value to avoid taking the logarithm of zero

    Returns:
    - Cross entropy between p and q
    """
    q = np.maximum(q, epsilon)  # Avoid taking the logarithm of zero
    x_entropy = -np.sum(p * np.log(q))
    return x_entropy


def compute_log_fidelity(incoherent_dist, expected_probs, measured_probs):
    """
    Compute the log fidelity between the expected and measured distributions.

    Parameters:
    - incoherent_dist: numpy array, the incoherent distribution
    - expected_probs: numpy array, the expected probabilities
    - measured_probs: numpy array, the measured probabilities

    Returns:
    - The log fidelity between the expected and measured distributions
    """
    # Compute the cross entropy between the incoherent distribution and the expected probabilities
    xe_incoherent = cross_entropy(incoherent_dist, expected_probs)
    xe_measured = cross_entropy(measured_probs, expected_probs)
    xe_expected = cross_entropy(expected_probs, expected_probs)

    f_xeb = (xe_incoherent - xe_measured) / (xe_incoherent - xe_expected)
    return f_xeb
