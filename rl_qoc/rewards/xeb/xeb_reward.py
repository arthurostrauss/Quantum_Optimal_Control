from dataclasses import dataclass, field
from typing import Literal, Union, Dict, Optional

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
from ...helpers.circuit_utils import get_parallel_gate_combinations as gate_combinations


@dataclass
class XEBReward(Reward):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    gate_set_choice: Union[Literal["sw", "t"], Dict[int, Gate]] = "sw"
    gate_set_seed: int = 2000
    gate_set_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.gate_set_rng = np.random.default_rng(self.gate_set_seed)
        if isinstance(self.gate_set_choice, str):
            sy = RYGate(np.pi / 2).to_matrix()
            if self.gate_set_choice == "sw":
                sw = UnitaryGate(
                    np.array([[1, -np.sqrt(1j)], [np.sqrt(-1j), 1]]) / np.sqrt(2), "sw"
                )
                self.gate_set_choice = {0: SXGate(), 1: sy, 2: sw}
            elif self.gate_set_choice == "t":
                self.gate_set_choice = {0: SXGate(), 1: sy, 2: TGate()}

            else:
                raise ValueError("Invalid gate set choice")

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
        coupling_map = env_config.backend.coupling_map
        if coupling_map is None:
            coupling_map = CouplingMap.from_full(num_qubits)

        if baseline_circuit is not None:
            circuit_ref = baseline_circuit
        else:
            try:
                circuit_ref = qc.metadata["baseline_circuit"]
            except KeyError:
                raise ValueError("No baseline circuit defined")

        reward_data = []
        gate_choice = len(self.gate_set_choice)
        sq_gates = np.empty((seqs, depth, num_qubits), dtype=int)
        two_qubit_gate_pattern = 0

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
                run_qc, initial_layout=target.layout, scheduling=False
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
        raise NotImplementedError
