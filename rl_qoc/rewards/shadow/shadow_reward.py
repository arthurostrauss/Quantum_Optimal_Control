from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import Operator, SparsePauliOp

from .shadow_reward_data import ShadowRewardData, ShadowRewardDataList
from ...environment.target import GateTarget, StateTarget
from ...environment.configuration.qconfig import QEnvConfig
from ..base_reward import Reward
from ...helpers.circuit_utils import (
    handle_n_reps,
    causal_cone_circuit,
    get_input_states_cardinality_per_qubit,
)
from qiskit_aer import AerSimulator


@dataclass
class ShadowReward(Reward):
    """
    Configuration for computing the reward based on (Shadow) 
    all data in GRD and all calculations in GRWP
    """
    unitary_seed: int = 2000
    unitary_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.unitary_rng = np.random.default_rng(self.unitary_seed)

    @property
    def reward_args(self):
        return {"unitary_seed": self.unitary_seed}

    @property
    def reward_method(self):
        return "shadow"

    def set_reward_seed(self, seed: int):
        """
        Set the seed for the random number generator
        """
        self.unitary_seed = seed + 357
        self.unitary_rng = np.random.default_rng(self.unitary_seed)

    
    def get_reward_data(
        self,
        qc: QuantumCircuit, #simulate imperfect gate
        params: np.ndarray,
        target: StateTarget | GateTarget,
        env_config: QEnvConfig,
    ) -> ShadowRewardDataList:
        """
        Compute pubs related to the reward method

        Args: 
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
            baseline_circuit: Ideal circuit that qc should implement
        """
   
        
        backend_info = env_config.backend_info
        shadow_size = env_config.execution_config.sampling_paulis
        
        num_qubits = len(target.physical_qubits) if isinstance(target, StateTarget) else target.causal_cone_size
        reward_data = [] 

        if isinstance(target, StateTarget):
            unitary_ids = self.unitary_rng.choice(3, size=(shadow_size, num_qubits))
            unique_unitaries, counts = np.unique(unitary_ids, axis=0, return_counts=True)

            for unitary, shots in zip(unique_unitaries, counts):
                qc_copy = qc.copy()    
                for qubit, id in enumerate(unitary):
                    if id == 0: # X basis 
                        qc_copy.h(qubit)
                    elif id == 1 : # Y basis 
                        qc_copy.sdg(qubit)
                        qc_copy.h(qubit)
                    
                qc_copy.measure_all()
                circuit = backend_info.custom_transpile(qc_copy,
                                                        remove_final_measurements=False,
                                                        initial_layout=target.layout,
                                                        scheduling=False, 
                                                        optimization_level=0)
            
                pub = (circuit, params, shots)  
                
                reward_data.append(
                    ShadowRewardData(
                        pub,
                        unitary=unitary,
                        u_in=None,
                        b_in=None
                    )
                )
        else:
            # TODO: implement Shadow tomography for channel
            raise NotImplementedError("Shadow tomography for channel is not implemented yet")
        
        return ShadowRewardDataList(reward_data)

    def get_reward_with_primitive(
        self,
        reward_data: ShadowRewardDataList,
        primitive: BaseSamplerV2,
        target: StateTarget
    ) -> np.ndarray:
        
        shadow_size = reward_data.shadow_size
        job = primitive.run(reward_data.pubs)
        pub_results = job.result()
        batch_size = reward_data.pubs[0].parameter_values.shape[0]
        
        total_data_list = []
        for k in range(batch_size):
            for i in range(len(pub_results)):
                unique_unitary = reward_data[i].unitary
                unique_unitary = unique_unitary[::-1]
                unique_pub_res = pub_results[i]
                bitstring_counts = unique_pub_res.data.meas.get_counts(loc=k)
                b = list(bitstring_counts.keys())

                # Convert each bitstring to a list of ints
                bitstrings = [[int(bit) for bit in bitstring] for bitstring in b]

                #List of counts corresponding to each bitstring
                counts = list(bitstring_counts.values())

                for j in range(len(counts)):
                    for count in range(int(counts[j])):
                        total_data_list.append([unique_unitary, bitstrings[j]])   #repeat counts[j] times for every individual b (measurement outcome)

        observable_decomp = SparsePauliOp.from_operator(Operator(target.dm))
        partition = int(2 * np.log(2*len(observable_decomp.paulis)/0.01))
        pauli_coeff = observable_decomp.coeffs   #to also be used in shadow bound
        pauli_str = observable_decomp.paulis
        pauli_str_num = []
        mapping = {'X': 0, 'Y': 1, 'Z': 2, 'I': 3}

        for pauli in pauli_str:
            term_str = pauli.to_label()  # e.g., 'IZ'
            term_list = [mapping[c] for c in term_str]
            pauli_str_num.append(term_list)
        # Here, we let X, Y, Z, I = 0, 1, 2, 3

        #print(pauli_coeff, pauli_str)

        reward = np.zeros(batch_size)
        for i in range(batch_size):
            data_list = total_data_list[i*shadow_size:(i+1)*shadow_size]  #structure: [[u1, b11], [u2, b12], ...]
            U_list = [var[0] for var in data_list]
            b_list = [var[1] for var in data_list]
            shadow = (b_list, U_list)
            
            exp_vals = [estimate_shadow_obervable(shadow, pauli_str_num[obs], partition) for obs in range(len(pauli_str))]
            reward_i = np.dot(pauli_coeff, exp_vals)
            assert np.imag(reward_i) - 1e-10 < 0, "Reward is complex"
            reward[i] = reward_i.real
            
        return reward

def shadow_bound(error, observables, failure_rate=0.01):

    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error ** 2
    return int(np.ceil(N * K)), int(K), M


def estimate_shadow_obervable(shadow, observable, k):
    
    """
    Goal: From measurement_list and unitary_ids, split into N/k groups, find mean for each, then find median of means
    
    By the formulation in Pennylane notes, if the observable is not matching the unitary, the entire expectation value goes to 0 for that shadow.
    For example; 5 qubits, observable is [3, 0, 0, 3, 3] which correspond to IXXII
    And my unitary id (shadow) is [2, 0, 0, 3, 1].
    Then Tr(Orho) will give non zero value. Else, we will get 0 for the entire shadow.
    """
    shadow = np.array(shadow)
    shadow_size, num_qubits = shadow[0].shape
    b_lists, obs_lists = shadow
    shuffle_indices = np.random.permutation(b_lists.shape[0])
    b_shuffled = b_lists[shuffle_indices]
    obs_shuffled = obs_lists[shuffle_indices]

    means = []
    P = observable

    # loop over the splits of the shadow:
    for i in range(k):       # shadow_size // k = no of elements in each set; k = no of sets

        # assign the splits 
        start = i * (shadow_size //k)
        end = (i+1) * (shadow_size //k)
        b_lists_k, obs_lists_k = (
            b_shuffled[start: end],
            obs_shuffled[start: end],
        )

        exp_val = np.zeros(shadow_size // k)
        
        for n in range(shadow_size // k):
            
            b = b_lists_k[n]
            U = obs_lists_k[n]
            
            f = []

            for m in range(num_qubits):
                
                if P[m] == 3:
                    f.append(1/3)       #important. if P = I, there is additional Tr(I) = 2 in the sum now. We must compensate by changing to 1/3.
                elif P[m] == U[m] and b[m] == 0:
                    f.append(1)
                elif P[m] == U[m] and b[m] == 1:
                    f.append(-1)
                else:
                    f.append(0)
            
            exp_val[n] = (3**num_qubits) * np.prod(np.array(f))
        means.append(np.sum(exp_val)/(shadow_size // k))   

    return np.median(means) 

