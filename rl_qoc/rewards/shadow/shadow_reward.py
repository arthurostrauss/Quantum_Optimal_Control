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
        target: StateTarget,
        shadow_size: int,
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

        if not isinstance(target, StateTarget):
            raise ValueError("Shadow reward can only be computed for a target state")
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info
        n_shots = execution_config.n_shots  # currently not used
        seed = execution_config.seed
        
        num_qubits = len(target.physical_qubits)
        reward_data = [] 
        unique_u_rows, unique_u_count = unique_u_func(shadow_size, num_qubits)

        for i in range(len(unique_u_count)):
            qc_copy = qc.copy()    
            unique_unitary = unique_u_rows[i]
            unique_unitary_shots = unique_u_count[i]
            for qubit, id in enumerate(unique_unitary):
                gate_mapping(id, qc_copy, qubit)

            qc_copy.measure_all()            # does not work due to env_config issue?

            # qc_no_meas = qc.copy()
            # qc_no_meas.remove_final_measurements()
            
            pub = (qc_copy, params, unique_unitary_shots)  
            
            reward_data.append(
                ShadowRewardData(
                    pub,
                    unique_unitary                  
                )
            )
        
        return ShadowRewardDataList(reward_data)

    def get_reward_with_primitive(
        self,
        reward_data: ShadowRewardDataList,
        primitive: BaseSamplerV2,
        shadow_size: int,
        partition: int,
        target: StateTarget
    ) -> np.ndarray:
        
        shadow_size = reward_data.shadow_size
        job = primitive.run(reward_data.pubs)
        pub_results = job.result()
        batch_size = reward_data.pubs[0].parameter_values.shape[0]  # alternatively, output env_config from GRD and replace here: batch_size = reward_data.env_config.batch_size 
        n_shots = reward_data.shots


        """
        assert all(
            [pub.shots == n_shots for pub in reward_data.pubs]
        ), "All pubs should have the same number of shots"
        assert all(
            [pub.parameter_values.shape[0] == batch_size for pub in reward_data.pubs]
        ), "All pubs should have the same batch size"



        To be sorted out:
        1. what does job.result() return?
        We have job that includes the run version of all the pubs. each pub has 50 circuits, if we allow batch = 50, and length of params to = 50
        By right I will have a dataset that has
        1. all pubs
        2. all 50 circuits wrt params
        3. within one circuit, we expect samplerpub to give a dictionary of measurement outcomes and probability distributions
        ALL OF THEM STORED IN pub_results
        eg pub_results[1][2] may give first pub, second param, full dictionary

        eg
        param1 = [2,3]
        param2 = [2]
        params3 = [[1,2], [2,5], [3,2]]
        shots1 = 1
        shots2 = 3
        shots3 = 20
        pub1 = (qc1, param1, shots1)
        pub2 = (qc1, param2, shots2)
        pub3 = (qc1, params3, shots3)

        job = sampler.run([pub1, pub2, pub3, …])
        pub_results = job.result()

        #access pub result eg pub1
        pub1_res = pub_results[0]
        bitstrings1 = pub1_res.data.meas.get_bitstrings()
        counts1     = pub1_res.data.meas.get_counts()

        #access pub result eg pub3, all 3 param sets
        pub3_res = pub_results[2]
        bitstrings3_1 = pub3_res.data.meas.get_bitstrings(experiment=0)
        counts3_1     = pub3_res.data.meas.get_counts(experiment=0)
        """
        
        total_data_list = []
        for k in range(batch_size):             #iterate through params
            
            for i in range(len(pub_results)):   #iterate through each pub ie each unique unitary
                unique_unitary = reward_data.unitaries[i]
                unique_unitary = unique_unitary[::-1]
                unique_pub_res = pub_results[i]
                bitstring_counts     = unique_pub_res.data.meas.get_counts(loc=k)
                b = list(bitstring_counts.keys())

                #Convert each bitstring to a list of ints
                bitstrings = [[int(bit) for bit in bitstring] for bitstring in b]

                #List of counts corresponding to each bitstring
                counts = list(bitstring_counts.values())

                for j in range(len(counts)):
                    for count in range(int(counts[j])):
                        total_data_list.append([unique_unitary, bitstrings[j]])   #repeat counts[j] times for every individual b (measurement outcome)

        
        observable_decomp = SparsePauliOp.from_operator(Operator(target.dm))
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

        reward = []
        for i in range(batch_size):
            
            data_list = total_data_list[i*shadow_size:(i+1)*shadow_size]  #structure: [[u1, b11], [u2, b12], ...]
            U_list = [var[0] for var in data_list]
            b_list = [var[1] for var in data_list]
            shadow = (b_list, U_list)
            
            exp_vals = [estimate_shadow_obervable(shadow, pauli_str_num[obs], partition) for obs in range(len(pauli_str))]
            reward_i = np.dot(pauli_coeff, exp_vals)
            reward_i = round(reward_i, 4)
            #print(exp_vals)
            reward.append(reward_i)
            
        return reward


def unique_u_func(shadow_size, num_qubits):

    """ This should be fully identical in output to pennylane, which uses big endian"""

    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    unique_rows, counts = np.unique(unitary_ids, axis=0, return_counts=True)
    return unique_rows, counts

def gate_mapping(id, qc, qubit):
    if id == 0 : 
        qc.h(qubit)
    elif id == 1 : 
        qc.sdg(qubit)
        qc.h(qubit)
    elif id == 2: 
        pass

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

        exp_val = []
        
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
            
            exp_val.append((3**num_qubits) * np.prod(np.array(f)))
        means.append(sum(exp_val)/(shadow_size // k))   

    return np.median(means) 

