from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2

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

    #input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"
    input_states_seed: int = 2000
    input_states_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.input_states_rng = np.random.default_rng(self.input_states_seed)

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}

    @property
    def reward_method(self):
        return "shadow"

    def set_reward_seed(self, seed: int):
        """
        Set the seed for the random number generator
        """
        self.input_states_seed = seed + 357
        self.input_states_rng = np.random.default_rng(self.input_states_seed)

    
    def get_reward_data(
        self,
        qc: QuantumCircuit, #simulate imperfect gate
        unitary: List[List[int]],  #do i need this here?
        shadow_size: int,

        params: np.array,
        target: StateTarget,
        env_config: QEnvConfig,
        baseline_circuit: Optional[QuantumCircuit] = None,
    ) -> ShadowRewardDataList:
        """
        Compute pubs related to the reward method

        Args: 
            qc: Quantum circuit to be executed on quantum system
            unitary: String of Paulis to run with the gate
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            env_config: QEnvConfig containing the backend information and execution configuration
            baseline_circuit: Ideal circuit that qc should implement
        """

        if not isinstance(target, GateTarget):
            raise ValueError("Shadow reward can only be computed for a target gate")
        execution_config = env_config.execution_config
        backend_info = env_config.backend_info

        if baseline_circuit is not None:
            circuit_ref = baseline_circuit.copy()
        else:
            circuit_ref = qc.metadata["baseline_circuit"].copy()
        # num_qubits = target.causal_cone_size # not a property in statetarget
        
        
        num_qubits = len(target.physical_qubits)
        reward_data = [] 
        unique_u_rows, unique_u_count = unique_u(shadow_size, num_qubits)
        for i in range(unique_u_count):
            u = unique_u_rows[i]
            for qubit, id in enumerate(u):
                gate_mapping(id, qc, qubit)

            
            pub = (qc, params, unique_u_count)  
            
            reward_data.append(
                ShadowRewardData(
                    pub,
                    u                    
                )
            )

        return ShadowRewardDataList(reward_data)

    def get_reward_with_primitive(
        self,
        reward_data: ShadowRewardDataList,
        primitive: BaseSamplerV2,
        shadow_size: int,  #does it need to be here, or calculated in main instead?
        error: float,
        target: StateTarget
    ) -> np.array:
        

        job = primitive.run(reward_data.pubs)
        pub_results = job.result()
        batch_size = reward_data.pubs[0].parameter_values.shape[0]  #is this to get shadow size?
        n_shots = reward_data.pubs[0].shots
        assert all(
            [pub.shots == n_shots for pub in reward_data.pubs]
        ), "All pubs should have the same number of shots"
        assert all(
            [pub.parameter_values.shape[0] == batch_size for pub in reward_data.pubs]
        ), "All pubs should have the same batch size"

        total_data_list = []
        for i in range(len(pub_results)):
            results = pub_results[i].quasi_dists
            b = list(results.keys())
            counts = list(results.values()) * n_shots[i]   #obtains b list and repeats, as shown in CCS function
            u = reward_data[i]
            for j in range(len(counts)):
                total_data_list.append(u, b[j])  # need to check where params come in, may need an extra loop

        shadow_size = np.sum(reward_data.total_shots)
        reward = []
        observable_matrix = target.dm #may need to do an extra check, convert to dm matrix if needed
        observable = pauli_to_string(observable_matrix) # to clarify: may need to define an additional function for this
        batch = 50  # to be put into input?
        for i in range(batch):
            shadow = total_data_list[i:i+batch] #extract U and b out of row i, 
            N_k, k, M = shadow_bound(error, observable_matrix)
            reward.append(estimate_shadow_obervable(shadow, observable, k))
        return reward


def unique_u(shadow_size, num_qubits):

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
                    f.append(-1)
                elif P[m] == U[m] and b[m] == 1:
                    f.append(1)
                else:
                    f.append(0)
            
            exp_val.append((3**num_qubits) * np.prod(np.array(f)))

        means.append(sum(exp_val)/(shadow_size // k))   

    return np.median(means) 

