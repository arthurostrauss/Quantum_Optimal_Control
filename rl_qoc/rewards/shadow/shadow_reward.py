from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Literal, Tuple
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, Permutation
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import Operator, SparsePauliOp, DensityMatrix, Statevector, Choi, SuperOp, state_fidelity
from .shadow_reward_data import ShadowRewardData, ShadowRewardDataList
from .snapshot import Snapshot, SnapshotList
from ...environment.target import GateTarget, StateTarget
from ...environment.configuration.qconfig import QEnvConfig
from ..base_reward import Reward
from ...helpers.circuit_utils import (
    handle_n_reps,
    causal_cone_circuit,
    get_input_states_cardinality_per_qubit,
)
from qiskit_aer import AerSimulator
import sys

@dataclass
class ShadowReward(Reward):
    """
    Configuration for computing the reward based on (Shadow) 
    all data in GRD and all calculations in GRWP
    """
    unitary_seed: int = 2000
    unitary_rng: np.random.Generator = field(init=False)
    shuffling_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.unitary_rng = np.random.default_rng(self.unitary_seed)
        self.shuffling_rng = np.random.default_rng(self.unitary_seed + 1)

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
        qc: QuantumCircuit, #simulate imperfect gate/ channel
        params: np.ndarray,
        target: StateTarget | GateTarget,
        env_config: QEnvConfig,
        *args
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

        """
        CAUTION - The issue with endianness
        Qiskit uses little endian; Qubit 1 is on the right.
        For example, if we want to act a unitary XYZI onto qubits 0, 1, 2, 3 and measure in Z basis, the bitstring that comes out
        e.g. b = '0101', will be Qubit 3, 2, 1, 0. 
        This means that U is big endian, b that comes out is little endian. We need to flip U so it is little endian as well.
        """
   
        
        backend_info = env_config.backend_info
        shadow_size = env_config.execution_config.sampling_paulis
        
        num_qubits = len(target.physical_qubits) if isinstance(target, StateTarget) else target.causal_cone_size
        reward_data = [] 

        if isinstance(target, StateTarget):
            unitary_ids = self.unitary_rng.choice(3, size=(shadow_size, num_qubits))    # Draw a sample of Pauli operators encoded via mapping = {'X': 0, 'Y': 1, 'Z': 2, 'I': 3}
            unique_unitaries, counts = np.unique(unitary_ids, axis=0, return_counts=True)   # Tabulates unique Pauli Strings with their counts

            for unitary, shots in zip(unique_unitaries, counts):    # Apply unitary based on Pauli String = {'X': H, 'Y': H*Sdg, 'Z': 1, 'I': 1}
                qc_copy = qc.copy()    
                for qubit, id in enumerate(unitary):
                    if id == 0: # X basis 
                        qc_copy.h(qubit)
                    elif id == 1 : # Y basis 
                        qc_copy.sdg(qubit)
                        qc_copy.h(qubit)
                    
                qc_copy.measure_all()   # perform measurement
                circuit = backend_info.custom_transpile(qc_copy,
                                                        remove_final_measurements=False,
                                                        initial_layout=target.layout,
                                                        scheduling=False, 
                                                        optimization_level=0)
            
                pub = (circuit, params, shots)  # create pub object
                
                reward_data.append(
                    ShadowRewardData(
                        pub,
                        unitary=unitary[::-1],  # flip u to make it big endian, i.e. u[0] now acts on leftmost qubit; making it consistent with output bitstring
                        u_in=None,
                        b_in=None
                    )
                )
        
        
        else:
            # TODO: implement Shadow tomography for channel
            """Step 1/2/4: Sample U_in, U_out and b_in"""

            
            index_process = self.unitary_rng.choice(18, size=(shadow_size, num_qubits))
            unique_indexes, counts = np.unique(index_process, axis=0, return_counts=True)


            """Step 3: Construct input state rho_in = U_in_hermitian * |b_in><b_in| * U_in"""
            qubits = target.causal_cone_qubits

            for unique_index, shots in zip(unique_indexes, counts):

                bitstrings_in, unitary_in, unitary_out = np.unravel_index(unique_index, (2,3,3))
                qc_copy = qc.copy_empty_like()
                
                for i, bit in enumerate(bitstrings_in):  
                    if bit == 1:
                        qc_copy.x(i)
    
                for q, id in enumerate(unitary_in):
                    if id == 0: # X basis
                        qc_copy.h(qubits[q])
                    elif id == 1 : # Y basis
                        qc_copy.sdg(qubits[q])
                        qc_copy.h(qubits[q])

                qc_copy.compose(qc, inplace=True)
                     
                for q, id in enumerate(unitary_out):
                    if id == 0: # X basis 
                        qc_copy.h(qubits[q])
                    elif id == 1 : # Y basis
                        qc_copy.sdg(qubits[q])
                        qc_copy.h(qubits[q])

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
                        unitary=unitary_out[::-1],  # u in and u out is taken to be little endian, so the output reverses it to become big endian
                        u_in=unitary_in[::-1],
                        b_in=bitstrings_in[::-1]       #inverted because little endian as well.``
                    )
                )
        
        return ShadowRewardDataList(reward_data, target=target)

    def get_reward_with_primitive(
        self,
        reward_data: ShadowRewardDataList,
        primitive: BaseSamplerV2,

    ) -> np.ndarray:
        
        shadow_size = reward_data.shadow_size
        job = primitive.run(reward_data.pubs)   # use Sampler Primitive to run the compiled circuits
        pub_results = job.result()  # contains all the bitstrings
        batch_size = reward_data.pubs[0].parameter_values.shape[0]
        target = reward_data.target
        
        total_data_list = []
        for k in range(batch_size):
            for i in range(len(pub_results)):
                # Extract unique unitary and bitstring for each reward data
                unique_unitary = reward_data[i].unitary
                unique_pub_res = pub_results[i]
                bitstring_counts = unique_pub_res.data.meas.get_counts(loc=k)
                bitstring_list = list(bitstring_counts.keys())

                # Convert each bitstring to a list of ints 
                bitstrings = [[int(bit) for bit in bitstring] for bitstring in bitstring_list]

                #List of counts corresponding to each bitstring
                counts = list(bitstring_counts.values())

                for j in range(len(counts)):
                    for count in range(int(counts[j])):
                        total_data_list.append([unique_unitary, bitstrings[j]])   #repeat counts[j] times for every individual b (measurement outcome)

       # total_data_list gives a list of all (unitary, bitstring) pairs 

        function_observable = 1 # for testing the 3 functions only


        dict_function_observable = {
            1: estimate_shadow_observable, 
            2: estimate_shadow_observable_v2, 
            3: estimate_shadow_observable_v3
            }
        
        function_observable_name = dict_function_observable[function_observable]

        if function_observable_name == estimate_shadow_observable or function_observable_name == estimate_shadow_observable_v2:
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
            
            reward = np.zeros(batch_size)
            for i in range(batch_size):
                data_list = total_data_list[i*shadow_size:(i+1)*shadow_size]  # structure: [[u1, b11], [u2, b12], ...]
                U_list = [var[0] for var in data_list]
                b_list = [var[1] for var in data_list]
                shadow = (b_list, U_list)
            
                #print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                exp_vals = [function_observable_name(shadow, pauli_str_num[obs], partition) for obs in range(len(pauli_str))]
                reward_i = np.dot(pauli_coeff, exp_vals)
                assert np.imag(reward_i) - 1e-10 < 0, "Reward is complex"
                reward[i] = reward_i.real
                #print("Reward batch ", i, " is ", reward_i.real)
            

        
        elif function_observable_name == estimate_shadow_observable_v3:
        # V3 directly evolves the shadow U|b><b|U with respect to the target observable.
        # As such, the number of observables is just 1, and partition formula is taken directly and calculated here.
            M = 1
            partition = int(2 * np.log(2 * M / 0.01))

            reward = np.zeros(batch_size)
            for i in range(batch_size):
                data_list = total_data_list[i*shadow_size:(i+1)*shadow_size]  #structure: [[u1, b11], [u2, b12], ...]
                U_list = [var[0] for var in data_list]
                b_list = [var[1] for var in data_list]
                shadow = (b_list, U_list)
            
                print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                reward_i = function_observable_name(shadow, target.dm, partition, self.shuffling_rng)
                
                assert np.imag(reward_i) - 1e-10 < 0, "Reward is complex"
                reward[i] = reward_i.real
                print("Reward batch ", i, " is ", reward_i.real)

        return reward
    


    def get_reward_with_primitive_process(
        self,
        reward_data: ShadowRewardDataList,
        primitive: BaseSamplerV2,
        target:  GateTarget
    ) -> np.ndarray:
        
        shadow_size = reward_data.shadow_size
        job = primitive.run(reward_data.pubs)
        pub_results = job.result()
        batch_size = reward_data.pubs[0].parameter_values.shape[0]
        
        total_data_list = []
        z_list = []

        for k in range(batch_size):
            for i in range(len(pub_results)):
                unique_unitary_out = reward_data[i].unitary
                unique_unitary_in = reward_data[i].u_in
                unique_bitstring_in = reward_data[i].b_in
                unique_pub_res = pub_results[i]
                bitstring_counts = unique_pub_res.data.meas.get_counts(loc=k)
                b = list(bitstring_counts.keys())

                # Convert each bitstring to a list of ints 
                bitstrings = [[int(bit) for bit in bitstring] for bitstring in b]

                #List of counts corresponding to each bitstring
                counts = list(bitstring_counts.values())

                for j in range(len(counts)):
                    for count in range(int(counts[j])):
                        total_data_list.append([unique_unitary_in, unique_bitstring_in, unique_unitary_out, bitstrings[j]])   
                        #repeat counts[j] times for every individual b (measurement outcome)

                        #Calculate z: U_in.T |bin> tensor U_out.hermitian |bout>
                        #for pauli string, .T only adds a - sign for Y terms and does not change I, X and Z terms. Y terms are mapped as digit '1'
                        sign = (-1) ** np.sum(unique_unitary_in == 1)
                        #however, if we use |z><z| later, this -ve cancels out. So it is not modelled into z now.
                        unique_unitary_combined = np.concatenate((unique_unitary_in, unique_unitary_out))
                        unique_bitstring_combined = np.concatenate((unique_bitstring_in, np.array(bitstrings[j])))
                        z_list.append([unique_unitary_combined, unique_bitstring_combined, sign])  #store the sign to be used later

        target_choi = Choi(target.target_operator)
        target_choi_dm = DensityMatrix(target_choi.data)

        observable_decomp = SparsePauliOp.from_operator(Operator(target_choi_dm))   #to check what is the target gate fidelity like, is it a density matrix?
        partition = int(2 * np.log(2*len(observable_decomp.paulis)/0.01))
        pauli_coeff = observable_decomp.coeffs   # to also be used in shadow bound
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
            data_list = z_list[i*shadow_size:(i+1)*shadow_size]  #structure: [[u1, b11], [u2, b12], ...]
            U_list = [var[0] for var in data_list]
            b_list = [var[1] for var in data_list]
            sign_list = [var[2] for var in data_list]
            shadow = (b_list, U_list)
            print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            exp_vals = [estimate_shadow_obervable_process(shadow, pauli_str_num[obs], partition) for obs in range(len(pauli_str))]
            reward_i = np.dot(pauli_coeff, exp_vals)
            assert np.imag(reward_i) - 1e-10 < 0, "Reward is complex"
            reward[i] = reward_i.real
            print("Reward batch ", i, " is ", reward_i.real)
            
        return reward








def estimate_shadow_observable_v3(
        shadow: Tuple[List[List[int]], List[List[int]]],
        observable: DensityMatrix,
        k: int,
        shuffling_rng: np.random.Generator,
    ) -> float:
    
    """
    Goal: From measurement_list and unitary_ids, split into N/k groups, find mean for each, then find median of means
    V3 of this function aims to directly evolve the unitary, then the observable.
    """
    b_lists  = np.array(shadow[0])      # split shadow and convert bitstring and unitaries into arrays
    u_lists = np.array(shadow[1])
    shadow_size, num_qubits = b_lists.shape
    shuffle_indices = np.random.permutation(b_lists.shape[0])
    
    #shuffle_indices = shuffling_rng.permutation(b_lists.shape[0])   # Shuffle the indices so that median of means work. The indices were previously ordered for ease of execution.
    b_shuffled = b_lists[shuffle_indices]
    u_shuffled = u_lists[shuffle_indices]
    obs = SparsePauliOp.from_operator(observable.data)

            
    

    means = []
    # loop over the splits of the shadow:
    for i in range(k):       # shadow_size // k = no of elements in each set; k = no of sets

        # assign the splits 
        start = i * (shadow_size //k)
        end = (i+1) * (shadow_size //k)
        b_lists_k, u_lists_k = (
            b_shuffled[start: end],
            u_shuffled[start: end],
        )

        #exp_val = np.array([])
        exp_val = []
        for n in range(shadow_size // k):
            
            b = b_lists_k[n]
            U = u_lists_k[n]
            snapshot = Snapshot(b, U)       
            """
            print("Snapshot bitstring: ", b)
            print("Snapshot unitary: ", U)
            print("bitstring in snapshot: ", snapshot.bitstring)
            """
            rho_shadow = None
            rho_shadow_list = []
            counter = 0
            for i in range(snapshot.num_qubits):
                b_single = Statevector.from_label(snapshot.bitstring[i])
                U_single = snapshot.unitary_single[i]
                b_evolved_single = b_single.evolve(U_single)
                rho_local = 3 * DensityMatrix(b_evolved_single) - DensityMatrix(np.eye(2)) # creating the shadow per qubit as per Pennylane, "State Reconstruction from a Classical Shadow"
                """
                print("bitstring: ", snapshot.bitstring[i])
                print("b_single: ", b_single)
                print("unitary_string: ", snapshot.unitary_single[i])    
                print("U_single: ", U_single)
                print("Evolved local shadow state statevector: ", b_evolved_single)
                print("Evolved local shadow state density matrix: ", rho_local)
                """
                rho_shadow_list.append(rho_local)
                """if rho_shadow is None:
                    rho_shadow = rho_local
                else:
                    rho_shadow = rho_shadow.expand(rho_local)   # tensor product of per qubit shadow to get one copy of shadow"""

           
            rho_shadow = DensityMatrix.tensor(*rho_shadow_list[::-1]) if snapshot.num_qubits < 1 else rho_local #may or may not need to reverse the order here

            exp_val.append(rho_shadow.expectation_value(obs).real)
                #exp_val, np.trace(observable.data @ rho_shadow.data).real)  #Tr(Orho)
            """
            print("Evolved shadow state density matrix: ", rho_shadow)
            print("Observable: ", observable)
            print(exp_val)
            counter+=1"""


        print(exp_val)
        exp_val_list = np.sum(exp_val)/ (shadow_size // k)
        means.append(exp_val_list)

    return np.mean(means)






def estimate_shadow_observable_v2(
        shadow: Tuple[List[List[int]], List[List[int]]],
        observable: DensityMatrix,
        k: int,
    ) -> float:
    # slightly less accurate as original and slightly slower
    """
    Goal: From measurement_list and unitary_ids, split into N/k groups, find mean for each, then find median of means
    
    By the formulation in Pennylane notes, if the observable is not matching the unitary, the entire expectation value goes to 0 for that shadow.
    For example; 5 qubits, observable is [3, 0, 0, 3, 3] which correspond to IXXII
    And my unitary id (shadow) is [2, 0, 0, 3, 1].
    Then Tr(Orho) will give non zero value. Else, we will get 0 for the entire shadow.
    """
    b_lists  = np.array(shadow[0])
    obs_lists = np.array(shadow[1])
    shadow_size, num_qubits = b_lists.shape
    shuffle_indices = np.random.permutation(b_lists.shape[0])
    b_shuffled = b_lists[shuffle_indices]
    obs_shuffled = obs_lists[shuffle_indices]
    

    exp_val_means = []
    P = np.array(observable)
    mask = P != 3
    P_remove_I = P[mask]

    exp_vals = []
    for b_list, U_list in zip(b_shuffled, obs_shuffled):
        U_list = U_list[mask]
        b_list = b_list[mask]
        P_U_matching = (U_list == P_remove_I).astype(int)
        b_list = -b_list
        b_list_modified = np.where(b_list == 0, 1, b_list)
        vals = P_U_matching * b_list_modified
        if vals.size == 0:
            exp_val = 1
        else:
            exp_val = (3**len(P_remove_I)) * np.prod(vals)

        exp_vals.append(exp_val)
    #print(P, U_shuffled[-1], b_shuffled[-1], exp_vals[-1])
    
    for i in range(k):       # shadow_size // k = no of elements in each set; k = no of sets

        # assign the splits
        start = i * (shadow_size // k)
        end = (i + 1) * (shadow_size // k)
        exp_vals_k = exp_vals[start:end]   
        exp_val_means.append(np.mean(exp_vals_k))

    return np.median(exp_val_means)
        




def estimate_shadow_observable(
        shadow: Tuple[List[List[int]], List[List[int]]],
        observable: DensityMatrix,
        k: int,
    ) -> float:



    """
    Goal: From measurement_list and unitary_ids, split into N/k groups, find mean for each, then find median of means
    
    By the formulation in Pennylane notes, if the observable is not matching the unitary, the entire expectation value goes to 0 for that shadow.
    For example; 5 qubits, observable is [3, 0, 0, 3, 3] which correspond to IXXII
    And my unitary id (shadow) is [2, 0, 0, 3, 1].
    Then Tr(Orho) will give non zero value. Else, we will get 0 for the entire shadow.
    """
    b_lists  = np.array(shadow[0])
    obs_lists = np.array(shadow[1])
    shadow_size, num_qubits = b_lists.shape
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
            
            exp_val[n] = exp_single(b_lists_k[n], obs_lists_k[n], P)

        exp_val_list = np.sum(exp_val)/(shadow_size // k)
        means.append(exp_val_list)


    return np.median(means) 

def exp_single(b, U, P):
    num_qubits = len(P)
    prod = 1.0

    for m in range(num_qubits):
        if P[m] == 3:                # I
            prod *= 1/3
        elif P[m] != U[m]:           # mismatch → whole term is 0
            return 0.0
        else:                        # match and P != I
            prod *= 1.0 if b[m] == 0 else -1.0

    return (3.0 ** num_qubits) * prod












def estimate_shadow_obervable_process(shadow, observable, k):
    
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

            exp_val[n] = exp_single_process(b_lists_k[n], obs_lists_k[n], P)

            """
            #outdated
            b = b_lists_k[n]
            U = obs_lists_k[n]
            
            f = []

            for m in range(num_qubits//2):
                
                if P[m] == 3:
                    f.append(1/3)       #important. if P = I, there is additional Tr(I) = 2 in the sum now. We must compensate by changing to 1/3.
                elif P[m] == U[m] and b[m] == 0:
                    if U[m] ==1:
                        f.append(-1)
                    else:
                        f.append(1)
                elif P[m] == U[m] and b[m] == 1:
                    if U[m] ==1:
                        f.append(1)
                    else:
                        f.append(-1)
                else:
                    f.append(0)

            for m in range(num_qubits//2, num_qubits):
                
                if P[m] == 3:
                    f.append(1/3)       #important. if P = I, there is additional Tr(I) = 2 in the sum now. We must compensate by changing to 1/3.
                elif P[m] == U[m] and b[m] == 0:
                    f.append(1)
                elif P[m] == U[m] and b[m] == 1:
                    f.append(-1)
                else:
                    f.append(0)

            exp_val[n] = (3**num_qubits) * np.prod(np.array(f)) #* (-1)**count_negative_y

            """

            
            # add a correction for transpose part - to integrate into code if necessary
            # b_in  = b[:num_qubits//2]
            # count_negative_y = np.sum(b_in == 1)

            
        means.append(np.sum(exp_val)/(shadow_size // k))   
    #print(means)

    return np.median(means) 


def exp_single_process(b, U, P):
    num_qubits = len(P)
    prod = 1.0

    for m in range(num_qubits//2):
        if P[m] == 3:                # I
            prod *= 1/3
        elif P[m] != U[m]:           # mismatch → whole term is 0
            return 0.0
        elif U[m] ==1:               # account for 
            prod *= -1.0 if b[m] == 0 else 1.0
        else:                        # match and P != I
            prod *= 1.0 if b[m] == 0 else -1.0

    for m in range(num_qubits//2, num_qubits):
        if P[m] == 3:                # I
            prod *= 1/3
        elif P[m] != U[m]:           # mismatch → whole term is 0
            return 0.0
        else:                        # match and P != I
            prod *= 1.0 if b[m] == 0 else -1.0

    return (3.0 ** num_qubits) * prod