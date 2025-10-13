
from rl_qoc import QuantumEnvironment, QiskitConfig, QEnvConfig, ExecutionConfig, StateTarget, ShadowReward, GateTarget
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import RXGate, UGate, RZXGate
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, random_statevector, Choi, Operator, SuperOp, SparsePauliOp, Pauli

from gymnasium.spaces import Box
import numpy as np


""" Shadow Bound Calculation; Taken from Pennylane: Classical Shadows."""
def shadow_bound_state(error, observables, coeffs, failure_rate=0.01):
   
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(observables[i]) * coeffs[i]**2 for i in range(len(observables))) / error ** 2
    
    return max(int(np.ceil(N.real * K)), 10000), int(K), M           #sometimes N = 0. A limit of 10000 is set to prevent this

# ______________________________________________________________________________________________________________________________________________



# TEST 0: 1 qubits, 1 parameter only
def test_1_qubits():
# simplified 1 qubit circuit of one parameter
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        qc.ry(params[0], 0)

    # 1 qubit parametrized state of one parameter
    theta = np.pi/8 #generate a random target state; this is the goal we want to obtain
    tgt_state = np.cos(theta/2) * Statevector.from_label('0') + np.sin(theta/2) * Statevector.from_label('1')

    #params = np.array([[theta]])
    params = np.array([[np.random.rand()*np.pi] for i in range(2)]) # for only one parameter in the circuit, over a few batches
    return apply_parametrized_gate, tgt_state, params

# ______________________________________________________________________________________________________________________________________________



# TEST 1: 2 qubits, 1 parameter only
def test_2_qubits_1_param():
# simplified 2 qubit circuit of one parameter
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        #test circuit 1
        qc.ry(params[0], 0)
        qc.cx(0,1)
        
        #test circuit 2
        #qc.h(0)
        #qc.cx(0,1)
        #qc.rz(params[0], 1)

    # 2 qubit parametrized bell state of one parameter
    theta = np.pi #generate a random target state; this is the goal we want to obtain
    tgt_state = (np.cos(theta/2) * Statevector.from_label('00') + np.sin(theta/2) * Statevector.from_label('11'))  

    params =  np.array([[theta]])
    #params = np.array([[np.random.rand()*np.pi] for i in range(10)]) # for only one parameter in the circuit, over a few batches

    return apply_parametrized_gate, tgt_state, params



# ______________________________________________________________________________________________________________________________________________

# TEST 2: 2 qubits, 6 parameters

def test_2_qubits():
    #generic 2 qubit circuit of 6 parameters
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        qc.u(params[0], params[1],params[2], 0)
        qc.u(params[3], params[4],params[5], 1)
        qc.cx(0,1)

    # 2 qubit random state
    no_qubits = 2
    tgt_state = random_statevector(2**no_qubits)  

    params = np.array([[np.random.rand()*2* np.pi for n in range(6)] for i in range(5)])  # for a generic 2 qubit circuit, 6 params are required to define it.

    return apply_parametrized_gate, tgt_state, params

# _______________________________________________________________________________________________________________________________________________

# TEST 3: 3 qubits, 9 parameters
def test_3_qubits():
#generic 3 qubit circuit of 9 parameters
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        qc.u(params[0], params[1],params[2], 0)
        qc.u(params[3], params[4],params[5], 1)
        qc.cx(0,1)
        qc.cx(0,2)
        qc.u(params[6], params[7],params[8], 2)
        qc.cx(1,2)

    # 3 qubit random state
    no_qubits = 3
    tgt_state = random_statevector(2**no_qubits)  

    params = np.array([[np.random.rand()*2* np.pi for n in range(9)] for i in range(5)])  # for a generic 3 qubit circuit, 9 params are required to define it.
    return apply_parametrized_gate, tgt_state, params

# _______________________________________________________________________________________________________________________________________________

# TEST 4: 4 qubits, 12 parameters
def test_4_qubits():
#generic 4 qubit circuit of 12 parameters
    def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
        qc.u(params[0], params[1],params[2], 0)
        qc.u(params[3], params[4],params[5], 1)
        qc.cx(0,1)
        qc.cx(0,2)
        qc.cx(0,3)
        qc.u(params[6], params[7],params[8], 2)
        qc.cx(1,2)
        qc.cx(1,3)
        qc.u(params[9], params[10],params[11], 3)
        qc.cx(2,3)

    # 4 qubit random state
    no_qubits = 4
    tgt_state = random_statevector(2**no_qubits)  

    params = np.array([[np.random.rand()*2* np.pi for n in range(12)] for i in range(5)])  # for a generic 4 qubit circuit, 12 params are required to define it.
    return apply_parametrized_gate, tgt_state, params


# ______________________________________________________________________________________________________________________________________________


apply_parametrized_gate, tgt_state, params = test_4_qubits() 

# print("State Vector of target state: ",tgt_state)
backend_config = QiskitConfig(apply_parametrized_gate)
state_target = StateTarget(tgt_state)


observable_decomp = SparsePauliOp.from_operator(Operator(state_target.dm))
pauli_coeff = observable_decomp.coeffs   #to also be used in shadow bound
pauli_str = observable_decomp.paulis
observables = [Pauli(str).to_matrix() for str in pauli_str]



error = 0.1 # can change
#print("Density Matrix of target state: ", observables)
shadow_size, partition, no_observables = shadow_bound_state(error, observables, pauli_coeff)
print("Shadow Size, Partition, Number of Observables: ", shadow_size, partition, no_observables)

batch_size = len(params)

execution_config = ExecutionConfig(batch_size=batch_size,
                                   sampling_paulis=shadow_size, # do not use first
                                   n_shots=1,
                                   seed=42
                                   )
reward = ShadowReward()
env_config = QEnvConfig(state_target, 
                        backend_config=backend_config,
                        execution_config=execution_config,
                        action_space=Box(low=np.array([0 for i in range(len(params[0]))]), high=np.array([2*np.pi for i in range(len(params[0]))]), shape=(len(params[0]),)),
                        reward=reward)

env = QuantumEnvironment(env_config)

reward_data = reward.get_reward_data(env.circuit, params, state_target, env_config)
reward_array = reward.get_reward_with_primitive(reward_data, env.sampler)#, state_target)
print("Rewards:", reward_array)

binded_circuits = [env.circuit.assign_parameters(p) for p in params]
print("expected rewards:" , [round(state_fidelity(state_target.dm, Statevector(circ)), 4) for circ in binded_circuits])





