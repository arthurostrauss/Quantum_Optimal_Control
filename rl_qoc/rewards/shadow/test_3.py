
from rl_qoc import QuantumEnvironment, QiskitConfig, QEnvConfig, ExecutionConfig, StateTarget, ShadowReward, GateTarget
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import RXGate, UGate, RZXGate
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, random_statevector, Choi, Operator, SuperOp, SparsePauliOp, Pauli

from gymnasium.spaces import Box
import numpy as np


""" Shadow Bound Calculation; Taken from Pennylane: Classical Shadows."""
def shadow_bound_state(error, observables, failure_rate=0.01):
   
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error ** 2
    return max(int(np.ceil(N * K)), 10000), int(K), M           #sometimes N = 0. A limit of 10000 is set to prevent this

# ______________________________________________________________________________________________________________________________________________
"""
# TEST 0: 1 qubits, 1 parameter only

# simplified 1 qubit circuit of one parameter
def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
    qc.ry(np.pi/8, 0)

# 1 qubit parametrized state of one parameter
theta = np.pi/8 #generate a random target state; this is the goal we want to obtain
tgt_state = np.cos(theta/2) * Statevector.from_label('0') + np.sin(theta/2) * Statevector.from_label('1')

params = np.array([[theta]])
#params = np.array([[np.random.rand()*np.pi] for i in range(5)]) # for only one parameter in the circuit, over a few batches
"""
#np.random.seed(42)
# ______________________________________________________________________________________________________________________________________________

# TEST 1: 2 qubits, 1 parameter only

# simplified 2 qubit circuit of one parameter
def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
    qc.ry(2*params[0], 0)
    qc.cx(0,1)

# 2 qubit parametrized bell state of one parameter
theta = np.pi/8 #generate a random target state; this is the goal we want to obtain
tgt_state = (np.cos(theta) * Statevector.from_label('00') + np.sin(theta) * Statevector.from_label('11'))  

#params = np.array([[theta]])
params = np.array([[np.random.rand()*np.pi] for i in range(3)]) # for only one parameter in the circuit, over a few batches



# ______________________________________________________________________________________________________________________________________________
"""
# TEST 2: 2 qubits, 6 parameters
#generic 2 qubit circuit of 6 parameters
def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
    qc.u(params[0], params[1],params[2], 0)
    qc.u(params[3], params[4],params[5], 1)
    qc.cx(0,1)

# 2 qubit random state
no_qubits = 2
tgt_state = psi = random_statevector(2**no_qubits)  

params = np.array([[np.random.rand()*2* np.pi for n in range(6)] for i in range(2)])  # for a generic 2 qubit circuit, 6 params are required to define it.
"""

# ______________________________________________________________________________________________________________________________________________
print("State Vector of target state: ",tgt_state)
backend_config = QiskitConfig(apply_parametrized_gate)
state_target = StateTarget(tgt_state)

error = 0.1 # can change
observables = [state_target.dm.data]
print("Density Matrix of target state: ", observables)
shadow_size, partition, no_observables = shadow_bound_state(error, observables)
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
reward_array = reward.get_reward_with_primitive(reward_data, env.sampler, state_target)
print("Rewards:", reward_array)

binded_circuits = [env.circuit.assign_parameters(p) for p in params]
print("expected rewards:" , [round(state_fidelity(state_target.dm, Statevector(circ)), 4) for circ in binded_circuits])





