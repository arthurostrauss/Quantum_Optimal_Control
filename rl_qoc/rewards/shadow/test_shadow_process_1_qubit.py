
from rl_qoc import QuantumEnvironment, QiskitConfig, QEnvConfig, ExecutionConfig, StateTarget, ShadowReward, GateTarget
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import RXGate, UGate, RZXGate
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, random_statevector, Choi, Operator, SuperOp, SparsePauliOp, Pauli

from gymnasium.spaces import Box
import numpy as np



# example circuit of one parameter
def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
    qc.ry(2*params[0], 0)




def shadow_bound_state(error, observables, coeffs, failure_rate=0.01):
   
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(observables[i]) * coeffs[i] for i in range(len(observables))) / error ** 2
    
    return max(int(np.ceil(N.real * K)), 10000), int(K), M           #sometimes N = 0. A limit of 10000 is set to prevent this


"""  
#2 qubit parametrized bell state of one parameter
theta = np.pi/8 #generate a random target state; this is the goal we want to obtain
tgt_state = (np.cos(theta) * Statevector.from_label('00') + np.sin(theta) * Statevector.from_label('11'))  
"""

backend_config = QiskitConfig(apply_parametrized_gate)

params = np.array([np.random.rand()*2* np.pi])
qc = QuantumCircuit(1)
qc.ry(2*params[0], 0)

specific_gate = param_gate = qc.to_gate(label="U_entangle")

gate_target = GateTarget(specific_gate)


target_choi = Choi(gate_target.target_operator)
target_choi_dm = DensityMatrix(target_choi.data)

observable_decomp = SparsePauliOp.from_operator(Operator(target_choi_dm))
pauli_coeff = observable_decomp.coeffs   #to also be used in shadow bound
pauli_str = observable_decomp.paulis
observables = [Pauli(str).to_matrix() for str in pauli_str]

error = 0.5  # Set the error tolerance for the shadow bound state
print("Density Matrix of target state: ", target_choi_dm)
shadow_size, partition, no_observables = shadow_bound_state(error, observables, pauli_coeff)
print("Shadow Size, Partition, Number of Observables: ", shadow_size, partition, no_observables)

#params = np.array([[np.random.rand()*np.pi] for i in range(5)]) # for only one parameter in the circuit, over a few batches
#params = np.array([[np.random.rand()*2* np.pi for n in range(6)] for i in range(5)])  # for a generic circuit, 6 params are required to define it.

params = np.array([params, params - 0.2, params + 0.2, params - 1, params + 1])

batch_size = len(params)

execution_config = ExecutionConfig(batch_size=batch_size,
                                   sampling_paulis=shadow_size, # do not use first
                                   n_shots=1,
                                   seed=42
                                   )
reward = ShadowReward()
env_config = QEnvConfig(gate_target, 
                        backend_config=backend_config,
                        execution_config=execution_config,
                        action_space=Box(low=np.array([0 for i in range(len(params[0]))]), high=np.array([2*np.pi for i in range(len(params[0]))]), shape=(len(params[0]),)),
                        reward=reward)

env = QuantumEnvironment(env_config)

reward_data = reward.get_reward_data(env.circuit, params, gate_target, env_config)
# print(reward_data[1].pub.parameter_values)

reward_array = reward.get_reward_with_primitive_process(reward_data, env.sampler, gate_target)
print("Rewards:", reward_array)
"""
binded_circuits = [env.circuit.assign_parameters(p) for p in params]
print("expected rewards:" , [round(state_fidelity(state_target.dm, Statevector(circ)), 4) for circ in binded_circuits])
"""




