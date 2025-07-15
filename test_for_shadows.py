
from rl_qoc import QuantumEnvironment, QiskitConfig, QEnvConfig, ExecutionConfig, StateTarget, ShadowReward
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import RXGate, UGate, RZXGate
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix

from gymnasium.spaces import Box
import numpy as np

def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
    qc.ry(2*params[0], 0)
    qc.cx(0,1)

    
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
    return max(int(np.ceil(N * K)), 10000), int(K), M           #sometimes N = 0. A limit of 10000 is set to prevent this

theta = np.pi/8 #generate a random target state; this is the goal we want to obtain
tgt_state = (np.cos(theta) * Statevector.from_label('00') + np.sin(theta) * Statevector.from_label('11')) #bell state
#vec = np.array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])   #XX state

#tgt_state = Statevector(vec)

print(tgt_state)
backend_config = QiskitConfig(apply_parametrized_gate)
state_target = StateTarget(tgt_state)

error = 0.1 # change
observables = [state_target.dm.data]
print(observables)
shadow_size, partition, no_observables = shadow_bound(error, observables)
print(shadow_size, partition, no_observables)


#params = np.array([[0,1], [1,1]])
#params = np.array([[np.random.rand()*np.pi] for i in range(10)])
params = np.array([[np.pi/8]])
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
                        action_space=Box(low=np.array([0,0]), high=np.array([2*np.pi,2*np.pi]), shape=(2,)),
                        reward=reward)

env = QuantumEnvironment(env_config)

reward_data = reward.get_reward_data(env.circuit, params, state_target, shadow_size, env_config)
#print(reward_data[1].pub.parameter_values)
reward_array = reward.get_reward_with_primitive(reward_data, env.sampler, shadow_size, partition = partition, target = state_target)
print("Rewards:", reward_array)

binded_circuits = [env.circuit.assign_parameters(p) for p in params]
print("expected rewards:" , [round(state_fidelity(state_target.dm, Statevector(circ)), 4) for circ in binded_circuits])


"""
for troubleshooting
def qc_testing(params):

    qc = QuantumCircuit(2)
    qc.ry(2*params[0,0], 0)
    qc.cx(0,1)
    return qc, Statevector.from_instruction(qc), DensityMatrix.from_instruction(qc)

print("expected rewards 2:", [state_fidelity(state_target.dm, qc_testing(params)[1])])
"""

