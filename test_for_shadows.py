
from rl_qoc import QuantumEnvironment, QiskitConfig, QEnvConfig, ExecutionConfig, StateTarget, ShadowReward
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import RXGate, UGate, RZXGate
from qiskit.quantum_info import Statevector

from gymnasium.spaces import Box
import numpy as np

def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
    qc.h(qr[0])
    qc.rzx(params[0], qr[0], qr[1])
    #qc.rzx(params[1], qr[0], qr[1])
    
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

tgt_state = (Statevector.from_label('00') + Statevector.from_label('11'))/np.sqrt(2)    #bell state
print(tgt_state)
backend_config = QiskitConfig(apply_parametrized_gate)
state_target = StateTarget(tgt_state)

error = 0.05 # change
observables = [state_target.dm.data]
shadow_size, partition, no_observables = shadow_bound(error, observables)
print(shadow_size//partition, no_observables)
batch_size=5
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

#
#params = np.array([[0,1], [1,1]])

params = np.array([[0],[1],[2],[3],[4]])

reward_data = reward.get_reward_data(env.circuit, params, state_target, env_config)
#print(env.circuit)
print(reward_data[1].pub.parameter_values)
reward_array = reward.get_reward_with_primitive(reward_data, env.sampler, shadow_size, partition = partition, target = state_target)
print(reward_array)







