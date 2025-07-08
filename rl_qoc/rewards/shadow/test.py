
from rl_qoc import QuantumEnvironment, QiskitConfig, QEnvConfig, ExecutionConfig, StateTarget, ShadowReward
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import RXGate, UGate, RZXGate
from qiskit.quantum_info import Statevector

from gymnasium.spaces import Box
import numpy as np

def apply_parametrized_gate(qc: QuantumCircuit, params: ParameterVector, qr: QuantumRegister, *args, **kwargs):
    qc.h(qr[0])
    qc.rzx(params[0], qr[0], qr[1])
    

    
tgt_state = (Statevector.from_label('00') + Statevector.from_label('11'))/np.sqrt(2)
print(tgt_state)
backend_config = QiskitConfig(apply_parametrized_gate)
state_target = StateTarget(tgt_state)
execution_config = ExecutionConfig(batch_size=5,
                                   sampling_paulis=100, # Shadow size
                                   n_shots=1,
                                   seed=42
                                   )
reward = ShadowReward()
env_config = QEnvConfig(state_target, 
                        backend_config=backend_config,
                        execution_config=execution_config,
                        action_space=Box(low=np.array([0]), high=np.array([2*np.pi]), shape=(1,)),
                        reward=reward)

env = QuantumEnvironment(env_config)
params = np.array([[0.]])
reward_data = reward.get_reward_data(env.circuit, params, state_target, env_config)
print(env.circuit)
print(reward_data)







