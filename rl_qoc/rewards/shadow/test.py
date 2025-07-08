import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from shadow import SHADOWReward
from qiskit.quantum_info import Statevector
from shadow import SHADOWRewardData, SHADOWRewardDataList
from ...environment.target import GateTarget, StateTarget
from ...environment.configuration.qconfig import QEnvConfig

shadow_reward = SHADOWReward()
primitive = ...
dm_target = Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  #bell state for 2 qubits
qubits = [0,1]  #2 QUBITS
target = StateTarget(dm_target, qubits) #STATE TARGET
env_config = QEnvConfig(target, )

rewards = []
qc = QuantumCircuit(2)  # append as required

for _ in range(50):  # 50 epochs
    

    params = np.array([])   
    # Prepare shadow data
    reward_data_list = shadow_reward.get_reward_data(qc, params, target, env_config)

    # Estimate shadow expectation, get reward
    reward = shadow_reward.get_reward_with_primitive(reward_data_list, primitive)

    
    
    # params_for_gate = machine_learning(rewards)
    # correction_gate = some_func(params_for_gate)
    # qc = qc.append(correction_gate)
