#%%
from rl_qoc import QuantumEnvironment, ChannelReward, QiskitConfig, QEnvConfig, ExecutionConfig, GateTarget
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RXGate, UGate, RZXGate
from gymnasium.spaces import Box
import numpy as np
seed = 30980922
np.random.seed(seed)
params = ParameterVector("a", 1)
angle = np.random.uniform(-np.pi, np.pi, size=(1,))
angle = np.array([1.43])
gate = RZXGate(*angle)
target = GateTarget(gate, (0, 1))
def apply_qc(qc:QuantumCircuit, params, qreg):
    qc.rzx(params[0], qreg[0], qreg[1])
    # qc.rx(params[0], qreg[0])

action_space = Box(-np.pi, np.pi, shape=(1,))
backend_config = QiskitConfig(apply_qc)
exec = ExecutionConfig(sampling_paulis=10000, n_shots=1000, batch_size=1, n_reps=list(range(1, 30)),
                       dfe_precision=(0.01, 0.01), seed=seed,
                       c_factor=1)
reward = ChannelReward()
q_env_config = QEnvConfig(target, backend_config, action_space, reward=reward,
                          execution_config=exec)
q_env = QuantumEnvironment(q_env_config)


#%%
angle
#%%
q_env.circuit.draw("mpl")
#%%
data = reward.get_reward_data(q_env.circuit, np.zeros((1, 1)), q_env.target, q_env.config)
#%%
q_env.initial_reward_fit(np.ones(1), reward_method=["fidelity", "channel"])
#%%
from qiskit.quantum_info import pauli_basis
basis = pauli_basis(target.n_qubits)
Chi = target.Chi(1)
dim = 2** target.n_qubits
pauli_pairs = []
for i, chi in enumerate(Chi):
    if abs(chi)>1e-6:
        k,l = np.unravel_index(i, (dim**2, dim**2))
        pauli_pairs.append((basis[k], basis[l], chi))
pauli_pairs
#%%
data.fiducials
#%%
