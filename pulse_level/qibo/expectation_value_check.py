from qiskit import QuantumRegister
import numpy as np
from gymnasium.spaces import Box
from qiskit.quantum_info import Statevector, SparsePauliOp

from rl_qoc import QuantumEnvironment, BenchmarkConfig, StateRewardConfig
from qiskit.circuit import QuantumCircuit, ParameterVector, Gate
from qiskit.circuit.library import CZGate, RXGate, XGate
from rl_qoc import (
    QEnvConfig,
    ExecutionConfig,
    ChannelRewardConfig,
)
from rl_qoc.qibo import QiboConfig
from gymnasium.wrappers import ClipAction


def param_circuit(
    qc: QuantumCircuit, params: ParameterVector, qreg: QuantumRegister, **kwargs
):
    target = kwargs["target"]
    gate: Gate = target.get("gate", "x")
    if gate == "x":
        gate_name = "x"
    else:
        gate_name = gate.name
    physical_qubits = target["physical_qubits"]
    custom_gate = Gate(f"{gate_name}_cal", len(physical_qubits), params.params)
    qc.append(custom_gate, qreg)

    return qc


def get_backend():
    return "qibolab"


target = {"state": Statevector.from_label("1"), "physical_qubits": [0]}
instruction_durations = {}
action_space_low = np.array([0.0], dtype=np.float32)  # [amp, phase, phase, duration]
action_space_high = np.array([0.5], dtype=np.float32)  # [amp, phase, phase, duration]
action_space = Box(action_space_low, action_space_high)
qibo_config = QiboConfig(
    param_circuit,
    get_backend(),
    platform="qw11q",
    physical_qubits=(["D1"]),
    gate_rule="x",
    parametrized_circuit_kwargs={"target": target},
    instruction_durations=None,
)
q_env_config = QEnvConfig(
    target=target,
    backend_config=qibo_config,
    action_space=action_space,
    reward_config=StateRewardConfig(),
    benchmark_config=BenchmarkConfig(0),
    execution_config=ExecutionConfig(
        batch_size=10, sampling_paulis=50, n_shots=1000, n_reps=1
    ),
)

env = QuantumEnvironment(q_env_config)
rescaled_env = ClipAction(env)
estimator = env.estimator

observable = SparsePauliOp.from_list([("Z", 1)])
qc = QuantumCircuit(1)
qc.x(0)

# for amplitude in np.arange(0,0.09, 0.001):
amplitude = 0.0
results = env.step(np.array([[amplitude]] * 10))
print(results)
