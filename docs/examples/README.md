# Examples

Practical examples demonstrating the key capabilities of the rl_qoc package, based on the available notebooks and real usage patterns.

## 📁 Repository Structure

The examples are organized into different folders based on the abstraction level and use case:

```
├── gate_level/                    # Gate-level quantum control
│   ├── torch_gate_cal.ipynb      # Context-aware gate calibration with PyTorch
│   ├── standard/                  # Standard gate calibration examples
│   └── spillover_noise_use_case/  # Noise characterization examples
├── pulse_level/                   # Pulse-level quantum control  
│   ├── torch_pulse_cal.ipynb     # Pulse-level calibration with PyTorch
│   ├── qua/                       # QUA/OPX real-time control
│   ├── qibo/                      # QIBO integration examples
│   └── qiskit_pulse/             # Qiskit pulse-level examples
└── serverless_files/             # Cloud deployment examples
```

## 🎯 Key Examples

### 1. Context-Aware Gate Calibration (`gate_level/torch_gate_cal.ipynb`)

Demonstrates context-aware calibration of quantum gates using the PyTorch-based CustomPPO agent.

**Key Features:**
- Context-aware gate calibration for CX gates
- Custom parametrized circuit using U gates and RZX
- FakeJakarta backend simulation
- Real-time training visualization with TensorBoard

**Core Components:**
```python
from rl_qoc import ContextAwareQuantumEnvironment, CustomPPO
from rl_qoc.agent import ActorNetwork, CriticNetwork, Agent

# Define parametrized circuit
def param_circuit(qc, params, q_reg, **kwargs):
    """Custom CX gate parametrization using U gates and RZX."""
    optimal_params = np.pi * np.array([0.0, 0.0, 0.5, 0.5, -0.5, 0.5, -0.5])
    
    qc.u(optimal_params[0] + params[0], optimal_params[1] + params[1], 
         optimal_params[2] + params[2], q_reg[0])
    qc.u(optimal_params[3] + params[3], optimal_params[4] + params[4], 
         optimal_params[5] + params[5], q_reg[1])
    qc.rzx(optimal_params[6] + params[6], q_reg[0], q_reg[1])

# Create context-aware environment
env = ContextAwareQuantumEnvironment(env_config, target_circuit, training_steps_per_gate)
rescaled_env = RescaleAction(ClipAction(env), -1.0, 1.0)

# Manual PPO training loop (pre-CustomPPO integration)
for update in range(num_updates):
    # Reset environment and collect experiences
    # Compute advantages using GAE
    # Update policy using PPO loss
```

### 2. Pulse-Level Gate Calibration (`pulse_level/torch_pulse_cal.ipynb`)

Demonstrates pulse-level calibration of ECR gates using Qiskit Dynamics backend.

**Key Features:**
- Pulse-level ECR gate parametrization
- Qiskit Dynamics backend simulation
- Custom pulse schedule modification
- Echoed Cross Resonance (ECR) gate optimization

**Core Components:**
```python
from qiskit_dynamics import DynamicsBackend
from rl_qoc.helpers import get_ecr_params

def custom_schedule(backend, physical_qubits, params, keep_symmetry=True):
    """Define parametrization of pulse schedule for ECR gate."""
    pulse_features = ["amp", "angle", "tgt_amp", "tgt_angle"]
    new_params, _, _, _ = get_ecr_params(backend, physical_qubits)
    
    # Modify ECR pulse parameters
    for sched in ["cr45p", "cr45m"]:
        for i, feature in enumerate(pulse_features):
            new_params[(feature, qubits, sched)] += params[i]
    
    return calibrations.get_schedule("ecr", physical_qubits, assign_params=new_params)

# Create Dynamics backend
dynamics_backend = DynamicsBackend.from_backend(
    fake_backend_v2, 
    subsystem_list=physical_qubits,
    **dynamics_options
)
```

### 3. Standard Gate Calibration (`gate_level/standard/gate_level_learning.ipynb`)

Demonstrates the standard workflow using the modern CustomPPO agent.

**Key Features:**
- Modern CustomPPO agent usage
- YAML-based configuration
- Training with different constraints
- Performance monitoring and visualization

**Core Components:**
```python
from rl_qoc.agent import CustomPPO, TrainingConfig, TotalUpdates

# Load configuration from YAML
agent_config = PPOConfig.from_yaml("agent_config.yaml")

# Create PPO agent
ppo_agent = CustomPPO(agent_config, rescaled_env)

# Configure training
training_config = TrainingConfig(
    TotalUpdates(200),
    target_fidelities=[0.9],
    lookback_window=20
)

# Train with real-time plotting
results = ppo_agent.train(training_config, train_settings)
```

### 4. QUA Real-Time Control (`pulse_level/qua/`)

Real-time quantum control using Quantum Machines' QUA platform.

**Key Features:**
- Real-time parameter streaming to OPX
- Hardware-based action sampling
- CustomQMPPO agent for QUA integration
- Minimal latency quantum control

**Core Workflow:**
```python
from rl_qoc.qua import QMEnvironment, CustomQMPPO

# Configure QUA environment
qm_env = QMEnvironment(qua_config)

# Create QUA-specific PPO agent
qm_agent = CustomQMPPO(ppo_config, qm_env)

# Start real-time program
job = qm_env.start_program()

# Train with real-time control
results = qm_agent.train()

# Clean up
qm_env.close()
```

## 🔧 Configuration Examples

### Environment Configuration

```python
from rl_qoc import QEnvConfig, ExecutionConfig
from rl_qoc.environment.configuration.backend_config import QiskitRuntimeConfig

# Backend configuration
backend_config = QiskitRuntimeConfig(
    parametrized_circuit=param_circuit,
    backend=backend,
    primitive_options=estimator_options
)

# Execution configuration
execution_config = ExecutionConfig(
    batchsize=300,           # Batch size for RL training
    sampling_Paulis=50,      # Number of Pauli observables to sample
    N_shots=200,             # Shots per measurement
    seed=10                  # Random seed
)

# Main environment configuration
q_env_config = QEnvConfig(
    target={"gate": CXGate(), "physical_qubits": [0, 1]},
    backend_config=backend_config,
    action_space=action_space,
    execution_config=execution_config
)
```

### Agent Configuration (YAML)

```yaml
# agent_config.yaml
run_name: "gate_calibration_experiment"
learning_rate: 0.0003
batch_size: 10
n_epochs: 4
gamma: 0.99
gae_lambda: 0.95
ppo_epsilon: 0.2

# Network architecture
hidden_layers: [64, 64]
hidden_activation_functions: ["Tanh", "Tanh"]
output_activation_mean: "Tanh"
output_activation_std: "Softplus"

# Training constraints
total_updates: 1000
target_fidelities: [0.9, 0.95, 0.99]
lookback_window: 20

# Optimization settings
anneal_learning_rate: true
normalize_advantage: true
entropy_coefficient: 0.01
```

## 📊 Training and Monitoring

### Real-time Monitoring

```python
# TensorBoard integration
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"runs/{run_name}")

# Track key metrics during training
writer.add_scalar("charts/episodic_return", np.mean(reward), global_step)
writer.add_scalar("losses/circuit_fidelity", env.circuit_fidelity_history[-1], global_step)
writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
```

### Performance Analysis

```python
# Plot learning curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(env.circuit_fidelity_history, label="Circuit Fidelity")
plt.plot(np.mean(env.reward_history, axis=1), label="Average Reward")
plt.xlabel("Updates")
plt.ylabel("Performance")
plt.legend()

plt.subplot(1, 3, 2) 
plt.plot(np.cumsum(env.total_shots), label="Total Shots")
plt.xlabel("Updates")
plt.ylabel("Cumulative Shots")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(ppo_agent.training_results["std_action"], label="Action Std")
plt.xlabel("Updates") 
plt.ylabel("Exploration")
plt.legend()

plt.tight_layout()
plt.show()
```

## 🚀 Getting Started

### 1. Choose Your Abstraction Level
- **Gate-level**: Use `torch_gate_cal.ipynb` for circuit-level parameters
- **Pulse-level**: Use `torch_pulse_cal.ipynb` for pulse-level control

### 2. Select Your Backend
- **Simulation**: Use FakeBackend or AerSimulator
- **Hardware**: Configure IBM Quantum backend via QiskitRuntimeService
- **Pulse simulation**: Use Qiskit Dynamics DynamicsBackend

### 3. Configure Your Target
- **Gate calibration**: Specify gate and physical qubits
- **State preparation**: Provide target circuit or density matrix

### 4. Train Your Agent
- Use CustomPPO for modern PyTorch-based training
- Monitor progress with TensorBoard
- Analyze results with built-in plotting utilities

## 📈 Advanced Use Cases

### Spillover Noise Characterization
Located in `gate_level/spillover_noise_use_case/`, these examples demonstrate:
- Modeling spillover noise effects
- Context-dependent error characterization  
- Adaptive noise mitigation strategies

### Multi-Platform Integration
Examples showing integration with different quantum platforms:
- **QIBO**: `pulse_level/qibo/` for QIBO quantum framework
- **QUA**: `pulse_level/qua/` for real-time Quantum Machines control
- **IBM Quantum**: Throughout examples using QiskitRuntimeService

### Cloud Deployment
Examples in `serverless_files/` demonstrate:
- Serverless quantum control deployment
- Cloud-based training workflows
- Scalable quantum experiments

The examples provide a comprehensive foundation for implementing quantum control with reinforcement learning across different platforms, abstraction levels, and use cases.