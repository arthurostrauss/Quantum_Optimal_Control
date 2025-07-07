# API Reference

This section provides comprehensive documentation for all modules, classes, and functions in the `rl_qoc` package.

## Package Structure

```
rl_qoc/
├── environment/           # Quantum environment implementations
├── agent/                # RL agents and PPO implementation
├── qua/                  # QUA integration for real-time control
├── rewards/              # Reward schemes and utilities
├── helpers/              # Utility functions and tools
├── hpo/                  # Hyperparameter optimization
├── qibo/                 # QIBO integration
└── custom_jax_sim/       # Custom JAX-based simulation
```

## Core Modules

### [Environment](environment/)
Quantum environment implementations for different abstraction levels and backends:
- `BaseQuantumEnvironment`: Base class for all quantum environments
- `QuantumEnvironment`: Standard quantum environment with Qiskit backend
- `ContextAwareQuantumEnvironment`: Environment for context-aware gate calibration
- Configuration classes for backends and execution parameters

### [Agent](agent/)
Reinforcement learning agents and training algorithms:
- `CustomPPO`: Proximal Policy Optimization implementation for quantum control
- `Agent`: Base agent class with actor-critic networks
- Configuration classes for training parameters
- Logging and utilities for training

### [QUA Integration](qua/)
Real-time quantum control capabilities with QUA/Quantum Orchestration Platform:
- `QMEnvironment`: Quantum environment for QUA backends
- `CustomQMPPO`: PPO implementation with QUA real-time action sampling
- `QuaEstimator`: Estimator primitive for QUA backends
- Transpiler passes and utilities for QUA integration

### [Rewards](rewards/)
Reward computation schemes for different quantum control tasks:
- `StateReward`: Reward for quantum state preparation
- `ChannelReward`: Reward for quantum channel calibration
- `CAFEReward`: Contextual fidelity estimation reward
- `FidelityReward`: Direct fidelity-based rewards
- Real-time utilities for reward computation

### [Helpers](helpers/)
Utility functions and tools:
- Circuit manipulation and analysis utilities
- Pulse-level control utilities
- Transpiler passes for quantum circuit optimization
- TensorFlow utilities for legacy compatibility

### [HPO](hpo/)
Hyperparameter optimization capabilities:
- `HyperparameterOptimizer`: Optuna-based hyperparameter optimization
- `HPOConfig`: Configuration for hyperparameter optimization
- Integration with WandB for experiment tracking

## Key Classes and Functions

### Main Exports
```python
from rl_qoc import (
    # Environments
    QuantumEnvironment, 
    ContextAwareQuantumEnvironment,
    
    # Agents  
    CustomPPO,
    Agent,
    ActorNetwork,
    CriticNetwork,
    PPOConfig,
    
    # Rewards
    StateReward,
    ChannelReward, 
    CAFEReward,
    FidelityReward,
    ORBITReward,
    XEBReward,
    
    # Configuration
    QEnvConfig,
    ExecutionConfig,
    BenchmarkConfig,
    
    # QUA Integration
    QMEnvironment,
    CustomQMPPO,
    QMConfig,
    
    # Wrappers
    RescaleAndClipAction
)
```

### Configuration Classes
- `QEnvConfig`: Main environment configuration
- `ExecutionConfig`: Execution parameters (batch size, shots, etc.)
- `PPOConfig`: PPO algorithm configuration loaded from YAML
- `QMConfig`: QUA backend configuration for real-time control
- `BackendConfig`: Backend-specific configuration (Qiskit, Dynamics)

## Usage Patterns

### Basic Gate Calibration
```python
from rl_qoc import (
    QuantumEnvironment, CustomPPO, PPOConfig, 
    QEnvConfig, ExecutionConfig, RescaleAndClipAction
)

# Configure environment
q_env_config = QEnvConfig(
    target={"gate": CXGate(), "physical_qubits": [0, 1]},
    backend_config=backend_config,
    action_space=action_space,
    execution_config=ExecutionConfig(
        batchsize=300,
        sampling_Paulis=50, 
        N_shots=200
    )
)

# Create environment and agent
q_env = QuantumEnvironment(q_env_config)
rescaled_env = RescaleAndClipAction(q_env, -1.0, 1.0)

# Load PPO configuration from YAML
agent_config = PPOConfig.from_yaml("agent_config.yaml")
ppo_agent = CustomPPO(agent_config, rescaled_env)

# Train
results = ppo_agent.train()
```

### Context-Aware Gate Calibration
```python
from rl_qoc import ContextAwareQuantumEnvironment

# Configure for context-aware calibration
q_env = ContextAwareQuantumEnvironment(
    q_env_config, 
    circuit_context, 
    training_steps_per_gate=250
)
rescaled_env = RescaleAndClipAction(q_env, -1.0, 1.0)

# Use same PPO setup
ppo_agent = CustomPPO(agent_config, rescaled_env)
results = ppo_agent.train()
```

### Real-time QUA Control
```python
from rl_qoc.qua import QMEnvironment, CustomQMPPO

# Configure QUA environment
qm_env = QMEnvironment(qua_config)

# Create QUA PPO agent  
qm_agent = CustomQMPPO(ppo_config, qm_env)

# Start real-time program
qm_env.start_program()

# Train with real-time control
results = qm_agent.train()

# Close environment
qm_env.close()
```

## Advanced Features

- **Real-time Control Flow**: Embed RL decision-making directly in quantum circuits
- **Hardware-Aware Optimization**: Leverage platform-specific capabilities
- **Multi-Backend Support**: Seamless switching between simulation and hardware
- **Scalable Architectures**: From single qubits to multi-qubit systems
- **Custom Reward Schemes**: Flexible reward computation for various control tasks