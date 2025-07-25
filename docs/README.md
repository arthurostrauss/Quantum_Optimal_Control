# RL-QOC Documentation

Welcome to the documentation for the RL-QOC (Reinforcement Learning for Quantum Optimal Control) package. This documentation provides comprehensive guides and API references for using the framework.

## Quick Navigation

### Getting Started
- [Installation Guide](installation.md)
- [Quick Start Tutorial](tutorials/quickstart.md)
- [Basic Concepts](concepts/basic_concepts.md)

### Core Components
- [Environment System](api/environment.md) - Quantum environments for RL training
- [Agent Architecture](api/agent.md) - PPO-based RL agents
- [Reward Systems](api/rewards.md) - Different reward computation methods
- [Configuration System](api/configuration.md) - Environment and training configuration

### Platform Integrations
- [QUA Integration](integrations/qua.md) - Real-time control with Quantum Orchestration Platform
- [Qiskit Integration](integrations/qiskit.md) - IBM Qiskit backend support
- [Qibo Integration](integrations/qibo.md) - Qibo platform support

### Advanced Features
- [Context-Aware Calibration](advanced/context_aware.md) - Circuit context-aware gate calibration
- [Noise Modeling](advanced/noise_modeling.md) - Advanced noise simulation capabilities
- [Hyperparameter Optimization](advanced/hpo.md) - Automated hyperparameter tuning
- [Real-time Control Flow](advanced/real_time_control.md) - Real-time quantum circuit execution

### Tutorials and Examples
- [Gate-Level Calibration](tutorials/gate_level.md)
- [Pulse-Level Control](tutorials/pulse_level.md)
- [State Preparation](tutorials/state_preparation.md)
- [Spillover Noise Use Cases](tutorials/spillover_noise.md)

### API Reference
- [Complete API Documentation](api/index.md)
- [Class Hierarchies](api/class_hierarchies.md)
- [Function Reference](api/functions.md)

## Key Features

- **Multi-Platform Support**: Seamless integration with Qiskit, Qibo, and QUA platforms
- **Real-Time Control**: Action sampling offloaded directly to quantum control systems
- **Context-Aware Calibration**: Gate calibration tailored to specific circuit contexts
- **Advanced PPO Implementation**: Custom PPO agent optimized for quantum control tasks
- **Flexible Reward Systems**: Multiple reward computation methods (DFE, CAFE, ORBIT, XEB)
- **Hardware Runtime Optimization**: Training constraints based on actual hardware runtime
- **Noise Modeling**: Comprehensive noise simulation including spillover effects

## Contributing

Please see our [contributing guidelines](../CONTRIBUTING.md) for information on how to contribute to this project.