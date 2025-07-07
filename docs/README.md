# RL-QOC Documentation

Welcome to the comprehensive documentation for the **Reinforcement Learning for Quantum Optimal Control (RL-QOC)** package.

## Documentation Structure

### 📚 [API Reference](api/)
Complete API documentation for all modules and classes:
- **Environment**: Quantum environments and configurations
- **Agent**: RL agents and PPO implementation  
- **QUA Integration**: Real-time quantum control with QUA
- **Rewards**: Reward schemes and real-time utilities
- **Helpers**: Utility functions and transpiler passes
- **HPO**: Hyperparameter optimization

### 🎯 [Examples](examples/)
Practical examples demonstrating key capabilities:
- Basic quantum state preparation
- Gate-level and pulse-level calibration
- Context-aware quantum control
- QUA real-time integration examples
- Multi-platform usage (Qiskit, QUA, QIBO)

### 📖 [Tutorials](tutorials/)
Step-by-step guides for getting started:
- Setting up quantum environments
- Implementing custom reward schemes
- Configuring PPO for quantum control
- Real-time control with QUA
- Advanced topics and best practices

### 📋 [Reference](reference/)
Technical specifications and background:
- Mathematical foundations
- Algorithm details
- Configuration schemas
- Hardware integration guides

## Quick Start

To explore the API, start with the [Environment API](api/environment/) to understand how quantum environments are configured, then move to the [Agent API](api/agent/) to learn about the RL implementation.

For practical examples, check out the [Basic Examples](examples/basic/) section.

## Key Features Documented

- **Real-time Quantum Control**: Integration with QUA for embedding control flow directly in quantum circuits
- **Context-Aware Calibration**: Adaptive gate calibration that accounts for circuit context and noise correlations  
- **PPO on Quantum Platforms**: Proximal Policy Optimization with parts of the algorithm offloaded to the Quantum Orchestration Platform
- **Multi-Framework Support**: Seamless integration with Qiskit, QUA, and QIBO
- **Scalable Architectures**: From single-qubit to multi-qubit quantum systems

## Getting Help

- For API questions, see the [API Reference](api/)
- For implementation guidance, check the [Examples](examples/) 
- For conceptual understanding, read the [Tutorials](tutorials/)
- For technical details, consult the [Reference](reference/) section