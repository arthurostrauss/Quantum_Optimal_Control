# Basic Concepts

This document explains the fundamental concepts underlying the RL-QOC framework, including reinforcement learning principles and their adaptation to quantum control tasks.

## Reinforcement Learning Fundamentals

### The RL Paradigm

Reinforcement Learning (RL) is an interaction-based learning procedure resulting from the interplay between two entities:

#### The Environment
In our quantum control context, the environment represents the quantum system of interest:
- **Quantum System**: Formed of n qubits, characterized by quantum state ρ ∈ ℋ
- **Hilbert Space**: Dimension d = 2ⁿ
- **Observations**: Measurements o_i ∈ O extracted from the quantum system
- **Observation Space**: Could be bitstrings O = {0,1}^⊗n or IQ pairs for advanced state discrimination

#### The Agent
A classical neural network whose goal is to learn an optimal policy:
- **Policy**: Stochastic policy π_θ(a|o_i) parameterized by θ
- **Actions**: a ∈ U (typically circuit/pulse parameters)
- **Goal**: Find actions enabling successful target state preparation or gate calibration

### Key RL Concepts

#### State vs Observation
- **State**: Complete description of the environment (typically unknown in quantum systems)
- **Observation**: Partial information available through measurements
- **Partial Observability**: Quantum environments are inherently partially observable since we cannot directly access quantum states

#### Reward Signal
The reward R provides learning guidance:
- **Design Principle**: Should be maximized when achieving the desired quantum control objective
- **Quantum Specificity**: Often based on fidelity measures between target and achieved states/gates
- **Challenge**: Direct fidelity computation requires full state knowledge (not available experimentally)

## Proximal Policy Optimization (PPO)

### Why PPO for Quantum Control?

PPO is particularly suited for quantum control because:

1. **Sample Efficiency**: Quantum experiments are expensive (time, resources)
2. **Stable Learning**: Prevents destructive policy updates that could damage quantum hardware
3. **Continuous Control**: Natural fit for continuous parameter optimization
4. **Trust Region**: Ensures gradual, safe policy improvements

### PPO Algorithm Principles

#### Core Objective
PPO optimizes the following objective function:

```
L^CLIP(θ) = 𝔼_t [min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio
- `Â_t` is the advantage estimate
- `ε` is the clipping parameter (typically 0.2)

#### Key Components

**1. Policy Clipping**
Prevents large policy updates by clipping the probability ratio:
```python
ratio = new_prob / old_prob
clipped_ratio = torch.clamp(ratio, 1 - ppo_epsilon, 1 + ppo_epsilon)
policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
```

**2. Value Function Learning**
Learns a value function V(s) to estimate expected returns:
```python
value_loss = F.mse_loss(values, returns)
```

**3. Advantage Estimation**
Uses Generalized Advantage Estimation (GAE) for variance reduction:
```python
advantages = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
```

### How PPO Learns from Quantum Rewards

#### Learning Process

1. **Exploration Phase**
   - Agent samples actions from current policy π_θ
   - Actions correspond to quantum circuit/pulse parameters
   - Multiple actions sampled to form a batch (efficient for quantum execution)

2. **Quantum Execution**
   - Batch of quantum circuits executed with sampled parameters
   - Measurements performed to estimate reward (e.g., via DFE)
   - Rewards computed for each action in the batch

3. **Policy Update**
   - Advantages computed from rewards and value estimates
   - Policy updated using PPO objective with clipped updates
   - Value function updated to better predict returns

4. **Iteration**
   - Process repeats with updated policy
   - Gradually converges to optimal quantum control parameters

#### Quantum-Specific Adaptations

**Batch Processing**
```python
# Standard RL: Single action per step
action = agent.get_action(obs)
reward = env.step(action)

# Quantum RL: Batch actions for efficiency
actions = agent.get_batch_actions(obs_batch)  # Shape: (batch_size, n_actions)
rewards = env.step_batch(actions)  # Efficient quantum execution
```

**Reward Estimation**
```python
# Direct Fidelity Estimation for reward computation
pauli_observables = sample_pauli_operators(target_state)
expectation_values = measure_pauli_expectations(quantum_circuit, pauli_observables)
fidelity_estimate = compute_dfe_fidelity(expectation_values, target_state)
reward = fidelity_estimate  # Use fidelity as reward
```

## Quantum Control Problem Formulation

### State Preparation Task

**Objective**: Prepare target quantum state |ψ_target⟩

**Formulation**:
- **Environment**: Quantum system initialized in |0⟩^⊗n
- **Actions**: Parameters for parametrized quantum circuit U(θ)
- **Goal**: U(θ)|0⟩^⊗n ≈ |ψ_target⟩
- **Reward**: Fidelity F(|ψ_target⟩⟨ψ_target|, ρ_prepared)

**Mathematical Foundation**:
```
argmax_θ 𝔼_{π_θ}[R] = |ψ_target⟩
```

Where the expectation is taken over actions sampled from policy π_θ.

### Gate Calibration Task

**Objective**: Calibrate quantum gate G to match target G_target

**Formulation**:
- **Environment**: Quantum system with parametrized gate G(θ)
- **Actions**: Gate parameters θ
- **Goal**: G(θ) ≈ G_target for all input states
- **Reward**: Average gate fidelity over input states

**Mathematical Foundation**:
```
F_avg(G, G_target) = ∫ dψ ⟨ψ|G†_target G(θ)|ψ⟩⟨ψ|G†(θ) G_target|ψ⟩
```

## Direct Fidelity Estimation (DFE)

### Motivation

Direct fidelity computation requires full state knowledge:
```
F(ρ, σ) = tr(ρσ)
```

This is experimentally inaccessible. DFE provides an unbiased estimator using only measurable quantities.

### DFE Protocol

**Step 1: Pauli Decomposition**
Express fidelity in terms of Pauli expectation values:
```
F(ρ, σ) = Σ_k χ_ρ(k) χ_σ(k) = Σ_k (⟨W_k⟩_ρ ⟨W_k⟩_σ) / d
```

Where χ_ρ(k) = ⟨W_k⟩_ρ/√d is the characteristic function.

**Step 2: Pauli Sampling**
Sample Pauli operators W_k with probability:
```
Pr(k) = [χ_ρ(k)]² = [⟨W_k⟩_ρ]² / d
```

**Step 3: Expectation Value Measurement**
For each sampled W_k, measure ⟨W_k⟩_σ experimentally.

**Step 4: Fidelity Estimation**
Construct unbiased estimator:
```
R = (⟨W_k⟩_σ ⟨W_k⟩_ρ) / (d · Pr(k))
```

**Unbiasedness Property**:
```
𝔼_{k∼Pr(k)}[R] = F(ρ, σ)
```

### Implementation in RL Context

```python
def compute_dfe_reward(target_state, measured_state, n_pauli_samples=100):
    """Compute DFE-based reward for state preparation."""
    
    # Step 1: Compute characteristic function for target
    chi_target = compute_characteristic_function(target_state)
    
    # Step 2: Sample Pauli operators based on probabilities
    pauli_probs = chi_target**2 / (2**n_qubits)**2
    sampled_paulis = sample_paulis(pauli_probs, n_pauli_samples)
    
    # Step 3: Measure expectation values
    measured_expectations = []
    for pauli in sampled_paulis:
        exp_val = measure_pauli_expectation(measured_state, pauli)
        measured_expectations.append(exp_val)
    
    # Step 4: Compute unbiased fidelity estimate
    fidelity_estimates = []
    for i, pauli in enumerate(sampled_paulis):
        target_exp = chi_target[pauli.index] * sqrt(2**n_qubits)
        measured_exp = measured_expectations[i]
        prob = pauli_probs[pauli.index]
        
        estimate = (target_exp * measured_exp) / (2**n_qubits * prob)
        fidelity_estimates.append(estimate)
    
    return np.mean(fidelity_estimates)
```

## Context-Aware Quantum Control

### Motivation

Traditional gate calibration assumes:
- **Gate Independence**: Each gate calibrated in isolation
- **Context Independence**: Gate performance independent of surrounding operations
- **Static Noise**: Noise characteristics don't depend on circuit context

**Reality**: Quantum gates exhibit context-dependent behavior due to:
- **Crosstalk**: Neighboring operations affect gate performance
- **Non-Markovian Noise**: Gate errors depend on operation history
- **Temporal Correlations**: Gate performance varies with timing in sequence

### Context-Aware Formulation

**Standard Gate Calibration**:
```
Optimize G(θ) to maximize F(G(θ), G_target)
```

**Context-Aware Gate Calibration**:
```
Optimize G(θ) within circuit context C to maximize F(C[G(θ)], C[G_target])
```

Where C[G] represents gate G executed within circuit context C.

### Implementation Strategy

**Step 1: Context Definition**
Define quantum circuit context containing target gate:
```python
context_circuit = QuantumCircuit(n_qubits)
# Add context operations before target gate
context_circuit.h(0)
context_circuit.cx(0, 1)
# Target gate location
context_circuit.rx(Parameter("theta"), target_qubit)
# Add context operations after target gate  
context_circuit.cx(1, 2)
context_circuit.measure_all()
```

**Step 2: Causal Cone Analysis**
Identify qubits that interact with target gate:
```python
causal_cone = analyze_causal_cone(context_circuit, target_gate_location)
# Only qubits in causal cone affect gate performance
```

**Step 3: Context-Aware Training**
Train gate parameters considering full context:
```python
# Environment includes circuit context
env = ContextAwareQuantumEnvironment(
    circuit_context=context_circuit,
    target_gate=target_gate,
    virtual_target_qubits=target_qubits
)

# Agent learns parameters for gate within context
agent.train(env)
```

## Hardware Runtime Optimization

### Motivation

Quantum control experiments are constrained by:
- **Limited Coherence Time**: Quantum states decay rapidly
- **Hardware Access Limits**: Expensive quantum computer time
- **Shot Budget**: Limited number of measurements possible

### Runtime-Aware Training

**Traditional Training**: Fixed number of policy updates
```python
training_config = TrainingConfig(
    training_constraint=TotalUpdates(1000)
)
```

**Hardware-Aware Training**: Limited by actual hardware runtime
```python
training_config = TrainingConfig(
    training_constraint=HardwareRuntime(max_runtime=3600)  # 1 hour
)
```

### Runtime Tracking

**Circuit Execution Time**:
```python
def estimate_circuit_runtime(circuit, backend):
    """Estimate circuit execution time on hardware."""
    
    # Get instruction durations from backend
    durations = backend.instruction_durations
    
    total_time = 0
    for instruction in circuit.data:
        gate_duration = durations.get(instruction.operation.name, qubits)
        total_time += gate_duration
    
    return total_time
```

**Training Budget Management**:
```python
class HardwareAwarePPO(CustomPPO):
    def train(self, training_config):
        total_runtime = 0
        max_runtime = training_config.training_constraint.max_hardware_runtime
        
        while total_runtime < max_runtime:
            # Execute training step
            step_runtime = self.execute_training_step()
            total_runtime += step_runtime
            
            # Check if budget exhausted
            if total_runtime >= max_runtime:
                break
```

## Advanced Concepts

### Multi-Circuit Training

Train on multiple circuit contexts simultaneously:
```python
circuit_contexts = [context1, context2, context3]
target = GateTarget(
    gate=RXGate(Parameter("theta")),
    circuit_context=circuit_contexts  # Multiple contexts
)

# Agent learns robust parameters across all contexts
env = ContextAwareQuantumEnvironment(target=target)
```

### Adaptive Exploration

Adjust exploration based on learning progress:
```python
def adaptive_exploration_schedule(episode, performance_history):
    """Adjust exploration noise based on learning progress."""
    
    recent_performance = np.mean(performance_history[-50:])
    
    if recent_performance > 0.95:  # High performance, reduce exploration
        return 0.1
    elif recent_performance < 0.8:  # Poor performance, increase exploration
        return 0.5
    else:  # Moderate performance, standard exploration
        return 0.2
```

### Hierarchical Control

Multi-level quantum control optimization:
```python
# High-level: Circuit structure optimization
circuit_agent = StructureOptimizationAgent()

# Low-level: Gate parameter optimization  
gate_agent = ParameterOptimizationAgent()

# Coordinated training
for episode in range(n_episodes):
    circuit_structure = circuit_agent.get_action(state)
    gate_params = gate_agent.get_action(state, circuit_structure)
    
    reward = env.step(circuit_structure, gate_params)
    
    circuit_agent.update(reward)
    gate_agent.update(reward)
```

This foundation enables sophisticated quantum control strategies that adapt to the unique challenges and opportunities of quantum computing platforms.