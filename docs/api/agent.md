# Agent System API

The agent system implements Proximal Policy Optimization (PPO) specifically adapted for quantum control tasks.

## PPO Agent Architecture

### CustomPPO

Main PPO implementation optimized for quantum environments.

```python
class CustomPPO:
    """
    Custom PPO agent implementation for quantum control.
    
    Implements Proximal Policy Optimization with modifications for:
    - Batch action submission to quantum environments
    - Hardware runtime constraints
    - Quantum-specific reward processing
    """
```

**Key Features:**
- Batch action processing for efficient quantum circuit execution
- Hardware-aware training constraints
- Learning rate annealing based on convergence criteria
- TensorBoard integration for monitoring

**Initialization:**
```python
def __init__(
    self,
    agent_config: Union[Dict, PPOConfig, str],
    env: BaseQuantumEnvironment,
    chkpt_dir: Optional[str] = "tmp/ppo",
    save_data: Optional[bool] = False
)
```

**Parameters:**
- `agent_config`: PPO configuration (dict, PPOConfig object, or YAML file path)
- `env`: Quantum environment for training
- `chkpt_dir`: Directory for saving checkpoints
- `save_data`: Whether to save training data

### Actor-Critic Networks

#### ActorNetwork

Neural network for policy representation.

```python
class ActorNetwork(nn.Module):
    """
    Actor network for policy learning.
    
    Outputs mean and standard deviation for continuous action spaces.
    """
```

**Architecture:**
- Configurable hidden layers with customizable activation functions
- Separate outputs for action mean and standard deviation
- Support for different output activation functions

**Key Methods:**

#### `forward(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
Forward pass through the network.

**Returns:**
- `mean_action`: Mean of action distribution
- `std_action`: Standard deviation of action distribution  
- `critic_value`: Value function estimate

#### CriticNetwork

Neural network for value function estimation.

```python
class CriticNetwork(nn.Module):
    """
    Critic network for value function estimation.
    
    Estimates the expected return from given states.
    """
```

## Training Configuration

### PPOConfig

Configuration dataclass for PPO training.

```python
@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    
    # Network architecture
    hidden_layers: Sequence[int] = (64, 64)
    input_activation_function: str = "identity"
    hidden_activation_functions: str = "tanh"
    output_activation_mean: str = "tanh"
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    minibatch_size: int = 64
    n_epochs: int = 10
    
    # PPO-specific parameters
    ppo_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Regularization
    normalize_advantage: bool = True
    clip_value_loss: bool = True
    max_grad_norm: float = 0.5
```

### TrainingConfig

Configuration for training procedures.

```python
@dataclass  
class TrainingConfig:
    """Configuration for training procedures."""
    
    training_constraint: Union[TotalUpdates, HardwareRuntime]
    target_fidelities: Optional[List[float]] = None
    lr_annealing: bool = False
    std_annealing: bool = False
    convergence_check: bool = False
```

#### Training Constraints

**TotalUpdates**: Train for a fixed number of policy updates
```python
@dataclass
class TotalUpdates:
    total_updates: int
```

**HardwareRuntime**: Train until maximum hardware runtime is reached
```python
@dataclass  
class HardwareRuntime:
    max_hardware_runtime: float  # in seconds
```

## PPO Algorithm Details

### Core PPO Principles

PPO is a policy gradient method that maintains a balance between exploration and exploitation while ensuring stable learning. Key principles:

1. **Policy Gradient**: Updates policy parameters in the direction of higher expected rewards
2. **Trust Region**: Constrains policy updates to prevent destructive changes
3. **Clipped Objective**: Uses ratio clipping instead of KL divergence constraints
4. **Actor-Critic**: Combines policy learning (actor) with value function estimation (critic)

### Quantum-Specific Adaptations

#### Batch Action Processing
```python
# Standard RL: Single action per step
action = agent.get_action(obs)
obs, reward, done, info = env.step(action)

# Quantum RL: Batch actions for efficient circuit execution  
actions = agent.get_batch_actions(obs_batch)  # Shape: (batch_size, n_actions)
obs_batch, rewards, dones, infos = env.step(actions)
```

#### Hardware Runtime Tracking
```python
class CustomPPO:
    def train(self, training_config: TrainingConfig):
        if isinstance(training_config.training_constraint, HardwareRuntime):
            # Train until hardware runtime limit reached
            while self.total_hardware_runtime < training_config.training_constraint.max_hardware_runtime:
                # Training loop with runtime monitoring
                pass
```

#### Learning Rate Annealing
```python
def update_learning_rate(self, progress: float):
    """Update learning rate based on training progress."""
    if self.config.lr_annealing:
        new_lr = self.config.learning_rate * (1 - progress)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
```

### QUA-Specific PPO Implementation

#### CustomQMPPO

Specialized PPO for QUA platform integration.

```python
class CustomQMPPO(CustomPPO):
    """
    PPO implementation optimized for QUA platform.
    
    Implements real-time action sampling directly on quantum hardware.
    """
```

**Key Features:**
- Action sampling offloaded to OPX hardware
- Hardware-optimized Box-Muller transformation for Gaussian sampling
- Minimal latency between policy updates and quantum execution

#### Real-Time Action Sampling

```python
def process_action(self, probs: Normal) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process actions for real-time execution on QUA platform.
    
    Implements efficient Gaussian sampling using lookup tables and
    linear congruential generators for hardware execution.
    """
    # Hardware-optimized Box-Muller transformation
    # Uses pre-computed lookup tables for trigonometric functions
    # Implements on OPX for real-time action sampling
```

## Training Workflow

### Standard Training Loop

```python
# Initialize agent and environment
agent = CustomPPO(ppo_config, env)
training_config = TrainingConfig(
    training_constraint=TotalUpdates(1000),
    lr_annealing=True
)

# Training
agent.train(training_config)
```

### Hardware-Constrained Training

```python
# Train until hardware budget exhausted
training_config = TrainingConfig(
    training_constraint=HardwareRuntime(max_hardware_runtime=3600),  # 1 hour
    target_fidelities=[0.99, 0.995, 0.999],
    convergence_check=True
)

agent.train(training_config)
```

### QUA Real-Time Training

```python
# QUA-optimized training with real-time action sampling
qm_agent = CustomQMPPO(ppo_config, qm_env)

# Start QUA program for real-time execution
qm_env.start_program()

# Training with hardware-accelerated action sampling
qm_agent.train(training_config)

# Close QUA environment
qm_env.close()
```

## Monitoring and Logging

### TensorBoard Integration

```python
# Enable TensorBoard logging
ppo_config.tensorboard_log = True

# Logged metrics:
# - episodic_return: Reward per episode
# - policy_loss: Policy gradient loss
# - value_loss: Critic loss  
# - entropy: Policy entropy
# - learning_rate: Current learning rate
# - hardware_runtime: Cumulative hardware runtime
```

### WandB Integration

```python
@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging."""
    project_name: str
    run_name: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List[str]] = None
```

## Advanced Features

### Hyperparameter Optimization

Integration with hyperparameter optimization frameworks:

```python
from rl_qoc.hpo import HyperparameterOptimizer, HPOConfig

hpo_config = HPOConfig(
    study_name="quantum_control_hpo",
    n_trials=100,
    parameter_ranges={
        "learning_rate": (1e-5, 1e-2),
        "batch_size": [128, 256, 512],
        "ppo_epsilon": (0.1, 0.3)
    }
)

optimizer = HyperparameterOptimizer(hpo_config)
best_params = optimizer.optimize(env, base_config)
```

### Convergence Criteria

Automatic convergence detection based on action standard deviation:

```python
def check_convergence(self) -> bool:
    """Check if training has converged based on action std."""
    if len(self.action_std_history) < 50:
        return False
    
    recent_std = np.mean(self.action_std_history[-50:])
    return recent_std < self.convergence_threshold
```

## Usage Examples

### Basic PPO Training

```python
from rl_qoc import CustomPPO, PPOConfig
from rl_qoc.agent.ppo_config import TrainingConfig, TotalUpdates

# Configure PPO
ppo_config = PPOConfig(
    learning_rate=3e-4,
    batch_size=256,
    n_epochs=10,
    ppo_epsilon=0.2
)

# Create agent
agent = CustomPPO(ppo_config, env)

# Train agent  
training_config = TrainingConfig(
    training_constraint=TotalUpdates(1000)
)
agent.train(training_config)

# Save trained model
agent.save_checkpoint()
```

### Advanced Training with Monitoring

```python
# Advanced configuration with monitoring
ppo_config = PPOConfig(
    learning_rate=3e-4,
    lr_annealing=True,
    tensorboard_log=True,
    wandb_config=WandBConfig(
        project_name="quantum_gate_calibration",
        tags=["rx_gate", "context_aware"]
    )
)

# Training with convergence checking
training_config = TrainingConfig(
    training_constraint=HardwareRuntime(1800),  # 30 minutes
    target_fidelities=[0.99, 0.995],
    convergence_check=True,
    lr_annealing=True
)

agent = CustomPPO(ppo_config, env)
agent.train(training_config)
```