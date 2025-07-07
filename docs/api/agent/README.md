# Agent API

The agent module implements reinforcement learning algorithms optimized for quantum control tasks. The core implementation is based on Proximal Policy Optimization (PPO) with custom adaptations for quantum environments.

## Core Classes

### CustomPPO

Advanced PPO implementation designed specifically for quantum control applications.

```python
class CustomPPO:
    """
    Custom Proximal Policy Optimization (PPO) agent for quantum control.
    
    Features:
    - Batch action submission to quantum environments
    - Hardware-aware training constraints
    - Real-time integration with quantum platforms
    - Automatic hyperparameter optimization
    - Advanced logging and monitoring
    """
```

#### Key Methods

```python
def __init__(
    self, 
    agent_config: Union[Dict, PPOConfig, str],
    env: BaseQuantumEnvironment,
    chkpt_dir: Optional[str] = "tmp/ppo",
    chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
    save_data: bool = False
):
    """Initialize PPO agent with configuration and environment."""

def train(
    self, 
    training_config: Optional[TrainingConfig] = None,
    train_function_settings: Optional[TrainFunctionSettings] = None
) -> Dict:
    """Train the agent using PPO algorithm."""

def process_action(self, probs: Normal) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process actions before sending to environment."""

def close(self) -> None:
    """Clean up agent resources."""
```

#### Usage Example

```python
from rl_qoc import CustomPPO, PPOConfig, QuantumEnvironment

# Configure PPO agent
ppo_config = PPOConfig(
    learning_rate=3e-4,
    batch_size=10,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_epsilon=0.2,
    entropy_coefficient=0.01
)

# Create agent
agent = CustomPPO(ppo_config, env)

# Train with different constraints
results = agent.train(
    training_config=TrainingConfig(
        total_updates=1000,
        target_fidelities=[0.95, 0.99]
    )
)
```

### Agent (Neural Networks)

Base agent class containing the actor-critic neural network architecture.

```python
class Agent(nn.Module):
    """
    Actor-Critic neural network for quantum control.
    
    Architecture:
    - Shared hidden layers
    - Separate actor and critic heads
    - Customizable activation functions
    - Batch normalization support
    """
```

#### Key Components

```python
def __init__(
    self,
    observation_space: gym.Space,
    hidden_layers: List[int],
    n_actions: int,
    hidden_activation: str = "ReLU",
    output_activation_mean: str = "Tanh", 
    output_activation_std: str = "Softplus"
):
    """Initialize actor-critic network."""

def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass returning mean, std, and value."""

def get_value(self, x: torch.Tensor) -> torch.Tensor:
    """Get value estimate from critic."""

def get_action_and_value(
    self, 
    x: torch.Tensor, 
    action: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get action, log probability, entropy, and value."""
```

### CustomQMPPO (QUA Integration)

Specialized PPO implementation for real-time quantum control with QUA.

```python
class CustomQMPPO(CustomPPO):
    """
    PPO implementation with QUA real-time action sampling.
    
    Key Features:
    - Policy parameters streamed to OPX in real-time
    - Actions sampled directly on quantum hardware
    - Linear Congruential Generator for deterministic sampling
    - Fixed-point arithmetic for hardware compatibility
    """
```

#### Real-time Action Processing

```python
def process_action(self, probs: Normal) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process actions for real-time execution on QUA platform.
    
    This method streams policy parameters (mean, std) to the OPX
    and generates actions using hardware-based sampling.
    """
```

#### QUA-Specific Features

- **Hardware-Based Sampling**: Actions sampled directly on OPX using LCG
- **Fixed-Point Arithmetic**: Ensures deterministic behavior on quantum hardware
- **Real-time Parameter Streaming**: Policy parameters sent to OPX in real-time
- **Minimal Latency**: Optimized for low-latency quantum control

#### Usage Example

```python
from rl_qoc.qua import CustomQMPPO, QMEnvironment

# Create QUA environment
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

## Configuration Classes

### PPOConfig

Comprehensive configuration for PPO training.

```python
@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    
    # Core PPO parameters
    learning_rate: float = 3e-4
    batch_size: int = 10
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epsilon: float = 0.2
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    hidden_activation_functions: List[str] = field(default_factory=lambda: ["ReLU", "ReLU"])
    output_activation_mean: str = "Tanh"
    output_activation_std: str = "Softplus"
    
    # Training constraints
    training_constraint: Union[TotalUpdates, HardwareRuntime] = TotalUpdates(1000)
    
    # Optimization settings
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    grad_clip: float = 0.5
    entropy_coefficient: float = 0.01
    critic_loss_coefficient: float = 0.5
    
    # Advanced features
    anneal_learning_rate: bool = True
    normalize_advantage: bool = True
    clip_vloss: bool = True
    
    # Monitoring and logging
    save_data: bool = False
    wandb_config: WandBConfig = field(default_factory=WandBConfig)
    run_name: str = "ppo_quantum_control"
```

### TrainingConfig

Configuration for training procedures and constraints.

```python
@dataclass  
class TrainingConfig:
    """Training configuration and constraints."""
    
    # Training constraints
    total_updates: Optional[int] = None
    max_hardware_runtime: Optional[float] = None  # in seconds
    
    # Performance targets
    target_fidelities: Optional[List[float]] = None
    lookback_window: int = 10
    
    # Early stopping
    convergence_threshold: float = 1e-6
    patience: int = 50
    
    # Curriculum learning
    curriculum_schedule: Optional[Dict] = None
```

### TrainFunctionSettings

Settings for training loop behavior.

```python
@dataclass
class TrainFunctionSettings:
    """Settings for training function behavior."""
    
    # Display options
    plot_real_time: bool = False
    print_debug: bool = False
    num_prints: int = 100
    
    # Data management  
    clear_history: bool = False
    save_data: bool = False
    
    # HPO mode
    hpo_mode: bool = False
```

## Training Constraints

### TotalUpdates

Constraint based on number of policy updates.

```python
@dataclass
class TotalUpdates:
    """Training constraint based on total updates."""
    
    total_updates: int
    constraint_name: str = "total_updates"
    
    @property
    def constraint_value(self) -> int:
        return self.total_updates
```

### HardwareRuntime

Constraint based on hardware execution time.

```python
@dataclass
class HardwareRuntime:
    """Training constraint based on hardware runtime."""
    
    hardware_runtime: float  # in seconds
    constraint_name: str = "hardware_runtime"
    
    @property  
    def constraint_value(self) -> float:
        return self.hardware_runtime
```

## Advanced Features

### Curriculum Learning

```python
# Configure curriculum learning
curriculum_config = {
    "episode_length": {
        "schedule": [1, 2, 4, 8, 16],
        "trigger": "performance_threshold",
        "threshold": 0.9
    },
    "noise_level": {
        "schedule": [0.0, 0.01, 0.05, 0.1],
        "trigger": "update_count",
        "interval": 200
    }
}

training_config = TrainingConfig(
    curriculum_schedule=curriculum_config
)
```

### Hyperparameter Optimization

```python
def create_agent(trial):
    """Create agent with trial-specific hyperparameters."""
    
    config = PPOConfig(
        learning_rate=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_int('batch_size', 5, 50),
        n_epochs=trial.suggest_int('n_epochs', 2, 10),
        ppo_epsilon=trial.suggest_float('ppo_epsilon', 0.1, 0.3),
        entropy_coefficient=trial.suggest_float('ent_coef', 0.0, 0.1)
    )
    
    return CustomPPO(config, env)

# Run optimization
from rl_qoc import HyperparameterOptimizer
optimizer = HyperparameterOptimizer(
    create_agent, 
    objective_metric="final_fidelity"
)
best_params = optimizer.optimize(n_trials=100)
```

### Real-time Monitoring

```python
# Configure WandB logging
wandb_config = WandBConfig(
    enabled=True,
    project="quantum_control",
    entity="your_team",
    api_key="your_api_key"
)

ppo_config = PPOConfig(
    save_data=True,
    wandb_config=wandb_config,
    run_name="context_aware_calibration"
)

# Train with logging
agent = CustomPPO(ppo_config, env)
results = agent.train()

# Access training results
print(f"Final fidelity: {results['fidelity_history'][-1]}")
print(f"Total shots used: {results['total_shots'][-1]}")
print(f"Hardware runtime: {results['hardware_runtime'][-1]} seconds")
```

### Custom Action Processing

```python
class CustomActionPPO(CustomPPO):
    """PPO with custom action processing for specific hardware."""
    
    def process_action(self, probs: Normal):
        """Custom action processing for specialized hardware."""
        
        # Get base action
        action, logprob = super().process_action(probs)
        
        # Apply custom transformations
        action = self.hardware_specific_transform(action)
        
        # Update environment state
        self.unwrapped_env.custom_state_update(action)
        
        return action, logprob
        
    def hardware_specific_transform(self, action):
        """Apply hardware-specific transformations."""
        # Custom implementation
        return transformed_action
```

## Integration with Quantum Platforms

### QUA/OPX Integration

```python
# Real-time quantum control with QUA
from rl_qoc.qua import QMEnvironment, CustomQMPPO

# Configure QUA environment
qua_config = QEnvConfig(
    backend_config=QMConfig(
        qm_config=qm_configuration,
        input_type=InputType.REALTIME
    )
)

qm_env = QMEnvironment(qua_config)
qm_agent = CustomQMPPO(ppo_config, qm_env)

# Start real-time program and train
qm_env.start_program()
results = qm_agent.train()
```

### Multi-Platform Support

```python
# Switch between different quantum platforms
platform_configs = {
    "qiskit": QEnvConfig(backend_config=QiskitBackendConfig()),
    "qua": QEnvConfig(backend_config=QMConfig()),
    "qibo": QEnvConfig(backend_config=QiboBackendConfig())
}

# Train on multiple platforms
results = {}
for platform, config in platform_configs.items():
    env = create_environment(config)
    agent = CustomPPO(ppo_config, env)
    results[platform] = agent.train()
```

## Performance Optimization

### Batch Processing

```python
# Optimize batch sizes for hardware
ppo_config = PPOConfig(
    batch_size=optimal_batch_size_for_hardware(backend),
    minibatch_size=compute_optimal_minibatch_size()
)
```

### Memory Management

```python
# Configure for large-scale experiments
ppo_config = PPOConfig(
    gradient_accumulation_steps=4,
    checkpoint_frequency=100,
    memory_efficient_mode=True
)
```

### Parallel Training

```python
# Multi-GPU training
ppo_config = PPOConfig(
    device="cuda:0",
    parallel_envs=True,
    distributed_training=True
)
```