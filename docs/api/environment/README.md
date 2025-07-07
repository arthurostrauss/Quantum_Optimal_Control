# Environment API

The environment module provides the foundation for quantum control experiments with reinforcement learning. It implements quantum environments that interface between RL agents and quantum systems (simulated or real hardware).

## Core Classes

### BaseQuantumEnvironment

The base class for all quantum environments in the RL-QOC framework.

```python
class BaseQuantumEnvironment:
    """
    Base quantum environment for reinforcement learning-based quantum control.
    
    This environment provides the standard Gym interface adapted for quantum control tasks,
    with support for batch execution, custom reward schemes, and multiple backend types.
    """
```

#### Key Methods

```python
def __init__(self, config: QEnvConfig):
    """Initialize the quantum environment with configuration."""

def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """Execute actions on the quantum system and return observations and rewards."""

def reset(self, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """Reset the environment to initial state."""

def close(self) -> None:
    """Clean up environment resources."""

def get_optimal_action(self) -> np.ndarray:
    """Get theoretically optimal action if known."""

def episode_length(self, global_step: int) -> int:
    """Get current episode length based on curriculum."""
```

#### Key Properties

```python
@property
def observation_space(self) -> gym.Space:
    """Observation space of the environment."""

@property  
def action_space(self) -> gym.Space:
    """Action space of the environment."""

@property
def n_actions(self) -> int:
    """Number of action parameters."""

@property
def n_qubits(self) -> int:
    """Number of qubits in the quantum system."""

@property
def backend_info(self) -> BackendInfo:
    """Information about the quantum backend."""
```

### QuantumEnvironment

Standard quantum environment implementation for gate-level and pulse-level control.

```python
class QuantumEnvironment(BaseQuantumEnvironment):
    """
    Quantum environment for gate-level and pulse-level quantum control.
    
    Supports:
    - Qiskit backends (simulators and hardware)
    - Custom parametrized circuits
    - Multiple reward schemes
    - Batch execution for efficient training
    """
```

#### Usage Example

```python
from rl_qoc import QuantumEnvironment, QEnvConfig, StateTarget

# Configure environment
config = QEnvConfig(
    target=StateTarget(
        state_circuit=target_circuit,
        target_qubits=[0, 1]
    ),
    backend_config=BackendConfig(
        backend_name="aer_simulator"
    ),
    reward_config=StateReward(),
    execution_config=ExecutionConfig(
        n_shots=1024,
        batch_size=10
    )
)

# Create environment
env = QuantumEnvironment(config)

# Use in RL training loop
obs, _ = env.reset()
action = agent.get_action(obs)
next_obs, reward, done, truncated, info = env.step(action)
```

### ContextAwareQuantumEnvironment

Environment for context-aware quantum gate calibration that adapts to circuit context.

```python
class ContextAwareQuantumEnvironment(BaseQuantumEnvironment):
    """
    Context-aware quantum environment for adaptive gate calibration.
    
    Features:
    - Circuit context analysis
    - Noise-adaptive calibration
    - Multi-circuit training
    - Contextual reward computation
    """
```

#### Key Features

- **Circuit Context Analysis**: Automatically analyzes circuit structure to identify relevant context
- **Adaptive Calibration**: Adjusts gate parameters based on surrounding operations
- **Multi-Circuit Training**: Trains on multiple circuit contexts simultaneously
- **Noise Correlation Modeling**: Accounts for spatially and temporally correlated noise

#### Usage Example

```python
from rl_qoc import ContextAwareQuantumEnvironment, GateTarget

# Configure for context-aware gate calibration
config = QEnvConfig(
    target=GateTarget(
        gate=RXGate(Parameter('theta')),
        target_qubits=[1]
    ),
    context_config=ContextConfig(
        circuit_contexts=[circuit1, circuit2, circuit3],
        context_aware=True,
        noise_characterization=True
    )
)

env = ContextAwareQuantumEnvironment(config)
```

## Configuration Classes

### QEnvConfig

Main configuration class for quantum environments.

```python
@dataclass
class QEnvConfig:
    """Configuration for quantum environments."""
    
    target: Union[StateTarget, GateTarget]
    backend_config: BackendConfig
    reward_config: BaseReward
    execution_config: ExecutionConfig
    
    # Optional configurations
    context_config: Optional[ContextConfig] = None
    dfe_config: Optional[DFEConfig] = None
    curriculum_config: Optional[CurriculumConfig] = None
```

### BackendConfig

Configuration for quantum backends.

```python
@dataclass  
class BackendConfig:
    """Backend configuration for quantum execution."""
    
    backend_name: str = "aer_simulator"
    backend_options: Dict = field(default_factory=dict)
    noise_model: Optional[NoiseModel] = None
    
    # Hardware-specific options
    optimization_level: int = 1
    resilience_level: int = 1
    transpiler_options: Dict = field(default_factory=dict)
```

### ExecutionConfig

Configuration for quantum circuit execution.

```python
@dataclass
class ExecutionConfig:
    """Execution configuration for quantum circuits."""
    
    n_shots: int = 1024
    batch_size: int = 10
    n_reps: List[int] = field(default_factory=lambda: [1])
    
    # Parallel execution
    max_parallel_experiments: int = 100
    
    # Optimization settings
    optimization_level: int = 1
    scheduling: bool = True
```

## Target Classes

### StateTarget

Configuration for quantum state preparation tasks.

```python
@dataclass
class StateTarget:
    """Target configuration for state preparation."""
    
    # Target state specification (choose one)
    state_circuit: Optional[QuantumCircuit] = None
    density_matrix: Optional[np.ndarray] = None
    state_vector: Optional[np.ndarray] = None
    
    # Qubit specification
    target_qubits: List[int] = field(default_factory=list)
    
    # Additional options
    normalize: bool = True
    fidelity_threshold: float = 0.99
```

### GateTarget

Configuration for quantum gate calibration tasks.

```python
@dataclass
class GateTarget:
    """Target configuration for gate calibration."""
    
    gate: Gate
    target_qubits: List[int]
    
    # Input state configuration
    input_states: Optional[List[QuantumCircuit]] = None
    input_states_choice: str = "pauli6"  # or "pauli4", "random"
    
    # Calibration options
    process_fidelity_threshold: float = 0.99
    average_gate_fidelity_threshold: float = 0.99
    
    # Context specification
    circuit_context: Optional[QuantumCircuit] = None
    causal_cone_qubits: Optional[List[int]] = None
```

## Backend Integration

### Qiskit Backend Support

```python
# Simulator backends
config.backend_config.backend_name = "aer_simulator"
config.backend_config.backend_name = "statevector_simulator"

# Hardware backends (requires IBM account)
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backend = service.backend("ibmq_lima")
config.backend_config.backend = backend
```

### Pulse-Level Control

```python
# Enable pulse-level control
config.abstraction_level = "pulse"
config.backend_config.use_dynamics = True

# Custom Hamiltonian
from qiskit_dynamics import DynamicsBackend
dynamics_backend = DynamicsBackend.from_backend(hardware_backend)
config.backend_config.backend = dynamics_backend
```

## Advanced Features

### Curriculum Learning

```python
@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    episode_length_schedule: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_reps_schedule: List[int] = field(default_factory=lambda: [1, 5, 10])
    difficulty_increase_condition: str = "performance_threshold"
    performance_threshold: float = 0.95
```

### Direct Fidelity Estimation

```python
@dataclass
class DFEConfig:
    """Configuration for Direct Fidelity Estimation."""
    
    epsilon: float = 0.01  # Precision parameter
    delta: float = 0.01    # Confidence parameter
    pauli_sampling_method: str = "efficient"  # or "uniform"
    min_pauli_measurements: int = 10
```

### Noise Characterization

```python
# Built-in noise models
config.backend_config.noise_model = NoiseModel.from_backend(backend)

# Custom noise characterization
config.context_config.characterize_crosstalk = True
config.context_config.characterize_non_markovian = True
```

## Integration Examples

### With PPO Agent

```python
from rl_qoc import CustomPPO, PPOConfig

# Create environment
env = QuantumEnvironment(env_config)

# Create PPO agent
ppo_config = PPOConfig(
    learning_rate=3e-4,
    batch_size=10,
    n_epochs=4
)
agent = CustomPPO(ppo_config, env)

# Train
results = agent.train()
```

### With Hyperparameter Optimization

```python
from rl_qoc import HyperparameterOptimizer

# Define search space for environment parameters
def create_env(trial):
    n_shots = trial.suggest_int('n_shots', 100, 2000)
    batch_size = trial.suggest_int('batch_size', 5, 20)
    
    config = QEnvConfig(
        execution_config=ExecutionConfig(
            n_shots=n_shots,
            batch_size=batch_size
        )
    )
    return QuantumEnvironment(config)

# Optimize
optimizer = HyperparameterOptimizer(create_env, objective_metric="final_fidelity")
best_params = optimizer.optimize(n_trials=100)
```