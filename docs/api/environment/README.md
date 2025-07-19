# Environment API

The environment module provides the foundation for quantum control experiments with reinforcement learning. It implements quantum environments that interface between RL agents and quantum systems (simulated or real hardware).

## Core Classes

### Core Environment Classes

The environment module provides quantum environments that interface between RL agents and quantum systems, supporting both simulation and real hardware execution.

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
from rl_qoc import QuantumEnvironment, QEnvConfig, ExecutionConfig
from qiskit.circuit.library import CXGate
from gymnasium.spaces import Box

# Configure environment for gate calibration
q_env_config = QEnvConfig(
    target={"gate": CXGate(), "physical_qubits": [0, 1]},
    backend_config=backend_config,
    action_space=Box(low=-0.1, high=0.1, shape=(n_actions,)),
    execution_config=ExecutionConfig(
        batchsize=300,
        sampling_Paulis=50,
        N_shots=200,
        seed=10
    )
)

# Create environment
q_env = QuantumEnvironment(q_env_config)

# Use in RL training loop
obs, _ = q_env.reset()
action = agent.get_action(obs)
next_obs, reward, done, truncated, info = q_env.step(action)
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
from rl_qoc import ContextAwareQuantumEnvironment

# Configure for context-aware gate calibration
q_env = ContextAwareQuantumEnvironment(
    q_env_config,           # Same config as QuantumEnvironment
    circuit_context,        # Circuit containing the target gate
    training_steps_per_gate=250  # Steps per gate instance
)

# The environment automatically analyzes circuit context
# and provides contextual calibration for the target gate
```

## Configuration Classes

### QEnvConfig

Main configuration class for quantum environments.

```python
class QEnvConfig:
    """Configuration for quantum environments."""
    
    def __init__(
        self,
        target: Dict,                    # Target gate or state definition
        backend_config: BackendConfig,   # Backend configuration
        action_space: gym.Space,         # Action space for RL agent
        execution_config: ExecutionConfig,  # Execution parameters
        benchmark_config: Optional[BenchmarkConfig] = None,
        training_with_cal: bool = False  # Whether to include calibration
    ):
        # Configuration for quantum control tasks
```

### ExecutionConfig

Configuration for quantum execution parameters.

```python
class ExecutionConfig:
    """Execution configuration for quantum experiments."""
    
    def __init__(
        self,
        batchsize: int,           # Batch size for RL training
        sampling_Paulis: int,     # Number of Pauli observables to sample
        N_shots: int,             # Shots per measurement
        n_reps: int = 1,          # Circuit repetitions
        seed: Optional[int] = None  # Random seed
    ):
        # Parameters for quantum measurements and execution
```

### BackendConfig

Configuration for different quantum backends.

```python
# For Qiskit backends
from rl_qoc.environment.configuration.backend_config import QiskitRuntimeConfig

backend_config = QiskitRuntimeConfig(
    parametrized_circuit=apply_parametrized_circuit,
    backend=backend,
    primitive_options=estimator_options
)

# For Dynamics backends (pulse-level)
from rl_qoc.environment.configuration.backend_config import DynamicsConfig

backend_config = DynamicsConfig(
    parametrized_circuit=apply_parametrized_circuit,
    backend=dynamics_backend
)
```

## Target Specification

In rl_qoc, targets are specified as dictionaries rather than dedicated classes, providing flexibility for different types of quantum control tasks.

### Gate Calibration Target

```python
from qiskit.circuit.library import CXGate, ECRGate

# Basic gate target
target = {
    "gate": CXGate(),                # Target gate to calibrate
    "physical_qubits": [0, 1]        # Physical qubits to apply gate on
}

# Alternative gates
ecr_target = {
    "gate": ECRGate(), 
    "physical_qubits": [0, 1]
}
```

### State Preparation Target

```python
# State target using circuit preparation
target = {
    "circuit": bell_state_circuit,   # Circuit to prepare target state
    "register": [0, 1]               # Qubits involved in state preparation
}

# Alternative: density matrix specification
target = {
    "dm": target_density_matrix,     # Target density matrix
    "register": [0, 1, 2]           # Qubits for the state
}
```

### Target Properties

The environment automatically determines:
- **Reward method**: Based on target type (gate → channel fidelity, state → state fidelity)
- **Input states**: Automatically generated for gate calibration (Pauli-6 basis by default)
- **Measurement observables**: Optimally sampled based on target
- **Fidelity benchmarking**: Computed during training for performance tracking

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