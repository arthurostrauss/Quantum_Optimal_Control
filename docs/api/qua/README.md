# QUA Integration API

The QUA integration module enables real-time quantum control by embedding reinforcement learning decision-making directly into quantum circuits through the Quantum Orchestration Platform (QOP). This represents a breakthrough in quantum control by moving parts of the PPO algorithm execution onto the quantum platform for minimal latency.

## Core Classes

### QMEnvironment

Quantum environment specifically designed for QUA/OPX integration with real-time control capabilities.

```python
class QMEnvironment(ContextAwareQuantumEnvironment):
    """
    Quantum environment for real-time control with QUA/OPX.
    
    Key Features:
    - Real-time policy parameter streaming to OPX
    - Hardware-based action sampling with minimal latency
    - Integration with Quantum Orchestration Platform
    - Support for DGX and real-time input types
    - Context-aware quantum circuit execution
    """
```

#### Key Methods

```python
def __init__(self, training_config: QEnvConfig, job: Optional[RunningQmJob] = None):
    """Initialize QUA environment with configuration."""

def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """
    Execute one step with real-time control.
    
    Streams policy parameters to OPX and executes quantum operations
    with hardware-based action sampling for minimal latency.
    """

def start_program(self) -> RunningQmJob:
    """Start the QUA program for real-time execution."""

def close(self) -> bool:
    """Close the environment and halt the QUA program."""

def rl_qoc_training_qua_prog(self, num_updates: int = 1000) -> Program:
    """Generate QUA program tailored for RL-based calibration."""
```

#### Real-time Control Flow

```python
# The QUA program embeds RL control directly in quantum circuits
def rl_qoc_training_qua_prog(self, num_updates: int) -> Program:
    """
    Generate QUA program with embedded RL control flow.
    
    This program runs on the OPX and:
    1. Receives policy parameters from the classical computer
    2. Samples actions in real-time using hardware RNG
    3. Applies actions to quantum gates
    4. Measures quantum state
    5. Computes rewards
    6. Streams results back to classical agent
    """
```

#### Usage Example

```python
from rl_qoc.qua import QMEnvironment, QMConfig
from qm import QuantumMachine, generate_qua_script

# Configure QUA environment
qua_config = QEnvConfig(
    target=GateTarget(gate=RXGate(Parameter('theta')), target_qubits=[0]),
    backend_config=QMConfig(
        qm_config=quantum_machine_config,
        input_type=InputType.REALTIME,
        num_updates=1000
    ),
    reward_config=ChannelReward()
)

# Create environment
qm_env = QMEnvironment(qua_config)

# Start real-time program
job = qm_env.start_program()

# Environment is now ready for real-time training
obs, _ = qm_env.reset()
action = agent.get_action(obs)
next_obs, reward, done, truncated, info = qm_env.step(action)

# Close when done
qm_env.close()
```

### CustomQMPPO

PPO implementation optimized for QUA platforms with hardware-based action sampling.

```python
class CustomQMPPO(CustomPPO):
    """
    PPO with QUA real-time action sampling.
    
    Features:
    - Policy parameters streamed to OPX in real-time
    - Actions sampled on quantum hardware using LCG
    - Fixed-point arithmetic for hardware compatibility
    - Deterministic reproducible sampling
    - Minimal classical-quantum communication overhead
    """
```

#### Hardware-Optimized Action Processing

```python
def process_action(self, probs: Normal) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process actions for real-time QUA execution.
    
    Key innovations:
    - Streams policy parameters (μ, σ) to OPX
    - Uses Linear Congruential Generator for deterministic sampling
    - Employs fixed-point arithmetic for hardware compatibility
    - Generates actions directly on quantum platform
    """
```

#### Real-time Sampling Algorithm

The QUA integration implements a sophisticated sampling algorithm that runs directly on the OPX:

```python
# Sampling algorithm running on OPX hardware
def qua_sampling_algorithm():
    """
    Hardware-based action sampling with:
    - Linear Congruential Generator (LCG) for reproducibility
    - Box-Muller transform for Gaussian sampling
    - Fixed-point arithmetic for deterministic behavior
    - Lookup tables for trigonometric functions
    """
    
    # Parameters: a=137939405, c=12345, m=2^28
    # Generates deterministic pseudo-random sequences
    # Converts to Gaussian distribution using hardware-optimized Box-Muller
    # Returns actions as fixed-point numbers
```

### QuaEstimator

Estimator primitive adapted for QUA backends to compute expectation values.

```python
class QuaEstimator:
    """
    Estimator primitive for QUA backends.
    
    Provides standard Qiskit Estimator interface while leveraging
    QUA's real-time capabilities for efficient expectation value computation.
    """
```

#### Key Methods

```python
def __init__(self, backend: QMBackend, options: Optional[Dict] = None):
    """Initialize QUA estimator with backend."""

def run(
    self,
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    observables: Union[str, List[str]],
    parameter_values: Optional[np.ndarray] = None
) -> EstimatorResult:
    """Run expectation value computation on QUA backend."""
```

## Configuration Classes

### QMConfig

Configuration class for QUA/OPX backend integration.

```python
@dataclass
class QMConfig:
    """Configuration for QUA backend."""
    
    # QUA configuration
    qm_config: Dict  # Quantum machine configuration
    input_type: InputType = InputType.REALTIME
    
    # Execution parameters
    num_updates: int = 1000
    test_mode: bool = False
    verbosity: int = 1
    
    # Hardware options
    compiler_options: Dict = field(default_factory=dict)
    opnic_dev_path: Optional[str] = None  # For DGX integration
    
    # Real-time streaming
    stream_processing: bool = True
    buffer_size: int = 1000
```

### InputType

Enumeration for different input types supported by QUA.

```python
class InputType(Enum):
    """Input types for QUA integration."""
    
    REALTIME = "realtime"  # Real-time streaming to OPX
    DGX = "dgx"           # DGX hardware integration
    SIMULATION = "simulation"  # QUA simulation mode
```

## Real-time Control Features

### Policy Parameter Streaming

```python
# Stream policy parameters to OPX in real-time
def stream_policy_parameters(self, mu: List[float], sigma: List[float]):
    """
    Stream policy parameters to OPX for real-time action sampling.
    
    Args:
        mu: Mean values for action distribution
        sigma: Standard deviation values for action distribution
    
    The OPX receives these parameters and generates actions using:
    - Hardware-based Linear Congruential Generator
    - Box-Muller transform for Gaussian distribution
    - Fixed-point arithmetic for reproducibility
    """
    
    policy_params = {
        "mu": [FixedPoint(val) for val in mu],
        "sigma": [FixedPoint(val) for val in sigma]
    }
    
    self.policy.push_to_opx(policy_params, job=self.qm_job, qm=self.qm)
```

### Hardware-Based Random Number Generation

```python
# Linear Congruential Generator optimized for OPX
def lcg(seed: int, a: int, c: int, m: int) -> int:
    """
    Linear Congruential Generator for deterministic sampling.
    
    Parameters optimized for OPX hardware:
    - a = 137939405 (multiplier)
    - c = 12345 (increment)  
    - m = 2^28 (modulus)
    
    Ensures reproducible sequences across hardware runs.
    """
    
def lcg_fixed_point(seed: int, a: int, c: int, m: int) -> Tuple[float, int]:
    """
    Generate fixed-point number between 0 and 1.
    
    Uses 4.28 fixed-point format for hardware compatibility.
    Returns both the value and updated seed.
    """
```

### Real-time Circuit Compilation

```python
def get_real_time_circuit(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    target: Union[GateTarget, StateTarget],
    config: QEnvConfig
) -> QuantumCircuit:
    """
    Compile quantum circuits for real-time execution.
    
    Features:
    - Dynamic circuit construction with runtime variables
    - Context-aware gate parameter optimization
    - Real-time measurement and feedback
    - Minimal latency circuit execution
    """
```

## Advanced QUA Features

### Context-Aware Real-time Control

```python
# Real-time circuit adaptation based on context
def context_aware_qua_program(
    circuit_contexts: List[QuantumCircuit],
    policy_params: ParameterTable,
    reward_computation: QuaParameter
) -> Program:
    """
    Generate QUA program with context-aware control.
    
    Adapts gate parameters in real-time based on:
    - Circuit context analysis
    - Historical performance data  
    - Current noise conditions
    - Hardware calibration state
    """
```

### Multi-Qubit Real-time Control

```python
# Coordinated control of multiple qubits
def multi_qubit_real_time_control(
    target_qubits: List[int],
    policy_params: ParameterTable,
    crosstalk_compensation: bool = True
) -> Program:
    """
    Real-time control for multi-qubit operations.
    
    Features:
    - Coordinated gate parameter updates
    - Real-time crosstalk compensation
    - Synchronized measurement and feedback
    - Scalable to large qubit counts
    """
```

### Dynamic Reward Computation

```python
def real_time_reward_computation(
    measurement_data: QuaParameter,
    target_observables: List[str],
    reward_method: str
) -> Program:
    """
    Compute rewards in real-time on OPX.
    
    Supported reward methods:
    - Direct fidelity estimation
    - State overlap computation
    - Channel fidelity measurement
    - Custom reward functions
    """
```

## Integration Examples

### Basic Real-time Training

```python
from rl_qoc.qua import QMEnvironment, CustomQMPPO

# Setup
qm_env = QMEnvironment(qua_config)
qm_agent = CustomQMPPO(ppo_config, qm_env)

# Start real-time program
job = qm_env.start_program()

# Training loop with real-time control
for epoch in range(num_epochs):
    obs, _ = qm_env.reset()
    
    for step in range(episode_length):
        # Policy parameters streamed to OPX
        action, log_prob = qm_agent.get_action(obs)
        
        # Quantum operations executed with real-time feedback
        next_obs, reward, done, truncated, info = qm_env.step(action)
        
        # Update policy based on real-time results
        qm_agent.update_policy(reward, log_prob)
        
        obs = next_obs
        if done or truncated:
            break

# Cleanup
qm_env.close()
```

### Advanced Context-Aware Calibration

```python
# Context-aware gate calibration with QUA
def context_aware_qua_calibration():
    
    # Multiple circuit contexts
    circuit_contexts = [
        create_vqe_circuit(),
        create_qaoa_circuit(), 
        create_quantum_simulation_circuit()
    ]
    
    # Configure for context awareness
    config = QEnvConfig(
        target=GateTarget(gate=RXGate(Parameter('theta')), target_qubits=[0]),
        backend_config=QMConfig(
            qm_config=qm_config,
            input_type=InputType.REALTIME
        ),
        context_config=ContextConfig(
            circuit_contexts=circuit_contexts,
            context_aware=True,
            real_time_adaptation=True
        )
    )
    
    # Train with real-time context adaptation
    qm_env = QMEnvironment(config)
    qm_agent = CustomQMPPO(ppo_config, qm_env)
    
    # Real-time training with context switching
    qm_env.start_program()
    results = qm_agent.train()
    qm_env.close()
    
    return results
```

### Hardware-Software Co-optimization

```python
def hardware_software_cooptimization():
    """
    Optimize both hardware parameters and software algorithms.
    
    This example shows how QUA integration enables co-optimization
    of quantum hardware settings and RL algorithm parameters.
    """
    
    # Hardware parameter space
    hardware_params = {
        'readout_length': [100, 200, 400],
        'control_amplitude': [0.1, 0.2, 0.3],
        'detuning': [-1, 0, 1]
    }
    
    # Software algorithm parameters  
    algorithm_params = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'batch_size': [5, 10, 20],
        'ppo_epsilon': [0.1, 0.2, 0.3]
    }
    
    best_performance = 0
    best_config = None
    
    for hw_config in itertools.product(*hardware_params.values()):
        for alg_config in itertools.product(*algorithm_params.values()):
            
            # Configure QUA environment with hardware parameters
            qua_config = create_qua_config(hw_config)
            qm_env = QMEnvironment(qua_config)
            
            # Configure PPO with algorithm parameters  
            ppo_config = create_ppo_config(alg_config)
            qm_agent = CustomQMPPO(ppo_config, qm_env)
            
            # Train and evaluate
            qm_env.start_program()
            results = qm_agent.train()
            qm_env.close()
            
            # Track best performance
            final_fidelity = results['fidelity_history'][-1]
            if final_fidelity > best_performance:
                best_performance = final_fidelity
                best_config = (hw_config, alg_config)
    
    return best_config, best_performance
```

## Performance Optimizations

### Minimal Latency Configuration

```python
# Optimize for minimal latency
qm_config = QMConfig(
    input_type=InputType.REALTIME,
    stream_processing=True,
    buffer_size=100,  # Small buffer for low latency
    compiler_options={
        'optimize_for_latency': True,
        'parallel_execution': True
    }
)
```

### High-Throughput Configuration

```python
# Optimize for high throughput
qm_config = QMConfig(
    input_type=InputType.DGX,
    stream_processing=True,
    buffer_size=10000,  # Large buffer for throughput
    compiler_options={
        'optimize_for_throughput': True,
        'batch_processing': True
    }
)
```

### Memory-Efficient Real-time Execution

```python
# Memory-efficient configuration for long runs
qm_config = QMConfig(
    input_type=InputType.REALTIME,
    memory_efficient_mode=True,
    streaming_compression=True,
    garbage_collection_interval=1000
)
```

The QUA integration represents a paradigm shift in quantum control, moving beyond classical-quantum interfaces to truly integrated quantum-classical computing systems where reinforcement learning algorithms execute partially on quantum hardware for unprecedented control precision and speed.