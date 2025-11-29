# Rewards API

The rewards module implements various reward computation schemes optimized for different quantum control tasks. These rewards serve as the objective function for reinforcement learning algorithms, providing feedback on the quality of quantum operations.

## Core Reward Classes

### BaseReward

Abstract base class for all reward schemes.

```python
class BaseReward:
    """
    Base class for quantum control reward schemes.
    
    Provides the interface for computing rewards from quantum measurement data,
    with support for batch processing, real-time computation, and hardware integration.
    """
```

#### Key Methods

```python
def __init__(self, reward_method: str, c_factor: float = 1.0):
    """Initialize base reward with method and normalization factor."""

def get_reward(
    self,
    measurement_data: np.ndarray,
    target_data: np.ndarray,
    config: QEnvConfig
) -> float:
    """Compute reward from measurement and target data."""

def get_reward_data(
    self,
    circuit: QuantumCircuit,
    actions: np.ndarray,
    target: Union[GateTarget, StateTarget],
    config: QEnvConfig,
    additional_input: Optional[Any] = None
) -> RewardData:
    """Prepare data needed for reward computation."""

def get_real_time_circuit(
    self,
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    target: Union[GateTarget, StateTarget],
    config: QEnvConfig,
    skip_transpilation: bool = False
) -> QuantumCircuit:
    """Generate circuit for real-time reward computation."""
```

### StateReward

Reward scheme for quantum state preparation tasks using Direct Fidelity Estimation.

```python
class StateReward(BaseReward):
    """
    State preparation reward using Direct Fidelity Estimation (DFE).
    
    Features:
    - Pauli operator sampling for fidelity estimation
    - Statistical noise handling
    - Batch processing support
    - Real-time computation capabilities
    """
```

#### Key Features

- **Direct Fidelity Estimation**: Uses Pauli expectation values to estimate state fidelity
- **Efficient Sampling**: Intelligent Pauli operator selection based on target state
- **Noise Resilience**: Statistical methods to handle measurement noise
- **Scalable**: Efficient computation for multi-qubit states

#### Usage Example

```python
from rl_qoc import StateReward, StateTarget

# Configure state reward
state_reward = StateReward(
    reward_method="state",
    c_factor=1.0,
    sampling_pauli_space=100,  # Number of Pauli measurements
    n_shots=1024              # Shots per measurement
)

# Define target state
target = StateTarget(
    state_circuit=bell_state_circuit,
    target_qubits=[0, 1]
)

# Use in environment configuration
config = QEnvConfig(
    target=target,
    reward_config=state_reward
)
```

### ChannelReward

Reward scheme for quantum gate/channel calibration using process tomography methods.

```python
class ChannelReward(BaseReward):
    """
    Channel calibration reward for quantum gate optimization.
    
    Features:
    - Process tomography-based fidelity estimation
    - Average gate fidelity computation
    - Multiple input state sampling
    - Context-aware calibration support
    """
```

#### Key Features

- **Process Fidelity**: Direct estimation of quantum channel fidelity
- **Average Gate Fidelity**: Comprehensive gate characterization
- **Input State Diversity**: Tomographically complete input state sets
- **Context Sensitivity**: Adapts to circuit context for realistic calibration

#### Usage Example

```python
from rl_qoc import ChannelReward, GateTarget
from qiskit.circuit.library import RXGate

# Configure channel reward
channel_reward = ChannelReward(
    reward_method="channel",
    input_states_choice="pauli6",  # or "pauli4", "random"
    process_fidelity_threshold=0.99
)

# Define gate target
target = GateTarget(
    gate=RXGate(Parameter('theta')),
    target_qubits=[0],
    input_states_choice="pauli6"
)

# Use in environment
config = QEnvConfig(
    target=target,
    reward_config=channel_reward
)
```

### CAFEReward

Context-Aware Fidelity Estimation reward for circuit-specific gate calibration.

```python
class CAFEReward(BaseReward):
    """
    Context-Aware Fidelity Estimation (CAFE) reward.
    
    Features:
    - Circuit context-dependent fidelity estimation
    - Cycle benchmarking integration
    - Temporal and spatial noise correlation modeling
    - Real-time context adaptation
    """
```

#### Key Features

- **Context Awareness**: Accounts for surrounding gates and circuit structure
- **Cycle Benchmarking**: Uses cycle-based fidelity estimation
- **Noise Correlation**: Models spatially and temporally correlated noise
- **Adaptive**: Dynamically adjusts to changing circuit contexts

#### Usage Example

```python
from rl_qoc import CAFEReward, GateTarget

# Configure CAFE reward
cafe_reward = CAFEReward(
    reward_method="cafe",
    context_aware=True,
    cycle_benchmarking=True,
    noise_characterization=True
)

# Define contextual gate target
target = GateTarget(
    gate=RXGate(Parameter('theta')),
    target_qubits=[0],
    circuit_context=surrounding_circuit
)

# Use with context-aware environment
config = QEnvConfig(
    target=target,
    reward_config=cafe_reward,
    context_config=ContextConfig(
        circuit_contexts=[context1, context2, context3]
    )
)
```

### FidelityReward

Direct fidelity-based reward for high-precision quantum control.

```python
class FidelityReward(BaseReward):
    """
    Direct fidelity reward for precise quantum control.
    
    Features:
    - Direct state/process fidelity computation
    - High precision measurements
    - Custom fidelity metrics
    - Hardware-optimized computation
    """
```

### ORBITReward

Reward scheme for orbital-based quantum control tasks.

```python
class ORBITReward(BaseReward):
    """
    ORBIT (Orbital-Based) reward for specialized quantum control.
    
    Features:
    - Orbital state preparation and manipulation
    - Custom observables for orbital systems
    - Multi-level quantum system support
    - Physics-aware reward computation
    """
```

## Real-time Reward Computation

### Real-time Utilities

```python
def get_real_time_reward_circuit(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    target: Union[GateTarget, StateTarget],
    env_config: QEnvConfig,
    reward_method: Optional[str] = None
) -> QuantumCircuit:
    """
    Generate quantum circuit for real-time reward computation.
    
    Features:
    - Dynamic circuit construction with runtime variables
    - Context-dependent measurement selection
    - Optimized for minimal latency
    - Hardware-specific optimizations
    """
```

#### Real-time Circuit Construction

```python
# Real-time circuit with dynamic measurements
def construct_real_time_circuit():
    """
    Construct circuit for real-time reward computation.
    
    The circuit includes:
    - Input state preparation with runtime variables
    - Parametrized quantum operations
    - Context-dependent measurements
    - Real-time reward computation
    """
    
    qc = QuantumCircuit()
    
    # Runtime variables for input states
    input_state_vars = [qc.add_input(f"input_state_{i}", Uint(4)) 
                       for i in range(n_qubits)]
    
    # Runtime variables for observables
    observable_vars = [qc.add_input(f"observable_{i}", Uint(4)) 
                      for i in range(n_qubits)]
    
    # Dynamic input state preparation
    for q_idx, qubit in enumerate(qc.qubits):
        with qc.switch(input_state_vars[q_idx]) as case_input:
            for i, input_circuit in enumerate(input_circuits):
                with case_input(i):
                    qc.compose(input_circuit, [qubit], inplace=True)
    
    # Parametrized quantum operations
    qc.compose(parametrized_circuit, inplace=True)
    
    # Dynamic observable measurements
    for q_idx, qubit in enumerate(qc.qubits):
        with qc.switch(observable_vars[q_idx]) as case_obs:
            for i, basis_rotation in enumerate(pauli_rotations):
                with case_obs(i):
                    qc.compose(basis_rotation, [qubit], inplace=True)
    
    return qc
```

### QUA Integration for Real-time Rewards

```python
def rl_qoc_training_qua_prog(
    circuit: QuantumCircuit,
    policy_params: ParameterTable,
    reward_param: QuaParameter,
    circuit_params: CircuitParams,
    config: QEnvConfig,
    num_updates: int
) -> Program:
    """
    Generate QUA program with embedded reward computation.
    
    The program runs entirely on OPX hardware and includes:
    - Policy parameter reception from classical computer
    - Action sampling using hardware RNG
    - Quantum circuit execution with real-time parameters
    - Reward computation from measurement results
    - Result streaming back to classical agent
    """
```

## Reward Data Structures

### RewardData

Base data structure for reward computation information.

```python
@dataclass
class RewardData:
    """Base class for reward computation data."""
    
    total_shots: int
    n_batches: int
    batch_size: int
    
    # Measurement configuration
    measurement_basis: List[str]
    observables: List[str]
    
    # Target information
    target_values: np.ndarray
    target_probabilities: np.ndarray
```

### StateRewardData

Specialized data structure for state reward computation.

```python
@dataclass
class StateRewardData(RewardData):
    """Data structure for state reward computation."""
    
    # Pauli sampling information
    pauli_indices: List[int]
    pauli_probabilities: List[float]
    pauli_observables: List[str]
    
    # Target state characteristics
    target_pauli_expectations: Dict[str, float]
    sampling_overhead: float
```

### ChannelRewardData

Specialized data structure for channel reward computation.

```python
@dataclass  
class ChannelRewardData(RewardData):
    """Data structure for channel reward computation."""
    
    # Input state information
    input_states: List[QuantumCircuit]
    input_indices: List[int]
    
    # Process tomography data
    process_matrix_elements: np.ndarray
    input_output_pairs: List[Tuple[int, int]]
    
    # Gate fidelity information
    average_gate_fidelity: float
    process_fidelity: float
```

### CAFERewardData

Specialized data structure for context-aware reward computation.

```python
@dataclass
class CAFERewardData(RewardData):
    """Data structure for CAFE reward computation."""
    
    # Context information
    circuit_context: QuantumCircuit
    causal_cone_qubits: List[int]
    baseline_circuit: QuantumCircuit
    
    # Cycle benchmarking data
    cycle_lengths: List[int]
    cycle_fidelities: List[float]
    
    # Noise characterization
    crosstalk_matrix: Optional[np.ndarray]
    temporal_correlations: Optional[np.ndarray]
```

## Advanced Reward Features

### Adaptive Reward Scaling

```python
class AdaptiveRewardScaling:
    """
    Adaptive reward scaling for improved training stability.
    
    Features:
    - Dynamic reward normalization
    - Performance-based scaling adaptation
    - Noise-aware adjustments
    - Training phase-dependent scaling
    """
    
    def __init__(self, 
                 initial_scale: float = 1.0,
                 adaptation_rate: float = 0.01,
                 min_scale: float = 0.1,
                 max_scale: float = 10.0):
        self.scale = initial_scale
        self.adaptation_rate = adaptation_rate
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def update_scale(self, 
                    current_performance: float,
                    target_performance: float,
                    training_phase: str):
        """Update reward scaling based on current performance."""
        
        performance_ratio = current_performance / target_performance
        
        if performance_ratio < 0.5:  # Low performance
            self.scale *= (1 + self.adaptation_rate)
        elif performance_ratio > 0.95:  # High performance
            self.scale *= (1 - self.adaptation_rate)
            
        self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
```

### Multi-Objective Rewards

```python
class MultiObjectiveReward(BaseReward):
    """
    Multi-objective reward combining multiple criteria.
    
    Features:
    - Weighted combination of multiple objectives
    - Pareto optimization support
    - Dynamic weight adjustment
    - Constraint handling
    """
    
    def __init__(self, 
                 objectives: List[BaseReward],
                 weights: List[float],
                 constraint_penalties: Optional[Dict] = None):
        self.objectives = objectives
        self.weights = weights
        self.constraint_penalties = constraint_penalties or {}
        
    def compute_reward(self, *args, **kwargs):
        """Compute weighted multi-objective reward."""
        
        individual_rewards = []
        for objective in self.objectives:
            reward = objective.compute_reward(*args, **kwargs)
            individual_rewards.append(reward)
            
        # Weighted combination
        total_reward = sum(w * r for w, r in zip(self.weights, individual_rewards))
        
        # Apply constraint penalties
        for constraint, penalty in self.constraint_penalties.items():
            if self.check_constraint_violation(constraint, *args, **kwargs):
                total_reward -= penalty
                
        return total_reward
```

### Curriculum Reward Scheduling

```python
class CurriculumRewardScheduler:
    """
    Curriculum-based reward scheduling for progressive learning.
    
    Features:
    - Progressive difficulty increase
    - Performance-triggered transitions
    - Multi-stage reward design
    - Adaptive scheduling
    """
    
    def __init__(self, 
                 reward_stages: List[BaseReward],
                 transition_criteria: List[Dict],
                 current_stage: int = 0):
        self.reward_stages = reward_stages
        self.transition_criteria = transition_criteria
        self.current_stage = current_stage
        
    def get_current_reward(self) -> BaseReward:
        """Get reward for current curriculum stage."""
        return self.reward_stages[self.current_stage]
        
    def update_stage(self, performance_metrics: Dict):
        """Update curriculum stage based on performance."""
        
        if self.current_stage < len(self.reward_stages) - 1:
            criteria = self.transition_criteria[self.current_stage]
            
            if self.check_transition_criteria(criteria, performance_metrics):
                self.current_stage += 1
                print(f"Advanced to curriculum stage {self.current_stage}")
```

## Integration Examples

### Basic State Preparation

```python
# Configure state preparation reward
config = QEnvConfig(
    target=StateTarget(
        state_circuit=ghz_state_circuit,
        target_qubits=[0, 1, 2]
    ),
    reward_config=StateReward(
        sampling_pauli_space=50,
        n_shots=1024,
        c_factor=1.0
    )
)

env = QuantumEnvironment(config)
agent = CustomPPO(ppo_config, env)
results = agent.train()
```

### Context-Aware Gate Calibration

```python
# Configure context-aware calibration
config = QEnvConfig(
    target=GateTarget(
        gate=RXGate(Parameter('theta')),
        target_qubits=[1],
        circuit_context=vqe_ansatz
    ),
    reward_config=CAFEReward(
        context_aware=True,
        cycle_benchmarking=True
    ),
    context_config=ContextConfig(
        circuit_contexts=[vqe_ansatz, qaoa_ansatz],
        context_aware=True
    )
)

env = ContextAwareQuantumEnvironment(config)
agent = CustomPPO(ppo_config, env)
results = agent.train()
```

### Real-time QUA Reward Computation

```python
# Configure for real-time QUA execution
qua_config = QEnvConfig(
    target=ChannelTarget(
        gate=CNOTGate(),
        target_qubits=[0, 1]
    ),
    reward_config=ChannelReward(
        input_states_choice="pauli6",
        real_time_computation=True
    ),
    backend_config=QMConfig(
        input_type=InputType.REALTIME
    )
)

qm_env = QMEnvironment(qua_config)
qm_agent = CustomQMPPO(ppo_config, qm_env)

# Start real-time program with embedded reward computation
qm_env.start_program()
results = qm_agent.train()
qm_env.close()
```

The rewards module provides a comprehensive framework for quantum control objectives, enabling precise and efficient optimization of quantum operations across different platforms and use cases.