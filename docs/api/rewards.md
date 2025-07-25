# Reward System API

The reward system provides various methods for computing rewards in quantum control tasks, ranging from direct fidelity estimation to advanced benchmarking protocols.

## Base Reward Classes

### Reward

Abstract base class for all reward methods.

```python
@dataclass
class Reward(ABC):
    """
    Base class for reward computation methods.
    
    Defines the interface for computing rewards from quantum circuit execution results.
    """
```

**Key Properties:**
- `reward_method: str` - String identifier for the reward method
- `dfe: bool` - Whether the method uses Direct Fidelity Estimation
- `reward_args: dict` - Additional arguments for reward computation

**Key Methods:**

#### `get_reward_data(qc, params, target, env_config, *args) -> RewardDataList`
Compute reward data (measurement circuits and observables) for the reward method.

#### `get_reward_with_primitive(reward_data, primitive) -> np.ndarray`
Compute final reward values using a quantum primitive (Estimator/Sampler).

## Fidelity-Based Rewards

### StateReward

Direct Fidelity Estimation for state preparation tasks.

```python
@dataclass
class StateReward(Reward):
    """
    State fidelity reward using Direct Fidelity Estimation (DFE).
    
    Estimates fidelity between prepared and target states using
    Pauli expectation value sampling.
    """
    
    reward_method: str = "state"
```

**Algorithm:**
1. Sample Pauli operators based on target state characteristic function
2. Measure expectation values of sampled Paulis
3. Compute unbiased fidelity estimator

**Mathematical Foundation:**
For target state |ψ⟩ and prepared state ρ, fidelity is:
```
F(|ψ⟩⟨ψ|, ρ) = ⟨ψ|ρ|ψ⟩ = Σₖ χ_ψ(k) χ_ρ(k)
```

Where χ_ψ(k) = ⟨W_k⟩_ψ/√d is the characteristic function.

**Usage:**
```python
state_reward = StateReward()
reward_data = state_reward.get_reward_data(
    qc=circuit,
    params=parameters,
    target=state_target,
    env_config=config
)
```

### ChannelReward

Channel fidelity estimation for gate calibration.

```python
@dataclass  
class ChannelReward(Reward):
    """
    Channel fidelity reward for quantum gate calibration.
    
    Estimates process fidelity by averaging state fidelities
    over a set of input states.
    """
    
    reward_method: str = "channel"
    num_eigenstates_per_pauli: int = 1
```

**Key Features:**
- Supports gates with causal cone size ≤ 3
- Uses eigenstate preparation for input states
- Averages over tomographically complete input set

**Algorithm:**
1. Prepare random input states from Pauli eigenstates
2. Apply parametrized gate
3. Compute state fidelity for each input-output pair
4. Average fidelities to estimate process fidelity

**Usage:**
```python
channel_reward = ChannelReward(num_eigenstates_per_pauli=2)
reward_data = channel_reward.get_reward_data(
    qc=gate_circuit,
    params=gate_parameters,  
    target=gate_target,
    env_config=config
)
```

### FidelityReward

Standard fidelity computation using quantum primitives.

```python
@dataclass
class FidelityReward(Reward):
    """
    Standard fidelity reward using direct computation.
    
    Computes fidelity through full state vector simulation
    or process tomography.
    """
    
    reward_method: str = "fidelity"
```

## Advanced Benchmarking Rewards

### CAFEReward

Context-Aware Fidelity Estimation for repeated gate sequences.

```python
@dataclass
class CAFEReward(Reward):
    """
    Context-Aware Fidelity Estimation (CAFE) reward.
    
    Estimates fidelity for sequences of repeated gates,
    useful for gate calibration in circuit contexts.
    """
    
    reward_method: str = "cafe"
    n_reps: int = 1
```

**Key Features:**
- Handles repeated gate sequences
- Context-aware calibration
- Efficient for gate repetition scenarios

**Algorithm:**
1. Create circuits with n_reps repetitions of target gate
2. Compose with input state preparation
3. Add inverse reference circuit
4. Measure in computational basis
5. Compute fidelity from success probability

**Usage:**
```python
cafe_reward = CAFEReward(n_reps=10)
# Automatically handles repeated gate sequences
```

### ORBITReward

Optimized Randomized Benchmarking with Interleaved Twirling.

```python
@dataclass
class ORBITReward(Reward):
    """
    ORBIT (Optimized Randomized Benchmarking with Interleaved Twirling) reward.
    
    Provides robust benchmarking for gate calibration using
    randomized benchmarking protocols.
    """
    
    reward_method: str = "orbit"
```

**Key Features:**
- Robust to state preparation and measurement errors
- Interleaved randomized benchmarking
- Statistical error mitigation

### XEBReward

Cross-Entropy Benchmarking for quantum supremacy circuits.

```python
@dataclass
class XEBReward(Reward):
    """
    Cross-Entropy Benchmarking (XEB) reward.
    
    Benchmarks quantum circuits using cross-entropy
    between ideal and experimental output distributions.
    """
    
    reward_method: str = "xeb"
    xeb_fidelity_type: Literal["log", "linear"] = "linear"
    gate_set_choice: Union[Literal["sw", "t"], Dict[int, Gate]] = "sw"
```

**Key Features:**
- Quantum supremacy circuit benchmarking
- Configurable gate sets (sqrt(W), T gates)
- Linear or logarithmic fidelity computation

**Algorithm:**
1. Generate random quantum circuits
2. Compute ideal output probabilities
3. Measure experimental output distribution  
4. Calculate cross-entropy benchmarking fidelity

## Real-Time Reward Computation

### Real-Time Circuit Generation

For platforms supporting real-time control flow:

```python
def get_real_time_reward_circuit(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    target: Union[GateTarget, StateTarget, List[GateTarget]],
    env_config: QEnvConfig,
    reward_method: Optional[Literal["channel", "state", "cafe"]] = None
) -> QuantumCircuit:
    """
    Generate quantum circuit for real-time reward computation.
    
    Creates circuits with classical control flow for platforms
    like QUA that support real-time decision making.
    """
```

**Supported Methods:**
- `"state"`: Real-time state fidelity estimation
- `"channel"`: Real-time channel fidelity estimation  
- `"cafe"`: Real-time CAFE protocols

**Features:**
- Classical register allocation
- Control flow integration
- Optimized for minimal latency

## Reward Data Structures

### RewardData

Base class for storing reward computation data.

```python
@dataclass
class RewardData:
    """Base class for reward computation data."""
    
    circuits: List[QuantumCircuit]
    observables: Optional[SparsePauliOp] = None
    parameter_values: Optional[np.ndarray] = None
    shots: int = 1000
```

### Specialized Reward Data

#### StateRewardData
```python
@dataclass
class StateRewardData(RewardData):
    """Data for state fidelity reward computation."""
    
    target_state: DensityMatrix
    pauli_indices: List[int]
    characteristic_function: np.ndarray
```

#### ChannelRewardData  
```python
@dataclass
class ChannelRewardData(RewardData):
    """Data for channel fidelity reward computation."""
    
    input_states: List[DensityMatrix]
    target_gate: Gate
    input_indices: List[int]
    observables_indices: List[List[int]]
```

## Integration with Quantum Primitives

### Estimator Integration

```python
# Using Qiskit Estimator
from qiskit.primitives import Estimator

estimator = Estimator()
reward_data = reward.get_reward_data(circuit, params, target, config)
reward_values = reward.get_reward_with_primitive(reward_data, estimator)
```

### Sampler Integration

```python
# Using Qiskit Sampler for CAFE/ORBIT
from qiskit.primitives import Sampler

sampler = Sampler()
reward_data = cafe_reward.get_reward_data(circuit, params, target, config)
reward_values = cafe_reward.get_reward_with_primitive(reward_data, sampler)
```

### QUA Primitive Integration

```python
# Using QUA-optimized primitives
from rl_qoc.qua import QMEstimator

qm_estimator = QMEstimator(qm_backend)
# Automatically handles real-time control flow
reward_values = reward.get_reward_with_primitive(reward_data, qm_estimator)
```

## Usage Examples

### State Preparation Reward

```python
from rl_qoc import StateTarget, StateReward
from qiskit.quantum_info import DensityMatrix

# Define target state
target_dm = DensityMatrix.from_label("0+")
target = StateTarget(dm=target_dm)

# Configure state reward
state_reward = StateReward()

# Use in environment
env_config = QEnvConfig(
    target=target,
    reward=state_reward,
    n_actions=2
)
```

### Gate Calibration Reward

```python
from rl_qoc import GateTarget, ChannelReward
from qiskit.circuit.library import RXGate

# Define target gate
target = GateTarget(gate=RXGate(Parameter("theta")), physical_qubits=[0])

# Configure channel reward
channel_reward = ChannelReward(num_eigenstates_per_pauli=2)

# Use in environment
env_config = QEnvConfig(
    target=target,
    reward=channel_reward,
    n_actions=1
)
```

### Context-Aware Calibration

```python
from rl_qoc import CAFEReward

# CAFE reward for repeated gate sequences
cafe_reward = CAFEReward(n_reps=5)

# Ideal for gates in circuit contexts with repetition
env_config = QEnvConfig(
    target=context_gate_target,
    reward=cafe_reward,
    n_actions=2
)
```

### Advanced Benchmarking

```python
from rl_qoc import XEBReward

# XEB reward for quantum supremacy circuits
xeb_reward = XEBReward(
    xeb_fidelity_type="linear",
    gate_set_choice="sw"  # sqrt(W) gate set
)

# Use for complex circuit benchmarking
env_config = QEnvConfig(
    target=complex_circuit_target,
    reward=xeb_reward,
    n_actions=10
)
```

## Reward Method Selection Guide

| Use Case | Recommended Reward | Key Benefits |
|----------|-------------------|--------------|
| State Preparation | `StateReward` | Direct fidelity estimation, shot-efficient |
| Single Gate Calibration | `ChannelReward` | Process fidelity, input state averaging |
| Repeated Gates | `CAFEReward` | Context-aware, efficient for repetitions |
| Robust Benchmarking | `ORBITReward` | Error mitigation, statistical robustness |
| Quantum Supremacy | `XEBReward` | Complex circuits, cross-entropy benchmarking |
| Basic Validation | `FidelityReward` | Simple, direct computation |

## Advanced Configuration

### Shot Budget Optimization

```python
# Automatic shot budget allocation
reward_data = reward.get_reward_data(circuit, params, target, config)
total_shots = reward.get_shot_budget(reward_data.pubs)

# Manual shot allocation
config.execution_config.n_shots = total_shots // len(reward_data.circuits)
```

### Custom Reward Implementation

```python
@dataclass
class CustomReward(Reward):
    """Custom reward implementation."""
    
    reward_method: str = "custom"
    
    def get_reward_data(self, qc, params, target, env_config, *args):
        # Implement custom reward data generation
        circuits = [qc]  # Custom circuit modifications
        observables = SparsePauliOp(["Z", "X"])  # Custom observables
        
        return CustomRewardData(
            circuits=circuits,
            observables=observables,
            parameter_values=params
        )
    
    def get_reward_with_primitive(self, reward_data, primitive):
        # Implement custom reward computation
        job = primitive.run(reward_data.pubs)
        results = job.result()
        
        # Custom reward calculation from results
        reward = custom_calculation(results)
        return np.array([reward])
```