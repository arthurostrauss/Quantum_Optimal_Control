# Environment System API

The environment system forms the core of the RL-QOC framework, providing quantum environments that interface with various quantum computing platforms.

## Base Environment Classes

### BaseQuantumEnvironment

The abstract base class for all quantum environments.

```python
class BaseQuantumEnvironment(ABC, Env):
    """
    Abstract base class for quantum environments implementing the Gymnasium interface.
    
    This class provides the foundation for quantum control environments, handling
    circuit execution, reward computation, and interaction with quantum backends.
    """
```

**Key Attributes:**
- `config: QEnvConfig` - Environment configuration
- `target: Union[GateTarget, StateTarget]` - Target for optimization
- `backend: Optional[BackendV2]` - Quantum backend for execution
- `reward_method: str` - Method for computing rewards
- `n_actions: int` - Number of action parameters
- `action_space: Box` - Action space definition

**Key Methods:**

#### `step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]`
Execute one environment step with the given action.

**Parameters:**
- `action`: Action vector to apply to the quantum system

**Returns:**
- `observation`: Next observation
- `reward`: Reward from the current step
- `terminated`: Whether episode has terminated
- `truncated`: Whether episode was truncated
- `info`: Additional information dictionary

#### `reset() -> Tuple[np.ndarray, dict]`
Reset the environment to initial state.

#### `simulate_circuit(qc: QuantumCircuit, params: np.ndarray) -> np.ndarray`
Simulate a quantum circuit with given parameters.

### QuantumEnvironment

Standard quantum environment for basic quantum control tasks.

```python
class QuantumEnvironment(BaseQuantumEnvironment):
    """
    Standard quantum environment for state preparation and gate calibration.
    
    Supports both gate-level and pulse-level control abstractions.
    """
```

**Use Cases:**
- Quantum state preparation
- Single gate calibration
- Basic quantum control tasks

### ContextAwareQuantumEnvironment

Advanced environment for context-aware quantum control.

```python
class ContextAwareQuantumEnvironment(BaseQuantumEnvironment):
    """
    Context-aware quantum environment for circuit-context-dependent calibration.
    
    Enables calibration of quantum gates within specific circuit contexts,
    accounting for temporal and spatial correlations.
    """
```

**Key Features:**
- Circuit context awareness
- Multiple target gate locations
- Causal cone analysis
- Context parameter binding

**Additional Attributes:**
- `circuits: List[QuantumCircuit]` - Circuit contexts
- `target_instructions: List[Instruction]` - Target instruction instances
- `virtual_target_qubits: List[Qubit]` - Virtual target qubit mapping
- `causal_cone_qubits: List[Qubit]` - Qubits in the causal cone

**Key Methods:**

#### `define_circuits() -> List[QuantumCircuit]`
Define the circuits to be used in the environment with parametrized gates.

#### `update_gate_calibration(gate_names: Optional[List[str]] = None)`
Update gate calibrations in the backend.

## Target System

### BaseTarget

Abstract base class for optimization targets.

```python
class BaseTarget(ABC):
    """Base class for defining optimization targets."""
```

### GateTarget

Target for gate calibration tasks.

```python
class GateTarget(BaseTarget):
    """
    Target for quantum gate calibration.
    
    Defines a quantum gate to be calibrated within optional circuit contexts.
    """
```

**Key Attributes:**
- `gate: Gate` - Target gate to calibrate
- `physical_qubits: List[int]` - Physical qubit indices
- `circuit_context: Optional[List[QuantumCircuit]]` - Circuit contexts
- `causal_cone_size: int` - Size of the causal cone

**Key Methods:**

#### `Chi(n_reps: int = 1) -> np.ndarray`
Compute characteristic function for the target gate.

#### `fidelity(state: DensityMatrix) -> float`
Compute fidelity between given state and target.

### StateTarget

Target for state preparation tasks.

```python
class StateTarget(BaseTarget):
    """
    Target for quantum state preparation.
    
    Defines a target quantum state to be prepared.
    """
```

**Key Attributes:**
- `dm: DensityMatrix` - Target density matrix
- `circuit: Optional[QuantumCircuit]` - Circuit for preparing target state

## Platform-Specific Environments

### QMEnvironment

Environment for QUA platform integration.

```python
class QMEnvironment(ContextAwareQuantumEnvironment):
    """
    Quantum environment for real-time control with QUA platform.
    
    Enables action sampling directly on the Quantum Orchestration Platform
    for minimal latency and maximum efficiency.
    """
```

**Key Features:**
- Real-time action sampling on OPX
- Parameter streaming to quantum control hardware
- Hardware-optimized execution

**Key Attributes:**
- `qm: QuantumMachine` - Quantum machine instance
- `policy: ParameterTable` - Policy parameters for real-time sampling
- `input_type: InputType` - Type of input for streaming

### QiboEnvironment

Environment for Qibo platform integration.

```python
class QiboEnvironment(QuantumEnvironment):
    """
    Quantum environment for Qibo platform.
    
    Provides integration with Qibo and Qibolab for quantum control.
    """
```

## Environment Configuration

### QEnvConfig

Main configuration class for quantum environments.

```python
@dataclass
class QEnvConfig:
    """Configuration for quantum environments."""
    
    # Core configuration
    target: Union[GateTarget, StateTarget]
    reward: Reward
    backend_config: BackendConfig
    execution_config: ExecutionConfig
    
    # Training parameters
    n_actions: int
    batch_size: int = 256
    seed: int = 1234
```

### ExecutionConfig

Configuration for quantum circuit execution.

```python
@dataclass
class ExecutionConfig:
    """Configuration for quantum circuit execution."""
    
    n_shots: int = 1000
    n_reps: int = 1
    c_factor: float = 1.0
    control_flow_enabled: bool = False
    sampling_paulis: int = 100
```

## Usage Examples

### Basic Gate Calibration

```python
from rl_qoc import QuantumEnvironment, GateTarget, ChannelReward
from qiskit.circuit.library import RXGate

# Define target gate
target = GateTarget(gate=RXGate(Parameter("theta")), physical_qubits=[0])

# Create environment
env = QuantumEnvironment(QEnvConfig(
    target=target,
    reward=ChannelReward(),
    n_actions=1
))

# Training loop
obs, info = env.reset()
for step in range(1000):
    action = agent.get_action(obs)  # Get action from RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Context-Aware Calibration

```python
from rl_qoc import ContextAwareQuantumEnvironment
from qiskit import QuantumCircuit

# Create circuit context
context_circuit = QuantumCircuit(2)
context_circuit.h(0)
context_circuit.rx(Parameter("theta"), 1)  # Target gate
context_circuit.cx(0, 1)

# Define target
target = GateTarget(
    gate=RXGate(Parameter("theta")),
    physical_qubits=[1],
    circuit_context=context_circuit
)

# Create context-aware environment
env = ContextAwareQuantumEnvironment(QEnvConfig(
    target=target,
    reward=ChannelReward(),
    n_actions=1
))
```

### QUA Real-Time Control

```python
from rl_qoc.qua import QMEnvironment, QMConfig
from qiskit_qm_provider import FluxTunableTransmonBackend

# Configure QUA backend
backend = FluxTunableTransmonBackend(...)
qm_config = QMConfig(backend=backend)

# Create QUA environment
env = QMEnvironment(QEnvConfig(
    target=target,
    backend_config=qm_config,
    n_actions=1
))

# Start real-time program
env.start_program()

# Training with real-time action sampling
obs, info = env.reset()
for step in range(1000):
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # Actions are sampled directly on OPX hardware
```