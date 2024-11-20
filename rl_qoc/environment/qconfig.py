from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Any, Literal, Tuple

import torch
from gymnasium.spaces import Box
from qiskit import QiskitError
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import DensityMatrix, Statevector

from qiskit_ibm_runtime import Options
from qiskit.providers import BackendV2
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
    QuantumRegister,
    Gate,
)
from qiskit.transpiler import InstructionDurations, PassManager


@dataclass
class BackendConfig(ABC):
    """
    Abstract base class for backend configurations.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager: Pass manager to transpile the circuit
        instruction_durations: Dictionary containing the durations of the instructions in the circuit

    """

    parametrized_circuit: Callable[
        [
            QuantumCircuit,
            ParameterVector,
            QuantumRegister,
            Any,
        ],
        None,
    ]
    backend: Optional[BackendV2] = None
    parametrized_circuit_kwargs: Dict = field(default_factory=dict)
    skip_transpilation: bool = False
    pass_manager: PassManager = PassManager()
    instruction_durations: Optional[InstructionDurations] = None

    @property
    @abstractmethod
    def config_type(self):
        return "backend"


@dataclass
class QiskitConfig(BackendConfig):
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function
        pass_manager
        instruction_durations: Dictionary containing the durations of the instructions in the circuit
    """

    @property
    def config_type(self):
        return "qiskit"


@dataclass
class DynamicsConfig(BackendConfig):
    """
    Qiskit Dynamics configuration elements.

    Args:
        estimator_options: Options to feed the Estimator primitive
        solver: Relevant only if dealing with pulse simulation (typically with DynamicsBackend), gives away solver used
        to run simulations for computing exact fidelity benchmark
        channel_freq: Relevant only if dealing with pulse simulation, Dictionary containing information mapping
        the channels and the qubit frequencies
        calibration_files: load existing gate calibrations from json file for DynamicsBackend
        baseline gate calibrations for running algorithm

    """

    calibration_files: Optional[str] = None
    do_calibrations: bool = True

    @property
    def config_type(self):
        return "dynamics"


@dataclass
class QiskitRuntimeConfig(BackendConfig):
    """
    Qiskit Runtime configuration elements.

    Args:
        options: Options to feed the Qiskit Runtime job
    """

    estimator_options: Optional[Options] = None

    @property
    def config_type(self):
        return "runtime"


@dataclass
class TargetConfig:
    """
    Configuration for the target state or gate to prepare

    Args:
        gate: Target gate to prepare
        physical_qubits: Physical qubits on which the target gate is applied
    """

    physical_qubits: List[int]

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key {key} not found in " f" Target configuration")

    def get(self, attribute_name, default_val=None):
        try:
            return getattr(self, attribute_name)
        except AttributeError:
            return default_val


@dataclass
class GateTargetConfig(TargetConfig):
    """
    Configuration for the target gate to prepare

    Args:
        gate: Target gate to prepare
        physical_qubits: Physical qubits on which the target gate is applied
    """

    gate: Gate | str

    def __post_init__(self):
        if isinstance(self.gate, str):
            if self.gate.lower() == "cnot":
                self.gate = "cx"
            elif self.gate.lower() == "cphase":
                self.gate = "cz"
            elif self.gate.lower() == "x/2":
                self.gate = "sx"
            try:
                self.gate = get_standard_gate_name_mapping()[self.gate.lower()]
            except KeyError as e:
                raise ValueError(f"Gate {self.gate} not recognized") from e


@dataclass
class StateTargetConfig(TargetConfig):
    """
    Configuration for the target state to prepare

    Args:
        state: Target state to prepare
        physical_qubits: Physical qubits on which the target state is prepared
    """

    state: QuantumCircuit | DensityMatrix | Statevector | str

    def __post_init__(self):
        if isinstance(self.state, str):
            try:
                self.state = Statevector.from_label(self.state)
            except QiskitError as e:
                raise ValueError(f"State {self.state} not recognized") from e


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarking the policy through fidelity estimation

    Args:
        benchmark_cycle: benchmark_cycle (int, optional): Number of epochs between two fidelity benchmarking.
    """

    benchmark_cycle: int = 1
    benchmark_batch_size: int = 1
    check_on_exp: bool = False
    tomography_analysis: str = "default"
    dfe_precision: Tuple[float, float] = field(default=(1e-2, 1e-2))


@dataclass
class ExecutionConfig:
    """
    Configuration for the execution of the policy

    Args:
        batch_size: Batch size (iterate over a bunch of actions per policy to estimate expected return). Defaults to 50.
        sampling_Paulis: Number of Paulis to sample for the fidelity estimation scheme. Defaults to 100.
        n_shots: Number of shots per Pauli for the fidelity estimation. Defaults to 1.
        c_factor: Renormalization factor. Defaults to 0.5.
        seed: Seed for Observable sampling. Defaults to 1234.
    """

    batch_size: int = 100
    sampling_paulis: int = 100
    n_shots: int = 1
    n_reps: int = 1
    c_factor: float = 0.5
    seed: int = 1234


@dataclass
class RewardConfig(ABC):
    """
    Configuration for how to compute the reward in the RL workflow
    """

    def __post_init__(self):
        if self.reward_method == "channel" or self.reward_method == "state":
            self.dfe = True
        else:
            self.dfe = False

    @property
    def reward_args(self):
        return {}

    @property
    @abstractmethod
    def reward_method(self) -> str:
        raise NotImplementedError


@dataclass
class FidelityConfig(RewardConfig):
    """
    Configuration for computing the reward based on fidelity estimation
    """

    @property
    def reward_method(self):
        return "fidelity"


@dataclass
class StateConfig(RewardConfig):
    """
    Configuration for computing the reward based on state fidelity estimation
    """

    @property
    def reward_method(self):
        return "state"


@dataclass
class ChannelConfig(RewardConfig):
    """
    Configuration for computing the reward based on channel fidelity estimation
    """

    num_eigenstates_per_pauli: int = 1

    @property
    def reward_args(self):
        return {"num_eigenstates_per_pauli": self.num_eigenstates_per_pauli}

    @property
    def reward_method(self):
        return "channel"


@dataclass
class XEBConfig(RewardConfig):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    @property
    def reward_method(self):
        return "xeb"


@dataclass
class CAFEConfig(RewardConfig):
    """
    Configuration for computing the reward based on Context-Aware Fidelity Estimation (CAFE)
    """

    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}

    @property
    def reward_method(self):
        return "cafe"


@dataclass
class ORBITConfig(RewardConfig):
    """
    Configuration for computing the reward based on ORBIT
    """

    use_interleaved: bool = False

    @property
    def reward_method(self):
        return "orbit"


def default_reward_config():
    return StateConfig()


def default_benchmark_config():
    return BenchmarkConfig()


@dataclass
class QEnvConfig:
    """
    Quantum Environment configuration.
    This is used to define all hyperparameters characterizing the QuantumEnvironment.

    Args:
        target (Dict): Target state or target gate to prepare
        backend_config (BackendConfig): Backend configuration
        action_space (Space): Action space
        execution_config (ExecutionConfig): Execution configuration
        reward_config (RewardConfig): Reward configuration
        benchmark_config (BenchmarkConfig): Benchmark configuration
        training_with_cal (bool): Training with calibration or not
        device (torch.device): Device on which the simulation is run
    """

    target: (
        Dict[str, List | Gate | QuantumRegister | QuantumCircuit | str] | TargetConfig
    )
    backend_config: BackendConfig
    action_space: Box
    execution_config: ExecutionConfig
    reward_config: RewardConfig = field(default_factory=default_reward_config)
    benchmark_config: BenchmarkConfig = field(default_factory=default_benchmark_config)
    training_with_cal: bool = True
    device: Optional[torch.device] = None

    def __post_init__(self):
        if isinstance(self.target, Dict):
            if "gate" in self.target:
                self.target = GateTargetConfig(**self.target)
            else:
                self.target = StateTargetConfig(**self.target)

    @property
    def backend(self) -> Optional[BackendV2]:
        return self.backend_config.backend

    @property
    def parametrized_circuit(self):
        return self.backend_config.parametrized_circuit

    @property
    def parametrized_circuit_kwargs(self):
        """
        Additional keyword arguments to feed the parametrized_circuit function
        Returns: Dictionary of additional keyword arguments with their values
        """
        return self.backend_config.parametrized_circuit_kwargs

    @parametrized_circuit_kwargs.setter
    def parametrized_circuit_kwargs(self, value: Dict):
        self.backend_config.parametrized_circuit_kwargs = value

    @property
    def physical_qubits(self):
        return self.target["physical_qubits"]

    @property
    def batch_size(self):
        return self.execution_config.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.execution_config.batch_size = value

    @property
    def sampling_paulis(self):
        return self.execution_config.sampling_paulis

    @sampling_paulis.setter
    def sampling_paulis(self, value: int):
        self.execution_config.sampling_paulis = value

    @property
    def n_shots(self):
        return self.execution_config.n_shots

    @n_shots.setter
    def n_shots(self, value: int):
        self.execution_config.n_shots = value

    @property
    def n_reps(self) -> int:
        return self.execution_config.n_reps

    @n_reps.setter
    def n_reps(self, value: int):
        self.execution_config.n_reps = value

    @property
    def c_factor(self):
        return self.execution_config.c_factor

    @property
    def seed(self):
        return self.execution_config.seed

    @seed.setter
    def seed(self, value: int):
        self.execution_config.seed = value

    @property
    def benchmark_cycle(self):
        return self.benchmark_config.benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, value: int):
        self.benchmark_config.benchmark_cycle = value

    @property
    def benchmark_batch_size(self):
        return self.benchmark_config.benchmark_batch_size

    @benchmark_batch_size.setter
    def benchmark_batch_size(self, value: int):
        self.benchmark_config.benchmark_batch_size = value

    @property
    def tomography_analysis(self):
        return self.benchmark_config.tomography_analysis

    @property
    def check_on_exp(self):
        return self.benchmark_config.check_on_exp

    @property
    def reward_method(self):
        return self.reward_config.reward_method

    @reward_method.setter
    def reward_method(
        self, value: Literal["fidelity", "channel", "state", "xeb", "cafe", "orbit"]
    ):
        if value == "fidelity":
            self.reward_config = FidelityConfig()
        elif value == "channel":
            self.reward_config = ChannelConfig()
        elif value == "state":
            self.reward_config = StateConfig()
        elif value == "xeb":
            self.reward_config = XEBConfig()
        elif value == "cafe":
            self.reward_config = CAFEConfig()
        elif value == "orbit":
            self.reward_config = ORBITConfig()
        else:
            raise ValueError(f"Reward method {value} not recognized")

    @property
    def dfe(self):
        """
        Indicates if Direct Fidelity Estimation is used for the reward computation (true if reward_method is "channel"
        or "state" and false otherwise)
        Returns: Boolean indicating if DFE is used

        """
        return self.reward_config.dfe

    @property
    def n_actions(self):
        """
        Number of actions in the action space (number of parameters to tune in the parametrized circuit)
        """
        return self.action_space.shape[-1]

    @property
    def channel_estimator(self):
        return self.reward_method == "channel"

    @property
    def fidelity_access(self):
        return self.reward_method == "fidelity"

    @property
    def instruction_durations_dict(self):
        return self.backend_config.instruction_durations

    @instruction_durations_dict.setter
    def instruction_durations_dict(self, value: InstructionDurations):
        self.backend_config.instruction_durations = value

    @property
    def pass_manager(self):
        return self.backend_config.pass_manager

    @pass_manager.setter
    def pass_manager(self, value: PassManager):
        self.backend_config.pass_manager = value
