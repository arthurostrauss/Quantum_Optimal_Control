from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Any, Literal, Tuple, Union, Iterable

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
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from .calibration_pubs import (CalibrationEstimatorPub, CalibrationEstimatorPubLike, 
                               CalibrationSamplerPub, CalibrationSamplerPubLike)

PubLike = Union[EstimatorPubLike, SamplerPubLike, CalibrationEstimatorPubLike, CalibrationSamplerPubLike]
Pub = Union[EstimatorPub, SamplerPub, CalibrationEstimatorPub, CalibrationSamplerPub]
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
    pass_manager: Optional[PassManager] = None
    instruction_durations: Optional[InstructionDurations] = None

    @property
    @abstractmethod
    def config_type(self):
        return "backend"
    
    
    def process_pubs(self, pubs: Iterable[Pub|PubLike]) -> Iterable[Pub]:
        """
        Process the pub to the correct type for the backend
        """
        return pubs
    
    def as_dict(self):
        return {
            "parametrized_circuit": self.parametrized_circuit,
            "backend": self.backend,
            "parametrized_circuit_kwargs": self.parametrized_circuit_kwargs,
            "pass_manager": self.pass_manager,
            "instruction_durations": self.instruction_durations,
        }


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
    
    def process_pubs(self, pubs: Iterable[Pub|PubLike]) -> List[Pub]:
        """
        Process the pub to the correct type for the backend
        """
        new_pubs = []
        for pub in pubs:
            if isinstance(pub, (CalibrationEstimatorPubLike, CalibrationEstimatorPub)):
                new_pubs.extend(CalibrationEstimatorPub.coerce(pub).to_pub_list())
            elif isinstance(pub, (CalibrationSamplerPubLike, CalibrationSamplerPub)):
                new_pubs.extend(CalibrationSamplerPub.coerce(pub).to_pub_list())
            elif isinstance(pub, (EstimatorPubLike, EstimatorPub)):
                new_pubs.append(EstimatorPub.coerce(pub))
            elif isinstance(pub, (SamplerPubLike, SamplerPub)):
                new_pubs.append(SamplerPub.coerce(pub))
            else:
                raise ValueError(f"Pub type {type(pub)} not recognized")
        return new_pubs


@dataclass
class DynamicsConfig(QiskitConfig):
    """
    Qiskit Dynamics configuration elements.

    Args:
        the channels and the qubit frequencies
        calibration_files: load existing gate calibrations from json file for DynamicsBackend
        baseline gate calibrations for running algorithm

    """

    calibration_files: Optional[str] = None
    do_calibrations: bool = True

    @property
    def config_type(self):
        return "dynamics"
    
    def as_dict(self):
        return super().as_dict() | {
            "calibration_files": self.calibration_files,
            "do_calibrations": self.do_calibrations,
        }


@dataclass
class QiskitRuntimeConfig(QiskitConfig):
    """
    Qiskit Runtime configuration elements.

    Args:
        options: Options to feed the Qiskit Runtime job
    """

    estimator_options: Optional[Options] = None

    @property
    def config_type(self):
        return "runtime"
    
    def as_dict(self):
        return super().as_dict() | {
            "estimator_options": self.estimator_options,
        }


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
    
    def as_dict(self):
        return {
            "gate": self.gate.name,
            "physical_qubits": self.physical_qubits,
        }


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
            
    def as_dict(self):
        return {
            "state": self.state.data,
            "physical_qubits": self.physical_qubits,
        }


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
    method: Literal["tomography", "rb"] = "rb"


@dataclass
class ExecutionConfig:
    """
    Configuration for the execution of the policy

    Args:
        batch_size: Batch size (iterate over a bunch of actions per policy to estimate expected return). Defaults to 50.
        sampling_paulis: Number of Paulis to sample for the fidelity estimation scheme. For ORBIT, this would be the number of
            random Clifford sequences to sample.
        n_shots: Number of shots per Pauli for the fidelity estimation. Defaults to 1.
        n_reps: Number of repetitions of cycle circuit (can be an integer or a list of integers to play multiple lengths)
        c_factor: Renormalization factor. Defaults to 0.5.
        seed: Seed for Observable sampling. Defaults to 1234.
    """

    batch_size: int = 100
    sampling_paulis: int = 100
    n_shots: int = 10
    n_reps: int = 1
    c_factor: float = 1.0
    seed: int = 1234
    
    def as_dict(self):
        return {
            "batch_size": self.batch_size,
            "sampling_paulis": self.sampling_paulis,
            "n_shots": self.n_shots,
            "n_reps": self.n_reps,
            "c_factor": self.c_factor,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)
    


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
        """
        String identifier for the reward method
        """
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
class StateRewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on state fidelity estimation
    """

    input_states_choice: Literal["pauli4", "pauli6", "2-design"] = "pauli4"

    @property
    def reward_method(self):
        return "state"

    @property
    def reward_args(self):
        return {"input_states_choice": self.input_states_choice}


@dataclass
class ChannelRewardConfig(RewardConfig):
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
class XEBRewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    @property
    def reward_method(self):
        return "xeb"


@dataclass
class CAFERewardConfig(RewardConfig):
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
class ORBITRewardConfig(RewardConfig):
    """
    Configuration for computing the reward based on ORBIT
    """

    use_interleaved: bool = False

    @property
    def reward_method(self):
        return "orbit"


def default_reward_config():
    return StateRewardConfig()


def default_benchmark_config():
    return BenchmarkConfig()


reward_dict = {
    "fidelity": FidelityConfig,
    "channel": ChannelRewardConfig,
    "state": StateRewardConfig,
    "xeb": XEBRewardConfig,
    "cafe": CAFERewardConfig,
    "orbit": ORBITRewardConfig,
}


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

    target: GateTargetConfig | StateTargetConfig
    backend_config: BackendConfig
    action_space: Box
    execution_config: ExecutionConfig
    reward_config: (
        RewardConfig | Literal["channel", "orbit", "state", "cafe", "xeb", "fidelity"]
    ) = "state"
    benchmark_config: BenchmarkConfig = field(default_factory=default_benchmark_config)
    env_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.target, Dict):
            if "gate" in self.target:
                self.target = GateTargetConfig(**self.target)
            else:
                self.target = StateTargetConfig(**self.target)
        if isinstance(self.reward_config, str):
            self.reward_config = reward_dict[self.reward_config]()

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
        assert value > 0, "Batch size must be greater than 0"
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
        assert value > 0, "Number of shots must be greater than 0"
        self.execution_config.n_shots = value

    @property
    def n_reps(self) -> int|List[int]:
        return self.execution_config.n_reps

    @n_reps.setter
    def n_reps(self, value: int):
        if isinstance(value, int):
            assert value > 0, "Number of repetitions must be greater than 0"
        else:
            assert all(v > 0 for v in value), "Number of repetitions must be greater than 0"
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
            self.reward_config = ChannelRewardConfig()
        elif value == "state":
            self.reward_config = StateRewardConfig()
        elif value == "xeb":
            self.reward_config = XEBRewardConfig()
        elif value == "cafe":
            self.reward_config = CAFERewardConfig()
        elif value == "orbit":
            self.reward_config = ORBITRewardConfig()
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

    def as_dict(self):
        return {
            "target": {
                "physical_qubits": self.physical_qubits,
                "gate": (
                    self.target.gate.name
                    if isinstance(self.target, GateTargetConfig)
                    else self.target.state.data
                ),
            },
            "backend_config": {
                "backend": (
                    self.backend.name
                    if isinstance(self.backend, BackendV2)
                    else self.backend
                ),
            },
            "action_space": {
                "low": self.action_space.low.tolist(),
                "high": self.action_space.high.tolist(),
            },
            "execution_config": {
                "batch_size": self.batch_size,
                "sampling_paulis": self.sampling_paulis,
                "n_shots": self.n_shots,
                "n_reps": self.n_reps,
                "c_factor": self.c_factor,
                "seed": self.seed,
            },
            "reward_config": self.reward_method,
            "benchmark_config": {
                "benchmark_cycle": self.benchmark_cycle,
                "benchmark_batch_size": self.benchmark_batch_size,
                "tomography_analysis": self.tomography_analysis,
                "check_on_exp": self.check_on_exp,
            },
            "metadata": self.env_metadata,
        }
