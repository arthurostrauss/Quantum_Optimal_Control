from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Any, Literal, Tuple

from quam.components.channels import Channel as QuamChannel
import torch
from gymnasium.spaces import Box
from qiskit import pulse

from qiskit_ibm_runtime import Options
from qiskit.providers import BackendV2
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
    QuantumRegister,
    Gate,
)
from qiskit.transpiler import InstructionDurations
from qiskit_dynamics import Solver


@dataclass
class BackendConfig(ABC):
    """
    Abstract base class for backend configurations.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a quantum circuit (Qiskit or QUA)
        backend: Quantum backend, if None is provided, then statevector simulation is used (not doable for pulse sim)
        parametrized_circuit_kwargs: Additional arguments to feed the parametrized_circuit function

    """

    parametrized_circuit: Callable
    backend: Optional[BackendV2]
    parametrized_circuit_kwargs: Dict = field(default_factory=dict)
    instruction_durations_dict: Optional[InstructionDurations] = None


@dataclass
class QiskitConfig(BackendConfig):
    """
    Qiskit configuration elements.

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QuantumCircuit instance
        estimator_options: Options to feed the Estimator primitive
        solver: Relevant only if dealing with pulse simulation (typically with DynamicsBackend), gives away solver used
        to run simulations for computing exact fidelity benchmark
        channel_freq: Relevant only if dealing with pulse simulation, Dictionary containing information mapping
        the channels and the qubit frequencies
        calibration_files: load existing gate calibrations from json file for DynamicsBackend
        baseline gate calibrations for running algorithm

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
    estimator_options: Optional[Options] = None
    solver: Optional[Solver] = None
    channel_freq: Optional[Dict] = field(default_factory=dict)
    calibration_files: Optional[str] = None
    do_calibrations: bool = True


@dataclass
class QuaConfig(BackendConfig):
    """
    QUA Configuration

    Args:
        parametrized_circuit: Function applying parametrized transformation to a QUA program
        backend: Quantum Machine backend
        hardware_config: Hardware configuration
        channel_mapping: Dictionary mapping channels to quantum elements
    """

    channel_mapping: Dict[pulse.channels.Channel, QuamChannel] = (
        None  # channel to quantum element mapping (e.g. DriveChannel(0) -> 'd0')
    )


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
class RewardConfig:
    """
    Configuration for how to compute the reward in the RL workflow
    """

    reward_method: Literal["fidelity", "channel", "state", "xeb", "cafe", "orbit"]

    def __post_init__(self):
        if self.reward_method == "fidelity":
            self.dfe = False

        elif self.reward_method == "channel" or self.reward_method == "state":
            self.dfe = True
        else:
            self.dfe = False


@dataclass
class FidelityConfig(RewardConfig):
    """
    Configuration for computing the reward based on fidelity estimation
    """

    reward_method: Literal["fidelity"] = field(default="fidelity", init=False)


@dataclass
class StateConfig(RewardConfig):
    """
    Configuration for computing the reward based on state fidelity estimation
    """

    reward_method: Literal["state"] = field(default="state", init=False)


@dataclass
class ChannelConfig(RewardConfig):
    """
    Configuration for computing the reward based on channel fidelity estimation
    """

    reward_method: Literal["channel"] = field(default="channel", init=False)
    num_eigenstates_per_pauli: int


@dataclass
class XEBConfig(RewardConfig):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    reward_method: Literal["xeb"] = field(default="xeb", init=False)
    num_sequences: int = 10
    depth: int = 1


@dataclass
class CAFEConfig(RewardConfig):
    """
    Configuration for computing the reward based on Context-Aware Fidelity Estimation (CAFE)
    """

    reward_method: Literal["cafe"] = field(default="cafe", init=False)
    input_states_choice: str = "all"


@dataclass
class ORBITConfig(RewardConfig):
    """
    Configuration for computing the reward based on ORBIT
    """

    reward_method: Literal["orbit"] = field(default="orbit", init=False)
    num_sequences: int = 3
    depth: int = 1
    use_interleaved: bool = False


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

    target: Dict[str, List | Gate | QuantumRegister | QuantumCircuit]
    backend_config: BackendConfig
    action_space: Box
    execution_config: ExecutionConfig
    reward_config: RewardConfig = field(default_factory=default_reward_config)
    benchmark_config: BenchmarkConfig = field(default_factory=default_benchmark_config)
    training_with_cal: bool = True
    device: Optional[torch.device] = None

    @property
    def backend(self):
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

    @property
    def sampling_paulis(self):
        return self.execution_config.sampling_paulis

    @property
    def n_shots(self):
        return self.execution_config.n_shots

    @property
    def n_reps(self):
        return self.execution_config.n_reps

    @property
    def c_factor(self):
        return self.execution_config.c_factor

    @property
    def seed(self):
        return self.execution_config.seed

    @property
    def benchmark_cycle(self):
        return self.benchmark_config.benchmark_cycle

    @property
    def benchmark_batch_size(self):
        return self.benchmark_config.benchmark_batch_size

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
        self.reward_config.reward_method = value

    @property
    def dfe(self):
        """
        Indicates if Direct Fidelity Estimation is used for the reward computation (true if reward_method is "channel"
        or "state" and false otherwise)
        Returns:

        """
        return self.reward_config.dfe

    @property
    def n_actions(self):
        return self.action_space.shape[-1]

    @property
    def channel_estimator(self):
        return self.reward_method == "channel"

    @property
    def fidelity_access(self):
        return self.reward_method == "fidelity"

    @property
    def instruction_durations_dict(self):
        return self.backend_config.instruction_durations_dict

    @instruction_durations_dict.setter
    def instruction_durations_dict(self, value: InstructionDurations):
        self.backend_config.instruction_durations_dict = value
