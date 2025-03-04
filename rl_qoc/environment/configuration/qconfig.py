from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal
from gymnasium.spaces import Box
from qiskit.providers import BackendV2
from qiskit.transpiler import InstructionDurations, PassManager
from .backend_config import BackendConfig

from .target_config import GateTargetConfig, StateTargetConfig
from .execution_config import ExecutionConfig
from .benchmark_config import BenchmarkConfig


def default_benchmark_config():
    return BenchmarkConfig()


@dataclass
class QEnvConfig:
    """
    Quantum Environment configuration.
    This is used to define all hyperparameters characterizing the QuantumEnvironment.

    Args:
        target (Dict): Target state or target gate to prepare
        backend_config (rl_qoc.environment.configuration.backend_config.BackendConfig): Backend configuration
        action_space (Space): Action space
        execution_config (ExecutionConfig): Execution configuration
        reward_config (Reward): Reward configuration
        benchmark_config (BenchmarkConfig): Benchmark configuration
    """

    target: GateTargetConfig | StateTargetConfig
    backend_config: BackendConfig
    action_space: Box
    execution_config: ExecutionConfig
    reward_config: (
        Literal["channel", "orbit", "state", "cafe", "xeb", "fidelity"] | "Reward"
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
            from ...rewards import reward_dict

            self.reward_config = reward_dict[self.reward_config]()
        else:
            from ...rewards import Reward

            if not isinstance(self.reward_config, Reward):
                raise ValueError(
                    "Reward configuration must be a string or a Reward instance"
                )

    @property
    def backend(self) -> Optional[BackendV2]:
        return self.backend_config.backend

    @backend.setter
    def backend(self, backend: BackendV2):
        self.backend_config.backend = backend
        self.backend_config.parametrized_circuit_kwargs["backend"] = backend

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
    def n_reps(self) -> List[int]:
        return self.execution_config.n_reps

    @n_reps.setter
    def n_reps(self, value: int | List[int]):
        if isinstance(value, int):
            assert value > 0, "Number of repetitions must be greater than 0"
        else:
            assert all(
                v > 0 for v in value
            ), "Number of repetitions must be greater than 0"
        self.execution_config.n_reps = [value] if isinstance(value, int) else value

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
        try:
            from ...rewards import reward_dict

            self.reward_config = reward_dict[value]()
        except KeyError as e:
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

    def as_dict(self, to_json: bool = False):
        config = {
            "target": {
                "physical_qubits": self.physical_qubits,
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

        if isinstance(self.target, GateTargetConfig):
            config["target"]["gate"] = self.target.gate.name
        elif isinstance(self.target, StateTargetConfig) and not to_json:
            config["target"]["state"] = self.target.state.data

        return config
