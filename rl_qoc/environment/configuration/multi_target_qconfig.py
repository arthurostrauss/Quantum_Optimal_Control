from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal, Callable, Any, TYPE_CHECKING
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from .backend_config import BackendConfig
from .execution_config import ExecutionConfig
from .benchmark_config import BenchmarkConfig
from ..target import MultiTarget
from ...helpers import load_q_env_from_yaml_file, select_backend

if TYPE_CHECKING:
    from ...rewards import Reward
    from qiskit.providers import BackendV2
    from qiskit.transpiler import PassManager, InstructionDurations


def default_benchmark_config():
    return BenchmarkConfig()


@dataclass
class MultiTargetQEnvConfig:
    """
    Quantum Environment configuration for MultiTarget.
    This is similar to QEnvConfig but works with MultiTarget and automatically
    infers the action_space from the InstructionReplacements in all GateTargets.
    """

    target: MultiTarget
    backend_config: BackendConfig
    execution_config: ExecutionConfig
    reward: Reward = "state"
    benchmark_config: BenchmarkConfig = field(default_factory=default_benchmark_config)
    env_metadata: Dict = field(default_factory=dict)
    action_space: DictSpace = field(default=False, init=False)  # Will be inferred automatically as DictSpace

    def __post_init__(self):
        if isinstance(self.reward, str):
            from ...rewards import reward_dict
            self.reward = reward_dict[self.reward]()
        else:
            from ...rewards import Reward
            if not isinstance(self.reward, Reward):
                raise ValueError("Reward configuration must be a string or a Reward instance")
        
        # Always infer action_space from InstructionReplacements
        self.action_space = self._infer_action_space()
    
    def _infer_action_space(self) -> DictSpace:
        """
        Infer the action space directly from the MultiTarget custom circuits.
        Parameters are extracted from the Parameter objects attached to those circuits.
        """
        param_names: List[str] = []
        if isinstance(self.target, MultiTarget):
            param_names = self.target.action_parameter_names
        
        if not param_names:
            param_names = ["param_0"]
        
        param_space = {
            name: Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
            for name in param_names
        }
        return DictSpace(param_space)

    @property
    def backend(self) -> Optional[BackendV2]:
        return self.backend_config.backend

    @backend.setter
    def backend(self, backend: BackendV2):
        self.backend_config.backend = backend
        self.backend_config.parametrized_circuit_kwargs["backend"] = backend

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
        """List of possible number of repetitions / circuit depths for reward computation"""
        return self.execution_config.n_reps

    @n_reps.setter
    def n_reps(self, value: int | List[int]):
        if isinstance(value, int):
            assert value > 0, "Number of repetitions must be greater than 0"
        else:
            assert all(v > 0 for v in value), "Number of repetitions must be greater than 0"
        self.execution_config.n_reps = [value] if isinstance(value, int) else value

    @property
    def current_n_reps(self) -> int:
        """Current number of repetitions / circuit depth for reward computation"""
        return self.execution_config.current_n_reps

    @property
    def c_factor(self):
        """Reward scaling factor"""
        return self.execution_config.c_factor

    @property
    def seed(self):
        """Random seed superseding the whole training"""
        return self.execution_config.seed

    @seed.setter
    def seed(self, value: int):
        self.execution_config.seed = value

    @property
    def benchmark_cycle(self):
        """Number of steps between two fidelity benchmarks"""
        return self.benchmark_config.benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, value: int):
        self.benchmark_config.benchmark_cycle = value

    @property
    def check_on_exp(self):
        """Check if benchmarking should be performed through experiment instead of simulation."""
        return self.benchmark_config.check_on_exp

    @property
    def reward_method(self):
        return self.reward.reward_method

    @reward_method.setter
    def reward_method(
        self, value: Literal["fidelity", "channel", "state", "xeb", "cafe", "orbit", "shadow"]
    ):
        try:
            from ...rewards import reward_dict
            self.reward = reward_dict[value]()
        except KeyError as e:
            raise ValueError(f"Reward method {value} not recognized")

    @property
    def dfe(self):
        """Indicates if Direct Fidelity Estimation is used for the reward computation"""
        return self.reward.dfe

    @property
    def n_actions(self) -> int:
        """Number of actions in the action space (total number of parameters)"""
        if isinstance(self.action_space, DictSpace):
            return len(self.action_space.spaces)
        else:
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
        self.backend_config.custom_instruction_durations = value

    @property
    def pass_manager(self):
        return self.backend_config.pass_manager

    @pass_manager.setter
    def pass_manager(self, value: PassManager):
        self.backend_config.pass_manager = value

    def as_dict(self):
        config = {
            "target": self.target,  # MultiTarget doesn't have as_dict yet, may need to implement
            "backend_config": self.backend_config.as_dict(),
            "action_space": {
                "type": "DictSpace",
                "spaces": {name: {"low": space.low.tolist(), "high": space.high.tolist()} 
                          for name, space in self.action_space.spaces.items()}
                if isinstance(self.action_space, DictSpace)
                else {"low": self.action_space.low.tolist(), "high": self.action_space.high.tolist()},
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
                "check_on_exp": self.check_on_exp,
            },
            "metadata": self.env_metadata,
        }
        return config
