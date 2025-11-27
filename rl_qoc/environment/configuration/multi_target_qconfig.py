from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal, Callable, Any, TYPE_CHECKING
import numpy as np
from gymnasium.spaces import Box

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
    action_space: Optional[Box] = None  # Will be inferred if None

    def __post_init__(self):
        if isinstance(self.reward, str):
            from ...rewards import reward_dict
            self.reward = reward_dict[self.reward]()
        else:
            from ...rewards import Reward
            if not isinstance(self.reward, Reward):
                raise ValueError("Reward configuration must be a string or a Reward instance")
        
        # Infer action_space from InstructionReplacements if not provided
        if self.action_space is None:
            self.action_space = self._infer_action_space()
    
    def _infer_action_space(self) -> Box:
        """
        Infer the action space from all InstructionReplacements in the MultiTarget.
        The action space dimension is the sum of all parameter dimensions from all targets.
        """
        total_params = 0
        for gate_target in self.target.gate_targets:
            if gate_target.instruction_replacement is not None:
                # Get parameters from the instruction replacement
                params = gate_target.instruction_replacement.params_to_cycle
                if params is not None:
                    # Handle different parameter formats
                    if isinstance(params, list):
                        # If it's a list, take the first element to determine structure
                        if len(params) > 0:
                            first_param = params[0]
                            if isinstance(first_param, (list, tuple)):
                                # Count parameters in the first set
                                total_params += len(first_param)
                            elif isinstance(first_param, dict):
                                total_params += len(first_param)
                            else:
                                # Single parameter
                                total_params += 1
                    elif isinstance(params, dict):
                        total_params += len(params)
                    elif hasattr(params, '__len__'):
                        # ParameterVector or similar
                        total_params += len(params)
                    else:
                        # Single parameter
                        total_params += 1
        
        # If no parameters found, default to a single parameter
        if total_params == 0:
            total_params = 1
        
        # Default to low=-pi, high=pi for all parameters
        return Box(
            low=-np.pi * np.ones(total_params, dtype=np.float32),
            high=np.pi * np.ones(total_params, dtype=np.float32),
            dtype=np.float32,
        )

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
    def n_actions(self):
        """Number of actions in the action space"""
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
                "check_on_exp": self.check_on_exp,
            },
            "metadata": self.env_metadata,
        }
        return config
