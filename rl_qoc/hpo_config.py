from typing import Literal, Union
from dataclasses import asdict, dataclass

import numpy as np

from rl_qoc.base_q_env import BaseQuantumEnvironment
from rl_qoc.context_aware_quantum_environment import ContextAwareQuantumEnvironment
from rl_qoc.quantumenvironment import QuantumEnvironment

QUANTUM_ENVIRONMENT = Union[
    BaseQuantumEnvironment,
    QuantumEnvironment,
    ContextAwareQuantumEnvironment,
]


@dataclass
class DirectoryPaths:
    save_results_path: str = None
    agent_config_path: str = None
    hpo_config_path: str = None


@dataclass
class HardwarePenaltyWeights:
    shots_penalty: float = 0.01
    missed_fidelity_penalty: float = 1e4
    fidelity_reward: float = 2 * 1e4

    def __post_init__(self):
        assert np.all(
            [
                self.shots_penalty >= 0,
                self.missed_fidelity_penalty >= 0,
                self.fidelity_reward >= 0,
            ]
        ), "All penalty weights must be non-negative."


@dataclass
class HPOConfig:
    q_env: QUANTUM_ENVIRONMENT
    num_trials: int
    hardware_penalty_weights: HardwarePenaltyWeights
    hpo_paths: DirectoryPaths
    saving_mode: Literal["all", "best"] = "all"
    log_results: bool = True

    def __post_init__(self):
        assert (
            isinstance(self.num_trials, int) and self.num_trials > 0
        ), "num_trials must be an integer greater than 0"

    @property
    def as_dict(self):
        return asdict(self)

    @property
    def shots_penalty(self):
        return self.hardware_penalty_weights.shots_penalty

    @property
    def missed_fidelity_penalty(self):
        return self.hardware_penalty_weights.missed_fidelity_penalty

    @property
    def fidelity_reward(self):
        return self.hardware_penalty_weights.fidelity_reward

    @property
    def save_results_path(self):
        return self.hpo_paths.save_results_path

    @property
    def agent_config_path(self):
        return self.hpo_paths.agent_config_path

    @property
    def hpo_config_path(self):
        return self.hpo_paths.hpo_config_path
