from typing import Literal, Optional, Union
from dataclasses import asdict, dataclass, field

import numpy as np

from base_q_env import BaseQuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from quantumenvironment import QuantumEnvironment

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
    fidelity_reward: float = 2*1e4

    def __post_init__(self):
        assert (
        np.all([self.shots_penalty >= 0, self.missed_fidelity_penalty >= 0, self.fidelity_reward >= 0])
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
        assert isinstance(self.num_trials, int) and self.num_trials > 0, \
        "num_trials must be an integer greater than 0"

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

@dataclass
class TotalUpdates:
    total_updates: int = None

    def __post_init__(self):
        assert isinstance(self.total_updates, int) and self.total_updates > 0, \
            "total_updates must be an integer greater than 0"

@dataclass
class HardwareRuntime:
    hardware_runtime: Union[int, float] = None

    def __post_init__(self):
        assert (
            self.hardware_runtime > 0
        ), "hardware_runtime must be greater than 0"
    
@dataclass
class TrainFunctionSettings:
    plot_real_time: bool = False
    print_debug: Optional[bool] = False
    num_prints: Optional[int] = 40
    hpo_mode: Optional[bool] = False
    clear_history: Optional[bool] = False

    def __post_init__(self):
        assert isinstance(self.num_prints, int) and self.num_prints > 0, \
            "num_prints must be an integer greater than 0"

@dataclass
class TrainingDetails:
    training_constraint: Union[TotalUpdates, HardwareRuntime] = None
    target_fidelities: Optional[list] = field(default_factory=lambda: [0.999, 0.9999, 0.99999])
    lookback_window: Optional[int] = 10
    anneal_learning_rate: Optional[bool] = False
    std_actions_eps: Optional[float] = 1e-2

@dataclass
class TrainingConfig:
    training_mode: Literal["normal_calibration", "spillover_noise_use_case"] = None # Forces the user to make an explicit choice
    training_details: TrainingDetails = None
    
    @property
    def training_constraint(self):
        return self.training_details.training_constraint

    @property
    def target_fidelities(self):
        return self.training_details.target_fidelities
    
    @property
    def lookback_window(self):
        return self.training_details.lookback_window
    
    @property
    def anneal_learning_rate(self):
        return self.training_details.anneal_learning_rate
    
    @property
    def std_actions_eps(self):
        return self.training_details.std_actions_eps
    
    @property
    def phi_gamma_tuple(self):
        return self.training_details.phi_gamma_tuple
    
    @property
    def as_dict(self):
        return asdict(self)