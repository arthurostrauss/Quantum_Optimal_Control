from dataclasses import dataclass
from typing import Optional, Union
from dataclasses import field, asdict


@dataclass
class TotalUpdates:
    total_updates: int = None

    def __post_init__(self):
        assert (
            isinstance(self.total_updates, int) and self.total_updates > 0
        ), "total_updates must be an integer greater than 0"


@dataclass
class HardwareRuntime:
    hardware_runtime: Union[int, float] = None

    def __post_init__(self):
        assert self.hardware_runtime > 0, "hardware_runtime must be greater than 0"


@dataclass
class TrainFunctionSettings:
    plot_real_time: bool = False
    print_debug: Optional[bool] = False
    num_prints: Optional[int] = 40
    hpo_mode: Optional[bool] = False
    clear_history: Optional[bool] = False

    def __post_init__(self):
        assert (
            isinstance(self.num_prints, int) and self.num_prints > 0
        ), "num_prints must be an integer greater than 0"


@dataclass
class TrainingConfig:
    training_constraint: Union[TotalUpdates, HardwareRuntime] = field(
        default_factory=lambda: TotalUpdates(250)
    )
    target_fidelities: Optional[list] = field(
        default_factory=lambda: [0.999, 0.9999, 0.99999]
    )
    lookback_window: Optional[int] = 10
    anneal_learning_rate: Optional[bool] = False
    std_actions_eps: Optional[float] = 1e-2

    @property
    def as_dict(self):
        return asdict(self)
