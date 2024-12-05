from dataclasses import dataclass
from typing import Optional, Union
from dataclasses import field, asdict
from abc import ABC, abstractmethod


@dataclass
class TrainingConstraint(ABC):
    """
    Abstract class for training constraints
    """

    @property
    @abstractmethod
    def constraint_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def constraint_value(self):
        raise NotImplementedError


@dataclass
class TotalUpdates(TrainingConstraint):
    """
    Total updates constraint for training (number of epochs)

    :param total_updates: Total number of updates
    """

    total_updates: int = None

    def __post_init__(self):
        assert (
            isinstance(self.total_updates, int) and self.total_updates > 0
        ), "total_updates must be an integer greater than 0"

    @property
    def constraint_name(self):
        return "TotalUpdates"

    @property
    def constraint_value(self):
        return self.total_updates


@dataclass
class HardwareRuntime(TrainingConstraint):
    """
    Hardware runtime constraint for training (in seconds). To use this mode, you should ensure that the selected backend
    does have a non-trivial InstructionDurations attribute (the user can set this attribute manually in the config as
    well).

    :param hardware_runtime: Hardware runtime constraint for training (in seconds)
    """

    hardware_runtime: Union[int, float] = None

    def __post_init__(self):
        assert self.hardware_runtime > 0, "hardware_runtime must be greater than 0"

    @property
    def constraint_name(self):
        return "HardwareRuntime"

    @property
    def constraint_value(self):
        return self.hardware_runtime


@dataclass
class TrainFunctionSettings:
    """
    Training function settings

    :param plot_real_time: Whether to plot the training progress in real time
    :param print_debug: Whether to print debug information
    :param num_prints: Number of prints to be displayed during training
    :param hpo_mode: Whether to use hyperparameter optimization mode
    :param clear_history: Whether to clear the history of the training (e.g. rewards, losses, etc.)
    :param save_data: Whether to save the data during training through tensorboard
    """

    plot_real_time: bool = False
    print_debug: Optional[bool] = False
    num_prints: Optional[int] = 10
    hpo_mode: Optional[bool] = False
    clear_history: Optional[bool] = False
    save_data: Optional[bool] = False

    def __post_init__(self):
        assert (
            isinstance(self.num_prints, int) and self.num_prints > 0
        ), "num_prints must be an integer greater than 0"


@dataclass
class TrainingConfig:
    """
    PPO training configuration

    :param training_constraint: TotalUpdates or HardwareRuntime
    :param target_fidelities: List of target fidelities to be achieved during training
    :param lookback_window: Number of episodes to look back to estimate if the agent has settled for a policy
    :param anneal_learning_rate: Whether to anneal the learning rate during training
    :param std_actions_eps: Threshold for deciding if the standard deviation of the actions is indicating a stuck policy

    """

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
