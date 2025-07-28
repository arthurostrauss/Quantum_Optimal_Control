from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Callable
from dataclasses import field
from abc import ABC, abstractmethod
import torch.nn as nn
import torch.optim as optim
from .ppo_initialization import (
    get_optimizer,
    get_module,
    reverse_module_dict,
    reverse_optim_dict,
)
from ..helpers import load_from_yaml_file
import wandb


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

    total_updates: int = 50

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

    @classmethod
    def from_value(cls, value):
        """
        Create a TotalUpdates object from a dictionary
        """
        return cls(total_updates=value)


@dataclass
class HardwareRuntime(TrainingConstraint):
    """
    Hardware runtime constraint for training (in seconds). To use this mode, you should ensure that the selected backend
    does have a non-trivial InstructionDurations attribute (the user can set this attribute manually in the config as
    well).

    :param hardware_runtime: Hardware runtime constraint for training (in seconds)
    """

    hardware_runtime: Union[int, float]

    def __post_init__(self):
        assert self.hardware_runtime > 0, "hardware_runtime must be greater than 0"

    @property
    def constraint_name(self):
        return "HardwareRuntime"

    @property
    def constraint_value(self):
        return self.hardware_runtime

    @classmethod
    def from_value(cls, value):
        """
        Create a HardwareRuntime object from a dictionary
        """
        return cls(hardware_runtime=value)


@dataclass
class TrainFunctionSettings:
    """
    Training function settings

    :param plot_real_time: Whether to plot the training progress in real time
    :param print_debug: Whether to print debug information
    :param num_prints: Number of prints to be displayed during training
    :param hpo_mode: Whether to use hyperparameter optimization mode
    :param clear_history: Whether to clear the history of the training (e.g. rewards, losses, etc.)
    :param save_data: Whether to save the data during training through tensorboard/wandb
    """

    plot_real_time: bool = False
    print_debug: bool = False
    num_prints: int = 10
    hpo_mode: bool = False
    clear_history: bool = False
    save_data: bool = False

    def __post_init__(self):
        assert (
            isinstance(self.num_prints, int) and self.num_prints > 0
        ), "num_prints must be an integer greater than 0"

    def as_dict(self):
        return {
            "plot_real_time": self.plot_real_time,
            "print_debug": self.print_debug,
            "num_prints": self.num_prints,
            "hpo_mode": self.hpo_mode,
            "clear_history": self.clear_history,
            "save_data": self.save_data,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a TrainFunctionSettings object from a dictionary
        """
        # Ensure all keys are written in lowercase
        config_dict = {k.lower(): v for k, v in config_dict.items()}
        return cls(**config_dict)


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
    target_fidelities: Optional[list] = field(default_factory=lambda: [0.999, 0.9999, 0.99999])
    lookback_window: Optional[int] = 10
    anneal_learning_rate: Optional[bool] = False
    std_actions_eps: Optional[float] = 1e-2

    def as_dict(self):
        return {
            "training_constraint_name": self.training_constraint.constraint_name,
            "training_constraint_value": self.training_constraint.constraint_value,
            "target_fidelities": self.target_fidelities,
            "lookback_window": self.lookback_window,
            "anneal_learning_rate": self.anneal_learning_rate,
            "std_actions_eps": self.std_actions_eps,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a TrainingConfig object from a dictionary
        """
        config_dict = {k.lower(): v for k, v in config_dict.items()}
        if "total_updates" in config_dict:
            config_dict["training_constraint"] = TotalUpdates.from_value(
                config_dict.pop("total_updates")
            )
        else:
            config_dict["training_constraint"] = HardwareRuntime.from_value(
                config_dict.pop("hardware_runtime")
            )
        return cls(**config_dict)


@dataclass
class WandBConfig:
    """
    Weights and Biases configuration

    :param project: Name of the project
    :param entity: Name of the entity
    :param tags: List of tags
    :param notes: Notes for the project
    """

    enabled: bool = False
    project: str = "Quantum-RL"
    entity: Optional[str] = None
    tags: Optional[list] = None
    notes: Optional[str] = None
    api_key: Optional[str] = None

    def as_dict(self):
        return {
            "project_name": self.project,
            "entity": self.entity,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a WandBConfig object from a dictionary
        """
        # Ensure all keys are written in lowercase
        config_dict = {k.lower(): v for k, v in config_dict.items()}
        return cls(**config_dict)


@dataclass
class PPOConfig:
    """
    PPO configuration

    :param training_config: TrainingConfig
    :param train_function_settings: TrainFunctionSettings
    :param wandb_config: WandBConfig
    """

    run_name: str = "test"
    num_updates: int = 500
    n_epochs: int = 8
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_value_loss: bool = True
    clip_value_coef: float = 0.5
    normalize_advantage: bool = True
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    gradient_clip: float = 0.5
    include_critic: bool = True
    hidden_layers: list = field(default_factory=lambda: [64, 64])
    hidden_activation_functions: list = field(default_factory=lambda: ["tanh", "tanh"])
    input_activation_function: nn.Module | str = "identity"
    output_activation_mean: nn.Module | str = "tanh"
    output_activation_std: nn.Module | str = "identity"
    optimizer: str | optim.Optimizer | Callable = "adam"
    minibatch_size: int = 16
    checkpoint_dir: str = "tmp/ppo"
    training_config: Optional[TrainingConfig] = field(default_factory=TrainingConfig)
    train_function_settings: Optional[TrainFunctionSettings] = field(
        default_factory=TrainFunctionSettings
    )
    wandb_config: Optional[WandBConfig] = field(default_factory=WandBConfig)

    def __post_init__(self):
        """
        Check validity of the configuration
        """
        if not len(self.hidden_layers) == len(self.hidden_activation_functions):
            raise ValueError("Number of hidden layers and activation functions must be the same")
        if isinstance(self.optimizer, str):
            self.optimizer = get_optimizer(self.optimizer)
        if self.wandb_config is not None and not isinstance(self.wandb_config, WandBConfig):
            self.wandb_config = WandBConfig.from_dict(self.wandb_config)
        self.hidden_activation_functions = [
            get_module(activation) for activation in self.hidden_activation_functions
        ]
        self.input_activation_function = get_module(self.input_activation_function)
        self.output_activation_mean = get_module(self.output_activation_mean)
        self.output_activation_std = (
            get_module(self.output_activation_std)
            if self.output_activation_std is not None
            else None
        )

    def as_dict(self):
        return {
            "training_config": self.training_config.as_dict(),
            "train_function_settings": self.train_function_settings.as_dict(),
            "wandb_config": (
                self.wandb_config.as_dict() if self.wandb_config is not None else None
            ),
            "run_name": self.run_name,
            "num_updates": self.num_updates,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "clip_value_loss": self.clip_value_loss,
            "clip_value_coef": self.clip_value_coef,
            "normalize_advantage": self.normalize_advantage,
            "entropy_coef": self.entropy_coef,
            "value_loss_coef": self.value_loss_coef,
            "gradient_clip": self.gradient_clip,
            "include_critic": self.include_critic,
            "hidden_layers": self.hidden_layers,
            "hidden_activation_functions": [
                reverse_module_dict[type(activation)]
                for activation in self.hidden_activation_functions
            ],
            "input_activation_function": reverse_module_dict[type(self.input_activation_function)],
            "output_activation_mean": reverse_module_dict[type(self.output_activation_mean)],
            "output_activation_std": reverse_module_dict[type(self.output_activation_std)],
            "optimizer": reverse_optim_dict[self.optimizer],
            "minibatch_size": self.minibatch_size,
            "checkpoint_dir": self.checkpoint_dir,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a PPOConfig object from a dictionary
        """
        # Ensure all keys are written in lowercase
        config_dict = {k.lower(): v for k, v in config_dict.items()}
        if "wandb_config" in config_dict:
            config_dict["wandb_config"] = WandBConfig.from_dict(config_dict["wandb_config"])
        if "training_config" in config_dict:
            config_dict["training_config"] = TrainingConfig.from_dict(
                config_dict["training_config"]
            )
        if "train_function_settings" in config_dict:
            config_dict["train_function_settings"] = TrainFunctionSettings.from_dict(
                config_dict["train_function_settings"]
            )
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, file_path):
        """
        Create a PPOConfig object from a YAML file
        """
        config_dict = load_from_yaml_file(file_path)
        config_dict = {k.lower(): v for k, v in config_dict.items()}
        return cls.from_dict(config_dict)

    def initialize_wandb(self):
        """
        Initialize Weights and Biases
        """
        wandb.init(
            project=self.wandb_config.project,
            entity=self.wandb_config.entity,
            tags=self.wandb_config.tags,
            notes=self.wandb_config.notes,
        )

    def log_wandb(self, **kwargs):
        """
        Log data to Weights and Biases
        """
        wandb.log(kwargs)

    def finish_wandb(self):
        """
        Finish Weights and Biases
        """
        wandb.finish()
