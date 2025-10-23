from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal


@dataclass
class ExecutionConfig:
    """
    Configuration for the execution of the policy.

    Attributes:
        batch_size: The batch size for training.
        sampling_paulis: The number of Pauli strings to sample for fidelity estimation.
        n_shots: The number of shots for each circuit execution.
        n_reps: A list of the number of repetitions for the circuit.
        c_factor: The renormalization factor for the reward.
        seed: The seed for the random number generator.
        dfe_precision: The precision for direct fidelity estimation.
        control_flow_enabled: Whether to enable control flow in the circuit.
        n_reps_mode: The mode for handling repetitions.
    """

    batch_size: int = 100
    sampling_paulis: int = 100
    n_shots: int = 10
    n_reps: List[int] = 1
    c_factor: float = 1.0
    seed: int = 1234
    dfe_precision: Optional[Tuple[float, float]] = None
    control_flow_enabled: bool = True
    n_reps_mode: Literal["joint", "sequential"] = "sequential"

    def __post_init__(self):
        if isinstance(self.n_reps, int):
            self.n_reps = [self.n_reps]
        self._n_reps_index = 0

    def as_dict(self):
        """
        Returns a dictionary representation of the execution configuration.

        Returns:
            A dictionary representation of the execution configuration.
        """
        return {
            "batch_size": self.batch_size,
            "sampling_paulis": self.sampling_paulis,
            "n_shots": self.n_shots,
            "n_reps": self.n_reps,
            "c_factor": self.c_factor,
            "seed": self.seed,
            "dfe_precision": self.dfe_precision,
            "control_flow_enabled": self.control_flow_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Creates an ExecutionConfig from a dictionary.

        Args:
            data: The dictionary to create the ExecutionConfig from.

        Returns:
            An ExecutionConfig object.
        """
        return cls(**data)

    @property
    def n_reps_index(self) -> int:
        """The index of the current number of repetitions."""
        return self._n_reps_index

    @n_reps_index.setter
    def n_reps_index(self, value: int):
        assert 0 <= value < len(self.n_reps), "Index out of bounds"
        self._n_reps_index = value

    @property
    def current_n_reps(self) -> int:
        """The current number of repetitions."""
        return self.n_reps[self.n_reps_index]
