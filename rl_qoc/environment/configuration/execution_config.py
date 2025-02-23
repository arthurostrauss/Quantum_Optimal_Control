from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal


@dataclass
class ExecutionConfig:
    """
    Configuration for the execution of the policy

    Args:
        batch_size: Batch size (iterate over a bunch of actions per policy to estimate expected return). Defaults to 50.
        sampling_paulis: Number of Paulis to sample for the fidelity estimation scheme. For ORBIT, this would be the number of
            random Clifford sequences to sample.
        n_shots: Number of shots per Pauli for the fidelity estimation. Defaults to 1.
        n_reps: Number of repetitions of cycle circuit (can be an integer or a list of integers to play multiple lengths)
        c_factor: Renormalization factor. Defaults to 0.5.
        seed: General seed for Environment internal sampling mechanisms (e.g. input state, observable, n_reps). Defaults to 1234.
        dfe_precision: Precision for the DFE. Defaults to None.
            Should be a tuple indicating expected additive error and failure probability.
        control_flow_enabled: Flag to enable control flow design of runnable circuit (relevant for involving
            real time control flow). Defaults to False.
    """

    batch_size: int = 100
    sampling_paulis: int = 100
    n_shots: int = 10
    n_reps: List[int] | int = 1
    c_factor: float = 1.0
    seed: int = 1234
    dfe_precision: Optional[Tuple[float, float]] = None
    control_flow_enabled: bool = False
    n_reps_mode: Literal["joint", "sequential"] = "sequential"

    def __post_init__(self):
        if isinstance(self.n_reps, int):
            self.n_reps = [self.n_reps]
        self._n_reps_index = 0

    def as_dict(self):
        return {
            "batch_size": self.batch_size,
            "sampling_paulis": self.sampling_paulis,
            "n_shots": self.n_shots,
            "n_reps": self.n_reps,
            "c_factor": self.c_factor,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

    @property
    def n_reps_index(self) -> int:
        return self._n_reps_index

    @n_reps_index.setter
    def n_reps_index(self, value: int):
        assert 0 <= value < len(self.n_reps), "Index out of bounds"
        self._n_reps_index = value

    @property
    def current_n_reps(self) -> int:
        return self.n_reps[self.n_reps_index]
