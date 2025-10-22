from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Literal


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarking the policy through fidelity estimation.

    Attributes:
        benchmark_cycle: Number of epochs between two fidelity benchmarking runs.
        benchmark_batch_size: The batch size for benchmarking.
        check_on_exp: Whether to check on the experiment.
        tomography_analysis: The analysis method for tomography.
        dfe_precision: The precision for direct fidelity estimation.
        method: The benchmarking method to use.
    """

    benchmark_cycle: int = 0  # 0 means no benchmarking
    benchmark_batch_size: int = 1
    check_on_exp: bool = False
    tomography_analysis: str = "default"
    dfe_precision: Tuple[float, float] = field(default=(1e-2, 1e-2))
    method: Literal["tomography", "rb"] = "rb"
