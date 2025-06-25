from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Literal


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarking the policy through fidelity estimation

    Args:
        benchmark_cycle: benchmark_cycle (int, optional): Number of epochs between two fidelity benchmarking.
    """

    benchmark_cycle: int = 0  # 0 means no benchmarking
    benchmark_batch_size: int = 1
    check_on_exp: bool = False
    tomography_analysis: str = "default"
    dfe_precision: Tuple[float, float] = field(default=(1e-2, 1e-2))
    method: Literal["tomography", "rb"] = "rb"
