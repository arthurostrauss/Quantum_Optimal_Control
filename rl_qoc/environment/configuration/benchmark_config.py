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

    def benchmark_qm_macro(self):
        """
        QUA macro to perform benchmarking cycle based on the benchmark config type
        """
        raise NotImplementedError("This benchmark config does not have a QUA macro for benchmarking")

    def benchmark_qm_stream_processing(self):
        """
        QUA stream processing block to save the benchmarking results
        """
        raise NotImplementedError("This benchmark config does not have a QUA stream processing block for benchmarking")

    def benchmark_qm_post_processing(self) -> float:
        """
        Post-processing block to compute the benchmarking results, yielding the estimated fidelity for the policy at 
        the current cycle
        """
        raise NotImplementedError("This benchmark config does not have a post-processing block for benchmarking")