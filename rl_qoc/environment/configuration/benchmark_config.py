from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .qconfig import QEnvConfig
    from ..target import GateTarget


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarking the policy through fidelity estimation

    Args:
        benchmark_cycle: benchmark_cycle (int, optional): Number of epochs between two fidelity benchmarking.
    """

    benchmark_cycle: int = 0  # 0 means no benchmarking

    def benchmark_qm_macro(self, policy):
        """
        QUA macro to perform benchmarking cycle based on the benchmark config type

        Args:
            policy: Policy to benchmark
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

    @property
    def method(self) -> str:
        """
        Method for benchmarking
        """
        raise NotImplementedError("This benchmark config does not have a method for benchmarking")

    @property
    def check_on_exp(self) -> bool:
        """
        Check if benchmarking should be performed through experiment instead of simulation.
        By default, benchmarking is performed through simulation.
        """
        return False

    
class SingleQubitRBBenchmarkConfig(BenchmarkConfig):
    """
    Configuration for benchmarking the policy throug real-time single qubit Randomized Benchmarking (in QUA)
    """
    interleaved_gate_operation: Literal["I", "x180", "y180", "x90", "-x90", "y90", "-y90"] = "x180"
    """The single qubit gate to interleave. Default is 'x180'."""
    use_state_discrimination: bool = False
    """Perform qubit state discrimination. Default is True."""
    use_strict_timing: bool = False
    """Use strict timing in the QUA program. Default is False."""
    num_random_sequences: int = 100
    """Number of random RB sequences. Default is 100."""
    num_shots: int = 20
    """Number of averages. Default is 20."""
    max_circuit_depth: int = 1000
    """Maximum circuit depth (number of Clifford gates). Default is 1000."""
    delta_clifford: int = 20
    """Delta clifford (number of Clifford gates between the RB sequences). Default is 20."""
    seed: Optional[int] = None
    """Seed for the random number generator. Default is None."""

    @property
    def method(self) -> str:
        """
        Method for benchmarking
        """
        return "rb"

    @property
    def check_on_exp(self) -> bool:
        """
        Check if benchmarking should be performed through experiment instead of simulation.
        By default, benchmarking is performed through simulation.
        """
        return True
    
    def benchmark_qm_macro(self, policy):
        """
        QUA macro to perform benchmarking cycle based on the benchmark config type
        """
        raise NotImplementedError("This benchmark config does not have a QUA macro for benchmarking")
    
class CAFEBenchmarkConfig(BenchmarkConfig):
    """
    Configuration for benchmarking the policy through CAFE fidelity estimation
    """
    n_reps: Sequence[int] = field(default_factory=lambda: [3, 5, 7, 9])
    @property
    def method(self) -> str:
        """
        Method for benchmarking
        """
        return "cafe"
    
    @property
    def check_on_exp(self) -> bool:
        """
        Check if benchmarking should be performed through experiment instead of simulation.
        By default, benchmarking is performed through simulation.
        """
        return True
    
    def benchmark_qm_macro(self, policy, config: QEnvConfig):
        """
        QUA macro to perform benchmarking cycle based on the benchmark config type
        """
        from qm.qua import program, declare, for_, stream_processing, assign, fixed
        from ...qua.qua_utils import rand_gauss_moller_box, rescale_and_clip_wrapper
        from qiskit_qm_provider import QMBackend
        from ...qua.qm_config import QMConfig
        from ...rewards.cafe.cafe_reward import CAFEReward
        from ..target import GateTarget

        if not isinstance(config.backend_config, QMConfig):
            raise ValueError("Backend config must be a QMConfig")
        if not isinstance(config.target, GateTarget):
            raise ValueError("CAFE reward is only supported for GateTarget")

        reward = CAFEReward("2-design")

    
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