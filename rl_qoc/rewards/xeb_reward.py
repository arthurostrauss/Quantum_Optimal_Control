from dataclasses import dataclass
from typing import List

import numpy as np
from qiskit import QuantumCircuit

from .base_reward import Pub
from ..environment.configuration.execution_config import ExecutionConfig
from ..environment.target import StateTarget, GateTarget
from ..environment.backend_info import BackendInfo
from ..rewards.base_reward import Reward


@dataclass
class XEBReward(Reward):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    @property
    def reward_method(self):
        return "xeb"

    def get_reward_pubs(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: StateTarget | GateTarget,
        backend_info: BackendInfo,
        execution_config: ExecutionConfig,
        *args,
    ) -> List[Pub]:
        """
        Compute pubs related to the reward method
        """
        raise NotImplementedError
