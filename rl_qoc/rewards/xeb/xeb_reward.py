from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2

from ..base_reward import Reward
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import GateTarget
from .xeb_reward_data import XEBRewardDataList


@dataclass
class XEBReward(Reward):
    """
    Configuration for computing the reward based on cross-entropy benchmarking
    """

    @property
    def reward_method(self):
        return "xeb"

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: GateTarget,
        env_config: QEnvConfig,
        *args,
    ) -> XEBRewardDataList:
        """
        Compute pubs related to the reward method
        """
        raise NotImplementedError
    
    def get_reward_with_primitive(
        self,
        reward_data: XEBRewardDataList,
        primitive: BaseSamplerV2,
    ) -> np.array:
        raise NotImplementedError