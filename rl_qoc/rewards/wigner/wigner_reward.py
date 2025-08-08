from qiskit import QuantumCircuit
import numpy as np
from ..base_reward import Reward
from ...environment.configuration.qconfig import QEnvConfig
from ...environment.target import Target
from .wigner_reward_data import WignerRewardDataList


class WignerReward(Reward):
    """
    Configuration for computing the reward based on Wigner
    """

    @property
    def reward_method(self):
        return "wigner"
    
    def get_reward_data(self, qc: QuantumCircuit, params: np.ndarray, target: Target, env_config: QEnvConfig, *args) -> WignerRewardDataList:
        pass    