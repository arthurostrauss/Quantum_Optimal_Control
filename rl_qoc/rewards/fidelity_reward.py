from dataclasses import dataclass
from .base_reward import Reward


@dataclass
class FidelityReward(Reward):
    """
    Configuration for computing the reward based on fidelity estimation
    """

    @property
    def reward_method(self):
        return "fidelity"
