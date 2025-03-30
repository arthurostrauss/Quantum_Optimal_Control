from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Optional
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.base import BaseEstimatorV2, BaseSamplerV2

from .reward_data import RewardData, RewardDataList
from ..environment.configuration.qconfig import QEnvConfig
from ..environment.target import StateTarget, GateTarget
import numpy as np

PubLike = Union[SamplerPubLike, EstimatorPubLike]
Pub = Union[SamplerPub, EstimatorPub]
Primitive = Union[BaseEstimatorV2, BaseSamplerV2]
Target = Union[StateTarget, GateTarget]


@dataclass
class Reward(ABC):
    """
    Configuration for how to compute the reward in the RL workflow
    """

    print_debug = False

    def __post_init__(self):
        if self.reward_method == "channel" or self.reward_method == "state":
            self.dfe = True
        else:
            self.dfe = False
        self._total_shots = 0

    @property
    def reward_args(self):
        return {}

    @property
    @abstractmethod
    def reward_method(self) -> str:
        """
        String identifier for the reward method
        """
        raise NotImplementedError

    @property
    def total_shots(self) -> int:
        """
        Total number of shots involved for the last call of the reward computation
        """
        return self._total_shots

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.array,
        target: Target,
        env_config: QEnvConfig,
        *args,
    ) -> RewardDataList:
        """
        Compute pubs related to the reward method

        Args:
            qc: Quantum circuit to be executed on quantum system
            params: Parameters to feed the parametrized circuit
            target: Target gate or state to prepare
            backend_info: Backend information
            execution_config: Execution configuration
            n_reps: Number of repetitions of cycle circuit

        Returns:
            List of pubs related to the reward method
        """
        pass

    def get_reward_with_primitive(
        self,
        reward_data: RewardDataList,
        primitive: Primitive,
    ) -> np.array:
        pass

    def get_shot_budget(self, pubs: List[Pub]):
        """
        Compute the total number of shots to be used for the reward computation
        """
        pass

    def set_reward_seed(self, seed: int):
        pass
