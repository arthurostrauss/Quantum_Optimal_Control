from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike

PubLike = Union[SamplerPubLike, EstimatorPubLike]
Pub = Union[SamplerPub, EstimatorPub]
from ..environment.backend_info import BackendInfo
from ..environment.target import StateTarget, GateTarget
from ..environment.configuration.execution_config import ExecutionConfig
import numpy as np


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

    def get_shot_budget(self, pubs: List[Pub]):
        """
        Compute the total number of shots to be used for the reward computation
        """
        pass

    def set_reward_seed(self, seed: int):
        pass
