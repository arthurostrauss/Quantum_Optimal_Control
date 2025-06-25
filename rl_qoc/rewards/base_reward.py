from __future__ import annotations

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
    def dfe(self) -> bool:
        """
        Whether the reward method is a direct fidelity estimation scheme
        """
        return self.reward_method in ["state", "channel"]

    def get_reward_data(
        self,
        qc: QuantumCircuit,
        params: np.ndarray,
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
    ) -> np.ndarray:
        pass

    def get_shot_budget(self, pubs: List[Pub]):
        """
        Compute the total number of shots to be used for the reward computation
        """
        pass

    def set_reward_seed(self, seed: int):
        pass

    def get_real_time_circuit(
        self,
        circuits: QuantumCircuit | List[QuantumCircuit],
        target: Target | List[Target],
        env_config: QEnvConfig,
        skip_transpilation: bool = False,
        *args,
    ) -> QuantumCircuit:
        """
        Get a circuit containing real-time control flow and logic to compute the reward.
        To be used with the QMEnvironment for optimizing compilation latency.
        Args:
            circuits: List of quantum circuits to be executed on quantum system (or a single circuit)
                      A switch statement is used to select the circuit to be executed at runtime.
            target:  List of target gates or states to prepare (or a single target), used to inform the transpilation
                     pipeline
            env_config: Environment configuration, containing information for transpilation process and
                        execution configuration (notably the desired number of repetitions of the cycle circuit)
            skip_transpilation: If True, the circuit will not be transpiled and will be returned as is.
                                Used when there is no need to transpile the circuit, for example when using IQCC in
                                sync-hook mode.
            *args: Optional arguments for the reward method

        Returns:
            A QuantumCircuit object containing the real-time control flow and logic to compute the reward.

        """
        raise NotImplementedError(
            "get_real_time_circuit is not implemented for this reward method."
        )
