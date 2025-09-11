from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Optional, TYPE_CHECKING
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.base import BaseEstimatorV2, BaseSamplerV2

from .reward_data import RewardData, RewardDataList
from ..environment.configuration.qconfig import QEnvConfig
from ..environment.target import StateTarget, GateTarget
import numpy as np

if TYPE_CHECKING:
    from qiskit_qm_provider.parameter_table import ParameterTable, Parameter as QuaParameter
    from ..qua.circuit_params import CircuitParams
    from qm import Program

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

    def get_shot_budget(self, pubs: List[Pub]) -> int:
        """
        Compute the total number of shots to be used for the reward computation
        """
        raise NotImplementedError("This reward method does not support shot budget")

    def set_reward_seed(self, seed: int):
        raise NotImplementedError("This reward method does not support setting reward seed")

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
        raise NotImplementedError("This reward method does not support real-time circuit")
            "get_real_time_circuit is not implemented for this reward method."
        )

    def qm_step(
        self,
        reward_data: RewardDataList,
        fetching_index: int,
        fetching_size: int,
        circuit_params: CircuitParams,
        reward: QuaParameter,
        config: QEnvConfig,
        **push_args,
    ):
        """
        This method is used to compute the classical processing that surrounds the QUA program execution in a step.
        It is used in the QMEnvironment.step() function.
        
        Args:
            reward_data: Reward data to be used to compute the reward (can be used to send inputs to the QUA program and also to post-process measurement outcomes/counts coming out of the QUA program)
            fetching_index: Index of the first measurement outcome to be fetched in stream processing / DGX Quantum stream
            fetching_size: Number of measurement outcomes to be fetched
            circuit_params: Parameters defining the quantum program to be executed, those are entrypoints towards streaming values to control-flow that define the program adaptively (e.g. input state, number of repetitions, observable, etc.)
            reward: Reward parameter to be used to fetch measurement outcomes from the QUA program and compute the reward
            config: Environment configuration
            **push_args: Additional arguments to pass necessary entrypoints to communicate with the OPX (e.g. job, qm, verbosity, etc.)

        Returns:
            Reward array of shape (batch_size,)
            
        """
        raise NotImplementedError("This reward method does not support QM step")

    def rl_qoc_training_qua_prog(
        self,
        qc: QuantumCircuit,
        policy: ParameterTable,
        reward: QuaParameter,
        circuit_params: CircuitParams,
        config: QEnvConfig,
        num_updates: int = 1000,
        test: bool = False,
    ) -> Program:
        raise NotImplementedError("This reward method does not have a QUA program for training")
