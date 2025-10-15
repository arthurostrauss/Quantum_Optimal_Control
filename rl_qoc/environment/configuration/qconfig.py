from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal, Callable, Any, TYPE_CHECKING
from gymnasium.spaces import Box
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers import BackendV2
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.version import get_version_info
from qiskit_ibm_runtime import IBMBackend

from .backend_config import (
    BackendConfig,
    DynamicsConfig,
    QiskitConfig,
)

from ..target import GateTarget, StateTarget
from .execution_config import ExecutionConfig
from .benchmark_config import BenchmarkConfig
from ...helpers import load_q_env_from_yaml_file, select_backend

if TYPE_CHECKING:
    from ...rewards import Reward, REWARD_STRINGS


def default_benchmark_config():
    return BenchmarkConfig()


@dataclass
class QEnvConfig:
    """
    Quantum Environment configuration.
    This is used to define all hyperparameters characterizing the QuantumEnvironment.

    Args:
        target (Dict): Target state or target gate to prepare
        backend_config (rl_qoc.environment.configuration.backend_config.BackendConfig): Backend configuration
        action_space (Space): Action space
        execution_config (ExecutionConfig): Execution configuration
        reward (Reward): Reward configuration
        benchmark_config (BenchmarkConfig): Benchmark configuration
    """

    target: GateTarget | StateTarget
    backend_config: BackendConfig
    action_space: Box
    execution_config: ExecutionConfig
    reward: Reward = "state"
    benchmark_config: BenchmarkConfig = field(default_factory=default_benchmark_config)
    env_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.target, Dict):
            if "gate" in self.target:
                self.target = GateTarget(**self.target)
            else:
                self.target = StateTarget(**self.target)
        if isinstance(self.reward, str):
            from ...rewards import reward_dict

            self.reward = reward_dict[self.reward]()
        else:
            from ...rewards import Reward

            if not isinstance(self.reward, Reward):
                raise ValueError("Reward configuration must be a string or a Reward instance")

    @property
    def backend(self) -> Optional[BackendV2]:
        return self.backend_config.backend

    @backend.setter
    def backend(self, backend: BackendV2):
        self.backend_config.backend = backend
        self.backend_config.parametrized_circuit_kwargs["backend"] = backend

    @property
    def parametrized_circuit(self):
        if self.backend_config.parametrized_circuit is not None:
            return self.backend_config.parametrized_circuit
        else:
            op_name = self.target.gate.name if isinstance(self.target, GateTarget) else "state_prep"
            custom_op = op_name + "_cal"
            from ...helpers.circuit_utils import add_custom_gate

            return lambda qc, params, q_reg, **kwargs: add_custom_gate(
                qc, custom_op, q_reg, params, self.target.physical_qubits, self.backend
            )

    @property
    def parametrized_circuit_kwargs(self):
        """
        Additional keyword arguments to feed the parametrized_circuit function
        Returns: Dictionary of additional keyword arguments with their values
        """
        return self.backend_config.parametrized_circuit_kwargs

    @parametrized_circuit_kwargs.setter
    def parametrized_circuit_kwargs(self, value: Dict):
        self.backend_config.parametrized_circuit_kwargs = value

    @property
    def physical_qubits(self):
        return self.target.physical_qubits

    @property
    def batch_size(self):
        return self.execution_config.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        assert value > 0, "Batch size must be greater than 0"
        self.execution_config.batch_size = value

    @property
    def sampling_paulis(self):
        return self.execution_config.sampling_paulis

    @sampling_paulis.setter
    def sampling_paulis(self, value: int):
        self.execution_config.sampling_paulis = value

    @property
    def n_shots(self):
        return self.execution_config.n_shots

    @n_shots.setter
    def n_shots(self, value: int):
        assert value > 0, "Number of shots must be greater than 0"
        self.execution_config.n_shots = value

    @property
    def n_reps(self) -> List[int]:
        """
        List of possible number of repetitions / circuit depths for reward computation
        """
        return self.execution_config.n_reps

    @n_reps.setter
    def n_reps(self, value: int | List[int]):
        if isinstance(value, int):
            assert value > 0, "Number of repetitions must be greater than 0"
        else:
            assert all(v > 0 for v in value), "Number of repetitions must be greater than 0"
        self.execution_config.n_reps = [value] if isinstance(value, int) else value

    @property
    def current_n_reps(self) -> int:
        """
        Current number of repetitions / circuit depth for reward computation
        """
        return self.execution_config.current_n_reps

    @property
    def c_factor(self):
        """
        Reward scaling factor
        """
        return self.execution_config.c_factor

    @property
    def seed(self):
        """
        Random seed superseding the whole training
        """
        return self.execution_config.seed

    @seed.setter
    def seed(self, value: int):
        self.execution_config.seed = value

    @property
    def benchmark_cycle(self):
        return self.benchmark_config.benchmark_cycle

    @benchmark_cycle.setter
    def benchmark_cycle(self, value: int):
        self.benchmark_config.benchmark_cycle = value

    @property
    def benchmark_batch_size(self):
        return self.benchmark_config.benchmark_batch_size

    @benchmark_batch_size.setter
    def benchmark_batch_size(self, value: int):
        self.benchmark_config.benchmark_batch_size = value

    @property
    def tomography_analysis(self):
        return self.benchmark_config.tomography_analysis

    @property
    def check_on_exp(self):
        return self.benchmark_config.check_on_exp

    @property
    def reward_method(self):
        return self.reward.reward_method

    @reward_method.setter
    def reward_method(
        self, value: Literal["fidelity", "channel", "state", "xeb", "cafe", "orbit", "shadow"]
    ):
        try:
            from ...rewards import reward_dict

            self.reward = reward_dict[value]()
        except KeyError as e:
            raise ValueError(f"Reward method {value} not recognized")

    @property
    def dfe(self):
        """
        Indicates if Direct Fidelity Estimation is used for the reward computation (true if reward_method is "channel"
        or "state" and false otherwise)
        Returns: Boolean indicating if DFE is used

        """
        return self.reward.dfe

    @property
    def n_actions(self):
        """
        Number of actions in the action space (number of parameters to tune in the parametrized circuit)
        """
        return self.action_space.shape[-1]

    @property
    def channel_estimator(self):
        return self.reward_method == "channel"

    @property
    def fidelity_access(self):
        return self.reward_method == "fidelity"

    @property
    def instruction_durations_dict(self):
        return self.backend_config.instruction_durations

    @instruction_durations_dict.setter
    def instruction_durations_dict(self, value: InstructionDurations):
        self.backend_config.instruction_durations = value

    @property
    def pass_manager(self):
        return self.backend_config.pass_manager

    @pass_manager.setter
    def pass_manager(self, value: PassManager):
        self.backend_config.pass_manager = value

    def as_dict(self):
        config = {
            "target": self.target.as_dict(),
            "backend_config": self.backend_config.as_dict(),
            "action_space": {
                "low": self.action_space.low.tolist(),
                "high": self.action_space.high.tolist(),
            },
            "execution_config": {
                "batch_size": self.batch_size,
                "sampling_paulis": self.sampling_paulis,
                "n_shots": self.n_shots,
                "n_reps": self.n_reps,
                "c_factor": self.c_factor,
                "seed": self.seed,
            },
            "reward_config": self.reward_method,
            "benchmark_config": {
                "benchmark_cycle": self.benchmark_cycle,
                "benchmark_batch_size": self.benchmark_batch_size,
                "tomography_analysis": self.tomography_analysis,
                "check_on_exp": self.check_on_exp,
            },
            "metadata": self.env_metadata,
        }

        return config

    @classmethod
    def from_yaml(
        cls,
        config_file_path: str,
        parametrized_circ_func: Optional[
            Callable[
                [
                    QuantumCircuit,
                    ParameterVector | List[Parameter],
                    QuantumRegister,
                    Dict[str, Any],
                ],
                None,
            ]
        ],
        backend: Optional[BackendV2 | Callable[[Any], BackendV2]] = None,
        pass_manager: Optional[PassManager] = None,
        instruction_durations: Optional[InstructionDurations] = None,
        **backend_callback_kwargs: Any,
    ) -> QEnvConfig:
        """
        Get Quantum Environment configuration from a YAML file and additional parameters.

        Args:
            config_file_path (str): Path to the YAML configuration file
            parametrized_circ_func (Callable): Function to create the parametrized circuit
            backend (BackendV2 or Callable): Optional custom backend or function to create a backend (if None,
                                              backend will be created from the backend_config in the YAML file)
            pass_manager (PassManager): Pass manager instance for custom transpilation
            instruction_durations (InstructionDurations): Instruction durations to specify durations for custom gates
            **backend_callback_kwargs: Additional keyword arguments to pass to the backend callback function

        Returns:
            QEnvConfig: Quantum Environment configuration object
        """

        params, backend_params, runtime_options = load_q_env_from_yaml_file(config_file_path)

        if isinstance(backend, Callable):
            backend = backend(**backend_callback_kwargs)
        elif backend is None:
            backend = select_backend(**backend_params)
        if get_version_info() < "2.0.0":
            from qiskit_dynamics import DynamicsBackend

            if isinstance(backend, DynamicsBackend):
                backend_config = DynamicsConfig(
                    parametrized_circ_func,
                    backend,
                    pass_manager=pass_manager,
                    instruction_durations=instruction_durations,
                )
                return cls(backend_config=backend_config, **params)
        if isinstance(backend, BackendV2):
            backend_config = QiskitConfig(
                parametrized_circ_func,
                backend,
                pass_manager=pass_manager,
                instruction_durations=instruction_durations,
            )

        else:
            raise ValueError("Backend type not recognized")

        return cls(backend_config=backend_config, **params)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        backend_config_type: Literal["qiskit", "dynamics", "runtime", "qm"] = "qiskit",
        parametrized_circ_func: Optional[
            Callable[
                [
                    QuantumCircuit,
                    ParameterVector | List[Parameter],
                    QuantumRegister,
                    Dict[str, Any],
                ],
                None,
            ]
        ] = None,
        backend: Optional[BackendV2 | Callable[[Any], BackendV2]] = None,
        pass_manager: Optional[PassManager] = None,
        instruction_durations: Optional[InstructionDurations] = None,
        **backend_callback_kwargs: Any,
    ) -> QEnvConfig:
        """
        Get Quantum Environment configuration from a dictionary and additional parameters.
        """
        import numpy as np

        if "target" not in config_dict:
            raise ValueError("Configuration dictionary must contain a 'target' key")
        if "backend_config" not in config_dict:
            raise ValueError("Configuration dictionary must contain a 'backend_config' key")

        target = config_dict["target"]
        backend_config = config_dict["backend_config"]

        if isinstance(target, dict):
            if "gate" in target:
                target = GateTarget(**target)
            else:
                target = StateTarget(**target)

        if isinstance(backend_config, dict):
            if backend_config_type == "qiskit":
                backend_config = QiskitConfig(
                    **backend_config,
                    parametrized_circuit=parametrized_circ_func,
                    backend=backend,
                    pass_manager=pass_manager,
                    instruction_durations=instruction_durations,
                )
            elif backend_config_type == "dynamics":
                backend_config = DynamicsConfig(
                    **backend_config,
                    parametrized_circuit=parametrized_circ_func,
                    backend=backend,
                    pass_manager=pass_manager,
                    instruction_durations=instruction_durations,
                )
            elif backend_config_type == "runtime":
                backend_config = QiskitRuntimeConfig(
                    **backend_config,
                    parametrized_circuit=parametrized_circ_func,
                    backend=backend,
                    pass_manager=pass_manager,
                    instruction_durations=instruction_durations,
                )
            elif backend_config_type == "qm":
                from ...qua.qm_config import QMConfig

                backend_config = QMConfig(
                    **backend_config,
                    parametrized_circuit=parametrized_circ_func,
                    backend=backend,
                    pass_manager=pass_manager,
                    instruction_durations=instruction_durations,
                )

        return cls(
            target=target,
            backend_config=backend_config,
            action_space=Box(
                low=np.array(config_dict["action_space"]["low"]),
                high=np.array(config_dict["action_space"]["high"]),
                dtype=np.float32,
            ),
            execution_config=ExecutionConfig(**config_dict["execution_config"]),
            reward=config_dict.get("reward", "state"),
            benchmark_config=BenchmarkConfig(**config_dict.get("benchmark_config", {})),
            env_metadata=config_dict.get("metadata", {}),
        )
