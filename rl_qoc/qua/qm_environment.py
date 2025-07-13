from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qm import QuantumMachine, Program
from quam.utils.qua_types import QuaVariableInt

from .circuit_params import CircuitParams
from ..environment import (
    ContextAwareQuantumEnvironment,
    QEnvConfig,
)

from .qm_config import QMConfig
from qm.jobs.running_qm_job import RunningQmJob
from typing import List, Optional, Union
from qiskit_qm_provider import (
    Parameter as QuaParameter,
    ParameterTable,
    Direction,
    InputType,
    ParameterPool,
    QMBackend,
)
from ..rewards import CAFERewardDataList, ChannelRewardDataList, StateRewardDataList

# @callable_from_qua
# def qua_print(*args):
#     text = ""
#     for i in range(0, len(args)-1, 2):
#         text += f"{args[i]}= {args[i+1]} | "
#     if len(args) % 2 == 1:
#         text += f"{args[-1]} | "
#     print(text)

class QMEnvironment(ContextAwareQuantumEnvironment):

    def __init__(
        self,
        training_config: QEnvConfig,
        job: Optional[RunningQmJob] = None,
    ):
        ParameterPool.reset()
        super().__init__(training_config)

        mu = QuaParameter(
            "mu",
            [0.0] * self.n_actions,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        sigma = QuaParameter(
            "sigma",
            [1.0] * self.n_actions,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        self.policy = ParameterTable([mu, sigma], name="policy")
        self.reward = QuaParameter(
            "reward",
            [0] * 2**self.n_qubits,
            input_type=self.input_type if self.input_type == InputType.DGX else None,
            direction=Direction.INCOMING,
        )

        self.real_time_circuit = self.config.reward.get_real_time_circuit(
            self.circuits,
            self.target,
            self.config,
            skip_transpilation=True,
        )
        self.circuit_params = CircuitParams.from_circuit(
            self.real_time_circuit, self.input_type, self.config
        )

        if self.input_type == InputType.DGX:
            ParameterPool.patch_opnic_wrapper(self.qm_backend_config.opnic_dev_path)

        self._qm_job: Optional[RunningQmJob] = job
        if self.backend is not None:
            self.backend.update_compiler_from_target(self.input_type)
            if (
                hasattr(self.real_time_circuit, "calibrations")
                and self.real_time_circuit.calibrations
            ):
                self.backend.update_calibrations(
                    qc=self.real_time_circuit, input_type=self.input_type
                )
        self._step_indices = {}
        self._total_data_points = 0

    def step(self, action):
        """
        Perform the action on the quantum environment
        """
        if self._qm_job is None:
            raise RuntimeError(
                "The QUA program has not been started yet. Call start_program() first."
            )

        push_args = {
            "job": self.qm_job,
            "qm": self.qm,
            "verbosity": self.qm_backend_config.verbosity,
        }
        mean_val = self.mean_action.tolist()
        std_val = self.std_action.tolist()

        additional_input = (
            self.config.execution_config.dfe_precision if self.config.dfe else self.baseline_circuit
        )
        reward_data = self.config.reward.get_reward_data(
            self.circuit,
            np.zeros((1, self.n_actions)),
            self.target,
            self.config,
            additional_input,
        )
        self._reward_data = reward_data

        # Push policy parameters to trigger real-time action sampling
        self.policy.push_to_opx({"mu": mean_val, "sigma": std_val}, **push_args)
        print("Just pushed policy parameters to OPX:", mean_val, std_val)

        # Push the data to compute reward to the OPX
        if hasattr(reward_data, "input_indices"):
            input_state_indices = reward_data.input_indices
            max_input_state = len(input_state_indices)

            num_obs_per_input_state = tuple(
                len(reward_data.observables_indices[i]) if self.config.dfe else 1
                for i in range(max_input_state)
            )
        else:
            num_obs_per_input_state = (1,)

        cumulative_datapoints = np.cumsum(num_obs_per_input_state).tolist()
        step_data_points = int(cumulative_datapoints[-1])
        self._step_indices[self.step_tracker] = (
            self._total_data_points,
            self._total_data_points + step_data_points,
        )
        self._total_data_points += step_data_points
        fetching_index, finishing_index = self._step_indices[self.step_tracker]
        fetching_size = finishing_index - fetching_index
        if self.qm_backend_config.verbosity > 0:
            print(f"Fetching index: {fetching_index}, finishing index: {finishing_index}")
            print(f"Fetching size: {fetching_size}")
            print(f"Step indices: {self._step_indices}")
            print(f"Total data points: {self._total_data_points}")
            

        reward = self.config.reward.qm_step(
            reward_data,
            fetching_index,
            fetching_size,
            self.circuit_params,
            self.reward,
            self.config,
            **push_args,
        )
        
        if np.mean(reward) > self._max_return:
            self._max_return = np.mean(reward)
            self._optimal_actions[self.circuit_choice] = self.mean_action

        # reward = np.clip(reward, 0.0, 1.0 - 1e-6)
        self.reward_history.append(reward)
        self.update_env_history(self.real_time_circuit, reward_data.total_shots)
        # reward = -np.log10(1.0 - reward)  # Convert to negative log10 scale

        return self._get_obs(), reward, True, False, self._get_info()

    def rl_qoc_training_qua_prog(self, num_updates: int = 1000) -> Program:
        """
        Generate a QUA program tailor-made for the RL-based calibration project
        """
        rl_qoc_training_prog = self.config.reward.rl_qoc_training_qua_prog(
            self.real_time_transpiled_circuit,
            self.policy,
            self.reward,
            self.circuit_params,
            self.config,
            num_updates,
            self.qm_backend_config.test_mode
        )

        return rl_qoc_training_prog

    @property
    def qm_backend_config(self) -> QMConfig:
        """
        Get the QM backend configuration
        """
        return self.config.backend_config

    @property
    def backend(self) -> QMBackend:
        return super().backend

    @property
    def real_time_transpiled_circuit(self) -> QuantumCircuit:
        """
        Get the real-time circuit transpiled for QUA execution
        """
        return self.backend_info.custom_transpile(
            self.real_time_circuit,
            optimization_level=1,
            initial_layout=self.layout,
            remove_final_measurements=False,
            scheduling=False,
        )

    def start_program(
        self,
    ) -> RunningQmJob:
        """
        Start the QUA program

        Returns:
            RunningQmJob: The running Qmjob
        """
        if self.input_type == InputType.DGX:
            ParameterPool.configure_stream()
        if hasattr(self.real_time_circuit, "calibrations") and self.real_time_circuit.calibrations:
            self.backend.update_calibrations(qc=self.real_time_circuit, input_type=self.input_type)
        self.backend.update_compiler_from_target()
        prog = self.rl_qoc_training_qua_prog(num_updates=self.qm_backend_config.num_updates)
        self.backend.close_all_qms()
        self._qm_job = self.qm.execute(
            prog, compiler_options=self.qm_backend_config.compiler_options
        )
        return self._qm_job

    def close(self) -> bool:
        """
        Close the environment (stop the running QUA program)
        Returns:

        """
        if self.input_type == InputType.DGX:
            ParameterPool.close_streams()
        finish = self.qm_job.halt()
        if not finish:
            print("Failed to halt the job")
        print("Job status: ", self.qm_job.status)
        self._qm_job = None
        return finish

    @property
    def input_type(self) -> InputType:
        """
        Get the input type for streaming to OPX
        """
        return self.qm_backend_config.input_type

    @property
    def qm_job(self) -> RunningQmJob:
        """
        Get the running QM job
        """
        return self._qm_job

    @property
    def qm(self) -> QuantumMachine:
        """
        Get the QM object
        """
        return self.backend.qm if self.backend is not None else None

    @property
    def all_parameters(self) -> List[QuaParameter | ParameterTable]:
        """
        Get all parameters that are used in the real-time circuit and that are not None
        """
        return [
            param
            for param in [
                self.real_time_circuit_parameters,
                self.max_input_state,
                self.input_state_vars,
                self.policy,
                self.reward,
                self.circuit_choice_var,
                self.n_reps_var,
                self.max_observables,
                self.n_shots,
                self.observable_vars,
            ]
            if param is not None
        ]
    
    def clear_history(self):
        """
        Clear the history of the environment
        """
        super().clear_history()
        self._step_indices = {}
        self._total_data_points = 0
