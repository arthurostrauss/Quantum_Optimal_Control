"""
Class to generate a RL environment suitable for usage with PyTorch, leveraging Qiskit modules to simulate
quantum system (could also include QUA code in the future)

Author: Arthur Strauss
Created on 28/11/2022
Last updated: 05/09/2024
"""

from __future__ import annotations

from dataclasses import asdict

# For compatibility for options formatting between Estimators.
from typing import List, Any, SupportsFloat, Tuple, Optional
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box
from qiskit import schedule, pulse, QuantumRegister

# Qiskit imports
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
)

# Qiskit Quantum Information, for fidelity benchmarking
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit.transpiler import InstructionProperties
from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.library import ProcessTomography

from .base_q_env import (
    BaseQuantumEnvironment,
    GateTarget,
    StateTarget,
)
from ..helpers import (
    simulate_pulse_input,
    get_optimal_z_rotation,
)
from ..helpers.circuit_utils import fidelity_from_tomography
from .configuration.qconfig import QEnvConfig, GateTargetConfig


class QuantumEnvironment(BaseQuantumEnvironment):

    def __init__(self, training_config: QEnvConfig):
        """
        Initialize the Quantum Environment
        Args:
            training_config: QEnvConfig object containing the training configuration
        """
        self._parameters = ParameterVector("θ", training_config.n_actions)

        super().__init__(training_config)

        self._target, self.circuits, self.baseline_circuits = (
            self.define_target_and_circuits()
        )
        # self.observation_space = Box(
        #     low=np.array([0, 0] + [-5] * (2 ** self.n_qubits) ** 2),
        #     high=np.array([1, 1] + [5] * (2 ** self.n_qubits) ** 2),
        #     dtype=np.float32,
        # )
        self.observation_space = Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32
        )

    @property
    def parameters(self):
        return self._parameters

    @property
    def trunc_index(self) -> int:
        return 0

    @property
    def tgt_instruction_counts(self) -> int:
        return 1

    @property
    def target(self) -> GateTarget:
        """
        Return current target to be calibrated
        """
        return self._target

    def episode_length(self, global_step: int) -> int:
        return 1

    def define_target_and_circuits(
        self,
    ) -> Tuple[
        GateTarget | StateTarget,
        List[QuantumCircuit],
        List[QuantumCircuit | DensityMatrix],
    ]:
        """
        Define the target to be used in the environment
        Returns:
            target: GateTarget or StateTarget object
        """
        input_states_choice = getattr(
            self.config.reward_config.reward_args, "input_states_choice", "pauli4"
        )
        q_reg = QuantumRegister(len(self.config.target.physical_qubits))
        if isinstance(self.config.target, GateTargetConfig):
            target = GateTarget(
                **self.config.target.as_dict(),
                input_states_choice=input_states_choice,
                tgt_register=q_reg,
            )

        else:
            target = StateTarget(**asdict(self.config.target))

        custom_circuit = QuantumCircuit(q_reg, name="custom_circuit")

        self.parametrized_circuit_func(
            custom_circuit,
            self.parameters,
            q_reg,
            **self._func_args,
        )

        if isinstance(target, StateTarget):
            ref_circuit = target.circuit.copy(name="baseline_circuit")
        else:
            ref_circuit = target.target_circuit.copy(name="baseline_circuit")

        custom_circuit.metadata["baseline_circuit"] = ref_circuit.copy()
        return target, [custom_circuit], [ref_circuit]

    def _get_obs(self):
        if isinstance(self.target, GateTarget) and self.config.reward_method == "state":
            return np.array(
                [
                    0.0,
                    0.0,
                ]
                + list(self._observable_to_observation())
            )
        else:
            return np.array([0, 0])

    def modify_environment_params(self, **kwargs):
        print(f"\n Number of repetitions: {self.n_reps}")

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._step_tracker += 1
        if self._episode_ended:
            print("Resetting environment")
            terminated = True
            return (
                self.reset()[0],
                np.zeros(self.batch_size),
                terminated,
                False,
                self._get_info(),
            )

        terminated = self._episode_ended = True
        params, batch_size = np.array(action), len(np.array(action))
        if params.shape != (batch_size, self.n_actions):
            raise ValueError(
                f"Action shape mismatch: {params.shape} != {(batch_size, self.n_actions)}"
            )
        reward = self.perform_action(action)

        if np.mean(reward) > self._max_return:
            self._max_return = np.mean(reward)
            self._optimal_action = self.mean_action
        self.reward_history.append(reward)
        assert (
            len(reward) == self.batch_size
        ), f"Reward table size mismatch {len(reward)} != {self.batch_size} "
        assert not np.any(np.isinf(reward)) and not np.any(
            np.isnan(reward)
        ), "Reward table contains NaN or Inf values"
        # Using Negative Log Error as the Reward
        optimal_error_precision = 1e-6
        max_fidelity = 1.0 - optimal_error_precision
        reward = np.clip(reward, a_min=0.0, a_max=max_fidelity)
        reward = -np.log(1.0 - reward)
        return self._get_obs(), reward, terminated, False, self._get_info()

    # def check_reward(self):
    #     if self.training_with_cal:
    #         print("Checking reward to adjust C Factor...")
    #         example_obs, _ = self.reset()
    #         if example_obs.shape != self.observation_space.shape:
    #             raise ValueError(
    #                 f"Training Config observation space ({self.observation_space.shape}) does not "
    #                 f"match Environment observation shape ({example_obs.shape})"
    #             )
    #         sample_action = np.random.normal(
    #             loc=(self.action_space.low + self.action_space.high) / 2,
    #             scale=(self.action_space.high - self.action_space.low) / 2,
    #             size=(self.batch_size, self.action_space.shape[-1]),
    #         )
    #
    #         batch_rewards = self.perform_action(sample_action)
    #         mean_reward = np.mean(batch_rewards)
    #         if not np.isclose(mean_reward, self.fidelity_history[-1], atol=1e-2):
    #             self.c_factor *= self.fidelity_history[-1] / mean_reward
    #             self.c_factor = np.round(self.c_factor, 1)
    #             print("C Factor adjusted to", self.c_factor)
    #         self.clear_history()
    #     else:
    #         pass

    def compute_benchmarks(self, qc: QuantumCircuit, params: np.array) -> np.array:
        """
        Method to store in lists all relevant data to assess performance of training (fidelity information)
        :param params: List of Action vectors to execute on quantum system
        :return: None
        """

        if self.config.check_on_exp:

            # Experiment based fidelity estimation
            try:
                if not self.config.reward_method == "fidelity":
                    if self.config.benchmark_config.benchmark_batch_size > 1:
                        angle_sets = np.clip(
                            np.random.normal(
                                self.mean_action,
                                self.std_action,
                                size=(self.config.benchmark_batch_size, self.n_actions),
                            ),
                            self.action_space.low,
                            self.action_space.high,
                        )
                    else:
                        angle_sets = [self.mean_action]
                else:
                    if len(params.shape) == 1:
                        params = np.expand_dims(params, axis=0)
                    angle_sets = params

                qc_input = [qc.assign_parameters(angle_set) for angle_set in angle_sets]
                print("Starting tomography...")
                fids = fidelity_from_tomography(
                    qc_input,
                    self.backend,
                    self.target.physical_qubits,
                    (
                        self.target.target_operator
                        if isinstance(self.target, GateTarget)
                        else self.target.dm
                    ),
                    analysis=self.config.tomography_analysis,
                    sampler=self.sampler,
                )
                print("Finished tomography")

            except Exception as e:
                self.close()
                raise e
            if isinstance(self.target, StateTarget):
                self.circuit_fidelity_history.append(np.mean(fids))
            else:
                self.avg_fidelity_history.append(np.mean(fids))

        else:  # Simulation based fidelity estimation (Aer for circuit level, Dynamics for pulse)
            print("Starting simulation benchmark...")
            if not self.config.reward_method == "fidelity":
                params = np.array(
                    [self.mean_action]
                )  # Benchmark policy only through mean action
            if self.abstraction_level == "circuit":  # Circuit simulation
                fids = self.simulate_circuit(qc, params)
            else:  # Pulse simulation
                fids = self.simulate_pulse_circuit(qc, params)
            if self.target.target_type == "state":
                print("State fidelity:", self.circuit_fidelity_history[-1])
            else:
                print("Avg gate fidelity:", self.avg_fidelity_history[-1])
            print("Finished simulation benchmark")
        return fids

    def update_gate_calibration(self, gate_name: Optional[str] = None):
        """
        Update backend target with the optimal action found during training

        :param gate_name: Name of custom gate to add to target (if None,
         use target gate and update its attached calibration)

        :return: Pulse calibration for the custom gate
        """
        if not isinstance(self.target, GateTarget):
            raise ValueError("Target type should be a gate for gate calibration task.")

        if self.abstraction_level == "pulse":
            assert self.backend is not None, "Backend must be set for pulse calibration"
            qc = self.circuits[0].assign_parameters(self.optimal_action, inplace=False)
            schedule_ = schedule(qc, self.backend)
            duration = schedule_.duration
            if isinstance(self.backend, DynamicsBackend):
                sim_data = simulate_pulse_input(
                    self.backend,
                    schedule_,
                    target=Operator(self.target.gate),
                )
                error = 1.0 - sim_data["gate_fidelity"]["optimal"]
                optimal_rots = sim_data["gate_fidelity"]["rotations"]
            else:
                exp = ProcessTomography(qc, self.backend, self.target.physical_qubits)
                exp_data = exp.run(shots=10000).block_for_results()
                process_matrix = exp_data.analysis_results("state").value
                opt_res = get_optimal_z_rotation(
                    process_matrix, self.target.target_operator, self.n_qubits
                )
                optimal_rots = opt_res.x
                error = 1.0 - opt_res.fun

            with pulse.build(self.backend) as sched:
                for i in range(self.n_qubits):
                    rz_cal = self.backend.target.get_calibration("rz", (i,))
                    pulse.call(
                        rz_cal,
                        value_dict={
                            parameter: optimal_rots[i]
                            for parameter in rz_cal.parameters
                        },
                    )
                pulse.call(schedule_)
                for i in range(self.n_qubits):
                    rz_cal = self.backend.target.get_calibration("rz", (i,))
                    pulse.call(
                        rz_cal,
                        value_dict={
                            parameter: optimal_rots[-1 - i]
                            for parameter in rz_cal.parameters
                        },
                    )

            instruction_prop = InstructionProperties(duration, error, sched)
            if gate_name is None:
                gate_name = self.target.gate.name

            if gate_name not in self.backend.operation_names:
                self.backend.target.add_instruction(
                    self.target.gate,
                    {tuple(self.physical_target_qubits): instruction_prop},
                    name=gate_name,
                )

            else:
                self.backend.target.update_instruction_properties(
                    gate_name,
                    tuple(self.physical_target_qubits),
                    instruction_prop,
                )

            return self.backend.target.get_calibration(
                gate_name, tuple(self.physical_target_qubits)
            )
        else:
            return self.circuits[0].assign_parameters(
                {self.parameters[0]: self.optimal_action}
            )
