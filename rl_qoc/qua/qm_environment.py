from __future__ import annotations

import numpy as np
from oqc import CompilationResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives.containers.bit_array import BitArray
from qm import QuantumMachine, Program, CompilerOptionArguments
from qm.qua._expressions import QuaArrayVariable
from quam.utils.qua_types import QuaVariableInt

from ..environment import (
    ContextAwareQuantumEnvironment,
    QEnvConfig,
)

from .qua_utils import binary, get_gaussian_sampling_input, clip_qua
from .qm_config import QMConfig
from qm.qua import *
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union
from qiskit_qm_provider import (
    Parameter as QuaParameter,
    ParameterTable,
    Direction,
    InputType,
    ParameterPool,
    QMBackend,
)
from qualang_tools.callable_from_qua import callable_from_qua, patch_qua_program_addons
from ..rewards import CAFERewardDataList, ChannelRewardDataList, StateRewardDataList

ALL_MEASUREMENTS_TAG = "measurements"
"""The tag to save all measurements results to."""
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"
RewardDataType = Union[CAFERewardDataList, ChannelRewardDataList, StateRewardDataList]

# @callable_from_qua
# def qua_print(*args):
#     text = ""
#     for i in range(0, len(args)-1, 2):
#         text += f"{args[i]}= {args[i+1]} | "
#     if len(args) % 2 == 1:
#         text += f"{args[-1]} | "
#     print(text)


def _get_state_int(qc: QuantumCircuit, result: CompilationResult, state_int: QuaVariableInt):
    for c, clbit in enumerate(qc.clbits):
        bit = qc.find_bit(clbit)
        if len(bit.registers) == 0:
            bit_output = result.result_program[f"{_QASM3_DUMP_LOOSE_BIT_PREFIX}{c}"]
        else:
            creg, creg_index = bit.registers[0]
            bit_output = result.result_program[creg.name][creg_index]
        assign(
            state_int,
            state_int + (1 << c) * Cast.to_int(bit_output),
        )
    return state_int


class QMEnvironment(ContextAwareQuantumEnvironment):

    def __init__(
        self,
        training_config: QEnvConfig,
        circuit_context: Optional[QuantumCircuit] = None,
        job: Optional[RunningQmJob] = None,
    ):
        ParameterPool.reset()
        super().__init__(training_config, circuit_context)
        if not isinstance(self.config.backend_config, QMConfig) or not isinstance(
            self.backend, QMBackend
        ):
            raise ValueError(
                "The backend should be a QMBackend object and the config should be a QMConfig object"
            )

        if not self.config.reward_method in ["state", "channel", "cafe"]:
            raise ValueError(
                "The reward method should be one of the following: state, channel, cafe"
            )
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
            input_type=self.input_type,
            direction=Direction.INCOMING,
        )

        self.real_time_circuit = self.config.reward.get_real_time_circuit(
            self.circuits,
            self.get_target(),
            self.config,
            skip_transpilation=True,
        )
        self.input_state_vars = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
            filter_function=lambda x: "input" in x.name,
            name="input_state_vars",
        )
        self.observable_vars: Optional[ParameterTable] = (
            ParameterTable.from_qiskit(
                self.real_time_circuit,
                input_type=self.input_type,
                filter_function=lambda x: "observable" in x.name,
                name="observable_vars",
            )
            if self.config.dfe
            else None
        )

        self.n_reps_var: Optional[QuaParameter] = (
            QuaParameter(
                "n_reps",
                self.n_reps,
                input_type=self.input_type,
                direction=Direction.OUTGOING,
            )
            if self.real_time_circuit.get_var("n_reps", None) is not None
            else None
        )

        self.real_time_circuit_parameters = ParameterTable.from_qiskit(
            self.real_time_circuit,
            input_type=self.input_type,
            filter_function=lambda x: isinstance(x, Parameter),
            name="real_time_circuit_parameters",
        )
        self.circuit_choice_var: Optional[QuaParameter] = (
            QuaParameter(
                "circuit_choice",
                0,
                input_type=self.input_type,
                direction=Direction.OUTGOING,
            )
            if self.real_time_circuit.get_var("circuit_choice", None) is not None
            else None
        )

        self.pauli_shots = QuaParameter(
            "pauli_shots",
            self.n_shots,
            input_type=self.input_type if self.config.dfe else None,
            direction=Direction.OUTGOING if self.config.dfe else None,
        )

        self.max_input_state = QuaParameter(
            "max_input_state",
            0,
            input_type=self.input_type,
            direction=Direction.OUTGOING,
        )
        self.max_observables = (
            QuaParameter(
                "max_observables",
                0,
                input_type=self.input_type,
                direction=Direction.OUTGOING,
            )
            if self.config.dfe
            else None
        )

        if self.input_type == InputType.DGX:
            ParameterPool.patch_opnic_wrapper(self.qm_backend_config.opnic_dev_path)

        self._qm_job: Optional[RunningQmJob] = job
        self._qm: Optional[QuantumMachine] = None
        self.backend.update_compiler_from_target(self.input_type)
        if hasattr(self.real_time_circuit, "calibrations") and self.real_time_circuit.calibrations:
            self.backend.update_calibrations(qc=self.real_time_circuit, input_type=self.input_type)

    def step(self, action):
        """
        Perform the action on the quantum environment
        """
        self._step_tracker += 1
        dim = 2**self.n_qubits
        if self._qm_job is None:
            raise RuntimeError(
                "The QUA program has not been started yet. Call start_program() first."
            )

        verbosity = self.qm_backend_config.verbosity

        push_args = {
            "job": self.qm_job,
            "verbosity": verbosity,
        }
        mean_val = self.mean_action.tolist()
        std_val = self.std_action.tolist()

        additional_input = (
            self.config.execution_config.dfe_precision if self.config.dfe else self.baseline_circuit
        )
        reward_data: RewardDataType = self.config.reward.get_reward_data(
            self.circuit,
            np.zeros((1, self.n_actions)),
            self.target,
            self.config,
            additional_input,
        )

        # Push policy parameters to trigger real-time action sampling
        self.policy.push_to_opx({"mu": mean_val, "sigma": std_val}, **push_args)
        print("Just pushed policy parameters to OPX:", mean_val, std_val)
        if self.circuit_choice_var is not None:
            self.circuit_choice_var.push_to_opx(self.trunc_index, **push_args)

        if self.n_reps_var is not None:
            self.n_reps_var.push_to_opx(self.n_reps, **push_args)

        # Push the reward data to the OPX
        input_state_indices = reward_data.input_indices
        max_input_state = len(input_state_indices)
        max_observables = (
            1 if not self.config.dfe else sum([len(obs) for obs in reward_data.observables_indices])
        )

        self.max_input_state.push_to_opx(max_input_state, **push_args)
        reward = np.zeros(shape=(self.batch_size,))
        collected_counts = []
        for i, input_state in enumerate(input_state_indices):
            input_state_dict = {
                input_var: input_state_val
                for input_var, input_state_val in zip(self.input_state_vars.parameters, input_state)
            }
            self.input_state_vars.push_to_opx(input_state_dict, **push_args)
            if self.config.dfe:
                observables = reward_data.observables_indices[i]
                num_obs = len(observables)
                self.max_observables.push_to_opx(num_obs, **push_args)
                self.pauli_shots.push_to_opx(reward_data.shots[i], **push_args)
                for j, observable in enumerate(observables):
                    observable_dict = {
                        observable_var: observable_val
                        for observable_var, observable_val in zip(
                            self.observable_vars.parameters, observable
                        )
                    }
                    self.observable_vars.push_to_opx(observable_dict, **push_args)

        collected_counts = self.reward.fetch_from_opx(
            **push_args,
            fetching_index=self.step_tracker - 1,
            fetching_size=max_input_state * max_observables,
        )

        # The counts are a flattened list of counts from the initial shape (max_input_state, max_observables, batch_size)
        # Reshape the counts to the original shape
        if self.config.dfe:
            # Initialiser la structure de données pour stocker les résultats restructurés
            reshaped_counts = [[] for _ in range(self.batch_size)]  # Une liste par batch

            # Index de position dans collected_counts
            current_idx = 0

            # Parcourir d'abord par input_state puis par observable selon la structure OPX
            for input_idx in range(max_input_state):
                num_obs_for_input = len(reward_data.observables_indices[input_idx])

                # Pour chaque input_state, initialiser une liste d'observables pour chaque batch
                for batch_idx in range(self.batch_size):
                    if len(reshaped_counts[batch_idx]) <= input_idx:
                        reshaped_counts[batch_idx].append([])

                # Parcourir les observables pour cet état d'entrée
                for obs_idx in range(num_obs_for_input):
                    # Pour chaque observable, extraire les données de chaque batch
                    for batch_idx in range(self.batch_size):
                        # Extraire les données pour cette combinaison (batch, input_state, observable)
                        obs_counts = np.array(collected_counts[current_idx : current_idx + dim])
                        current_idx += dim

                        # Ajouter à la structure organisée
                        reshaped_counts[batch_idx][input_idx].append(obs_counts)

            # Vérifier que toutes les données ont été utilisées
            if current_idx != len(collected_counts):
                raise ValueError(
                    f"Toutes les données n'ont pas été traitées. Traité: {current_idx}, Total: {len(collected_counts)}"
                )

            counts = reshaped_counts
        else:
            # Traitement pour le cas non-DFE (reste inchangé)
            shape = (max_input_state, self.batch_size, dim)
            transpose_shape = (1, 0, 2)
            counts = np.transpose(np.array(collected_counts).reshape(shape), transpose_shape)

        # Convert the counts to a dictionary of counts

        for batch_idx in range(self.batch_size):
            if self.config.dfe:
                exp_value = reward_data.id_coeff
                for i_idx in range(max_input_state):
                    for o_idx in range(max_observables):
                        counts_dict = {
                            binary(i, self.n_qubits): counts[batch_idx][i_idx][o_idx][i]
                            for i in range(dim)
                        }
                        obs = reward_data[i_idx].hamiltonian.group_commuting(True)[o_idx]
                        diag_obs = SparsePauliOp("I" * self.n_qubits, 0.0)
                        for obs_, coeff in zip((obs.paulis, obs.coeffs)):
                            diag_obs_label = ""
                            for char in obs_.to_label():
                                diag_obs_label += char if char == "I" else "Z"
                            diag_obs += SparsePauliOp(diag_obs_label, coeff)
                        bit_array = BitArray.from_counts(counts_dict, num_bits=self.n_qubits)
                        exp_value += bit_array.expectation_values(diag_obs)
                exp_value /= reward_data.pauli_sampling
                reward[batch_idx] = exp_value
            else:
                survival_probability = np.zeros(shape=(max_input_state,))
                for i_idx in range(max_input_state):
                    counts_dict = {
                        binary(i, self.n_qubits): counts[batch_idx][i_idx][i] for i in range(dim)
                    }
                    bit_array = BitArray.from_counts(counts_dict, num_bits=self.n_qubits)
                    survival_probability[i_idx] = (
                        bit_array.get_int_counts().get(0, 0) / self.n_shots
                    )
                reward[batch_idx] = np.mean(survival_probability)

        if np.mean(reward) > self._max_return:
            self._max_return = np.mean(reward)
            self._optimal_actions[self.trunc_index] = self.mean_action

        reward = np.clip(reward, 0.0, 1.0)
        self.reward_history.append(reward)
        reward = -np.log10(1.0 - reward)  # Convert to negative log10 scale

        return self._get_obs(), reward, True, False, self._get_info()

    # def step(self, action):
    #     """
    #     Perform the action on the quantum environment
    #     """
    #     self._step_tracker += 1
    #     dim = 2**self.n_qubits
    #     if self._qm_job is None:
    #         raise RuntimeError(
    #             "The QUA program has not been started yet. Call start_program() first."
    #         )
    #
    #     verbosity = self.qm_backend_config.verbosity
    #     push_args = {"job": self.qm_job,
    #                  "verbosity": verbosity}
    #
    #     mean_val = self.mean_action.tolist()
    #     std_val = self.std_action.tolist() # Assuming std_action is managed elsewhere or fixed
    #
    #     additional_input = (
    #         self.config.execution_config.dfe_precision if self.config.dfe else self.baseline_circuit
    #     )
    #     reward_data: RewardDataType = self.config.reward.get_reward_data(
    #         self.circuit,
    #         self.mean_action.reshape(1, -1), # Pass current mean_action
    #         self.target,
    #         self.config,
    #         additional_input,
    #     )
    #
    #     # Push policy parameters to trigger real-time action sampling
    #     self.policy.push_to_opx({"mu": mean_val, "sigma": std_val}, **push_args)
    #     if verbosity > 0:
    #         print(f"Step {self._step_tracker}: Pushed policy mu={mean_val}, sigma={std_val}")
    #
    #     if self.circuit_choice_var is not None:
    #         self.circuit_choice_var.push_to_opx(self.trunc_index, **push_args)
    #     if self.n_reps_var is not None:
    #         self.n_reps_var.push_to_opx(self.n_reps, **push_args)
    #
    #     input_state_indices = reward_data.input_indices
    #     max_input_state = len(input_state_indices)
    #     self.max_input_state.push_to_opx(max_input_state, **push_args)
    #
    #     # Calculate total number of (input_state, observable) measurement settings per batch item
    #     # This is used to determine the total number of dim-sized count arrays to fetch
    #     total_measurement_settings_per_batch_item = 0
    #     if self.config.dfe:
    #         total_measurement_settings_per_batch_item = sum(
    #             len(obs_list) for obs_list in reward_data.observables_indices
    #         )
    #     else:
    #         total_measurement_settings_per_batch_item = max_input_state
    #
    #     # --- Data Fetching Logic ---
    #     # collected_counts_raw will be a list of dim-sized arrays/lists
    #     collected_counts_raw: list[Union[np.ndarray, list[Union[int, float]]]] = []
    #
    #     if self.input_type == InputType.IO1 or self.input_type == InputType.IO2:
    #         # For IO types, fetch iteratively based on QUA program's execution order
    #         for i, input_state_val_list in enumerate(input_state_indices):
    #             input_state_dict = {
    #                 param.name: val
    #                 for param, val in zip(self.input_state_vars.parameters, input_state_val_list)
    #             }
    #             self.input_state_vars.push_to_opx(input_state_dict, **push_args)
    #             self.pauli_shots.push_to_opx(reward_data.shots[i], **push_args)
    #
    #             num_obs_for_this_input = 1
    #             if self.config.dfe:
    #                 observables_for_input = reward_data.observables_indices[i]
    #                 num_obs_for_this_input = len(observables_for_input)
    #                 self.max_observables.push_to_opx(num_obs_for_this_input, **push_args)
    #                 for observable_val_list in observables_for_input:
    #                     observable_dict = {
    #                         param.name: val
    #                         for param, val in zip(self.observable_vars.parameters, observable_val_list)
    #                     }
    #                     self.observable_vars.push_to_opx(observable_dict, **push_args)
    #                     for _ in range(self.batch_size):
    #                         fetched_data = self.reward.fetch_from_opx(**push_args)
    #                         collected_counts_raw.append(fetched_data)
    #             else: # Not DFE
    #                 for _ in range(self.batch_size):
    #                     fetched_data = self.reward.fetch_from_opx(**push_args)
    #                     collected_counts_raw.append(fetched_data)
    #
    #     elif self.input_type == InputType.INPUT_STREAM or self.input_type == InputType.DGX:
    #         # For stream types, push all control parameters first, then fetch all data
    #         for i, input_state_val_list in enumerate(input_state_indices):
    #             input_state_dict = {
    #                 param.name: val
    #                 for param, val in zip(self.input_state_vars.parameters, input_state_val_list)
    #             }
    #             self.input_state_vars.push_to_opx(input_state_dict, **push_args)
    #             self.pauli_shots.push_to_opx(reward_data.shots[i], **push_args)
    #
    #             if self.config.dfe:
    #                 observables_for_input = reward_data.observables_indices[i]
    #                 num_obs_for_this_input = len(observables_for_input)
    #                 self.max_observables.push_to_opx(num_obs_for_this_input, **push_args)
    #                 for observable_val_list in observables_for_input:
    #                     observable_dict = {
    #                         param.name: val
    #                         for param, val in zip(self.observable_vars.parameters, observable_val_list)
    #                     }
    #                     self.observable_vars.push_to_opx(observable_dict, **push_args)
    #         # Now fetch all data at once
    #         num_total_fetches = self.batch_size * total_measurement_settings_per_batch_item
    #         if num_total_fetches > 0:
    #             collected_counts_raw = self.reward.fetch_from_opx(
    #                 **push_args,
    #                 fetching_index=self.step_tracker,
    #                 fetching_size=num_total_fetches,
    #             )
    #             if not isinstance(collected_counts_raw, list) or len(collected_counts_raw) != num_total_fetches:
    #                 raise TypeError(f"Stream fetch expected a list of {num_total_fetches} arrays, "
    #                                 f"got {type(collected_counts_raw)} with len {len(collected_counts_raw)}")
    #         else:
    #             collected_counts_raw = []
    #     else:
    #         raise NotImplementedError(f"Fetching logic not implemented for input type {self.input_type}")
    #
    #     # --- Process Counts ---
    #     batch_rewards_calculated = np.zeros(shape=(self.batch_size,))
    #
    #     if not collected_counts_raw and total_measurement_settings_per_batch_item > 0 : # Should only happen if batch_size is 0
    #         if self.batch_size > 0:
    #             print("Warning: No counts collected, but expected data. Rewards will be zero.")
    #         # batch_rewards_calculated is already zeros
    #
    #     elif total_measurement_settings_per_batch_item == 0: # No actual measurements to process
    #         if self.config.dfe: # If DFE, reward might start from id_coeff
    #             batch_rewards_calculated.fill(reward_data.id_coeff)
    #             if reward_data.pauli_sampling > 0:
    #                 batch_rewards_calculated /= reward_data.pauli_sampling
    #         # else, it remains zeros, which is fine for no measurements.
    #
    #     elif self.config.dfe:
    #         reorganized_counts_dfe = self._reshape_and_reorganize_counts_dfe(
    #             collected_counts_raw, reward_data
    #         )
    #         for batch_idx in range(self.batch_size):
    #             current_batch_total_exp_value = reward_data.id_coeff
    #             for i_idx in range(max_input_state):
    #                 observables_for_input_state = reorganized_counts_dfe[batch_idx][i_idx]
    #                 expected_num_obs = len(reward_data.observables_indices[i_idx])
    #                 if len(observables_for_input_state) != expected_num_obs:
    #                     raise ValueError(f"DFE Count Mismatch for batch {batch_idx}, input_state {i_idx}: "
    #                                      f"Got {len(observables_for_input_state)} obs counts, expected {expected_num_obs}")
    #
    #                 for o_list_idx, counts_for_this_observable in enumerate(observables_for_input_state):
    #                     counts_dict = {
    #                         binary(k, self.n_qubits): int(count_val)
    #                         for k, count_val in enumerate(counts_for_this_observable)
    #                     }
    #                     master_obs_idx = reward_data.observables_indices[i_idx][o_list_idx]
    #                     hamiltonian_op: SparsePauliOp = reward_data.observables[master_obs_idx]
    #                     diag_obs = SparsePauliOp("I" * self.n_qubits, 0.0)
    #                     for pauli_term, coeff_val in zip(hamiltonian_op.paulis, hamiltonian_op.coeffs):
    #                         diag_obs_label = "".join(
    #                             ["Z" if p_char != "I" else "I" for p_char in pauli_term.to_label()]
    #                         )
    #                         diag_obs += SparsePauliOp(diag_obs_label, coeff_val)
    #                     try:
    #                         bit_array = BitArray.from_counts(counts_dict, num_bits=self.n_qubits)
    #                         exp_val_contrib = bit_array.expectation_values(diag_obs)
    #                         current_batch_total_exp_value += exp_val_contrib
    #                     except Exception as e:
    #                         print(f"DFE ExpVal Error: batch {batch_idx}, input {i_idx}, obs_list_idx {o_list_idx}: {e}")
    #             if reward_data.pauli_sampling > 0:
    #                 current_batch_total_exp_value /= reward_data.pauli_sampling
    #             batch_rewards_calculated[batch_idx] = current_batch_total_exp_value
    #     else:  # Not DFE - data is homogeneous
    #         # Original QUA order: (max_input_state -> batch_size) for the collected_counts_raw list
    #         # Each element of collected_counts_raw is a dim-sized array.
    #         # We want counts[batch_idx, i_idx, k_dim_component]
    #         shape_non_dfe = (max_input_state, self.batch_size, dim)
    #         try:
    #             counts_non_dfe_temp = np.array(collected_counts_raw, dtype=int).reshape(shape_non_dfe)
    #         except ValueError as e:
    #             raise ValueError(f"Reshape failed for non-DFE counts. Expected {np.prod(shape_non_dfe)} elements, "
    #                              f"got {len(collected_counts_raw)} arrays of varying/incorrect lengths or {len(collected_counts_raw)*dim} total elements if flattened. "
    #                              f"Collected_counts_raw length: {len(collected_counts_raw)}. Error: {e}")
    #
    #         # Transpose to put batch_size dimension first: (batch_size, max_input_state, dim)
    #         counts_non_dfe = np.transpose(counts_non_dfe_temp, (1, 0, 2))
    #
    #         for batch_idx in range(self.batch_size):
    #             survival_probability_for_batch_item = np.zeros(shape=(max_input_state,))
    #             for i_idx in range(max_input_state):
    #                 counts_for_this_input_state = counts_non_dfe[batch_idx, i_idx, :]
    #                 counts_dict = {
    #                     binary(k, self.n_qubits): int(count_val)
    #                     for k, count_val in enumerate(counts_for_this_input_state)
    #                 }
    #                 total_shots_for_instance = np.sum(counts_for_this_input_state) # Should be == reward_data.shots[i_idx]
    #                 prob_0_state = 0.0
    #                 if total_shots_for_instance > 0:
    #                     prob_0_state = (
    #                             counts_dict.get(binary(0, self.n_qubits), 0)
    #                             / total_shots_for_instance
    #                     )
    #                 survival_probability_for_batch_item[i_idx] = prob_0_state
    #             batch_rewards_calculated[batch_idx] = np.mean(survival_probability_for_batch_item)
    #
    #     # --- Post-process rewards ---
    #     if np.mean(batch_rewards_calculated) > self._max_return:
    #         self._max_return = np.mean(batch_rewards_calculated)
    #         self._optimal_actions[self.trunc_index] = self.mean_action.copy() # Store a copy
    #
    #     clipped_batch_rewards = np.clip(batch_rewards_calculated, 0.0, 1.0)
    #     self.reward_history.append(clipped_batch_rewards.copy())
    #     final_rewards_for_agent = -np.log10(1.0 - clipped_batch_rewards + 1e-9)
    #
    #     return self._get_obs(), final_rewards_for_agent, True, False, self._get_info()

    def rl_qoc_training_qua_prog(self, num_updates: int = 1000) -> Program:
        """
        Generate a QUA program tailor-made for the RL-based calibration project
        """
        self.reset_parameters()

        qc = self.real_time_circuit
        dim = int(2**self.n_qubits)
        for clbit in qc.clbits:
            if len(qc.find_bit(clbit).registers) >= 2:
                raise ValueError("Overlapping classical registers are not supported")

        with program() as rl_qoc_training_prog:
            # Declare the necessary variables (all are counters variables to loop over the corresponding hyperparameters)

            self.real_time_circuit_parameters.declare_variables(declare_streams=False)
            self.max_input_state.declare_variable()
            self.input_state_vars.declare_variables()
            self.policy.declare_variables()
            mu = self.policy.get_variable("mu")
            sigma = self.policy.get_variable("sigma")
            self.reward.declare_variable()
            for var in [self.circuit_choice_var, self.n_reps_var, self.max_observables]:
                if var is not None:
                    var.declare_variable()
            if self.observable_vars is not None:
                self.observable_vars.declare_variables()
                obs_idx = declare(int)

            # Number of shots for each observable (or total number of shots if no observables)
            self.pauli_shots.declare_variable()
            n_u = declare(int)
            input_state_count = declare(int)
            b = declare(int)
            state_int = declare(int, value=0)
            counts = declare(int, value=[0 for _ in range(2**self.n_qubits)])

            # Declare variables for efficient Gaussian random sampling
            j = declare(int)

            batch_r = Random(self.seed)

            if self.backend.init_macro is not None:
                self.backend.init_macro()
            # Infinite loop to run the training
            with for_(n_u, 0, n_u < num_updates, n_u + 1):
                self.policy.load_input_values()  # Load µ and σ
                batch_r.set_seed(self.seed + n_u)
                for var in [self.circuit_choice_var, self.n_reps_var, self.max_input_state]:
                    if var is not None:
                        var.load_input_value()

                with for_(
                    input_state_count,
                    0,
                    input_state_count < self.max_input_state.var,
                    input_state_count + 1,
                ):
                    # Load info about input states to prepare (single qubit indices)
                    self.input_state_vars.load_input_values()

                    if self.config.dfe:
                        self.max_observables.load_input_value()
                        self.pauli_shots.load_input_value()  # TODO: RELAX THIS TO PUT IT PER OBSERVABLE WITHIN LOOP BELOW)
                        with for_(obs_idx, 0, obs_idx < self.max_observables.var, obs_idx + 1):
                            # Load info about observable to measure (single qubit indices)
                            self.observable_vars.load_input_values()

                            self._rl_macro(
                                batch_r,
                                mu,
                                sigma,
                                state_int,
                                counts,
                            )

                    else:
                        self._rl_macro(
                            batch_r,
                            mu,
                            sigma,
                            state_int,
                            counts,
                        )

            with stream_processing():
                self.reward.stream_processing(buffer=(self.batch_size, dim))

        self.reset_parameters()
        return rl_qoc_training_prog

    def _rl_macro(
        self,
        batch_r,
        mu,
        sigma,
        state_int,
        counts,
    ):
        b = declare(int)
        j = declare(int)
        n_lookup, cos_array, ln_array = get_gaussian_sampling_input()
        uniform_r = declare(fixed)
        u1, u2 = declare(int), declare(int)
        tmp1 = declare(fixed, size=self.n_actions)
        tmp2 = declare(fixed, size=self.n_actions)
        lower_bound = declare(fixed, value=self.action_space.low)
        upper_bound = declare(fixed, value=self.action_space.high)

        with for_(b, 0, b < self.batch_size, b + 2):
            # Sample from a multivariate Gaussian distribution (Muller-Box method)
            with for_(j, 0, j < self.n_actions, j + 1):
                assign(uniform_r, batch_r.rand_fixed())
                assign(u1, Cast.unsafe_cast_int(uniform_r >> 19))
                assign(u2, Cast.unsafe_cast_int(uniform_r) & ((1 << 19) - 1))
                assign(
                    tmp1[j],
                    mu[j] + sigma[j] * ln_array[u1] * cos_array[u2 & (n_lookup - 1)],
                )
                assign(
                    tmp2[j],
                    mu[j]
                    + sigma[j] * ln_array[u1] * cos_array[(u2 + n_lookup // 4) & (n_lookup - 1)],
                )
                clip_qua(tmp1[j], lower_bound[j], upper_bound[j])
                clip_qua(tmp2[j], lower_bound[j], upper_bound[j])

            with for_(j, 0, j < 2, j + 1):
                # Assign the sampled actions to the action batch
                for i, parameter in enumerate(self.real_time_circuit_parameters.parameters):
                    parameter.assign(tmp1[i], condition=(j == 0), value_cond=tmp2[i])

                counts, state_int = self._run_circuit(state_int, counts)
                # qua_print("counts", counts, "state_int", state_int)
                self.reward.stream_back(counts)

    def _run_circuit(self, state_int: QuaVariableInt, counts: QuaArrayVariable):
        """
        Run the circuit and get the counts
        """
        qc = self.real_time_transpiled_circuit
        n_shots = declare(int)

        with for_(state_int, 0, state_int < 2**self.n_qubits, state_int + 1):
            assign(counts[state_int], 0)
        assign(state_int, 0)  # Reset state_int for the next shot
        with for_(n_shots, 0, n_shots < self.pauli_shots.var, n_shots + 1):
            param_inputs = [
                self.input_state_vars,
                self.real_time_circuit_parameters,
                self.circuit_choice_var,
                self.n_reps_var,
                self.observable_vars if self.config.dfe else None,
            ]
            param_inputs = [param for param in param_inputs if param is not None]

            result = self.backend.quantum_circuit_to_qua(qc, param_inputs)
            state_int = _get_state_int(qc, result, state_int)

            assign(counts[state_int], counts[state_int] + 1)
            assign(state_int, 0)  # Reset state_int for the next shot

        return counts, state_int

    def reset_parameters(self):
        self.policy.reset()
        self.reward.reset()
        self.real_time_circuit_parameters.reset()
        self.input_state_vars.reset()
        self.max_input_state.reset()
        if self.observable_vars is not None:
            self.observable_vars.reset()
            self.pauli_shots.reset()
            self.max_observables.reset()
        if self.n_reps_var is not None:
            self.n_reps_var.reset()
        if self.circuit_choice_var is not None:
            self.circuit_choice_var.reset()

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
        num_updates: int = 1000,
        compiler_options: Optional[CompilerOptionArguments] = None,
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
        prog = self.rl_qoc_training_qua_prog(num_updates=num_updates)
        self._qm_job = self.qm.execute(prog, compiler_options=compiler_options)
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
        return self.backend.qm

    # Add this method to your QMEnvironment class
    def _reshape_and_reorganize_counts_dfe(
        self,
        collected_counts_list_of_arrays: list[Union[np.ndarray, list[Union[int, float]]]],
        reward_data: RewardDataType,
    ) -> list[list[list[np.ndarray]]]:
        """
        Reorganizes fetched counts for DFE into a batch-first nested list structure.
        Assumes collected_counts_list_of_arrays is a list where each element
        is a dim-sized array/list of counts, ordered by QUA execution flow
        (input_state -> observable -> batch).

        Output structure:
        reorganized_counts[batch_idx][input_state_idx] = list_of_observable_counts_arrays
        where each counts_array is a 1D NumPy array of size 'dim' (integers).
        """
        dim = 2**self.n_qubits
        max_input_state = len(reward_data.input_indices)
        reorganized_counts: list[list[list[np.ndarray]]] = [
            [[] for _ in range(max_input_state)] for _ in range(self.batch_size)
        ]

        current_item_idx = 0

        for i_idx_qua_loop in range(max_input_state):
            # For DFE, the number of observables can vary per input state
            num_obs_for_this_input_state = len(reward_data.observables_indices[i_idx_qua_loop])

            for _o_idx_qua_loop in range(num_obs_for_this_input_state):
                for batch_idx_qua_loop in range(self.batch_size):
                    if current_item_idx >= len(collected_counts_list_of_arrays):
                        raise ValueError(
                            f"Not enough data in collected_counts_list_of_arrays for DFE. "
                            f"Attempting to access index: {current_item_idx}, "
                            f"Total items available: {len(collected_counts_list_of_arrays)}."
                        )
                    try:
                        counts_array_for_instance = np.array(
                            collected_counts_list_of_arrays[current_item_idx], dtype=int
                        ).reshape(dim)
                    except Exception as e:
                        problematic_item = collected_counts_list_of_arrays[current_item_idx]
                        raise ValueError(
                            f"Error processing DFE counts at index {current_item_idx}. Item: {problematic_item}, Type: {type(problematic_item)}. "
                            f"Expected a {dim}-element list/array of numbers."
                        ) from e

                    current_item_idx += 1
                    reorganized_counts[batch_idx_qua_loop][i_idx_qua_loop].append(
                        counts_array_for_instance
                    )

        if current_item_idx != len(collected_counts_list_of_arrays):
            print(
                f"Warning (DFE): Mismatch in consumed items from collected_counts_list_of_arrays. "
                f"Consumed: {current_item_idx}, Total available: {len(collected_counts_list_of_arrays)}."
            )
        return reorganized_counts
