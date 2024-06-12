import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from typing import Tuple, Union, Optional, Dict, List
import numpy as np
import math
import time
from datetime import datetime
import pickle
import optuna
from base_q_env import BaseQuantumEnvironment
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from helper_functions import (
    load_from_yaml_file,
    create_hpo_agent_config,
    save_to_pickle
)
from gate_level_abstraction.spillover_noise_use_case.noise_utils.utils import get_baseline_fid_from_phi_gamma
from ppoV2 import CustomPPOV2
from functools import partial

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

QUANTUM_ENVIRONMENT = Union[
    BaseQuantumEnvironment,
    QuantumEnvironment,
    ContextAwareQuantumEnvironment,
]

class HyperparameterOptimizer:
    """
    A class designed to optimize the hyperparameters of a Proximal Policy Optimization (PPO)
    agent operating within a quantum environment, leveraging the Optuna framework for
    hyperparameter optimization.

    This optimizer facilitates a systematic exploration of the hyperparameter space to identify
    configurations that minimize a custom cost function. The cost function is designed to
    account not only for the performance of the quantum operation (e.g., infidelity) but also
    for the experimental cost associated with certain hyperparameters, such as the number of
    shots in quantum measurements or batch sizes during training. The class supports saving
    the best found configurations and provides tools for logging the optimization process.

    Attributes:
        q_env (Any valid QuantumEnvironment instance):
            The quantum environment in which the PPO agent operates. This environment
            must comply with either the QuantumEnvironment or
            ContextAwareQuantumEnvironment interface.
        config_paths (Dict[str, str]): Dictionary containing paths to the configuration files
        log_progress (bool): Indicates whether the progress of the hyperparameter optimization
            should be logged. Useful for monitoring the optimization process in real time.
    """

    def __init__(
        self,
        q_env: QUANTUM_ENVIRONMENT,
        config_paths: Dict[str, str],
        saving_mode: str = "all",
        log_results: bool = True,
    ):
        """
        Initializes the HyperparameterOptimizer with configurations for the quantum environment,
        agent, hyperparameter optimization, and other settings.

        Parameters:
        - q_env: QuantumEnvironment or ContextAwareQuantumEnvironment instance.
        - config_paths: Dictionary containing paths to the configuration files for the agent, HPO and where to save results to.
        - experimental_penalty_weights: Optional dictionary specifying penalty weights for experimentally costly hyperparameters.
        - log_results: Flag indicating whether to log the optimization results at the end of the process.
        """
        self.q_env = q_env
        self.path_agent_config = config_paths["agent_config_path"]
        self.hpo_config = load_from_yaml_file(config_paths["hpo_config_path"])
        self.save_results_path = config_paths["save_results_path"]
        self.log_progress = log_results
        self.saving_mode = saving_mode
        if self.saving_mode not in ["all", "best"]:
            raise ValueError(
                f"Saving mode must be 'all' or 'best', but got '{self.saving_mode}'"
            )
        self.all_trials_data = []

    def _get_num_hpo_trials(self, num_hpo_trials: Optional[int] = None):
        if num_hpo_trials is not None:
            assert (
                isinstance(num_hpo_trials, int) and num_hpo_trials > 0
            ), "n_hpo_trials must be an integer greater than 0"
        return num_hpo_trials if num_hpo_trials is not None else self.hpo_config.get("n_trials", 10)
    
    def _set_sub_training_mode(self, training_details: Dict):
        self.total_updates = None
        self.max_hardware_runtime = None
        if 'total_updates' in training_details:
            self.total_updates = training_details["total_updates"]
            constraint = 'total_updates'
        elif 'max_hardware_runtime' in training_details:
            self.max_hardware_runtime = training_details["max_hardware_runtime"]
            constraint = 'max_hardware_runtime'
        else:
            raise ValueError(
                "ERROR: Please provide either 'total_updates' or 'max_hardware_runtime' in the 'training_details' for 'spillover_noise_use_case' training mode."
            )
        if 'total_updates' in training_details and 'max_hardware_runtime' in training_details:
            logging.warning(
                "Both 'total_updates' or 'max_hardware_runtime' was provided in the 'training_details'! Using 'total_updates' as default."
            )
        return constraint
    
    def _get_training_mode(self):
        training_details = self.training_config["training_details"]
        self.training_mode = self.training_config["training_mode"]
        self.phi_gamma_tuple = None
        if self.training_mode == 'spillover_noise_use_case':
            # Retrieve phi and gamma values
            if not 'phi_gamma_tuple' in training_details:
                raise ValueError(
                    "ERROR: Please set 'phi_gamma_tuple' in the 'training_details' for the 'spillover_noise_use_case' training mode."
                )
            self.phi_gamma_tuple = training_details["phi_gamma_tuple"]
            self.training_constraint = self._set_sub_training_mode(training_details)
            logging.warning(
                f"Constraint: {self.training_constraint}; Phi: {training_details['phi_gamma_tuple'][0]}; Gamma: {training_details['phi_gamma_tuple'][1]}"
            )

        elif self.training_config["training_mode"] == 'normal_calibration': 
            self.training_constraint = self._set_sub_training_mode(training_details)
            info_str = 'Total Updates' if self.training_constraint == 'total_updates' else 'Max Hardware Runtime'
            logging.warning(
                f"Constraint: {self.training_constraint}; {info_str}: {training_details[self.training_constraint]}"
            )
        else:
            raise ValueError(
                "ERROR: Training mode not recognized. Please choose between 'normal_calibration' and 'spillover_noise_use_case'."
        )
        logging.warning(
            f"------------------------- STARTING HPO IN MODE: {self.training_config['training_mode']} -------------------------"
        )

    def _get_study_name(self):
        if hasattr(self.q_env.unwrapped.target, "gate"):
            studyname = f'{self.target_operation["target_gate"].name}-calibration'
        elif hasattr(self.q_env.unwrapped.target, "dm"):
            studyname = "state-preparation"
        return studyname

    def _get_n_reps_noisy_use_case(self, phi_gamma_tuple: Tuple[float, float]):
        self.phi, self.gamma = phi_gamma_tuple
        self.baseline_fidelity = get_baseline_fid_from_phi_gamma(
            param_tuple=phi_gamma_tuple
        )
        # To which integer power do I need to raise self.baseline_fidelity to be below the lowest target fidelity?
        self.smallest_N_reps = math.ceil(
            math.log(min(self.target_fidelities), self.baseline_fidelity)
        )
        self.q_env.unwrapped.n_reps = min([self.smallest_N_reps, 150])
        if self.q_env.unwrapped.n_reps == 150:
            logging.warning('WARNING: n_reps set to 150. Consider increasing noise strength.')

    def _log_training_parameters(self):
        param_str = ""
        logging.warning("Parameters:")
        if self.phi_gamma_tuple:
            param_str = "phi: {}; gamma: {}; Baseline Fidelity: {}; ".format(
                self.phi, self.gamma, self.baseline_fidelity
            )
        param_str += "N_Reps: {}; Target Fidelities: {}; Lookback Window: {}".format(
                self.q_env.unwrapped.n_reps,
                self.target_fidelities,
                self.lookback_window,
            )
        
        logging.warning(param_str)

    def _initialize_optimization_process(self, training_config: Dict, num_hpo_trials: int, experimental_penalty_weights: Dict[str, float | int]):
        self.training_config = training_config
        training_details = self.training_config["training_details"]
        self._get_training_mode()
        self.experimental_penalty_weights = experimental_penalty_weights
        self.n_hpo_trials = self._get_num_hpo_trials(num_hpo_trials)     
        self.target_fidelities = training_details.get("target_fidelities", [0.99, 0.999, 0.9999])
        self.lookback_window = training_details.get('lookback_window', 10)


    def optimize_hyperparameters(
        self,
        num_hpo_trials: int,
        training_config: Dict,
        experimental_penalty_weights: Dict[str, float | int],
    ):
        """
        Starts the hyperparameter optimization process for the specified number of trials.

        Parameters:
        - num_hpo_trials: The number of trials to run for the hyperparameter optimization.

        Returns:
        - A dictionary containing the best configuration and its performance metric.
        """
        self._initialize_optimization_process(training_config, num_hpo_trials, experimental_penalty_weights)
        study = optuna.create_study(
            direction="minimize",
            study_name=f'{self._get_study_name()}_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}',
        )

        # Ensure that only the target fidelities that are greater than the baseline fidelity are considered
        if self.phi_gamma_tuple:
            self._get_n_reps_noisy_use_case(self.phi_gamma_tuple)
        
        self._log_training_parameters()
        time.sleep(4)
        # Start the hyperparameter optimization process
        start_time_hpo = time.time()        
        study.optimize(self._objective, n_trials=self.n_hpo_trials)
        self.post_processing(study, start_time_hpo)

        return self.data
    
    def post_processing(self, study: optuna.study.Study, start_time_hpo: float):
        self._catch_all_trials_failed(study)
        self.best_trial = study.best_trial
        self._save_results()
        self._log_results(study, start_time_hpo) if self.log_progress else None


    def _catch_all_trials_failed(self, study: optuna.study.Study):
        if all([trial.value == float("inf") for trial in study.trials]):
            error_str = "ERROR: HPO failed. All trials led to errors in the training process. No trials to save."
            logging.warning(error_str)
            return {
                "result": error_str,
            }

    def _objective(
        self,
        trial: optuna.trial.Trial,
    ):
        """
        Objective function for the hyperparameter optimization process. This function is called
        by Optuna for each trial and is responsible for training the agent with the trial's
        hyperparameters, evaluating its performance, and calculating the custom cost value
        including penalties.

        Parameters:
        - trial: An Optuna trial object containing the hyperparameters for this optimization iteration.

        Returns:
        - The custom cost value for the trial, incorporating the infidelity and penalties for
        experimentally costly hyperparameters.
        """

        # q_env.unwrapped.backend
        self.agent_config, self.hyperparams = create_hpo_agent_config(
            trial, self.hpo_config, self.path_agent_config
        )

        # Include batchsize, n_shots, and sampling_Pauli_space in the hpo scope
        self.q_env.unwrapped.batch_size = self.agent_config["BATCHSIZE"]
        self.q_env.unwrapped.n_shots = self.agent_config["N_SHOTS"]
        self.q_env.unwrapped.sampling_Pauli_space = self.agent_config["SAMPLE_PAULIS"]

        # train_fn = make_train_ppo(self.agent_config, self.q_env, hpo_mode=True)
        ppo_agent = CustomPPOV2(self.agent_config, self.q_env)
        start_time = time.time()
        training_results = ppo_agent.train(self.training_config)

        simulation_training_time = time.time() - start_time
        if training_results["avg_reward"] != -1.0:  # Check if training was successful
            trial.set_user_attr("training_results", training_results)
            trial.set_user_attr("simulation runtime", simulation_training_time)
        else:
            return float("inf")  # Catch errors in the trianing process

        custom_cost_value = self._calculate_custom_cost(
            training_results, self.experimental_penalty_weights
        )

        # Keep traaack of all trials data
        trial_data = {
            "trial_number": trial.number,
            "training_results": training_results,
            "hyper_params": trial.params,
            "custom_cost_value": custom_cost_value,
            "penalty_weights": self.experimental_penalty_weights,
            "simulation runtime": simulation_training_time,
        }
        self.all_trials_data.append(trial_data)

        return custom_cost_value

    def _calculate_custom_cost(
        self, training_results: dict, reward_and_penalty_params: dict
    ) -> float:
        """
        Calculates a custom cost with considerations for:
        - The number of shots used to achieve the highest target fidelity,
        - Rewarding the achievement of target fidelities,
        - Penalizing based on the closeness for unachieved target fidelities.

        Parameters:
        - training_results (dict): Dictionary containing training outcomes.
        - fidelity_reward (dict): Dictionary containing rewards and penalties for fidelities and shots used

        Returns:
        - float: The calculated custom cost.
        """

        total_cost = 0
        highest_fidelity_achieved_info = None
        target_fidelities = list(training_results["fidelity_info"].keys())

        fidelity_reward = reward_and_penalty_params["fidelity_reward"]
        base_shot_penalty = reward_and_penalty_params["penalty_n_shots"]
        penalty_per_missed_fidelity = reward_and_penalty_params[
            "penalty_per_missed_fidelity"
        ]

        # Identify the highest fidelity achieved and its info
        for fidelity in sorted(target_fidelities, reverse=True):
            info = training_results["fidelity_info"][fidelity]
            if info["achieved"]:
                highest_fidelity_achieved_info = info
                break

        if highest_fidelity_achieved_info:
            # Use shots up to the highest fidelity achieved
            shots_used = highest_fidelity_achieved_info["shots_used"]
        else:
            # If no fidelities were achieved, consider all shots used
            shots_used = sum(training_results["total_shots"])

        # Apply base penalty for the shots used
        total_cost += shots_used * base_shot_penalty

        # Calculate reward/penalty for each target fidelity
        for fidelity in target_fidelities:
            info = training_results["fidelity_info"][fidelity]
            if info["achieved"]:
                # Reward for achieving the fidelity, inversely proportional to shots used
                total_cost -= fidelity_reward
            else:
                # Apply penalty based on how close the training came to the fidelity
                highest_fidelity_reached = max(training_results["fidelity_history"])
                closeness = fidelity - highest_fidelity_reached
                total_cost += closeness * penalty_per_missed_fidelity

        return total_cost

    def _generate_filename(self):
        """Generate the file name where the best configuration will be saved."""
        if self.training_mode == 'spillover_noise_use_case' and hasattr(self, "phi") and hasattr(self, "gamma"):
            start_file_name = f"phi-{self.phi/np.pi}pi_gamma-{round(self.gamma, 2)}"
        elif self.training_mode == 'normal_calibration':
            start_file_name = f"gate_calibration"
        else:
            start_file_name = "state_preparation"

        start_file_name += f"_{self.training_constraint}-{self.max_hardware_runtime if self.max_hardware_runtime else self.total_updates}_"

        return (
            start_file_name
            + f"custom-cost-value-{round(self.best_trial.value, 6)}"
            + "_timestamp_"
            + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            + ".pickle.gzip"
        )

    def _save_results(self):
        """
        Saves either the best or all trial configuration(s), including hyperparameters, training results,
        penalty weights, and runtime, to a pickle file.

        This method is called after the completion of all optimization trials.
        """
        if not os.path.exists(self.save_results_path):
            os.makedirs(self.save_results_path)
            logging.warning(f"Folder '{self.save_results_path}' created.")
        if self.best_trial is not None:
            best_config = {
                "training_results": self.best_trial.user_attrs.get(
                    "training_results", {}
                ),
                "runtime": self.best_trial.user_attrs.get("runtime", 0),
                "hyper_params": self.best_trial.params,
                "custom_cost_value": self.best_trial.values[0],
                "penalty_weights": self.experimental_penalty_weights,
            }
            pickle_file_name = os.path.join(
                self.save_results_path, self._generate_filename()
            )
            # Only save if best_config not empty
            if len(best_config) != 0:
                if self.saving_mode == "all":
                    data_to_save = sorted(
                        self.all_trials_data, key=lambda x: x["custom_cost_value"]
                    )
                else:
                    data_to_save = [best_config]
                save_to_pickle(data_to_save, pickle_file_name)
                self.data = data_to_save
                logging.warning(f"{'All trials have' if self.saving_mode == 'all' else 'Best trial has'} been saved to {pickle_file_name}.")
            else:
                logging.warning("WARNING: No best trial data to save.")
        else:
            logging.warning("No best trial data to save.")

    def _log_results(self, study: optuna.study.Study, start_time: float):
        """
        Logs the results of the hyperparameter optimization process, including the best trial's
        custom cost value, hyperparameters, and the best action vector.

        Parameters:
        - study: The Optuna study object containing the results of the optimization.
        - start_time: The start time of the optimization process, used to calculate the total duration.
        """
        logging.warning("---------------- FINISHED HPO ----------------")
        logging.warning(
            "HPO completed in {} minutes.".format(round((time.time() - start_time) / 60, 2))
        )

        logging.warning("Best trial:")
        logging.warning("-------------------------")
        logging.warning("  Custom Cost Value: {}".format(study.best_trial.value))
        logging.warning("  Hyperparameters: ")
        for key, value in study.best_trial.params.items():
            logging.warning("    {}: {}".format(key, value))

        logging.warning(
            "The best action vector: {}".format(
                self.data[0]['training_results']['best_action_vector']
            )
        )

    @property
    def target_operation(self):
        """
        Returns information about the target gate and register from the quantum environment.

        Returns:
        - A dictionary containing 'target_gate' and 'target_register' keys with their corresponding values.
        """
        if hasattr(self.q_env.unwrapped.target, "gate"):
            return {
                "target_gate": self.q_env.unwrapped.target.gate,
                "target_register": self.q_env.unwrapped.circ_tgt_register,
            }
        elif hasattr(self.q_env.unwrapped.target, "dm"):
            return {
                "target_state": self.q_env.unwrapped.target.dm,
                "target_register": self.q_env.unwrapped.target.physical_qubits,
            }
