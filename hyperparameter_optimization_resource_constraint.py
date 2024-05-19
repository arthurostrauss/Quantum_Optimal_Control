import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from typing import Union, Optional, Dict, List
import numpy as np
import math
import time
from datetime import datetime
import pickle
import optuna
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from helper_functions import (
    load_from_yaml_file,
    create_hpo_agent_config,
    get_baseline_fid_from_phi_gamma,
)
from ppo_while_not_met_baseline import make_train_ppo

from functools import partial

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


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
        q_env (Union[QuantumEnvironment, ContextAwareQuantumEnvironment]):
            The quantum environment in which the PPO agent operates. This environment
            must comply with either the QuantumEnvironment or
            ContextAwareQuantumEnvironment interface.
        path_agent_config (str): Path to the YAML configuration file for initializing
            the PPO agent with default parameters.
        path_hpo_config (str): Path to the YAML configuration file for setting up the
            hyperparameter optimization process, defining the hyperparameter search space.
        save_results_path (str): The directory path where the results and configurations
            of the optimization process will be saved.
        experimental_penalty_weights (Optional[Dict[str, float]]): A dictionary mapping
            hyperparameter names to penalty weights, used to add a penalty to the cost
            function for using experimentally costly hyperparameters.
        log_progress (bool): Indicates whether the progress of the hyperparameter optimization
            should be logged. Useful for monitoring the optimization process in real time.
    """

    def __init__(
        self,
        q_env: Union[QuantumEnvironment, ContextAwareQuantumEnvironment],
        path_agent_config: str,
        path_hpo_config: str,
        save_results_path: str,
        saving_mode: str = "all",
        experimental_penalty_weights: Optional[Dict[str, float]] = None,
        log_progress: bool = True,
    ):
        """
        Initializes the HyperparameterOptimizer with configurations for the quantum environment,
        agent, hyperparameter optimization, and other settings.

        Parameters:
        - q_env: QuantumEnvironment or ContextAwareQuantumEnvironment instance.
        - path_agent_config: Path to the YAML file containing the agent configuration.
        - path_hpo_config: Path to the YAML file containing the hyperparameter optimization configuration.
        - save_results_path: Directory path where the results and configurations will be saved.
        - experimental_penalty_weights: Optional dictionary specifying penalty weights for experimentally costly hyperparameters.
        - log_progress: Flag indicating whether to log the optimization progress.
        """
        self.q_env = q_env
        self.path_agent_config = path_agent_config
        self.hpo_config = load_from_yaml_file(path_hpo_config)
        self.save_results_path = save_results_path
        self.experimental_penalty_weights = experimental_penalty_weights
        self.log_progress = log_progress
        self.saving_mode = saving_mode if not saving_mode else "all"
        if self.saving_mode not in ["all", "best"]:
            raise ValueError(
                f"Saving mode must be 'all' or 'best', but got '{self.saving_mode}'"
            )
        self.all_trials_data = []

    def optimize_hyperparameters(
        self,
        num_hpo_trials: int,
        phi_gamma_tuple: Optional[tuple],
        target_fidelities: List[float],
        lookback_window: int,
        max_runtime: int | float,
    ):
        """
        Starts the hyperparameter optimization process for the specified number of trials.

        Parameters:
        - num_hpo_trials: The number of trials to run for the hyperparameter optimization.

        Returns:
        - A dictionary containing the best configuration and its performance metric.
        """
        self.n_hpo_trials = (
            num_hpo_trials
            if num_hpo_trials is not None
            else self.hpo_config.get("n_trials", 1)
        )
        # Assert that n_hpo_trials is an integer and greater than 0
        assert (
            isinstance(self.n_hpo_trials, int) and self.n_hpo_trials > 0
        ), "n_hpo_trials must be an integer greater than 0"

        start_time_hpo = time.time()
        self.max_hardware_runtime = max_runtime
        logging.warning(
            "------------------------- STARTING HPO -------------------------"
        )
        logging.warning(
            "Max Hardware Runtime: {} mins".format(self.max_hardware_runtime / 60)
        )

        if hasattr(self.q_env.target, "gate"):
            studyname = f'{self.target_operation["target_gate"].name}-calibration'
        elif hasattr(self.q_env.target, "dm"):
            studyname = "state-preparation"

        study = optuna.create_study(
            direction="minimize",
            study_name=f'{studyname}_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}',
        )

        self.target_fidelities = target_fidelities

        # Ensure that only the target fidelities that are greater than the baseline fidelity are considered
        if phi_gamma_tuple:
            self.phi, self.gamma = phi_gamma_tuple
            self.baseline_fidelity = get_baseline_fid_from_phi_gamma(
                param_tuple=phi_gamma_tuple
            )
            # To which integer power do I need to raise self.baseline_fidelity to be below the lowest target fidelity?
            self.smallest_N_reps = math.ceil(
                math.log(min(target_fidelities), self.baseline_fidelity)
            )
            self.q_env.unwrapped.n_reps = min([self.smallest_N_reps, 150])

            logging.warning("Parameters:")
            logging.warning(
                "    phi: {} pi; gamma: {}; Baseline Fidelity: {}; N_Reps: {}; Target Fidelities: {}".format(
                    self.phi / np.pi,
                    self.gamma,
                    self.baseline_fidelity,
                    self.smallest_N_reps,
                    self.target_fidelities,
                )
            )

        logging.warning(f"N reps: {self.q_env.unwrapped.n_reps}")

        self.lookback_window = lookback_window if lookback_window is not None else 20
        logging.warning(f"Lookback window: {self.lookback_window}")

        if len(self.target_fidelities) == 0:
            logging.warning(
                "ERROR: No target fidelities greater than the baseline fidelity for the parameter set: phi = {}pi, gamma = {}".format(
                    phi_gamma_tuple[0] / np.pi, phi_gamma_tuple[1]
                )
            )
            return {
                "result": "ERROR: No target fidelities greater than the baseline fidelity for the parameter set: phi = {}pi, gamma = {}".format(
                    phi_gamma_tuple[0] / np.pi, phi_gamma_tuple[1]
                )
            }

        time.sleep(4)
        # Assuming target_fidelity and max_runtime are defined
        objective_with_params = partial(
            self._objective,
            target_fidelities=self.target_fidelities,
            lookback_window=self.lookback_window,
            max_runtime=max_runtime,
        )
        study.optimize(objective_with_params, n_trials=self.n_hpo_trials)

        # If all trials led to errors in the training process, fidelities for all trials are 0.0
        # If this is the case, then the HPO failed
        # Return a warning and return a dictionary with the result
        if study.best_trial.value == 0.0:
            logging.warning("ERROR: HPO failed. No trials to save.")
            return {
                "result": "ERROR: HPO failed. All trials led to errors in the training process.",
            }
        self.best_trial = study.best_trial
        self._save_results()
        if self.log_progress:
            self._logging_result(study, start_time_hpo)

        return self.data

    def _objective(
        self,
        trial: optuna.trial.Trial,
        target_fidelities: List[float] = [0.99, 0.999, 0.9999],
        lookback_window: int = 30,
        max_runtime: int | float = 600,
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

        train_fn = make_train_ppo(self.agent_config, self.q_env, hpo_mode=True)
        start_time = time.time()
        training_results = train_fn(
            target_fidelities=target_fidelities,
            lookback_window=lookback_window,
            max_hardware_runtime=max_runtime,
            print_debug=False,
            num_prints=50,
        )

        runtime = time.time() - start_time
        if training_results["avg_reward"] != -1.0:  # Check if training was successful
            trial.set_user_attr("training_results", training_results)
            trial.set_user_attr("simulation runtime", runtime)
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
            "simulation runtime": runtime,
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
        if hasattr(self, "phi") and hasattr(self, "gamma"):
            start_file_name = f"phi-{self.phi/np.pi}pi_gamma-{round(self.gamma, 2)}_max-hardware-runtime-{self.max_hardware_runtime}_"
        else:
            start_file_name = "state_preparation_"

        return (
            start_file_name
            + f"custom-cost-value-{round(self.best_trial.value, 6)}"
            + "_timestamp_"
            + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            + ".pickle"
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
                    # Only save the best configuration
                    all_trials_data = sorted(
                        self.all_trials_data, key=lambda x: x["custom_cost_value"]
                    )
                    self.data = all_trials_data
                    self.save_to_pickle(all_trials_data, pickle_file_name)
                    logging.warning(
                        f"All trials have been saved to {pickle_file_name}."
                    )
                else:
                    self.save_to_pickle(best_config, pickle_file_name)
                    self.data = best_config
                    logging.warning(f"Best trial has been saved to {pickle_file_name}.")
            else:
                logging.warning("WARNING: No best trial data to save.")
        else:
            logging.warning("No best trial data to save.")

    def _logging_result(self, study: optuna.study.Study, start_time: float):
        """
        Logs the results of the hyperparameter optimization process, including the best trial's
        custom cost value, hyperparameters, and the best action vector.

        Parameters:
        - study: The Optuna study object containing the results of the optimization.
        - start_time: The start time of the optimization process, used to calculate the total duration.
        """
        logging.warning("---------------- FINISHED HPO ----------------")
        logging.warning(
            "HPO completed in {} seconds.".format(round(time.time() - start_time, 2))
        )

        logging.warning("Best trial:")
        logging.warning("-------------------------")
        logging.warning("  Custom Cost Value: {}".format(study.best_trial.value))
        logging.warning("  Hyperparameters: ")
        for key, value in study.best_trial.params.items():
            logging.warning("    {}: {}".format(key, value))

        logging.warning(
            "The best action vector: {}".format(
                self.data["training_results"]["best_action_vector"]
            )
        )

    @staticmethod
    def save_to_pickle(data, file_path: str):
        """
        Saves the HyperparameterOptimizer instance to a pickle file.

        Parameters:
        - file_path: The path to the pickle file where the instance will be saved.
        """
        with open(file_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def target_operation(self):
        """
        Returns information about the target gate and register from the quantum environment.

        Returns:
        - A dictionary containing 'target_gate' and 'target_register' keys with their corresponding values.
        """
        if hasattr(self.q_env.target, "gate"):
            return {
                "target_gate": self.q_env.unwrapped.target.gate,
                "target_register": self.q_env.unwrapped.circ_tgt_register,
            }
        elif hasattr(self.q_env.target, "dm"):
            return {
                "target_state": self.q_env.unwrapped.target.dm,
                "target_register": self.q_env.unwrapped.target.physical_qubits,
            }
