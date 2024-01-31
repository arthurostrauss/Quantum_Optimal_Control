import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from typing import Union
import time
from datetime import datetime
import pickle
import optuna
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from helper_functions import (
    load_agent_from_yaml_file,
    create_hpo_agent_config,
    load_hpo_config_from_yaml_file,
)
from ppo import make_train_ppo

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


class HyperparameterOptimizer:
    """
    A class for optimizing the hyperparameters of a Proximal Policy Optimization (PPO) agent
    in a quantum environment using Optuna.

    The class is responsible for initializing the quantum environment and the PPO agent,
    setting up hyperparameter optimization trials, and saving the best configurations
    discovered during optimization.

    Attributes:
        gate_q_env_config (QEnvConfig): Configuration for the quantum environment.
        q_env (QuantumEnvironment): The quantum environment instance.
        ppo_params (dict): Parameters for the PPO agent.
        network_config (dict): Configuration for the neural network used in the PPO agent.
        hpo_config (dict): Configuration for hyperparameter optimization.
        save_results_path (str): Path to save the best configuration and results.
        log_progress (bool): Flag to indicate whether to log the progress of hyperparameter optimization.
        rescalse_action (dict): Dictionary containing information about whether and how to apply the RescaleAction wrapper.
        num_hpo_trials (int): The number of trials to run for hyperparameter optimization.
        best_trial (optuna.trial._frozen.FrozenTrial, optional): The best trial found during optimization.

    Methods:
        optimize_hyperparameters(): Runs the hyperparameter optimization process.
        best_hpo_configuration: Returns the best hyperparameter configuration and its performance metric.
        target_gate: Returns information about the target gate and register from the quantum environment.

    Example:
    >>> optimizer = HyperparameterOptimizer(
            q_env=QuantumEnvironment,
            path_agent_config="path/to/agent/config.yaml",
            path_hpo_config="path/to/hpo/config.yaml",
            save_results_path="path/to/save/results",
            log_progress=True
        )
    >>> optimizer.optimize_hyperparameters()

    """

    def __init__(
        self,
        q_env: Union[QuantumEnvironment, ContextAwareQuantumEnvironment],
        path_agent_config: str,
        path_hpo_config: str,
        save_results_path: str,
        log_progress: bool = True,
        ):
        self.q_env = q_env
        # Start with an initial agent configuration and then update it with the hyperparameters later in the workflow
        self.agent_config_init = load_agent_from_yaml_file(
            path_agent_config
        )
        self.hpo_config = load_hpo_config_from_yaml_file(path_hpo_config)
        self.save_results_path = save_results_path
        self.log_progress = log_progress

    def _objective(self, trial):
        # Fetch hyperparameters from the trial object
        self.agent_config, self.hyperparams = create_hpo_agent_config(
            trial, self.hpo_config, self.agent_config_init
        )

        # Overwrite the batch_size of the (unwrapped) environment with the one from the agent_config
        self.q_env.unwrapped.batch_size = self.agent_config["BATCHSIZE"]

        train_fn = make_train_ppo(self.agent_config, self.q_env)
        training_results = train_fn(
            total_updates=self.agent_config["N_UPDATES"],
            print_debug=True,
            num_prints=50,
        )
        if training_results["avg_reward"] != -1.0: # If the training was successful 
            # Save important information about the trial
            trial.set_user_attr("action_vector", training_results["best_action_vector"])
            trial.set_user_attr("avg_reward", training_results["avg_reward"])
            trial.set_user_attr("std_action", training_results["std_action"])
            trial.set_user_attr("action_history", training_results["action_history"])
            trial.set_user_attr("fidelity_history", training_results["fidelity_history"])

        # Use a relevant metric from training_results as the return value
        last_ten_percent = int(0.1 * len(training_results["fidelity_history"]))

        return training_results["fidelity_history"][
            -last_ten_percent
        ]  # Return the fidelity of the last 10% of updates

    def _save_best_configuration(self):
        if self.best_trial is not None:
            best_config = {
                "hyper_params": self.best_trial.params,
                "action_vector": self.best_trial.user_attrs["action_vector"],
                "avg_reward": self.best_trial.user_attrs["avg_reward"],
                "std_action": self.best_trial.user_attrs["std_action"],
                "action_history": self.best_trial.user_attrs["action_history"],
                "fidelity_history": self.best_trial.user_attrs["fidelity_history"],
            }

            if not os.path.exists(self.save_results_path):
                os.makedirs(self.save_results_path)
                logging.warning(f"Folder '{self.save_results_path}' created.")

            pickle_file_name = os.path.join(
                self.save_results_path,
                f"fidelity_{round(self.best_trial.value, 6)}.pickle",
            )
            with open(pickle_file_name, "wb") as handle:
                pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.warning(f"Best configuration saved to {pickle_file_name}")
        else:
            logging.warning("No best trial data to save.")

        return best_config

    def _logging_result(self, study, start_time):
        logging.warning("---------------- FINISHED HPO ----------------")
        logging.warning(
            "HPO completed in {} seconds.".format(round(time.time() - start_time, 2))
        )

        logging.warning("Best trial:")
        logging.warning("-------------------------")
        logging.warning("  Fidelity: {}".format(study.best_trial.value))
        logging.warning("  Hyperparameters: ")
        for key, value in study.best_trial.params.items():
            logging.warning("    {}: {}".format(key, value))

        best_action_vector = study.best_trial.user_attrs["action_vector"]
        logging.warning("The best action vector: {}".format(best_action_vector))

    def optimize_hyperparameters(self, num_hpo_trials: int = 1):
        if num_hpo_trials is not None:
            self.n_hpo_trials = num_hpo_trials
        else:
            self.n_hpo_trials = self.hpo_config.get("n_trials", 0)
        # Assert that n_hpo_trials is an integer and greater than 0
        assert (
            isinstance(self.n_hpo_trials, int) and self.n_hpo_trials > 0
        ), "n_hpo_trials must be an integer greater than 0"

        start_time = time.time()
        logging.warning("n_HPO_trials: {}".format(self.n_hpo_trials))
        logging.warning("---------------- STARTING HPO ----------------")

        study = optuna.create_study(
            direction="maximize",
            study_name=f'{self.target_gate["target_gate"].name}-calibration_{datetime.now().strftime("%d-%m-%Y-_%H:%M:%S")}',
        )
        study.optimize(self._objective, n_trials=self.n_hpo_trials)

        # If all trials led to errors in the training process, fidelities for all trials are 0.0
        # If this is the case, then the HPO failed
        # Return a warning and return a dictionary with the result
        if study.best_trial.value == 0.0:
            logging.warning("ERROR: HPO failed. No trials to save.")
            return {
                'result': 'ERROR: HPO failed. All hyperparameter trials led to errors in the training process.',
            }

        if self.log_progress:
            self._logging_result(study, start_time)

        self.best_trial = study.best_trial
        # Save the best configuration and return it as a dictionary for the user
        return self._save_best_configuration()

    @property
    def target_gate(self):
        return {
            'target_gate': self.q_env.target["gate"],
            'target_register': self.q_env.target["register"],
        }