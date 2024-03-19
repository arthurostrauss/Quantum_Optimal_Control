import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from typing import Union, Optional, Dict
import time
from datetime import datetime
import pickle
import optuna
from quantumenvironment import QuantumEnvironment
from context_aware_quantum_environment import ContextAwareQuantumEnvironment
from helper_functions import (
    load_from_yaml_file,
    create_hpo_agent_config,
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
        # Start with an initial agent configuration and then update it with the hyperparameters later in the workflow
        self.agent_config_init = load_from_yaml_file(path_agent_config)
        self.hpo_config = load_from_yaml_file(path_hpo_config)
        self.save_results_path = save_results_path
        self.experimental_penalty_weights = experimental_penalty_weights
        self.log_progress = log_progress

    def _objective(self, trial: optuna.trial.Trial):
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
        self.agent_config, self.hyperparams = create_hpo_agent_config(
            trial, self.hpo_config, self.agent_config_init
        )

        # Include batchsize, n_shots, and sampling_Pauli_space in the hpo scope
        self.q_env.unwrapped.batch_size = self.agent_config["BATCHSIZE"]
        self.q_env.unwrapped.n_shots = self.agent_config["N_SHOTS"]
        self.q_env.unwrapped.sampling_Pauli_space = self.agent_config["SAMPLE_PAULIS"]

        train_fn = make_train_ppo(self.agent_config, self.q_env)
        start_time = time.time()
        training_results = train_fn(
            total_updates=self.agent_config["N_UPDATES"],
            print_debug=True,
            num_prints=50,
        )
        runtime = time.time() - start_time
        if training_results["avg_reward"] != -1.0:  # Check if training was successful
            trial.set_user_attr("training_results", training_results)
            trial.set_user_attr("runtime", runtime)

        fidelity_last_ten_percent = training_results["fidelity_history"][
            -int(0.1 * len(training_results["fidelity_history"]))
        ]
        infidelity = 1 - fidelity_last_ten_percent

        experimental_penalty_terms = self._get_penalty(
            self.experimental_penalty_weights, runtime
        )
        custom_cost_value = infidelity + experimental_penalty_terms

        return custom_cost_value

    def _get_penalty(
        self, experimental_penalty_weights: Dict[str, float], runtime: float
    ):
        """
        Calculates the penalty for experimentally costly hyperparameters and runtime.

        Parameters:
        - experimental_penalty_weights: Dictionary specifying penalty weights for each hyperparameter.
        - runtime: The runtime of the trial in seconds.

        Returns:
        - Total penalty as a float, including both hyperparameter penalties and runtime penalty.
        """
        if experimental_penalty_weights is None:
            return 0
        total_penalty = sum(
            self.agent_config[key.upper()] * experimental_penalty_weights[key]
            for key in experimental_penalty_weights
            if key != "runtime"
        )
        if "runtime" in experimental_penalty_weights:
            total_penalty += runtime * experimental_penalty_weights["runtime"]
        return total_penalty

    def _generate_filename(self):
        """Generate the file name where the best configuration will be saved."""
        return (
            f"custom_cost_value_{round(self.best_trial.value, 6)}"
            + "_timestamp_"
            + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            + ".pickle"
        )

    def _save_best_configuration(self):
        """
        Saves the best trial configuration, including hyperparameters, training results,
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
            with open(pickle_file_name, "wb") as handle:
                pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.warning(f"Best configuration saved to {pickle_file_name}")
        else:
            logging.warning("No best trial data to save.")

        self.best_config = best_config

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
                self.best_config["training_results"]["best_action_vector"]
            )
        )

    def optimize_hyperparameters(self, num_hpo_trials: int = 1):
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
        logging.warning("n_HPO_trials: {}".format(self.n_hpo_trials))
        logging.warning("---------------- STARTING HPO ----------------")

        study = optuna.create_study(
            direction="minimize",
            study_name=f'{self.target_gate["target_gate"].name}-calibration_{datetime.now().strftime("%d-%m-%Y-_%H:%M:%S")}',
        )
        study.optimize(self._objective, n_trials=self.n_hpo_trials)

        # If all trials led to errors in the training process, fidelities for all trials are 0.0
        # If this is the case, then the HPO failed
        # Return a warning and return a dictionary with the result
        if study.best_trial.value == 0.0:
            logging.warning("ERROR: HPO failed. No trials to save.")
            return {
                "result": "ERROR: HPO failed. All trials led to errors in the training process.",
            }
        self.best_trial = study.best_trial
        self._save_best_configuration()
        if self.log_progress:
            self._logging_result(study, start_time_hpo)

        return self.best_config

    @property
    def target_gate(self):
        """
        Returns information about the target gate and register from the quantum environment.

        Returns:
        - A dictionary containing 'target_gate' and 'target_register' keys with their corresponding values.
        """
        return {
            "target_gate": self.q_env.unwrapped.target["gate"],
            "target_register": self.q_env.unwrapped.target["register"],
        }
