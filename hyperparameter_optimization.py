import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import time
import pickle
import optuna
from quantumenvironment import QuantumEnvironment
from helper_functions import load_agent_from_yaml_file, create_agent_config
from ppo import make_train_ppo
from qconfig import QEnvConfig

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s", # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

class HyperparameterOptimizer:
    def __init__(
            self, 
            gate_q_env_config: QEnvConfig, 
            path_agent_config: str, 
            save_results_path: str, 
            log_progress: bool = True,
            num_hpo_trials: int = None,
        ):
        self.gate_q_env_config = gate_q_env_config
        self.q_env = QuantumEnvironment(self.gate_q_env_config)
        self.ppo_params, self.network_config, self.hpo_config = load_agent_from_yaml_file(path_agent_config)
        self.save_results_path = save_results_path
        self.log_progress = log_progress

        if num_hpo_trials is not None:
            self.num_hpo_trials = num_hpo_trials
        else:
            self.num_hpo_trials = self.hpo_config.get('num_trials', 0)
        # Assert that num_hpo_trials is an integer and greater than 0
        assert isinstance(self.num_hpo_trials, int) and self.num_hpo_trials > 0, "num_hpo_trials must be an integer greater than 0"

        
    def _objective(self, trial):
        # Fetch hyperparameters from the trial object
        self.agent_config, self.hyperparams = create_agent_config(trial, self.hpo_config, self.network_config, self.ppo_params)

        self.q_env = QuantumEnvironment(self.gate_q_env_config)
        # Overwrite the batch_size of the environment with the one from the agent_config
        self.q_env.batch_size = self.agent_config['BATCHSIZE']

        train_fn = make_train_ppo(self.agent_config, self.q_env)
        training_results = train_fn(total_updates=self.agent_config['N_UPDATES'], print_debug=True, num_prints=50)

        # Save the action vector associated with this trial's fidelity for future retrieval
        trial.set_user_attr('action_vector', training_results['action_vector'])

        # Use a relevant metric from training_results as the return value
        last_ten_percent = int(0.1 * self.agent_config['N_UPDATES'])
        
        return training_results['avg_return'][-last_ten_percent]  # Return a metric to minimize or maximize

    def _save_best_configuration(self):
        if self.best_trial is not None:
            best_config = {
                'parameters': self.best_trial.params,
                'action_vector': self.best_trial.user_attrs['action_vector']
            }

            if not os.path.exists(self.save_results_path):
                os.makedirs(self.save_results_path)
                logging.warning(f"Folder '{self.save_results_path}' created.")

            pickle_file_name = os.path.join(self.save_results_path, f'reward_{round(self.best_trial.value, 6)}.pickle')
            with open(pickle_file_name, 'wb') as handle:
                pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Best configuration saved to {pickle_file_name}")
        else:
            print("No best trial data to save.")
    
    def _logging_progress(self, study, start_time):
        logging.warning('---------------- FINISHED HPO ----------------')
        logging.warning('HPO completed in {} seconds.'.format(round(time.time() - start_time, 2)))
        logging.warning("Best trial:")
        logging.warning("-------------------------")
        logging.warning("  Value: {}".format(study.best_trial.value))
        logging.warning("  Parameters: ")
        for key, value in study.best_trial.params.items():
            logging.warning("    {}: {}".format(key, value))

        best_action_vector = study.best_trial.user_attrs['action_vector']
        logging.warning('The best action vector is {}'.format(best_action_vector))
    
    def optimize_hyperparameters(self):
        start_time = time.time()
        logging.warning('num_HPO_trials: {}'.format(self.num_hpo_trials))
        logging.warning('---------------- STARTING HPO ----------------')

        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.num_hpo_trials)
        
        if self.log_progress:
            self._logging_progress(study, start_time)

        self.best_trial = study.best_trial
        self._save_best_configuration()

    @property
    def best_hpo_configuration(self):
        if self.best_trial is None:
            return "No HPO trial has been run yet."

        best_config = {
            'best_avg_return': self.best_trial.value,
            'best_hyperparams': self.best_trial.params,
        }
        return best_config
    
    @property
    def target_gate(self):
        return {
            'target_gate': self.gate_q_env_config.target['gate'],
            'target_register': self.gate_q_env_config.target['register']
        }