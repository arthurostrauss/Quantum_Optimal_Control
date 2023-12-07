import numpy as np
import argparse
import time
import pickle
import os
import sys

from rl_training import (
    train_agent, 
    get_network, 
    apply_parametrized_circuit
)

import optuna

module_path = os.path.abspath(os.path.join('/Users/lukasvoss/Documents/Master Wirtschaftsphysik/Masterarbeit Yale-NUS CQT/Quantum_Optimal_Control'))
if module_path not in sys.path:
    sys.path.append(module_path)
from helper_functions import select_optimizer
from quantumenvironment import QuantumEnvironment
from simulation_config import sim_config

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s", # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def parse_args():
    """
    Parse the command line arguments and return the parsed arguments.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed command line arguments.

    Raises:
        None
    """
    parser = argparse.ArgumentParser(description="Train a RL agent to optimally calibrate gate parameters of a quantum gate")

    parser.add_argument(
        "-t",
        "--num_trials",
        metavar="num_trials",
        type=int,
        default=20,
        help="number of HPO trials; default: 20",
        required=False,
    )

    return parser.parse_args()

def positive_integer(value):
    """
    Converts the given value to an integer and checks if it is a positive integer.

    Args:
        value (Any): The value to convert to an integer.

    Returns:
        int: The converted positive integer.

    Raises:
        argparse.ArgumentTypeError: If the converted integer is not positive.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def objective(trial):
    """
    Objective function for hyperparameter optimization using Optuna.
    
    This function sets up the quantum environment, defines the training parameters,
    and runs the training agent. It then calculates and returns the average of the
    last 10% of fidelities achieved during training as the value to be optimized.

    Args:
        trial (optuna.trial.Trial): An individual trial object with methods to
                                    suggest hyperparameters.

    Returns:
        float: The average of the last 10% of fidelities from the training results.
    """
    sim_config.parametrized_circuit = apply_parametrized_circuit
    q_env = QuantumEnvironment(simulation_config=sim_config)

    training_parameters = {
        'use_PPO': True,
        'n_epochs': trial.suggest_int('epochs', 400, 200),
        'batchsize': trial.suggest_int('batchsize', 50, 500),
        'opti_choice': trial.suggest_categorical('opti', ['Adam', 'SGD']),
        'eta': trial.suggest_float('eta', 1e-5, 5*1e-2, log=True),
        'eta_2': trial.suggest_float('eta_2', 1e-4, 1e-1, log=True) if trial.suggest_int("use_eta_2", 0, 1) else None,
        'epsilon': trial.suggest_float('epsilon', 0.005, 0.5),
        'grad_clip': trial.suggest_float('grad_clip', 0.005, 1),
        'critic_loss_coeff': trial.suggest_float('critic_loss_coeff', 0.1, 2),
        'sigma_eps': trial.suggest_float('sigma_eps', 1e-5, 1e-2),
    }
    training_parameters['optimizer'] =  select_optimizer(lr=training_parameters['eta'], optimizer=training_parameters['opti_choice'], 
                                                         grad_clip=training_parameters['grad_clip'], concurrent_optimization=True, 
                                                         lr2=training_parameters['eta_2'])

    # Get the RL model
    training_parameters['network'], training_parameters['init_msmt'] = get_network()

    # Call the training function from rl_training.py
    training_results = train_agent(q_env, training_parameters)

    # Save the action vector associated with this trial's fidelity for future retrieval
    trial.set_user_attr('action_vector', training_results['action_vector'])

    # Use a relevant metric from training_results as the return value
    last_ten_percent = int(0.1 * training_parameters['n_epochs'])

    return training_results['fidelities'][-last_ten_percent]

def save_pickle(best_trial, best_run):
    # Save the best run configuration as a hashed file (here pickle)
    pickle_file_name = f"gate_level_abstraction/hpo/hpo_results/fidelity_{best_trial.value}.pickle" 
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(best_run, handle, protocol=pickle.HIGHEST_PROTOCOL)

def hyperparameter_optimization(n_trials):
    """
    Runs the hyperparameter optimization using Optuna to find the best set of
    hyperparameters for the reinforcement learning training agent.

    Args:
        n_trials (int): The number of trials to run in the hyperparameter optimization.

    Returns:
        optuna.study.Study: The study object that contains all the information about
                            the optimization session, including the best trial.
    """
    logging.warning("----------------------------START HPO----------------------------")
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    logging.warning("----------------------------FINISH HPO----------------------------")
    runtime = time.time() - start_time
    
    return study, runtime

def log_results(study, runtime):
    """ Log the results of the HPO run """
    logging.warning(f"HPO RUNTIME: {int(runtime)} SEC")

    print("Number of finished trials: ", len(study.trials))
    print("\nBest trial:")
    print("-------------------------------------------- ")
    best_trial = study.best_trial

    print("Gate Fidelity: ", best_trial.value)
    print("Hyperparams: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return best_trial


if __name__ == "__main__":
    # Parse command-line arguments to get the number of trials for optimization, ensure a meaningful value for num_trials (pos. int.)
    num_trials = positive_integer(parse_args().num_trials)
    # Run hyperparameter optimization
    study, runtime = hyperparameter_optimization(n_trials=num_trials)
    best_trial = log_results(study, runtime)

    # Fetch and display the best trial's action vector
    best_action_vector = study.best_trial.user_attrs['action_vector']
    print(f'The best action vector is: {best_action_vector}')
    
    # Save best hyperparameters to hashed file (e.g., using pickle)    
    best_run = {
        'fidelity': best_trial.value,
        'action_vector': best_action_vector,
        'hyperparams': best_trial.params,
    }
    save_pickle(best_trial, best_run)