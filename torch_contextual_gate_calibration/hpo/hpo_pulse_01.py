"""
This script allows to run hyperparameter optimization (HPO) for the contextual gate calibration problem. 
Simulation parameters can be set in the simulation_config.py file. 
The HPO results are saved in the torch_contextual_gate_calibration/hpo_results folder.

The workflow is divided into three parts:
    1. Specify parameters for the gate calibration simulation in the simulation_config.py file
    2. Run this file to perform HPO specifying the number of trials via argparse
    3. This script orchestrates the workflow while pulse_parametrization_functions_v01.py contains the functions that are used in this script

Example usage: python torch_contextual_gate_calibration/hpo_pulse_01.py -t 2

Author: Lukas Voss
Created on 16/11/2023
"""

# %%
import argparse
import optuna
import pickle
import sys
import logging
import time

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pulse_parametrization_functions_v01 import (
   get_own_solver, get_target_gate, get_estimator_options, get_db_qiskitconfig, get_torch_env, get_network, clear_history, train_agent
)
from simulation_config import sim_config, get_circuit_context
# %%
# Create a custom logger with the level WARNING because INFO would trigger too many log message by qiskit itself
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
        default=2,
        help="number of HPO trials; default: 2",
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

    # %%
    target = sim_config.target # get_target_gate(gate=sim_config.target, register=sim_config.register)
    physical_qubits = tuple(target.get('register', None))

    # %%
    circuit_context = get_circuit_context() # circuit_context output to be specified in the simulation_config.py file by the user
    
    # %%
    dynamics_options, estimator_options, channel_freq, solver = get_estimator_options(sim_config.sampling_Paulis, sim_config.n_shots, physical_qubits, sim_config.backend)
    # %%
    gate_str = target.get('gate_str', None)
    q_env = get_db_qiskitconfig(sim_config.backend, target, physical_qubits, gate_str, estimator_options, channel_freq, solver, dynamics_options)
    # %%
    torch_env, observation_space, tgt_instruction_counts, batchsize, min_bound_actions, max_bound_actions, scale_factor, seed = get_torch_env(q_env, circuit_context, sim_config.n_actions)

    # %%
    agent = get_network(sim_config.device, observation_space, sim_config.n_actions)

    """
    -----------------------------------------------------------------------------------------------------
        Subject to Optuna HPO: Hyperparameters for RL agent
    -----------------------------------------------------------------------------------------------------
    """
    training_parameters = {
        'n_epochs': trial.suggest_int('n_epochs', 10, 15), # Choose small values for debugging
        'num_updates': trial.suggest_int('num_updates', 10, 25), # Choose small values for debugging
        'lr_actor': trial.suggest_float('lr_actor', 1e-4, 1e-2, log=True),
        'lr_critic': trial.suggest_float('lr_critic', 1e-4, 1e-2, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.1, 0.3),
        'critic_loss_coeff': trial.suggest_float('critic_loss_coeff', 0.1, 1.0),
        'minibatch_size': trial.suggest_int('minibatch_size', 20, 100),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        'clip_vloss': trial.suggest_categorical('clip_vloss', [True, False]),
        'grad_clip': trial.suggest_float('grad_clip', 0.1, 1.0),
        'clip_coef': trial.suggest_float('clip_coef', 0.1, 1.0),
        'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
    }

    optimizer = optim.Adam(agent.parameters(), lr=training_parameters['lr_actor'], eps=1e-5)

    # %%
    ### Training ###
    global_step, obs, actions, logprobs, rewards, dones, values, _, visualization_steps = clear_history(torch_env, tgt_instruction_counts, batchsize, sim_config.device)
    run_name = "test"
    writer = SummaryWriter(f"runs/{run_name}")
    training_results = train_agent(torch_env, global_step, training_parameters['num_updates'], seed, sim_config.device, batchsize, obs, agent, scale_factor, min_bound_actions, max_bound_actions, logprobs, actions, rewards, dones, values, training_parameters['n_epochs'], optimizer, training_parameters['minibatch_size'], training_parameters['gamma'], training_parameters['gae_lambda'], training_parameters['critic_loss_coeff'], training_parameters['epsilon'], training_parameters['clip_vloss'], training_parameters['grad_clip'], training_parameters['clip_coef'], training_parameters['normalize_advantage'], training_parameters['ent_coef'], writer, visualization_steps)

    # %%
    ### Save results ###
    # Save the action vector associated with this trial's fidelity for future retrieval
    trial.set_user_attr('action vector', training_results['mean_action'])
    trial.set_user_attr('sigma', training_results['sigma_action'])
 
    # Use a relevant metric from training_results as the return value
    last_ten_percent = int(0.1 * training_parameters['n_epochs'])

    return training_results['avg_return'][-last_ten_percent]


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

def save_pickle(best_trial, best_run):
    # Save the best run configuration as a hashed file (here pickle)
    pickle_file_name = f"torch_contextual_gate_calibration/hpo/hpo_results/return_{best_trial.value}.pickle" 
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(best_run, handle, protocol=pickle.HIGHEST_PROTOCOL)

def log_results(runtime, study):
    """ Log the results of the HPO run """
    logging.warning(f"HPO RUNTIME: {int(runtime)} SEC")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial

    print("Average Return: ", best_trial.value)
    print("Hyperparams: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return best_trial

if __name__ == "__main__":
    # Parse command-line arguments to get the number of trials for optimization, ensure a meaningful value for num_trials (pos. int.)
    num_trials = positive_integer(parse_args().num_trials)
    
    # Run hyperparameter optimization
    study, runtime = hyperparameter_optimization(n_trials=num_trials)
    best_trial = log_results(runtime, study)

    # Fetch and display the best trial's action vector
    best_action_vector = study.best_trial.user_attrs['action vector']

    # Save best hyperparameters to hashed file (e.g., using pickle)    
    best_run = {'avg_return': best_trial.value,
                'action_vector': best_action_vector.numpy(),
                'hyperparams': best_trial.params,
                }
    save_pickle(best_trial, best_run)
    
    print(f"The best action vector is: {best_action_vector.numpy()}")