# %%
from qiskit.circuit.library import XGate
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2

import argparse
import optuna
import pickle
import sys
import logging
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pulse_parametrization_functions_v01 import (
   get_target_gate, get_circuit_context, get_estimator_options, get_db_qiskitconfig, get_torch_env, get_network, clear_history, train_agent
)
from qconfig import SimulationConfig

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
    -----------------------------------------------------------------------------------------------------
        User Input: Simulation parameters
    -----------------------------------------------------------------------------------------------------
    """
    simulation_config = SimulationConfig(abstraction_level="pulse",
                                         target_gate=XGate(),
                                         register=[0],
                                         fake_backend=FakeJakarta(),
                                         fake_backend_v2=FakeJakartaV2(),
                                         n_actions=4,
                                         sampling_Paulis=50,
                                         n_shots=200,
                                         device=torch.device("cpu")
                                         )
    
    target = get_target_gate(gate=simulation_config.target_gate, register=simulation_config.register)
    physical_qubits = tuple(target["register"])

    # %%
    target_circuit = get_circuit_context(num_total_qubits=1)
    
    # %%
    qubit_properties, dynamics_options, estimator_options, channel_freq, solver = get_estimator_options(simulation_config.sampling_Paulis, simulation_config.n_shots, physical_qubits, simulation_config.fake_backend, simulation_config.fake_backend_v2)
    # %%
    _, _, q_env = get_db_qiskitconfig(simulation_config.fake_backend, target, physical_qubits, qubit_properties, estimator_options, channel_freq, solver, simulation_config.sampling_Paulis, simulation_config.abstraction_level, simulation_config.n_shots, dynamics_options)
    # %%
    torch_env, observation_space, _, tgt_instruction_counts, batchsize, min_bound_actions, max_bound_actions, scale_factor, seed = get_torch_env(q_env, target_circuit, simulation_config.n_actions)

    # %%
    # device = torch.device("cpu")
    _, _, agent = get_network(simulation_config.device, observation_space, simulation_config.n_actions)

    """
    -----------------------------------------------------------------------------------------------------
        Subject to Optuna HPO: Hyperparameters for RL agent
    -----------------------------------------------------------------------------------------------------
    """
    training_parameters = {
        'n_epochs': trial.suggest_int('n_epochs', 10, 15), # Choose small values for debugging
        'num_updates': trial.suggest_int('num_updates', 5, 15), # Choose small values for debugging
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
    # actor_optimizer = optim.Adam(actor_net.parameters(), lr=training_parameters['lr_actor'], eps=1e-5)
    # critic_optimizer = optim.Adam(critic_net.parameters(), lr=training_parameters['lr_actor'], eps=1e-5)

    # %%
    ### Training ###
    global_step, obs, actions, logprobs, rewards, dones, values, _, visualization_steps = clear_history(torch_env, tgt_instruction_counts, batchsize, simulation_config.device)
    run_name = "test"
    writer = SummaryWriter(f"runs/{run_name}")
    training_results = train_agent(torch_env, global_step, training_parameters['num_updates'], seed, simulation_config.device, batchsize, obs, agent, scale_factor, min_bound_actions, max_bound_actions, logprobs, actions, rewards, dones, values, training_parameters['n_epochs'], optimizer, training_parameters['minibatch_size'], training_parameters['gamma'], training_parameters['gae_lambda'], training_parameters['critic_loss_coeff'], training_parameters['epsilon'], training_parameters['clip_vloss'], training_parameters['grad_clip'], training_parameters['clip_coef'], training_parameters['normalize_advantage'], training_parameters['ent_coef'], writer, visualization_steps)

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
    pickle_file_name = f"torch_contextual_gate_calibration/hpo_results/return_{best_trial.value}.pickle" 
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(best_run, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # Parse command-line arguments to get the number of trials for optimization, ensure a meaningful value for num_trials (pos. int.)
    num_trials = positive_integer(parse_args().num_trials)
    # Run hyperparameter optimization
    study, runtime = hyperparameter_optimization(n_trials=num_trials)

    logging.warning(f"HPO RUNTIME: {int(runtime)} SEC")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial

    print("Average Return: ", best_trial.value)
    print("Hyperparams: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best hyperparameters to hashed file (e.g., using pickle)
    best_hyperparams = best_trial.params
    # Fetch and display the best trial's action vector
    best_action_vector = study.best_trial.user_attrs["action vector"]
    
    best_run = {'avg_return': best_trial.value,
                'action_vector': best_action_vector.numpy(),
                'hyperparams': best_hyperparams,
                }
    save_pickle(best_trial, best_run)
    
    print(f"The best action vector is: {best_action_vector.numpy()}")