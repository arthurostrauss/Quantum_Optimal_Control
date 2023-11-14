# %%
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2

import argparse
import optuna
import logging

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sx_parametrization_functions import (
   get_target_gate, get_circuit_context, transpile_circuit, get_estimator_options, get_db_qiskitconfig, get_torch_env, get_network, clear_history, train_agent
)

# Create a custom logger with the desired logging level
"""logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)"""

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

    fake_backend = FakeJakarta()
    fake_backend_v2 = FakeJakartaV2()
    
    target = get_target_gate(gate=XGate(), register=[0])
    physical_qubits = tuple(target["register"])
    # %%
    target_circuit = get_circuit_context(num_total_qubits=1)
    target_circuit.draw(output="mpl")
    # %%
    transpiled_circ = transpile_circuit(target_circuit, fake_backend)
    transpiled_circ.draw(output="mpl")

    # %%
    qubit_properties, dynamics_options, estimator_options, channel_freq, solver, sampling_Paulis, abstraction_level, N_shots = get_estimator_options(physical_qubits)
    # %%
    dynamics_backend, Qiskit_setup, q_env = get_db_qiskitconfig(target, physical_qubits, qubit_properties, estimator_options, channel_freq, solver, sampling_Paulis, abstraction_level, N_shots, dynamics_options)
    # %%
    torch_env, observation_space, action_space, tgt_instruction_counts, batchsize, min_bound_actions, max_bound_actions, scale_factor, seed = get_torch_env(q_env, target_circuit)

    # %%
    device = torch.device("cpu")
    actor_net, critic_net, agent = get_network(device, observation_space)

    """
    -----------------------------------------------------------------------------------------------------
    Hyperparameters for RL agent
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
    global_step, obs, actions, logprobs, rewards, dones, values, train_obs, visualization_steps = clear_history(torch_env, tgt_instruction_counts, batchsize, device)
    run_name = "test"
    writer = SummaryWriter(f"runs/{run_name}")
    training_results = train_agent(torch_env, global_step, training_parameters['num_updates'], seed, device, batchsize, obs, agent, scale_factor, min_bound_actions, max_bound_actions, logprobs, actions, rewards, dones, values, training_parameters['n_epochs'], optimizer, training_parameters['minibatch_size'], training_parameters['gamma'], training_parameters['gae_lambda'], training_parameters['critic_loss_coeff'], training_parameters['epsilon'], training_parameters['clip_vloss'], training_parameters['grad_clip'], training_parameters['clip_coef'], training_parameters['normalize_advantage'], training_parameters['ent_coef'], writer, visualization_steps)

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

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Average Return: ", trial.value)
    print("Hyperparams: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


if __name__ == "__main__":
    # Parse command-line arguments to get the number of trials for optimization, ensure a meaningful value for num_trials (pos. int.)
    num_trials = positive_integer(parse_args().num_trials)
    # Run hyperparameter optimization
    study = hyperparameter_optimization(n_trials=num_trials)

    # Fetch and display the best trial's action vector
    best_action_vector = study.best_trial.user_attrs["action vector"]
    print(f"The best action vector is: {best_action_vector}")