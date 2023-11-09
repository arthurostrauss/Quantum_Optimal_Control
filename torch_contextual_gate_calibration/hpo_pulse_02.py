# %%
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate
from qiskit.providers.fake_provider import FakeJakarta, FakeJakartaV2
import torch


from sx_parametrization_03 import (
   get_target_gate, get_circuit_context, transpile_circuit, get_estimator_options, get_db_qiskitconfig, get_torch_env, get_network, get_hyperparams, clear_history
)



fake_backend = FakeJakarta()
fake_backend_v2 = FakeJakartaV2()
# %%
target = get_target_gate(gate=XGate(), register=[0])
physical_qubits = tuple(target["register"]
                        )
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

# %%
n_epochs, num_updates, lr_actor, lr_critic, epsilon, critic_loss_coeff, optimizer, actor_optimizer, critic_optimizer, minibatch_size, gamma, gae_lambda, clip_vloss, grad_clip, clip_coef, normalize_advantage, ent_coef = get_hyperparams(agent, actor_net, critic_net)
# %%
print(torch_env.episode_length)
# %%
global_step, obs, actions, logprobs, rewards, dones, values, train_obs, visualization_steps = clear_history(torch_env, tgt_instruction_counts, batchsize, device)
# %%
training_results = train_agent(torch_env, global_step, num_updates, seed, device, batchsize, obs, agent, scale_factor, min_bound_actions, max_bound_actions, logprobs, actions, rewards, dones, values, n_epochs, optimizer, minibatch_size, gamma, gae_lambda, critic_loss_coeff, epsilon, clip_vloss, grad_clip, clip_coef, normalize_advantage, ent_coef, writer, visualization_steps)