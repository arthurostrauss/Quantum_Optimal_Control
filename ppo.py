import time
import sys
import numpy as np
from typing import Optional, Dict
import tqdm
import warnings
from IPython.display import clear_output

# Torch imports for building RL agent and framework
from gymnasium.spaces import Box
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.distributions import Normal
from agent import ActorNetwork, CriticNetwork, Agent
from quantumenvironment import QuantumEnvironment

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s", # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def get_module_from_str(module_str):
    module_dict = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
        "none": nn.ReLU,
        "softmax": nn.Softmax,
        "log_softmax": nn.LogSoftmax,
        "gelu": nn.GELU,
    }
    if module_str not in module_dict:
        raise ValueError(
            f"Agent Config `ACTIVATION` needs to be one of {module_dict.keys()}"
        )
    return module_dict[module_str]


def get_optimizer_from_str(optim_str):
    optim_dict = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "rmsprop": optim.RMSprop,
        "rprop": optim.Rprop,
        "sgd": optim.SGD,
    }
    if optim_str not in optim_dict:
        raise ValueError(
            f"Agent Config `OPTIMIZER` needs to be one of {optim_dict.keys()}"
        )

    return optim_dict[optim_str]


def make_train_ppo(
    agent_config: Dict,
    env: QuantumEnvironment,
    chkpt_dir: Optional[str] = "tmp/ppo",
    chkpt_dir_critic: Optional[str] = "tmp/critic_ppo",
):
    """
    Creates a training function for PPO algorithm.
    :param agent_config: Dictionary containing all the hyperparameters for the PPO algorithm
    :param env: Quantum Environment on which the algorithm is trained
    :param chkpt_dir: Directory where the policy network is saved
    :param chkpt_dir_critic: Directory where the critic network is saved
    :return: Training function for PPO algorithm
    """
    seed = env.seed
    n_actions = env.action_space.shape[-1]
    batchsize = env.batch_size
    num_time_steps = env.tgt_instruction_counts
    min_action = env.action_space.low
    max_action = env.action_space.high

    hidden_units = agent_config["N_UNITS"]
    activation_fn = agent_config["ACTIVATION"]
    activation_functions = [
        get_module_from_str(activation_fn)() for _ in range(len(hidden_units) + 1)
    ]
    include_critic = agent_config["INCLUDE_CRITIC"]
    minibatch_size = agent_config["MINIBATCH_SIZE"]
    if batchsize % minibatch_size != 0:
        raise ValueError(
            f"The current minibatch size of {minibatch_size} does not evenly divide the batchsize of {batchsize}"
        )

    run_name = agent_config["RUN_NAME"]
    writer = SummaryWriter(f"runs/{run_name}")

    # General RL Params
    n_epochs = agent_config["N_EPOCHS"]
    lr = agent_config["LR"]

    # PPO Specific Params
    ppo_epsilon = agent_config["CLIP_RATIO"]
    critic_loss_coef = agent_config["V_COEF"]
    gamma = agent_config["GAMMA"]
    gae_lambda = agent_config["GAE_LAMBDA"]

    # Clipping
    clip_vloss = agent_config["CLIP_VALUE_LOSS"]
    grad_clip = agent_config["GRADIENT_CLIP"]
    clip_coef = agent_config["CLIP_VALUE_COEF"]
    normalize_advantage = agent_config["NORMALIZE_ADVANTAGE"]
    ent_coef = agent_config["ENT_COEF"]

    actor_net = ActorNetwork(
        env.observation_space,
        hidden_units,
        n_actions,
        activation_functions,
        include_critic,
        chkpt_dir,
    )
    critic_net = CriticNetwork(
        env.observation_space, hidden_units, activation_functions, chkpt_dir_critic
    )
    if not include_critic:
        agent = Agent(actor_net, critic_net=critic_net)
    else:
        agent = Agent(actor_net, critic_net=None)

    optim_name = agent_config["OPTIMIZER"]
    optim_eps = 1e-5
    optimizer = get_optimizer_from_str(optim_name)(
        agent.parameters(), lr=lr, eps=optim_eps
    )

    def train(
        total_updates: int,
        print_debug: Optional[bool] = True,
        num_prints: Optional[int] = 40,
    ):
        try:
            env.clear_history()
            start = time.time()
            global_step = 0

            obs = torch.zeros((num_time_steps, batchsize) + env.observation_space.shape)
            actions = torch.zeros((num_time_steps, batchsize) + env.action_space.shape)
            logprobs = torch.zeros((num_time_steps, batchsize))
            rewards = torch.zeros((num_time_steps, batchsize))
            dones = torch.zeros((num_time_steps, batchsize))
            values = torch.zeros((num_time_steps, batchsize))

            ### Starting Learning ###
            avg_return = []
            fidelities = np.zeros(total_updates)
            for ii in tqdm.tqdm(range(1, total_updates + 1)):
                next_obs, _ = env.reset(seed=seed)
                num_steps = num_time_steps  # env.episode_length(global_step)
                batch_obs = torch.tile(torch.Tensor(next_obs), (batchsize, 1))
                batch_done = torch.zeros_like(dones[0])

                # print("episode length:", num_steps)
                for step in range(num_steps):
                    global_step += 1
                    obs[step] = batch_obs
                    dones[step] = batch_done

                    with torch.no_grad():
                        mean_action, std_action, critic_value = agent(batch_obs)
                        probs = Normal(mean_action, std_action)
                        action = torch.clip(
                            probs.sample(),
                            torch.Tensor(min_action),
                            torch.Tensor(max_action),
                        )
                        logprob = probs.log_prob(action).sum(1)
                        values[step] = critic_value.flatten()

                    actions[step] = action
                    logprobs[step] = logprob

                    next_obs, reward, terminated, truncated, infos = env.step(
                        action.cpu().numpy()
                    )
                    next_obs = torch.Tensor(next_obs)
                    done = int(np.logical_or(terminated, truncated))

                    batch_obs = torch.tile(next_obs, (batchsize, 1))
                    next_done = done * torch.ones_like(dones[0])
                    obs[step] = batch_obs
                    dones[step] = next_done
                    reward = torch.Tensor(reward)
                    rewards[step] = reward

                    # print(f"global_step={global_step}, episodic_return={np.mean(reward)}")
                    writer.add_scalar(
                        "charts/episodic_return", np.mean(reward.numpy()), global_step
                    )
                    writer.add_scalar("charts/episodic_length", num_steps, global_step)

                # bootstrap value if not done
                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards)
                    lastgaelam = 0
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                        )
                    returns = advantages + values

                # flatten the batch
                b_obs = obs.reshape((-1,) + env.observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + env.action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(batchsize)
                clipfracs = []
                for epoch in range(n_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batchsize, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        new_mean, new_sigma, new_value = agent(b_obs[mb_inds])
                        new_dist = Normal(new_mean, new_sigma)
                        new_logprob, entropy = new_dist.log_prob(b_actions[mb_inds]).sum(
                            1
                        ), new_dist.entropy().sum(1)
                        logratio = new_logprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [
                                ((ratio - 1.0).abs() > ppo_epsilon).float().mean().item()
                            ]

                        mb_advantages = b_advantages[mb_inds]
                        if normalize_advantage:  # Normalize advantage
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                            )

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1 - ppo_epsilon, 1 + ppo_epsilon
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = new_value.view(-1)
                        if clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -clip_coef,
                                clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - ent_coef * entropy_loss + v_loss * critic_loss_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
                        optimizer.step()

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
                if print_debug:
                    print("mean", mean_action[0])
                    #print("sigma", std_action[0])
                    print("Average return:", np.mean(env.reward_history, axis=1)[-1])
                    print("DFE Rewards Mean:", np.mean(env.reward_history, axis=1)[-1])
                    print(
                        "DFE Rewards standard dev", np.std(env.reward_history, axis=1)[-1]
                    )
                    print("Returns Mean:", np.mean(b_returns.numpy()))
                    print("Returns standard dev", np.std(b_returns.numpy()))
                    print("Advantages Mean:", np.mean(b_advantages.numpy()))
                    print("Advantages standard dev", np.std(b_advantages.numpy()))
                    # print(np.mean(env.reward_history, axis =1)[-1])
                    # print("Circuit fidelity:", env.circuit_fidelity_history[-1])

                    if global_step % num_prints == 0:
                        clear_output(wait=True)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(
                    "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar(
                    "losses/avg_return",
                    np.mean(env.reward_history, axis=1)[-1],
                    global_step,
                )
                # writer.add_scalar("losses/avg_gate_fidelity", env.avg_fidelity_history[-1], global_step)
                # writer.add_scalar("losses/circuit_fidelity", env.circuit_fidelity_history[-1], global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)

                avg_return.append(np.mean(env.reward_history, axis=1)[-1])
                print('Fidelity History:', env.avg_fidelity_history)

            env.close()
            writer.close()

            return {
                'avg_return': avg_return,
                # 'fidelities': fidelities,
                # returns the action vector that led to the highest gate fidelity during the training process
                'action_vector': np.mean(env.action_history[np.argmax(avg_return)], axis=0),
            }
        except Exception as e:
            logging.error(f'An error occurred during training: {e}')
            return {
                'avg_return': -1.0,  # penalized return value
                'action_vector': [0] * len(env.action_history[0][0]),
            }

    return train