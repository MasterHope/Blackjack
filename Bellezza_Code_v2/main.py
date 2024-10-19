import itertools
import math
import random
import sys
from collections import defaultdict
from functools import partial

import gymnasium as gym
import numpy as np
import torch as th
from tqdm import tqdm

from plotting import show, view_training, view_rewards_test


def linear_epsilon_decay(epsilons):
    epsilons['epsilon'] = max(epsilons['end'], epsilons['epsilon'] - epsilons['decay'])


def exp_decay(epsilons):
    epsilons['epsilon'] = max(epsilons['end'], epsilons['init'] * math.exp(-epsilons['decay'] * epsilons['step']))


def create_decay_params(decay_epsilon, end_epsilon, init_epsilon):
    epsilon_dict = create_epsilon_dict(init_epsilon, decay_epsilon, end_epsilon)
    decay_epsilon_fun = {'linear': partial(linear_epsilon_decay, epsilon_dict),
                         'exp': partial(exp_decay, epsilon_dict)}
    return decay_epsilon_fun, epsilon_dict


def t_difference(env, alpha, epsilon_di, decay_epsilon, gamma, n_episodes):
    env.action_space.seed(random.randint(0, sys.maxsize))
    q = defaultdict(lambda: th.zeros(env.action_space.n))
    rewards = []
    episode_lengths = []
    for i in tqdm(range(n_episodes), file=sys.stdout):
        epsilon_di['step'] = i
        s, _ = env.reset(seed=random.randint(0, sys.maxsize))
        done = False
        a = epsilon_policy(epsilon_di['epsilon'], env, q, s)
        total_reward = 0
        episode_length = 0
        while not done:
            next_s, reward, terminated, truncated, _ = env.step(a)
            next_a = epsilon_policy(epsilon_di['epsilon'], env, q, next_s)
            done = terminated or truncated
            future_q_value = (not terminated) * q[next_s][next_a]
            temporal_difference = (reward + gamma * future_q_value - q[s][a])
            q[s][a] = (q[s][a] + alpha * temporal_difference)
            s = next_s
            a = next_a
            episode_length += 1
            total_reward += reward
        rewards.append(total_reward)
        episode_lengths.append(episode_length)
        decay_epsilon()
    return q, rewards, episode_lengths


def generate_episode(env, epsilon, q):
    done = False
    episode = []
    observation, _ = env.reset(seed=random.randint(0, sys.maxsize))
    while not done:
        act = epsilon_policy(epsilon, env, q, observation)
        next_obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        episode.append([observation, act, reward])
        observation = next_obs
    return episode


def montecarlo(env, alpha, epsilon_di, decay_epsilon, gamma, n_episodes):
    env.action_space.seed(random.randint(0, sys.maxsize))
    q = defaultdict(lambda: th.zeros(env.action_space.n))
    rewards = []
    episode_lengths = []
    for i in tqdm(range(n_episodes), file=sys.stdout):
        epsilon_di['step'] = i
        total_rewards = 0
        episode = generate_episode(env, epsilon_di["epsilon"], q)
        all_state_actions = [(st, ac) for (st, ac, r) in episode]
        g = 0
        for j in range(len(episode), 0, -1):
            s, a, r = episode[j - 1]
            g = gamma * g + r
            total_rewards += r
            if not (s, a) in all_state_actions[0:j - 1]:
                q[s][a] = q[s][a] + alpha * (g - q[s][a])

        rewards.append(total_rewards)
        episode_lengths.append(len(episode))
        decay_epsilon()
    return q, rewards, episode_lengths


def update_q_mc(episode, g, q, alpha):
    for i in range(len(episode)):
        observ = episode[i][0]
        act = episode[i][1]
        q[observ][act] += alpha * (g - q[observ][act])


def epsilon_policy(epsilon, en, q_values, observ):
    if th.rand(1) < epsilon:
        return en.action_space.sample()
    else:
        return int(th.argmax(q_values[observ]))


def testing(env, policy, n_test):
    rewards = []
    env.action_space.seed(random.randint(0, sys.maxsize))
    percentage_win = 0
    for _ in tqdm(range(n_test), file=sys.stdout):
        total_episode_reward = 0
        s, _ = env.reset(seed=random.randint(0, sys.maxsize))
        a = policy[s]
        done = False
        reward = 0
        while not done:
            next_s, reward, terminated, truncated, _ = env.step(a)
            total_episode_reward += reward
            done = terminated or truncated
            # also take the best action until the episode ends.
            a = policy[next_s]
        # if at the end of the episode we have reward = 1, that means that we win!
        if reward == 1:
            percentage_win += 1
        rewards.append(reward)
    percentage_win /= n_test
    return percentage_win * 100, rewards


def get_q_values_policy(q):
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in q.items():
        state_value[obs] = float(th.max(action_values))
        policy[obs] = int(th.argmax(action_values))
    return state_value, policy


def testing_random(env, n_test):
    rewards = []
    env.action_space.seed(random.randint(0, sys.maxsize))
    percentage_win = 0
    for _ in tqdm(range(n_test), file=sys.stdout):
        total_episode_reward = 0
        s, _ = env.reset(seed=random.randint(0, sys.maxsize))
        # take random action
        a = random.randint(0, 1)
        done = False
        reward = 0
        while not done:
            next_s, reward, terminated, truncated, _ = env.step(a)
            total_episode_reward += reward
            done = terminated or truncated
            # take random action
            a = random.randint(0, 1)
        # if at the end of the episode we have reward = 1, that means that we win!
        if reward == 1:
            percentage_win += 1
        rewards.append(total_episode_reward)
    percentage_win /= n_test
    return percentage_win * 100, rewards


def create_epsilon_dict(init_epsilon, decay_epsilon, end_epsilon):
    return {'epsilon': init_epsilon, 'init': init_epsilon, 'decay': decay_epsilon, 'end': end_epsilon,
            'step': 0}


def mc_td_evaluation(env):
    n_episodes = 1_000_000
    n_test = 100_000
    init_epsilon = 1
    decay_epsilon = init_epsilon / (n_episodes / 2)
    end_epsilon = 0.1
    gamma = 0.95
    alpha = 0.01

    epsilon_dict = create_epsilon_dict(init_epsilon, decay_epsilon, end_epsilon)
    epsilon_decay = partial(linear_epsilon_decay, epsilon_dict)
    q_mc, rewards_mc, episode_mc = montecarlo(env, alpha, epsilon_dict, epsilon_decay, gamma, n_episodes)
    states_mc, policy_mc = get_q_values_policy(q_mc)
    show(state_value=states_mc, policy=policy_mc, title="MC")
    view_training(rewards_mc, episode_mc, "MC")

    epsilon_dict = create_epsilon_dict(init_epsilon, decay_epsilon, end_epsilon)
    epsilon_decay = partial(linear_epsilon_decay, epsilon_dict)
    q_td, rewards_td, episode_td = t_difference(env, alpha, epsilon_dict, epsilon_decay, gamma, n_episodes)
    states_td, policy_td = get_q_values_policy(q_td)
    show(state_value=states_td, policy=policy_td, title="TD")
    view_training(rewards_td, episode_td, "TD")

    # testing

    win_mc_perc, rewards_mc_test = testing(env, policy_mc, n_test)
    print(''.join([str(win_mc_perc), '% of wins for MC agent']))
    view_rewards_test(rewards_mc_test, "MC")

    win_td_perc, rewards_td_test = testing(env, policy_td, n_test)
    print(''.join([str(win_td_perc), '% of wins for TD agent']))
    view_rewards_test(rewards_td_test, "TD")

    win_random_perc, rewards_random_test = testing_random(env, n_test)
    print(''.join([str(win_random_perc), '% of wins for random agent']))
    view_rewards_test(rewards_random_test, "random")


def get_best_config(env):
    n_episodes = 1_000_000
    n_test = 100_000
    init_epsilon = 1
    decay_epsilon = init_epsilon / (n_episodes / 2)
    end_epsilon = 0.1
    decay_epsilons = ["linear", "exp"]
    alphas = np.linspace(0.001, 0.01, num=3)
    gammas = np.linspace(0, 1, num=3)
    win_mc_max = 0.0
    win_td_max = 0.0
    best_mc = {}
    best_td = {}
    best_mc_params = {}
    best_td_params = {}
    for alpha, gamma, decay in tqdm(itertools.product(alphas, gammas, decay_epsilons), file=sys.stdout,
                                    total=len(alphas) * len(gammas) * len(decay_epsilons)):

        decay_epsilon_fun, epsilon_dict = create_decay_params(decay_epsilon, end_epsilon, init_epsilon)
        q_mc, rewards_mc, episode_mc = montecarlo(env, alpha, epsilon_dict, decay_epsilon_fun[decay], gamma, n_episodes)
        states_mc, policy_mc = get_q_values_policy(q_mc)
        win_mc_perc, rewards_mc_test = testing(env, policy_mc, n_test)
        decay_epsilon_fun, epsilon_dict = create_decay_params(decay_epsilon, end_epsilon, init_epsilon)
        q_td, rewards_td, episode_td = t_difference(env, alpha, epsilon_dict, decay_epsilon_fun[decay], gamma,
                                                    n_episodes)
        states_td, policy_td = get_q_values_policy(q_td)
        win_td_perc, rewards_td_test = testing(env, policy_td, n_test)
        if win_mc_perc > win_mc_max:
            win_mc_max = win_mc_perc
            best_mc_params = {'alpha': alpha, 'gamma': gamma, 'decay': decay}
            best_mc['state'] = states_mc
            best_mc['policy'] = policy_mc
            best_mc['reward'] = rewards_mc
            best_mc['reward_testing'] = rewards_mc_test
            best_mc['episode_length'] = episode_mc
        if win_td_perc > win_td_max:
            win_td_max = win_td_perc
            best_td_params = {'alpha': alpha, 'gamma': gamma, 'decay': decay}
            best_td['state'] = states_td
            best_td['policy'] = policy_td
            best_td['reward'] = rewards_td
            best_td['reward_testing'] = rewards_td_test
            best_td['episode_length'] = episode_td
    show(best_mc['state'], best_mc['policy'], "MC")
    view_training(best_mc['reward'], best_mc['episode_length'], "MC")
    view_rewards_test(best_mc['reward_testing'], "MC")
    show(best_td['state'], best_td['policy'], "TD")
    view_training(best_td['reward'], best_td['episode_length'], "TD")
    view_rewards_test(best_td['reward_testing'], "TD")
    return win_mc_max, best_mc_params, win_td_max, best_td_params


# set the reproducibility for environment
seed = 101
random.seed(seed)
th.manual_seed(random.randint(0, sys.maxsize))
# create the environment of blackjack
environment = gym.make('Blackjack-v1', natural=False, sab=False)
best_config = str(input("Do you want the best config? 1.yes, Any key. if you want to evaluate a single scenario:"))
if best_config == "1":
    win_mc, mc_params, win_td, td_params = get_best_config(environment)
    print(''.join(["Monte Carlo wins ", str(win_mc), "% of times, with this hyper-param:", str(mc_params)]))
    print(''.join(["Temporal Difference wins ", str(win_td), "% of times, with this hyper-param:", str(td_params)]))
else:
    print("Evaluate a single scenario...")
    mc_td_evaluation(environment)
