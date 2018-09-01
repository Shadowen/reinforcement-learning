import csv
import itertools
import os

import __main__ as main
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['get_or_make_data_dir',
           'get_empty_data_file',
           'make_epsilon_greedy_policy',
           'run_episode',
           'log_episode_stats']

BASE_DATA_DIR = '/home/wesley/data/reinforcement-learning'


def get_or_make_data_dir(subdir: str = None) -> str:
    """
    Gets or creates a data path for the current experiment.
    :param subdir: if specified, creates a subdirectory within the data folder.
    """
    # Get the main file's name (without the extension).
    data_dir = os.path.join(BASE_DATA_DIR, os.path.basename(main.__file__).rsplit('.', 1)[0])
    if subdir is not None:
        data_dir = os.path.join(data_dir, subdir)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir + '/'


def get_empty_data_file(name: str) -> str:
    data_dir = get_or_make_data_dir()
    data_path = os.path.join(data_dir, name)
    return data_path


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def run_episode(env, q_estimator, render=True):
    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator, 0, env.action_space.n)

    # Run an episode.
    state = env.reset()
    total_reward = 0
    for t in itertools.count():
        # Take action.
        action_probs = policy(state)
        action = np.random.choice(env.action_space.n, p=action_probs)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

        if render:
            env.render(mode='human')
            plt.pause(0.05)
        state = next_state

    print("Reward: {}".format(total_reward))
    return t, total_reward


def log_episode_stats(destination, stats):
    with open(destination, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['episode_num'] + list(stats._fields))
        data = [d for d in stats._asdict().values()]
        episode_nums = np.arange(1, len(data[0]) + 1)
        writer.writerows(np.stack([episode_nums] + data).T)
