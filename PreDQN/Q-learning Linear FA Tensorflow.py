import itertools
import sys

import gym
import matplotlib
import numpy as np
import tensorflow as tf

from PreDQN.linear_estimator import Estimator
from PreDQN.util import *
from lib import plotting

matplotlib.style.use('ggplot')


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()

        # Run an episode.
        state = env.reset()
        for t in itertools.count():
            # Take action.
            action_probs = policy(state)
            action = np.random.choice(env.action_space.n, p=action_probs)
            next_state, reward, done, info = env.step(action)

            # Record stats.
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # Update Q function.
            td_target = reward + discount_factor * np.max(estimator.predict(next_state))
            estimator.update(state, action, td_target)

            if done:
                break

            state = next_state

    return stats


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")
    with tf.Session() as sess:
        estimator = Estimator(env=env, lr=0.01)

        sess.run(tf.global_variables_initializer())

        # Note: For the Mountain Car we don't actually need an epsilon > 0.0
        # because our initial estimate for all states is too "optimistic" which leads
        # to the exploration of all states.
        stats = q_learning(env, estimator, 100, epsilon=0.0)

        plotting.plot_cost_to_go_mountain_car(env, estimator)
        plotting.plot_episode_stats(stats, smoothing_window=25)
