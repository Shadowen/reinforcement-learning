import itertools
import random
import sys
from collections import deque
from collections import namedtuple

import gym
import matplotlib
import numpy as np
import tensorflow as tf

from PreDQN.batch_linear_estimator import BatchLinearEstimator
from PreDQN.util import *
from lib import plotting

matplotlib.style.use('ggplot')


def q_learning(env, q_estimator, target_estimator, num_episodes,
               update_target_estimator_every, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
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

    batch_size = 32
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    # Create a replay memory buffer.
    replay_memory = deque(maxlen=10000)
    # Fill replay buffer.
    state = env.reset()
    for i in itertools.count():
        if i % 100 == 0:
            print("Filling replay buffer... " + str(i) + "/" + str(replay_memory.maxlen), end="\r")
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        # Record the transition.
        replay_memory.append(Transition(state, action, reward, next_state, done))
        state = next_state
        if done:
            state = env.reset()
        if i >= replay_memory.maxlen:
            break
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    total_t = 0
    for i_episode in range(num_episodes):
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            q_estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print(
            "\rEpisode={}/{}\tTotal timesteps={}\tReward={}".format(i_episode + 1, num_episodes, total_t, last_reward),
            end="")
        sys.stdout.flush()

        # Run an episode.
        state = env.reset()
        for t in itertools.count():
            # Copy target network.
            if total_t % update_target_estimator_every == 0:
                target_estimator.copy_params()
                # print("\nCopied model parameters to target network.")

            # Take action.
            action_probs = policy(state)
            action = np.random.choice(env.action_space.n, p=action_probs)
            next_state, reward, done, info = env.step(action)

            # Record the transition.
            replay_memory.append(Transition(state, action, reward, next_state, done))
            # Record stats.
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            # Calculate q values and targets
            q_values_next = target_estimator.predict_batch(next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(
                q_values_next, axis=1)
            # Update Q function.
            states_batch = np.array(states_batch)
            q_estimator.update(states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_t += 1

    return stats


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")
    with tf.Session() as sess:
        q_estimator = BatchLinearEstimator(scope='q_estimator', env=env)
        target_estimator = BatchLinearEstimator(scope='target_estimator', env=env, copy_from=q_estimator)
        sess.run(tf.global_variables_initializer())

        stats = q_learning(env, q_estimator=q_estimator, target_estimator=target_estimator,
                           update_target_estimator_every=1, num_episodes=500,
                           epsilon=1.0, epsilon_decay=0.99)

        plotting.plot_cost_to_go_mountain_car(env, q_estimator)
        plotting.plot_episode_stats(stats, smoothing_window=25)
