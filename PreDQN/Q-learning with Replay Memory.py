import itertools
import random
import sys
from collections import deque
from collections import namedtuple

import gym
import matplotlib
import numpy as np
import tensorflow as tf

from PreDQN.batch_linear_estimator import Estimator
from lib import plotting

matplotlib.style.use('ggplot')


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
        q_values = estimator.predict([observation])[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


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

            # Record the transition.
            replay_memory.append(Transition(state, action, reward, next_state, done))
            # Record stats.
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            # Calculate q values and targets
            q_values_next = estimator.predict(next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(
                q_values_next, axis=1)
            # Update Q function.
            states_batch = np.array(states_batch)
            estimator.update(states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state

    return stats


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    import matplotlib
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict([_])[0]), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")
    with tf.Session() as sess:
        estimator = Estimator(env=env, lr=0.01)

        sess.run(tf.global_variables_initializer())

        # Note: For the Mountain Car we don't actually need an epsilon > 0.0
        # because our initial estimate for all states is too "optimistic" which leads
        # to the exploration of all states.
        stats = q_learning(env, estimator, num_episodes=500, epsilon=1.0, epsilon_decay=0.99)

        plot_cost_to_go_mountain_car(env, estimator)
        plotting.plot_episode_stats(stats, smoothing_window=25)
