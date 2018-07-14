import itertools

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['make_epsilon_greedy_policy', 'run_episode']


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
