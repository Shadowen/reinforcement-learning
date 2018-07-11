import itertools

import gym
import matplotlib.style
import matplotlib.pyplot as plt
import numpy as np

from PreDQN.oracle_estimator import OracleEstimator
from PreDQN.util import *

matplotlib.style.use('ggplot')


def run_episode(env, q_estimator):
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
        env.render(mode='human')
        plt.pause(0.1)

        if done:
            break

        state = next_state

    print("Reward: {}".format(total_reward))


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")

    oracle_estimator = OracleEstimator()

    while True:
        run_episode(env, oracle_estimator)
