import gym
import matplotlib.style

from PreDQN.oracle_estimator import OracleEstimator
from PreDQN.util import *

matplotlib.style.use('ggplot')

if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")

    oracle_estimator = OracleEstimator()

    while True:
        run_episode(env, oracle_estimator, render=True)
