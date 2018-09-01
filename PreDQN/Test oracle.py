import gym
import matplotlib.style
import numpy as np

from PreDQN.oracle_estimator import OracleEstimator
from PreDQN.util import *
from lib.plotting import EpisodeStats

matplotlib.style.use('ggplot')

if __name__ == '__main__':
    save_directory = get_or_make_data_dir('q_estimator')

    env = gym.envs.make('MountainCar-v0')

    oracle_estimator = OracleEstimator()

    num_episodes = 100
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    for i in range(num_episodes):
        t, total_reward = run_episode(env, oracle_estimator, render=False)
        stats.episode_lengths[i] = t
        stats.episode_rewards[i] = total_reward
    log_episode_stats(get_empty_data_file("stats.csv"), stats)

    oracle_estimator.save(save_directory)
