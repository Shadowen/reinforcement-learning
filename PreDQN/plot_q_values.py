import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PreDQN.nonlinear_estimator import NonlinearEstimator
from PreDQN.util import BASE_DATA_DIR

env = gym.envs.make("CartPole-v1")

with tf.Session():
    q_estimator = NonlinearEstimator('q_estimator', env)
    latest_checkpoint = tf.train.latest_checkpoint(BASE_DATA_DIR + "/Cartpole Double Q-Learning/q_estimator")
    print(latest_checkpoint)
    q_estimator.load(BASE_DATA_DIR + "/Cartpole Double Q-Learning/q_estimator")

    batch_size = 100
    variable = np.arange(*env.observation_space.shape[0], batch_size)
    states = np.zeros([batch_size, 4])
    states[:, 0] = variable
    q_values = q_estimator.predict_batch(states)

    plt.show(q_values)
