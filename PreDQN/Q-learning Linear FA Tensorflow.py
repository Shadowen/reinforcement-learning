import itertools
import sys

import gym
import matplotlib
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler

from lib import plotting

matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")


class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self, env, lr):
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized representation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        # Build model.
        self.ob_pl = tf.placeholder(dtype=tf.float32, shape=[None, 400],
                                    name="observation")
        self.prediction = tf.layers.dense(inputs=self.ob_pl, units=env.action_space.n,
                                          kernel_initializer=tf.zeros_initializer(),
                                          bias_initializer=tf.zeros_initializer)
        self.action_pl = tf.placeholder(dtype=tf.uint8, shape=[None], name="action")
        self.action_one_hot = tf.one_hot(self.action_pl, depth=env.action_space.n)
        self.target_pl = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
        self.loss = tf.reduce_sum((self.action_one_hot * (self.target_pl - self.prediction)) ** 2)

        # Set up optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.minimize = optimizer.minimize(self.loss)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        if a:
            raise NotImplementedError()
        else:
            feed = {self.ob_pl: [self.featurize_state(s)]}
            p = tf.get_default_session().run(self.prediction, feed_dict=feed)
            return p[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        feed = {self.ob_pl: [self.featurize_state(s)], self.action_pl: [a], self.target_pl: [y]}
        tf.get_default_session().run(self.minimize, feed_dict=feed)


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


with tf.Session() as sess:
    estimator = Estimator(env=env, lr=0.01)

    sess.run(tf.global_variables_initializer())

    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    stats = q_learning(env, estimator, 100, epsilon=0.0)

    plotting.plot_cost_to_go_mountain_car(env, estimator)
    plotting.plot_episode_stats(stats, smoothing_window=25)
