import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler

from PreDQN.estimator import Estimator


class LinearEstimator(Estimator):
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
                                          bias_initializer=tf.zeros_initializer())
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

    def predict(self, state):
        feed = {self.ob_pl: [self.featurize_state(state)]}
        p = tf.get_default_session().run(self.prediction, feed_dict=feed)
        return p[0]

    def predict_batch(self, state):
        raise NotImplementedError()

    def update(self, s, a, y):
        feed = {self.ob_pl: [self.featurize_state(s)], self.action_pl: [a], self.target_pl: [y]}
        tf.get_default_session().run(self.minimize, feed_dict=feed)
