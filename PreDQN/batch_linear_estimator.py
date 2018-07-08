import gym
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler

from PreDQN.estimator import Estimator


class BatchLinearEstimator(Estimator):
    def __init__(self, scope: str, env: gym.Env, copy_from: 'BatchLinearEstimator' = None):
        super().__init__()
        self.scope = scope

        if copy_from is None:
            self._init_feature_preprocessor(env)
        else:
            self.scaler = copy_from.scaler
            self.featurizer = copy_from.featurizer
        with tf.variable_scope(self.scope):
            self._init_model(env)

        if copy_from is not None:
            # Set up assignment operations for copying parameters.
            my_params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
            my_params = sorted(my_params, key=lambda v: v.name)
            their_params = [t for t in tf.trainable_variables() if t.name.startswith(copy_from.scope)]
            their_params = sorted(their_params, key=lambda v: v.name)

            self.copy_ops = []
            for my_v, thier_v in zip(my_params, their_params):
                op = tf.assign(my_v, thier_v)
                self.copy_ops.append(op)

    def _init_feature_preprocessor(self, env):
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

    def _init_model(self, env):
        # Build model.
        self.ob_pl = tf.placeholder(dtype=tf.float32, shape=[None, 400], name="observation")
        self.prediction = tf.layers.dense(inputs=self.ob_pl, units=env.action_space.n,
                                          kernel_initializer=tf.zeros_initializer(),
                                          bias_initializer=tf.zeros_initializer())
        self.action_pl = tf.placeholder(dtype=tf.uint8, shape=[None], name="action")
        action_one_hot = tf.one_hot(self.action_pl, depth=env.action_space.n)
        self.target_pl = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
        tiled_target = tf.tile(tf.expand_dims(self.target_pl, axis=1), multiples=(1, env.action_space.n))
        self.loss = tf.reduce_sum((action_one_hot * (tiled_target - self.prediction)) ** 2)

        # Set up optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.minimize = optimizer.minimize(self.loss)

    def _featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized

    def predict(self, state):
        return self.predict_batch([state])[0]

    def predict_batch(self, batch_state):
        feed = {self.ob_pl: self._featurize_state(batch_state)}
        p = tf.get_default_session().run(self.prediction, feed_dict=feed)
        return p

    def update(self, batch_state, batch_action, batch_target):
        feed = {self.ob_pl: self._featurize_state(batch_state), self.action_pl: batch_action,
                self.target_pl: batch_target}
        tf.get_default_session().run(self.minimize, feed_dict=feed)

    def copy_params(self):
        """Copies parameters from the estimator passed in at initialization."""
        tf.get_default_session().run(self.copy_ops)
