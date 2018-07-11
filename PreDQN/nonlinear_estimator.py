import gym
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler

from PreDQN.estimator import Estimator


class NonlinearEstimator(Estimator):
    def __init__(self, scope: str, env: gym.Env, copy_from: 'NonlinearEstimator' = None):
        super().__init__()
        self.scope = scope

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

    def _init_model(self, env):
        # Build model.
        self.ob_pl = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="observation")
        h1 = tf.layers.dense(inputs=self.ob_pl, units=16, activation=tf.sigmoid)
        h2 = tf.layers.dense(inputs=h1, units=32, activation=tf.sigmoid)
        self.prediction = tf.layers.dense(inputs=h2, units=env.action_space.n)

        # Set up loss function.
        self.action_pl = tf.placeholder(dtype=tf.uint8, shape=[None], name="action")
        action_one_hot = tf.one_hot(self.action_pl, depth=env.action_space.n)
        self.target_pl = tf.placeholder(dtype=tf.float32, shape=[None], name="target")
        tiled_target = tf.tile(tf.expand_dims(self.target_pl, axis=1), multiples=(1, env.action_space.n))
        self.loss = tf.reduce_sum((action_one_hot * (tiled_target - self.prediction)) ** 2)

        # Set up optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        self.minimize = optimizer.minimize(self.loss)

    def predict(self, state):
        return self.predict_batch([state])[0]

    def predict_batch(self, batch_state):
        feed = {self.ob_pl: batch_state}
        p = tf.get_default_session().run(self.prediction, feed_dict=feed)
        return p

    def update(self, batch_state, batch_action, batch_target):
        feed = {self.ob_pl: batch_state, self.action_pl: batch_action,self.target_pl: batch_target}
        tf.get_default_session().run(self.minimize, feed_dict=feed)

    def copy_params(self):
        """Copies parameters from the estimator passed in at initialization."""
        tf.get_default_session().run(self.copy_ops)
