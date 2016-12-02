import numpy as np
import os
import tensorflow as tf


class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self, sess, env, scope="Estimator", model_builder=None, summaries_dir=None):
        self.sess = sess
        self.env = env
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            (model_builder if model_builder else self._build_model)()
            if summaries_dir:
                if not os.path.exists(summaries_dir):
                    os.makedirs(summaries_dir)
                self.summary_writer = tf.train.SummaryWriter(summaries_dir)

    def _build_model(self):
        # Placeholders for our input
        self.X_pl = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.X_pl)[0]

        hidden_layer = tf.contrib.layers.fully_connected(self.X_pl, 20)
        self.predictions = tf.contrib.layers.fully_connected(hidden_layer, self.env.action_space.n)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.merge_summary([
            tf.scalar_summary("loss", self.loss),
            tf.histogram_summary("loss_hist", self.losses),
            tf.histogram_summary("q_values_hist", self.predictions),
            tf.scalar_summary("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, states):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        if len(states.shape) < len(self.X_pl.get_shape()):
            return self.sess.run(self.predictions, {self.X_pl: np.expand_dims(states, 0)})
        return self.sess.run(self.predictions, {self.X_pl: states})

    def update(self, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = self.sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

    def get_action(self, observation, epsilon=0.0):
        probs = np.ones(self.env.action_space.n, dtype=float) * epsilon / self.env.action_space.n
        q_values = self.predict(np.expand_dims(observation, 0))
        best_action = np.argmax(q_values)
        probs[best_action] += (1.0 - epsilon)
        return np.random.choice(np.arange(len(probs)), p=probs)

    def copy_parameters_from(self, other):
        self_params = sorted((t for t in tf.trainable_variables() if t.name.startswith(self.scope)), key=lambda v: v.name)
        other_params = sorted((t for t in tf.trainable_variables() if t.name.startswith(other.scope)), key=lambda v: v.name)

        update_ops = []
        for self_v, other_v in zip(self_params, other_params):
            op = self_v.assign(other_v)
            update_ops.append(op)

        self.sess.run(update_ops)
