# Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# Originally a Policy Gradient algorithm, upgraded to an Actor-Critic algorithm.

import tensorflow as tf
import numpy as np
import gym
from MyDQN.CircularBufferReplayMemory import CircularBufferReplayMemory
import collections
import os
import shutil

env = gym.make('CartPole-v1')

gamma = 1.0


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class PolicyEstimator():
    def __init__(self, learning_rate, s_size, a_size, scope='policy_estimator'):
        with tf.variable_scope(scope):
            # These lines established the feed-forward part of the network. The agent takes a state and produces an
            # action.
            self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            fc_1 = tf.layers.dense(inputs=self.state_in, units=8, activation=tf.nn.relu)
            self.output = tf.layers.dense(inputs=fc_1, units=a_size, activation=tf.nn.softmax)

            # The next six lines establish the training procedure. We feed the reward and chosen action into the network
            # to compute the loss, and use it to update the network.
            self.advantage_pl = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_pl = tf.placeholder(shape=[None], dtype=tf.int32)

            indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_pl
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), indexes)

            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.advantage_pl)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.gradients, self.trainable_variables = zip(*self.optimizer.compute_gradients(self.loss))
            self.clipped_gradients, self.global_norm = tf.clip_by_global_norm(self.gradients, clip_norm=1.0)
            self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.trainable_variables),
                global_step=tf.contrib.framework.get_global_step())

            self._create_summaries()

    def _create_summaries(self):
        # self.action_probs_summary = tf.summary.histogram('actions_probs', self.action_probs)
        self.max_gradient_norm_op = tf.reduce_max([tf.reduce_max(g) for g in self.clipped_gradients])
        self.max_gradient_norm_summary_op = tf.summary.scalar('maxgradient_norm', self.max_gradient_norm_op)

        self.global_norm_summary_op = tf.summary.scalar('global_norm', self.global_norm)

        self.training_summaries = tf.summary.merge([self.max_gradient_norm_summary_op, self.global_norm_summary_op])

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state_in: state})

    def update(self, state, target, action, sess=None, summary_writer=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state_in: state, self.advantage_pl: target, self.action_pl: action}
        global_step, _, summaries = sess.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.training_summaries], feed_dict)

        if summary_writer:
            summary_writer.add_summary(summaries, global_step)


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=1E-2, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="state")
            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")

            self.fc1 = tf.layers.dense(inputs=self.state, units=32, activation=tf.nn.relu)
            self.output_layer = tf.layers.dense(inputs=self.fc1, units=1, activation=None)

            self.value_estimate = tf.squeeze(self.output_layer, axis=1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.target))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

            self._create_summaries()

    def _create_summaries(self):
        self.value_summary = tf.summary.scalar('average_value_estimate', tf.reduce_mean(self.value_estimate))
        self.loss_summary = tf.summary.scalar('loss', self.loss)

    def predict(self, state, sess=None, summary_writer=None):
        sess = sess or tf.get_default_session()
        value_estimate, value_summary, global_step = sess.run(
            [self.value_estimate, self.value_summary, tf.contrib.framework.get_global_step()], {self.state: state})
        if summary_writer is not None:
            summary_writer.add_summary(value_summary, global_step)
        return value_estimate

    def update(self, state, target, sess=None, summary_writer=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss, loss_summary, global_step = sess.run(
            [self.train_op, self.loss, self.loss_summary, tf.contrib.framework.get_global_step()], feed_dict)
        if summary_writer is not None:
            summary_writer.add_summary(loss_summary, global_step)


tf.reset_default_graph()  # Clear the Tensorflow graph.

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(learning_rate=1e-2, s_size=4, a_size=2)  # Load the agent.
value_estimator = ValueEstimator(learning_rate=1e-2)
logdir = 'basic_policy_gradient_clipped_gradient'
if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.mkdir(logdir)
summary_writer = tf.summary.FileWriter(logdir)

reward_placeholder = tf.placeholder(dtype=tf.float32, name='reward')
reward_summary_op = tf.summary.scalar('reward', reward_placeholder)


def summarize_reward(reward, sess, summary_writer):
    reward_summary, global_step = sess.run([reward_summary_op, tf.contrib.framework.get_global_step()],
        feed_dict={reward_placeholder: reward})
    summary_writer.add_summary(reward_summary, global_step)


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
replay_memory = CircularBufferReplayMemory(10000)

total_episodes = 5000  # Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_reward = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    for i in range(total_episodes):
        state = env.reset()
        total_episode_reward = 0
        value_estimator.predict(state=np.expand_dims(state, axis=0), sess=sess, summary_writer=summary_writer)
        for j in range(max_ep):
            # Choose either a random action or one from our network.
            a_dist = sess.run(policy_estimator.output, feed_dict={policy_estimator.state_in: [state]})[0]
            action = np.random.choice(np.arange(len(a_dist)), p=a_dist)

            next_state, reward, done, _ = env.step(action)  # Get our reward for taking an action given a bandit.
            replay_memory.append(Transition(state, action, reward, next_state, done))
            state = next_state
            total_episode_reward += reward

            if done:
                states, actions, rewards, next_states, done = zip(*replay_memory.sample(500))
                # Update the value network
                value_predictions = value_estimator.predict(state=next_states, sess=sess)
                td_targets = rewards + np.logical_not(done).astype(np.int8) * gamma * value_predictions
                value_estimator.update(state=states, target=td_targets, sess=sess)
                # Update the network.
                feed_dict = {
                    policy_estimator.advantage_pl: td_targets, policy_estimator.action_pl: actions,
                    policy_estimator.state_in: states
                }
                policy_estimator.update(states, td_targets, actions, sess, summary_writer)

                summarize_reward(total_episode_reward, sess, summary_writer)
                total_reward.append(total_episode_reward)
                break

        # Update our running tally of scores.
        if i % 100 == 0:
            print(
                "episode={}\tglobal_step={}\tavg_reward={}".format(i, sess.run(tf.contrib.framework.get_global_step()),
                    np.mean(total_reward[-100:])))
