# Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# Originally a Policy Gradient algorithm, upgraded to an Actor-Critic algorithm.

import gym
import tensorflow as tf
import numpy as np
from MyDQN.CircularBufferReplayMemory import CircularBufferReplayMemory
import os
import shutil
import itertools
from functools import reduce
import operator
from collections import namedtuple, deque


class PolicyEstimator():
    def __init__(self, learning_rate, state_size, action_size, tf_session, scope='policy_estimator'):
        self.tf_session = tf_session

        with tf.variable_scope(scope):
            self.state_pl = tf.placeholder(shape=[None, reduce(operator.mul, state_size)], dtype=tf.float32,
                name='state')
            fc_1 = tf.layers.dense(inputs=self.state_pl, units=8, activation=tf.nn.relu)
            self.output_op = tf.layers.dense(inputs=fc_1, units=action_size, activation=None)
            self.softmax = tf.nn.softmax(self.output_op)

            self.advantage_pl = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_pl = tf.placeholder(shape=[None], dtype=tf.int32)

            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_pl,
                logits=self.output_op) * self.advantage_pl)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.gradients_ops, self.trainable_variables = zip(*self.optimizer.compute_gradients(self.loss_op))
            self.clipped_gradients_ops, self.gradient_global_norms_op = tf.clip_by_global_norm(self.gradients_ops,
                clip_norm=1.0)
            self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients_ops, self.trainable_variables),
                global_step=tf.contrib.framework.get_global_step())

            self.check_ops = [tf.check_numerics(o, 'Error') for o in
                              list(self.gradients_ops) + list(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))]

            self._create_summaries()

    def _create_summaries(self):
        self.action_probs_summary = tf.summary.histogram('actions_probs', self.softmax)

        self.max_gradient_norm_op = tf.reduce_max([tf.reduce_max(g) for g in self.gradients_ops])
        self.max_gradient_norm_summary_op = tf.summary.scalar('max_gradient_norm', self.max_gradient_norm_op)
        self.global_norm_summary_op = tf.summary.scalar('gradient_global_norm', self.gradient_global_norms_op)
        self.loss_summary_op = tf.summary.scalar('loss', self.loss_op)

        self.training_summaries_op = tf.summary.merge(
            [self.max_gradient_norm_summary_op, self.global_norm_summary_op, self.loss_summary_op])

    def predict(self, state):
        a_dist = self.tf_session.run(self.softmax, feed_dict={self.state_pl: [state]})[0]
        action = np.random.choice(np.arange(len(a_dist)), p=a_dist)
        return action

    def update(self, state, target, action, summary_writer=None, check_numerics=False):
        feed_dict = {self.state_pl: state, self.advantage_pl: target, self.action_pl: action}
        global_step, _, summaries, *_ = self.tf_session.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.training_summaries_op] + (
                self.check_ops if check_numerics else []), feed_dict=feed_dict)

        if summary_writer:
            summary_writer.add_summary(summaries, global_step)


class ValueEstimator():
    def __init__(self, learning_rate, state_size, tf_session, scope="value_estimator"):
        self.tf_session = tf_session

        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, reduce(operator.mul, state_size)], name="state")
            self.target = tf.placeholder(dtype=tf.float32, shape=[None], name="target")

            fc_1 = tf.layers.dense(inputs=self.state, units=8, activation=tf.nn.relu)
            self.output_layer = tf.squeeze(tf.layers.dense(inputs=fc_1, units=1, activation=None), axis=1)

            self.loss_op = tf.reduce_mean(tf.squared_difference(self.output_layer, self.target))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.gradients_ops, self.trainable_variables = zip(*self.optimizer.compute_gradients(self.loss_op))
            self.clipped_gradients_ops, self.gradient_global_norms_op = tf.clip_by_global_norm(self.gradients_ops,
                clip_norm=1.0)
            self.train_op = self.optimizer.minimize(self.loss_op, global_step=tf.contrib.framework.get_global_step())

            self._create_summaries()

    def _create_summaries(self):
        self.value_summary = tf.summary.scalar('initial_value_estimate', tf.reduce_mean(self.output_layer))

        self.loss_summary = tf.summary.scalar('loss', self.loss_op)
        self.global_norm_summary_op = tf.summary.scalar('gradient_global_norm', self.gradient_global_norms_op)

        self.train_summaries = tf.summary.merge([self.loss_summary, self.global_norm_summary_op])

    def predict(self, state, summary_writer=None):
        value_estimate, value_summary, global_step = self.tf_session.run(
            [self.output_layer, self.value_summary, tf.contrib.framework.get_global_step()], {self.state: state})
        if summary_writer is not None:
            summary_writer.add_summary(value_summary, global_step)
        return value_estimate

    def update(self, state, target, summary_writer=None):
        global_step, _, loss, summaries = self.tf_session.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.loss_op, self.train_summaries],
            feed_dict={self.state: state, self.target: target})
        if summary_writer is not None:
            summary_writer.add_summary(summaries, global_step)


tf.reset_default_graph()  # Clear the Tensorflow graph.


def summarize_episode(episode, sess, summary_writer):
    if summarize_episode.__dict__.get('episode_placeholder', None) is None:
        summarize_episode.episode_placeholder = tf.placeholder(dtype=tf.int16, name='episode')
        summarize_episode.episode_summary_op = tf.summary.scalar('episode', summarize_episode.episode_placeholder)

    episode_summary, global_step = sess.run(
        [summarize_episode.episode_summary_op, tf.contrib.framework.get_global_step()],
        feed_dict={summarize_episode.episode_placeholder: episode})
    summary_writer.add_summary(episode_summary, global_step)


def summarize_reward(reward, sess, summary_writer):
    if summarize_reward.__dict__.get('reward_placeholder', None) is None:
        summarize_reward.reward_placeholder = tf.placeholder(dtype=tf.float32, name='reward')
        summarize_reward.reward_summary_op = tf.summary.scalar('reward', summarize_reward.reward_placeholder)

    reward_summary, global_step = sess.run([summarize_reward.reward_summary_op, tf.contrib.framework.get_global_step()],
        feed_dict={summarize_reward.reward_placeholder: reward})
    summary_writer.add_summary(reward_summary, global_step)


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
replay_memory = CircularBufferReplayMemory(10000000)

gamma = 0.99  # The discount rate
total_episodes = None
max_timesteps_per_episode = 1000
train_every_n_steps = 5

env = gym.make('Acrobot-v1')

logdir = 'basic_policy_gradient_2'
if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.mkdir(logdir)
summary_writer = tf.summary.FileWriter(logdir)

with tf.Session() as sess:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator(learning_rate=1e-3, state_size=env.observation_space.shape,
        action_size=env.action_space.n, tf_session=sess)
    value_estimator = ValueEstimator(learning_rate=1e-2, state_size=env.observation_space.shape, tf_session=sess)

    summary_writer.add_graph(tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    reward_history = deque(maxlen=100)

    for episode_num in range(total_episodes) if total_episodes is not None else itertools.count():
        state = env.reset()
        value_estimator.predict([state], summary_writer=summary_writer)
        total_episode_reward = 0
        for timestep in range(
                max_timesteps_per_episode) if max_timesteps_per_episode is not None else itertools.count():
            # Get an action from the policy estimator
            action = policy_estimator.predict(state)
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            total_episode_reward += reward
            # Save the transition in the replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))
            state = next_state
            print('\rtimestep={}'.format(timestep), end='')

            if timestep % train_every_n_steps == 0 and episode_num > 100:
                states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = zip(
                    *replay_memory.sample(100))
                # Update the value network
                value_current_state = value_estimator.predict(states_batch)
                value_next_state = value_estimator.predict(next_states_batch)
                td_targets = rewards_batch + np.logical_not(done_batch).astype(np.int8) * gamma * value_next_state
                value_estimator.update(states_batch, td_targets, summary_writer=summary_writer)
                # Update the policy network
                td_error = td_targets - value_current_state
                policy_estimator.update(states_batch, td_error, actions_batch, summary_writer=summary_writer)

            if done:
                break

        summarize_reward(total_episode_reward, sess, summary_writer=summary_writer)
        summarize_episode(episode_num, sess, summary_writer=summary_writer)
        reward_history.append(total_episode_reward)

        # Update our running tally of scores.
        if episode_num % 10 == 0:
            print("\repisode={}\tglobal_step={}\tavg_reward={}".format(episode_num,
                sess.run(tf.contrib.framework.get_global_step()), np.mean(reward_history)))
