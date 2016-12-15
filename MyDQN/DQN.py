import gym
from MyDQN.Estimator import *
from MyDQN.CircularBufferReplayMemory import *
from MyDQN.SetupLogger import *
import numpy as np
from collections import namedtuple
import os
import datetime


def q_learning(env, q_estimator, target_estimator, num_episodes, discount_factor, epsilon_steps, replay_memory,
        copy_params_every, get_td_target):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        q_estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon_steps: The number of steps to decay epsilon over
        get_td_target: A static function from TDTarget; either dqn or double_dqn

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The epsilon decay schedule
    epsilon_schedule = np.linspace(1, 0.1, epsilon_steps)
    global_step_num = sess.run(tf.contrib.framework.get_global_step())

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # Populate the replay memory with initial experience
    state = env.reset()
    for i in range(replay_memory.max_length):
        if i % 100 == 0:
            print("\rPopulating replay memory {}/{}".format(i, replay_memory.max_length), end="")
        action = q_estimator.get_action(state, epsilon=1.0)
        next_state, reward, done, _ = env.step(action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
        else:
            state = next_state
    print("\rReplay memory full!")

    for episode_num in range(num_episodes):
        # Play one episode
        state = env.reset()
        done = False
        step_num = 0
        total_reward = 0
        action_counter = {}
        while not done:
            # Copy parameters from Q estimator to target estimator every once in a while
            if global_step_num % copy_params_every == 0:
                target_estimator.copy_parameters_from(q_estimator)

            # Take a step
            epsilon = epsilon_schedule[min(global_step_num, epsilon_steps - 1)]
            action = q_estimator.get_action(state, epsilon=epsilon)
            action_counter[action] = action_counter[action] + 1 if action in action_counter else 1
            next_state, reward, done, _ = env.step(action)
            # Add to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # Sample from the replay memory
            samples = replay_memory.sample(32)
            # Update the Q estimator using a TD-target
            states_batch, action_batch, targets_batch = get_td_target(samples, q_estimator, target_estimator,
                discount_factor)
            loss = q_estimator.update(states_batch, action_batch, targets_batch)

            # Some bookkeeping
            state = next_state
            total_reward += reward
            step_num += 1
            global_step_num += 1
            if step_num % 100 == 0:
                print("\rStep_num={}, {} = {}".format(step_num, action_counter, loss), end="")

        # Add summaries to Tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, node_name="epsilon", tag="epsilon")
        episode_summary.value.add(simple_value=total_reward, node_name="episode_reward", tag="episode_reward")
        # episode_summary.value.add(simple_value=step_num, node_name="episode_length", tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, global_step_num)
        q_estimator.summary_writer.flush()
        yield episode_num, total_reward


class TDTarget():
    @staticmethod
    def dqn(samples_batch, q_estimator, target_estimator, discount_factor):
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples_batch))
        q_values_next = target_estimator.predict(next_states_batch)
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next,
            axis=1)
        return states_batch, action_batch, targets_batch

    @staticmethod
    def double_dqn(samples_batch, q_estimator, target_estimator, discount_factor):
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples_batch))
        q_values_next = q_estimator.predict(next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        q_values_next_target = target_estimator.predict(next_states_batch)
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[
            np.arange(32), best_actions]
        return states_batch, action_batch, targets_batch


with tf.Session() as sess:
    """Set up loggers"""
    # My own logger!
    logger = SetupLogger()
    summaries_dir = os.path.abspath("./experiments/{}".format(str(datetime.datetime.now())))

    """Start the environment"""
    # Gym
    env = logger.logged_call(gym.envs.make, log_type=LogType.environment)(id="CartPole-v0")

    """Create the computation graph"""
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Pick some parameters
    def model_builder(state):
        hidden_layer_1 = tf.contrib.layers.fully_connected(state, 20, activation_fn=tf.nn.relu)
        hidden_layer_2 = tf.contrib.layers.fully_connected(hidden_layer_1, 10, activation_fn=tf.nn.relu)
        q_values = tf.contrib.layers.fully_connected(hidden_layer_2, env.action_space.n, activation_fn=None)
        return q_values


    optimizer = logger.logged_call(tf.train.RMSPropOptimizer, log_type=LogType.optimizer)(learning_rate=0.00025,
        decay=0.99, momentum=0.95, epsilon=0.01)
    estimator = logger.logged_call(Estimator, log_type=LogType.estimator)(sess=sess, env=env, scope="q_estimator",
        model_builder=model_builder, optimizer=optimizer, summaries_dir=summaries_dir)
    target_estimator = Estimator(sess=sess, env=env, model_builder=model_builder, scope="target_estimator")
    replay_memory = logger.logged_call(CircularBufferReplayMemory, log_type=LogType.replay_memory)(max_length=1000000)
    # replay_memory = logger.logged_call(SumTreeReplayMemory, log_type=LogTypes.replay_memory)(max_length=1000000)
    tf.global_variables_initializer().run()

    """Start the algorithm"""
    learning_algorithm = logger.logged_call(q_learning, log_type=LogType.algorithm)(env=env, q_estimator=estimator,
        target_estimator=target_estimator, replay_memory=replay_memory, num_episodes=50000, discount_factor=0.99,
        epsilon_steps=1000000, copy_params_every=10000, get_td_target=TDTarget.double_dqn)

    logger.dump_to_file(summaries_dir + '/configuration.txt')

    # Run the algorithm in a loop
    for episode_num, reward in learning_algorithm:
        print("Episode {} ({})".format(episode_num + 1, reward), end="\n")
