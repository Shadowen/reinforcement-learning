import gym

env = gym.envs.make("CartPole-v0")

from MyDQN.Estimator import *
import numpy as np
import sys
from collections import namedtuple
import random
import os
import datetime


def q_learning(env, q_estimator, target_estimator, num_episodes=10000, discount_factor=0.99, epsilon_steps=50000,
               replay_memory_size=50000, batch_size=32, copy_params_every=10000):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        q_estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The epsilon decay schedule
    epsilon_schedule = np.linspace(1, 0.1, epsilon_steps)
    global_step_num = sess.run(tf.contrib.framework.get_global_step())

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    # Populate the replay memory with initial experience
    state = env.reset()
    for i in range(replay_memory_size):
        if i % 100 == 0:
            print("\rPopulating replay memory {}/{}".format(i, replay_memory_size), end="")
        action = q_estimator.get_action(state, epsilon=1.0)
        next_state, reward, done, _ = env.step(action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            # print("Found the end in replay memory population!!")
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
            if global_step_num % copy_params_every == 0:
                print("Copied parameters!")
                target_estimator.copy_parameters_from(q_estimator)

            # Take a step
            epsilon = epsilon_schedule[min(global_step_num, epsilon_steps - 1)]
            action = q_estimator.get_action(state, epsilon=epsilon)
            action_counter[action] = action_counter[action] + 1 if action in action_counter else 1
            next_state, reward, done, _ = env.step(action)
            # Add to replay memory
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # Update
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            q_values_next = target_estimator.predict(next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(
                q_values_next, axis=1)
            loss = q_estimator.update(states_batch, action_batch, targets_batch)

            state = next_state
            total_reward += reward
            step_num += 1
            global_step_num += 1
            # if step_num % 100 == 0:
            print("\rStep_num={}, {} = {}".format(step_num, action_counter, loss), end="")

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, node_name="epsilon", tag="epsilon")
        episode_summary.value.add(simple_value=total_reward, node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=step_num, node_name="episode_length", tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, global_step_num)
        q_estimator.summary_writer.flush()
        yield episode_num, step_num, total_reward, loss


with tf.Session() as sess:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    estimator = Estimator(sess, env, scope="q_estimator", summaries_dir=os.path.abspath("./experiments/CartPole_dqn.{}".format(str(datetime.datetime.now()))))
    target_estimator = Estimator(sess, env, scope="target_estimator")
    tf.global_variables_initializer().run()

    for episode_num, steps, reward, loss in q_learning(env, estimator, target_estimator):
        # Print out which episode we're on, useful for debugging.
        print("Episode {} ({}) = {} ({})".format(episode_num + 1, steps, reward, loss), end="\n")
        sys.stdout.flush()
