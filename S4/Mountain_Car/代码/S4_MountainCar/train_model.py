import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from dqn_agent import DQNAgent
import numpy as np
import random

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

mpl.use('TkAgg')
# 固定随机数
random.seed(2212)
np.random.seed(2212)
tf.random.set_seed(2212)

# Hyperparameters / Constants
EPISODES = 10000 # 迭代次数
REPLAY_MEMORY_SIZE = 100000
MINIMUM_REPLAY_MEMORY = 1000
MINIBATCH_SIZE = 32
EPSILON = 1
EPSILON_DECAY = 0.99
MINIMUM_EPSILON = 0.001
DISCOUNT = 0.99
VISUALIZATION = False
ENV_NAME = 'MountainCar-v0'

# Environment
env = gym.make(ENV_NAME)
action_dim = env.action_space.n
observation_dim = env.observation_space.shape

sess = tf.compat.v1.Session()

replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

agent = DQNAgent(sess, action_dim, observation_dim)


def train_dqn_agent():
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
    X_cur_states = []
    X_next_states = []
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        X_cur_states.append(cur_state)
        X_next_states.append(next_state)

    X_cur_states = np.array(X_cur_states)
    X_next_states = np.array(X_next_states)

    cur_action_values = agent.model.predict(X_cur_states)

    next_action_values = agent.model.predict(X_next_states)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        if not done:

            cur_action_values[index][action] = reward + DISCOUNT * np.amax(next_action_values[index])
        else:

            cur_action_values[index][action] = reward

    agent.model.fit(X_cur_states, cur_action_values, verbose=0)


max_reward = -999999
reward_set=[]
#训练过程
for episode in range(EPISODES):
    cur_state = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:
        episode_length += 1

        if VISUALIZATION:
            env.render()

        if np.random.uniform(0, 1) < EPSILON:

            action = np.random.randint(0, action_dim)
        else:

            action = np.argmax(agent.model.predict(np.expand_dims(cur_state, axis=0))[0])

        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if done and episode_length < 200:

            reward = 250 + episode_reward

            if (episode_reward > max_reward):
                agent.model.save_weights(str(episode_reward) + "_agent_.h5")
        else:

            reward = 5 * abs(next_state[0] - cur_state[0]) + 3 * abs(cur_state[1])

        replay_memory.append((cur_state, action, reward, next_state, done))
        cur_state = next_state

        if (len(replay_memory) < MINIMUM_REPLAY_MEMORY):
            continue

        train_dqn_agent()

    if (EPSILON > MINIMUM_EPSILON and len(replay_memory) > MINIMUM_REPLAY_MEMORY):
        EPSILON *= EPSILON_DECAY

    max_reward = max(episode_reward, max_reward)
    reward_set.append(episode_reward)
    print('Episode', episode, 'Episodic Reward', episode_reward, 'Maximum Reward', max_reward, 'EPSILON', EPSILON)


plt.plot(reward_set)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(r'S4_MountainCar.png')
plt.show()

