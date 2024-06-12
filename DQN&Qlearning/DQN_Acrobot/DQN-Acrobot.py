import sys
import logging
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

torch.manual_seed(0)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('Acrobot-v1')
env.seed(0)

print("loading agent")

class QLearningAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        # Neural network to approximate Q-value function
        self.q_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100,],
            output_size=env.action_space.n)
        self.optimizer = optim.Adam(self.q_net.parameters(), learning_rate)
        self.criterion = nn.MSELoss()

    def build_net(self, input_size, hidden_sizes, output_size):
        layers = []
        for input_size, output_size in zip(
                [input_size,] + hidden_sizes, hidden_sizes + [output_size,]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def get_action(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_net(state_tensor)
        with torch.no_grad():
            next_q_values = self.q_net(next_state_tensor)
        target = reward + (1. - done) * self.gamma * torch.max(next_q_values)
        loss = self.criterion(q_values[0][action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = QLearningAgent(env)

def play_episode(env, agent):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    return total_reward

logging.info('==== train ====')
episode_rewards = []
for episode in range(200):
    episode_reward = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f',
                  episode, episode_reward)

plt.plot(episode_rewards)
plt.show()

env.close()