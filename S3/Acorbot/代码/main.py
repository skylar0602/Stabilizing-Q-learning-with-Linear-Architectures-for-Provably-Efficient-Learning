import sys
import logging
import itertools

import numpy as np

np.random.seed(0)
import pandas as pd
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

mpl.use('TkAgg')
#prevent bugs

torch.manual_seed(0)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')
# load agent
env = gym.make('Acrobot-v1')
env.seed(0)
for key in vars(env):
    logging.info('%s: %s', key, vars(env)[key])
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
# load agent
print("load agent")


class AdvantageActorCriticAgent:
    def __init__(self, env):
        self.gamma = 0.99

        self.actor_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100, ],
            output_size=env.action_space.n, output_activator=nn.Softmax(1))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 0.0001)
        self.critic_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100, ])
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 0.0002)
        self.critic_loss = nn.MSELoss()

    def build_net(self, input_size, hidden_sizes, output_size=1,
                  output_activator=None):
        layers = []
        for input_size, output_size in zip(
                [input_size, ] + hidden_sizes, hidden_sizes + [output_size, ]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        if output_activator:
            layers.append(output_activator)
        net = nn.Sequential(*layers)
        return net

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.discount = 1.

    def step(self, observation, reward, done):
        state_tensor = torch.as_tensor(observation, dtype=torch.float).reshape(1, -1)
        prob_tensor = self.actor_net(state_tensor)
        action_tensor = distributions.Categorical(prob_tensor).sample()
        action = action_tensor.numpy()[0]
        if self.mode == 'train':
            self.trajectory += [observation, reward, done, action]
            if len(self.trajectory) >= 8:
                self.learn()
            self.discount *= self.gamma
        return action

    def close(self):
        pass

    # write reward function
    def learn(self):
        state, _, _, action, next_state, reward, done, next_action \
            = self.trajectory[-8:]
        state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float).unsqueeze(0)

        next_v_tensor = self.critic_net(next_state_tensor)
        target_tensor = reward + (1. - done) * self.gamma * next_v_tensor
        v_tensor = self.critic_net(state_tensor)
        td_error_tensor = target_tensor - v_tensor

        # train
        pi_tensor = self.actor_net(state_tensor)[0, action]
        logpi_tensor = torch.log(pi_tensor.clamp(1e-6, 1.))
        actor_loss_tensor = -(self.discount * td_error_tensor * logpi_tensor).squeeze()
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward(retain_graph=True)
        self.actor_optimizer.step()

        pred_tensor = self.critic_net(state_tensor)
        critic_loss_tensor = self.critic_loss(pred_tensor, target_tensor)
        self.critic_optimizer.zero_grad()
        critic_loss_tensor.backward()
        self.critic_optimizer.step()


agent = AdvantageActorCriticAgent(env)


# train and test
def play_episode_train(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, done)
        if render:
            env.render()
        if done:
            break

        observation, reward, done, _, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            break
    agent.close()
    return episode_reward, elapsed_steps


def play_episode_test(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, done)
        if render:
            env.render()
        if done:
            break

        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            break
    agent.close()
    return episode_reward, elapsed_steps


# starting train
"""
script
"""
logging.info('==== train ====')
episode_rewards = []
upper_limit = 200
for episode in itertools.islice(itertools.count(), upper_limit):
    episode_reward, elapsed_steps = play_episode_train(env.unwrapped, agent,
                                                       max_episode_steps=env._max_episode_steps, mode='train', render=1)
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f, steps = %d',
                  episode, episode_reward, elapsed_steps)
    # if np.mean(episode_rewards[-10:]) > -120:#check if converge
    #     break
plt.plot(episode_rewards)
# plt.savefig(path)
plt.show()

# start train
"""测试过程"""
logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode_test(env, agent)
    episode_rewards.append(episode_reward)
    logging.debug('test episode %d: reward = %.2f, steps = %d',
                  episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f ± %.2f',
             np.mean(episode_rewards), np.std(episode_rewards))

env.close()
