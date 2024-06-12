import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gym

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


mpl.use('TkAgg')


class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # input (state) to hiden，hiden to out (action)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)

        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # size of experience in each memory  (state + next state + reward + action)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # when target network update

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    
    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # random
            action = np.random.randint(0, self.n_actions)
        else: # choose the best
            actions_value = self.eval_net(x) # use eval net to calculate action's value
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # pick highest action

        return action

    def store_transition(self, state, action, reward, next_state):
        # conclude experience
        transition = np.hstack((state, [action, reward], next_state))


        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # random pick experience under batch_size
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        # cal eval net and target net to get Q value differences
        q_eval = self.eval_net(b_state).gather(1, b_action) # re cal experience's eval net --- Q value
        q_next = self.target_net(b_next_state).detach() # detach target net not used
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for (target_replace_iter), uopdate target net，as copy eval net to target net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())


env = gym.make("Acrobot-v1")


n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

n_hidden = 50             # num of nn
batch_size = 32           # batch size
lr = 0.01                 
epsilon = 0.1             
gamma = 0.9               
target_replace_iter = 100 # target network --update interval
memory_capacity = 200    # num of record
n_episodes = 300          
step_record=[]
# build DQN
dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

#train
for i_episode in range(n_episodes):
    t = 0
    rewards = 0
    state = env.reset()
    while True:
        env.render()

        # choose action
        action = dqn.choose_action(state)
        next_state, reward, done, info = env.step(action)
        reward=-(next_state[0]+next_state[2])
        
        # save experience
        dqn.store_transition(state, action, reward, next_state)

        # add reward
        rewards = reward

        # enough experience and train
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        # next state
        state = next_state

        if done:
            step_record.append(t)
            print('Episode finished after {} timesteps, total rewards {} , round {}'.format(t+1, rewards,i_episode))
            break

        t += 1
plt.plot(range(len(step_record)), step_record, color='b')
plt.xlabel('episode')
plt.ylabel('steps')
plt.show()

env.close()

