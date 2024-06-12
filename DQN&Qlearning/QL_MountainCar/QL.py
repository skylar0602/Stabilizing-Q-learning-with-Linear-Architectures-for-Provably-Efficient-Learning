import numpy as np
import gym
import matplotlib.pyplot as plt

# Hyperparameters / Constants
EPISODES = 10000
EPSILON = 1
EPSILON_DECAY = 0.99
MINIMUM_EPSILON = 0.001
DISCOUNT = 0.99
VISUALIZATION = False
ENV_NAME = 'MountainCar-v0'
NUM_BINS = [20, 20]  # Discretization bins for position and velocity

# Environment
env = gym.make(ENV_NAME)

# Create Q-Table and rewards list
q_table = np.random.uniform(low=-1, high=1, size=(NUM_BINS + [env.action_space.n]))
rewards = []

# Discretizing function
def discretize_state(state):
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    state_adj = (state - state_low) / (state_high - state_low)
    discrete_state = tuple((state_adj * np.array(NUM_BINS)).astype(np.int))
    return discrete_state

for episode in range(EPISODES):
    state = discretize_state(env.reset())
    done = False
    total_reward = 0

    while not done:
        if VISUALIZATION:
            env.render()

        if np.random.random() < EPSILON:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)
        next_discrete_state = discretize_state(next_state)

        if done:
            q_update = reward
        else:
            q_update = reward + DISCOUNT * np.max(q_table[next_discrete_state])

        q_table[state][action] = (1 - 0.1) * q_table[state][action] + 0.1 * q_update
        state = next_discrete_state
        total_reward += reward

    rewards.append(total_reward)

    # Decay epsilon
    if EPSILON > MINIMUM_EPSILON:
        EPSILON *= EPSILON_DECAY

# Plot rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward by Episode')
plt.show()

env.close()