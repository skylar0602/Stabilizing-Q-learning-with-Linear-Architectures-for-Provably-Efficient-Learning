# coding: utf-8


import gym
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from collections import defaultdict
import scipy
import visualize
from Q_Learning import state_featurizer, policy_f, Estimator, q_learning_testing_rewards, q_learning, compare_results

mpl.use('TkAgg')

def normalization(env, observation_examples):
    #Standardization function, returns a scaler object
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)
    return scaler


def featurizer_function(normalized_data, featureVecDim):
    # Feature function, returns a FeatureUnion object
    featurizer_vector = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=0.5, n_components=10)),
        ("rbf2", RBFSampler(gamma=0.25, n_components=20)),
        ("rbf3", RBFSampler(gamma=0.1, n_components=20))
    ])

    position_vec = np.delete(normalized_data, 1, 1)  # Delete the second column which is velocity
    featurizer_vector.fit(position_vec)

    return featurizer_vector


def scaler_val(state, scaler):
    # Normalize the state
    return (state[0] - scaler.mean_[0]) / scaler.var_[0]


def UniformRandomPolicyGenerator(nA):
    # generate a uniform random policy
    def urpg(state):
        return np.ones(nA, dtype=float) / nA

    return urpg


def RandomVectorGenerator(featureVecDim):
    # generate a random vector
    return 2 * np.random.rand(featureVecDim) - 1


def max_dict(dictionary):
    # Returns the key with the highest value in the dictionary
    v_max = max(dictionary.values())
    for key, value in dictionary.items():
        if value == v_max:
            return key


def e_greedy_policy(estimator, nA, epsilon):
    # Generate an epsilon-greedy strategy
    def policy_maker(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_maker


def reward(alpha_vec, featurizer_vector, scaler):
    # returns a reward function
    def reward_fn(state):
        return np.dot(alpha_vec, featurizer_vector.transform([[scaler_val(state, scaler)]])[0])

    return reward_fn


def reward_plot(alpha_vec, featurizer_vector, scaler):
    # plot the reward function
    x = np.linspace(-1.2, 0.6, num=1000)
    i = 0
    y = np.zeros(1000)
    for position in x:
        y[i] = np.dot(alpha_vec, featurizer_vector.transform([[(position - scaler.mean_[0]) / scaler.var_[0]]])[0])
        i += 1
    plt.plot(x, y, linewidth=2.0)
    plt.xlabel("Position")
    plt.ylabel("Reward")
    plt.title("Reward function")
    plt.show()


def smooth(y, box_pts=4):
    # smooth function
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def savitzky_golay(y, window_size=51, order=5, deriv=0, rate=1):
    # smooth function
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)

    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')



def ValueFunctionGenerator(env, alpha_vec, policy, featurizer_vec, scaler, featureVecDim, num_trajectories,
                           discount_factor):
    v_basis = np.zeros(featureVecDim)
    # generate trajectory
    episode = defaultdict(list)
    for i in range(num_trajectories):
        state = env.reset()
        done = False
        for l in range(200):
            # generate policy
            prob = policy(state)
            action = np.random.choice(np.arange(len(prob)), p=prob)
            new_observations = env.step(action)
            next_state = new_observations[0]
            done = new_observations[2]
            if done == True:
                break
            episode[i].append((state, action))
            state = next_state
            l += 1
        env.close()
        j = 0
        # Calculate eigenvectors
        for state, action in episode[i]:
            v_basis += featurizer_vec.transform([[scaler_val(state, scaler)]])[0] * (discount_factor) ** j
            j += 1

    # Calculate the mean of the eigenvectors
    v_basis_net = v_basis / num_trajectories
    # computed value function
    V = np.dot(alpha_vec, v_basis_net)
    return V, v_basis_net


def irl(env, alpha_vec, featurizer_vector, scaler, featurizer, normalized_data, featureVecDim, policy_dbe,
        num_trajectories=10, num_episodes=20, max_epoch=10, discount_factor=1, penalty_factor=2, epsilon_v=0.0):
    nP = 0  #
    V_vec = defaultdict(float)  
    V_policy_basis = defaultdict(list)  
    V_input = np.zeros(featureVecDim)  

    # cal V_dbe&V_dbe_basis
    V_dbe, V_dbe_basis = ValueFunctionGenerator(env, alpha_vec, policy_dbe, featurizer_vector, scaler, featureVecDim,
                                                num_trajectories, discount_factor)
    print("-------------------------------")
    print("| V_dbe | ", V_dbe, " |")
    print("-------------------------------")
    print("")

    # looooping
    while True:
        print("############################################################################")
        print("Starting epoch {} .... ".format(nP))
        print("Alpha_vec value at the start of the epoch = {}".format(alpha_vec))

        if nP == 0:
            policy_ = UniformRandomPolicyGenerator(env.action_space.n)

        # cal V_vec&V_policy_basis
        V_vec[nP], V_policy_basis[nP] = ValueFunctionGenerator(env, alpha_vec, policy_, featurizer_vector, scaler,
                                                               featureVecDim, num_trajectories, discount_factor)
        print("New policy value based on previous alpha, V_vec[np] = {}".format(V_vec[nP]))

        print("_____________________________LP starts_______________________________")

        nP_best = max_dict(V_vec)

        print("nP_best =", nP_best)
        print("nP_best_value based on old alpha =", V_vec[nP_best])
        print("DBE_value based on old alpha =", V_dbe)

        if V_dbe - V_vec[nP] >= 0:
            V_input += V_policy_basis[nP] - V_dbe_basis
        else:
            V_input += penalty_factor * (V_policy_basis[nP] - V_dbe_basis)

        # linear programming
        res = scipy.optimize.linprog(V_input, bounds=(-1, 1), method="simplex")

        print("**********LP results******************************************************")
        print("                       ")
        # print(res)
        print("new alpha_vec = ", res.x)
        print("                       ")
        print("**************************************************************************")

        alpha_vec_new = res.x

        V_dbe_new = np.dot(alpha_vec_new, V_dbe_basis)

        alpha_vec = alpha_vec_new
        V_dbe = V_dbe_new

        for i, list_ in V_policy_basis.items():
            V_vec[i] = np.dot(list_, alpha_vec)

        print("According to new alpha, V_dbe = ", V_dbe_new)
        print("New V_vec[max] in existing values", V_vec[max_dict(V_vec)])
        print("_________________________________________________________________________")
        print("Plotting reward function based on alpha_vec start value.....")

        reward_plot(alpha_vec, featurizer_vector, scaler)

        print("Q learning starts..........")

        reward_fn = reward(alpha_vec, featurizer_vector, scaler)
        nP += 1

        estimator = Estimator(env, scaler, featurizer)


env = gym.make("MountainCar-v0").env

featureVecDim = 50

# generate 10000 samples
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

# normalized data
scaler = normalization(env, observation_examples)
normalized_data = scaler.fit_transform(observation_examples)

# generating feature function
featurizer_vector = featurizer_function(normalized_data, featureVecDim)

# Generate random alpha vectors
alpha_vec = RandomVectorGenerator(featureVecDim)
print("random Alpha vector: ")
print(alpha_vec)



featurizer = state_featurizer(normalized_data)
policy_dbe, estimator_dbe = policy_f(env, scaler, featurizer, print_ep_lens=False)

reward_fn, alpha_vec = irl(env, alpha_vec, featurizer_vector, scaler, featurizer,
                           normalized_data, featureVecDim, policy_dbe)

estimator_f = Estimator(env, scaler, featurizer)
reward_plot(alpha_vec, featurizer_vector, scaler)
print("start learning:")


success = q_learning_testing_rewards(env, estimator_f, reward_fn, num_episodes=200, render=True, ep_details=False)

print("output:")
visualize.plot_cost_to_go_mountain_car(env, estimator_f)
visualize.plot_episode_stats(success, smoothing_window=25)

# compare
a, b = compare_results(env, estimator_f, estimator_dbe, num_test_trajs=100, epsilon_test=0.0)
