import os.path

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from core.algorithms import DQN, BacktrackDQN, MultiBatchDQN, BacktrackSarsaDQN

ALGO_NAMES = ['BacktrackSarsaDQN', 'DQN', 'MultiBatchDQN', 'BacktrackDQN']
ENV_NAME = 'CartPole-v0'  # CartPole-v0, Acrobot-v1, MountainCar-v0
LOG_INTERVAL = 10

PARAMS = dict(
    lr=2e-3,
    gamma=0.99,
    #buffer_size=1e5,
    sample_size=64,
    eps_start=0.8,
    eps_end=0.05,
    eps_decay=0.95,
    backtrack_steps=3,
    use_double_dqn=True,
    alpha=0.5,
    beta=1e-2,
)

MAX_HORIZON = 10000
NUM_EPOCHS = 200
SEED_LIST = [227, 222, 1003, 1123]
BUFFER_SIZES = [3,4,5,6,7]

env = gym.make(ENV_NAME)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

algo_name = 'BacktrackSarsaDQN'

def main():
    records = {}

    for bs in BUFFER_SIZES:
        records[bs] = [[] for _ in range(len(SEED_LIST))]

        for seed_idx, seed in enumerate(SEED_LIST):
            env.seed(seed)
            torch.manual_seed(seed)

            print('Env Name: %s | Seed: %d | STATE_DIM: %d | ACTION_DIM: %d | Algo: %s | BUFFER_SIZE: 10^%d'
                  % (ENV_NAME, seed, STATE_DIM, ACTION_DIM, algo_name, bs))

            if algo_name == 'DQN':
                model = DQN(STATE_DIM, ACTION_DIM, buffer_size=10**bs, **PARAMS)
            elif algo_name == 'BacktrackDQN':
                model = BacktrackDQN(STATE_DIM, ACTION_DIM, buffer_size=10**bs, **PARAMS)
            elif algo_name == 'MultiBatchDQN':
                model = MultiBatchDQN(STATE_DIM, ACTION_DIM, buffer_size=10**bs, **PARAMS)
            elif algo_name == 'BacktrackSarsaDQN':
                model = BacktrackSarsaDQN(STATE_DIM, ACTION_DIM, buffer_size=10**bs, **PARAMS)
            else:
                raise NotImplementedError

            running_reward = 0
            for i_episode in range(NUM_EPOCHS):
                state = env.reset()
                ep_reward = 0
                for t in range(1, MAX_HORIZON):
                    action = model.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    model.save_transition(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                    model.update()
                    if done:
                        break

                model.end_episode()
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                if i_episode % LOG_INTERVAL == 0:
                    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))

                    records[bs][seed_idx].append(running_reward)

    for bs in BUFFER_SIZES:
        data = np.array(records[bs])

        y_mean = np.mean(data, axis=0)
        y_std = np.std(data, axis=0)
        x = range(len(y_mean))
        ax = plt.gca()
        ax.set_ylim([-350, 0])

        plt.plot(x, y_mean, label="10^{}".format(bs))
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()