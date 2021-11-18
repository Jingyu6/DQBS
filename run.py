import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from algorithm.q_learning import DQN, BacktrackDQN

ALGO_NAMES = ['DQN', 'BacktrackDQN']
ENV_NAME = 'CartPole-v0'
LOG_INTERVAL = 10
SAMPLE_SIZE = 64
LR = 2e-3
MAX_HORIZON = 10000
NUM_EPOCHS = 300
SEED_LIST = [227, 222, 1003, 1123]

env = gym.make(ENV_NAME)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

def main():
    records = {}

    for algo_name in ALGO_NAMES:
        records[algo_name] = [[] for _ in range(len(SEED_LIST))]

        for seed_idx, seed in enumerate(SEED_LIST):
            env.seed(seed)
            torch.manual_seed(seed)

            print('Env Name: %s | Seed: %d | STATE_DIM: %d | ACTION_DIM: %d | Algo: %s '
                  % (ENV_NAME, seed, STATE_DIM, ACTION_DIM, algo_name))

            if algo_name == 'BacktrackDQN':
                model = BacktrackDQN(STATE_DIM, ACTION_DIM, SAMPLE_SIZE, LR)
            else:
                model = DQN(STATE_DIM, ACTION_DIM, SAMPLE_SIZE, LR)

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

                    records[algo_name][seed_idx].append(running_reward)

    for algo_name in ALGO_NAMES:
        data = np.array(records[algo_name])

        y_mean = np.mean(data, axis=0)
        y_std = np.std(data, axis=0)
        x = range(len(y_mean))

        plt.plot(x, y_mean)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)

    plt.legend(ALGO_NAMES)
    plt.show()

if __name__ == '__main__':
    main()