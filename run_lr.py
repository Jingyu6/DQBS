import os.path
import argparse

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.algorithms import DQN, BacktrackDQN, MultiBatchDQN, BacktrackSarsaDQN

parser = argparse.ArgumentParser()
parser.add_argument("--env", "--env", type=str, default='acrobot', choices=['mountaincar', 'cartpole', 'acrobot'])
#parser.add_argument("--lr", "--lr", type=float, default=2e-3)
parser.add_argument("--gamma", "--gamma", type=float, default=0.99)
parser.add_argument("--epochs", "--epochs", type=int, default=200)
parser.add_argument("--buffer_size", "--buffer_size", type=int, default=1e5)
parser.add_argument("--sample_size", "--sample_size", type=int, default=64)
parser.add_argument("--eps_start", "--eps_start", type=float, default=0.8)
parser.add_argument("--eps_end", "--eps_end", type=float, default=0.05)
parser.add_argument("--eps_decay", "--eps_decay", type=float, default=0.95)
parser.add_argument("--backtrack_steps", "--backtrack_steps", type=int, default=3)
parser.add_argument("--use_double_dqn", "-use_double_dqn", action="store_true")
parser.add_argument("--alpha", "--alpha", type=float, default=0.5)
parser.add_argument("--beta", "--beta", type=float, default=1e-2)
args = parser.parse_args()

ALGO_NAMES = ['BacktrackSarsaDQN', 'DQN', 'MultiBatchDQN', 'BacktrackDQN']
algo_name = 'BacktrackSarsaDQN'

if args.env == 'mountaincar':
    ENV_NAME = 'MountainCar-v0'
elif args.env == 'cartpole':
    ENV_NAME = 'CartPole-v0'
else:
    ENV_NAME = 'Acrobot-v1'

PARAMS = vars(args)

LOG_INTERVAL = 10
MAX_HORIZON = 10000
NUM_EPOCHS = args.epochs
SEED_LIST = [227, 222, 1003, 1123]
#LEARNING_RATES = [5e-4, 8e-4, 2e-3, 5e-3, 8e-3, 2e-2]
LEARNING_RATES = [2e-4, 5e-4, 8e-4, 2e-3, 5e-3, 8e-3]

env = gym.make(ENV_NAME)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n


def main():
    records = {}

    for lr in LEARNING_RATES:
        records[lr] = [[] for _ in range(len(SEED_LIST))]

        for seed_idx, seed in enumerate(SEED_LIST):
            env.seed(seed)
            torch.manual_seed(seed)

            print('Env Name: %s | Seed: %d | STATE_DIM: %d | ACTION_DIM: %d | Algo: %s | LEARNING_RATE: %f'
                  % (ENV_NAME, seed, STATE_DIM, ACTION_DIM, algo_name, lr))

            if algo_name == 'DQN':
                model = DQN(STATE_DIM, ACTION_DIM, lr=lr, **PARAMS)
            elif algo_name == 'BacktrackDQN':
                model = BacktrackDQN(STATE_DIM, ACTION_DIM, lr=lr, **PARAMS)
            elif algo_name == 'MultiBatchDQN':
                model = MultiBatchDQN(STATE_DIM, ACTION_DIM, lr=lr, **PARAMS)
            elif algo_name == 'BacktrackSarsaDQN':
                model = BacktrackSarsaDQN(STATE_DIM, ACTION_DIM, lr=lr, **PARAMS)
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

                    records[lr][seed_idx].append(running_reward)

    for lr in LEARNING_RATES:
        data = np.array(records[lr])

        y_mean = np.mean(data, axis=0)
        y_std = np.std(data, axis=0)
        x = range(len(y_mean))
        ax = plt.gca()
        #ax.set_ylim([-350, 0])

        plt.plot(x, y_mean)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)

    #plt.legend(["5e-4", "8e-4", "2e-3", "5e-3", "8e-3", "2e-2"])
    plt.legend(["2e-4", "5e-4", "8e-4", "2e-3", "5e-3", "8e-3"])
    out_file = ENV_NAME if args.alpha == 0.0 else ENV_NAME + 'p'
    plt.savefig(os.path.join('result/lr', out_file + '.png'))
    # plt.show()


if __name__ == '__main__':
    main()
