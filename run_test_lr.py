import time
import os.path
import argparse

import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

from core.algorithms import DQN, BacktrackDQN, MultiBatchDQN, BacktrackSarsaDQN

parser = argparse.ArgumentParser()
parser.add_argument("--env", "--env", type=str, default='cartpole', choices=['mountaincar', 'cartpole', 'acrobot'])
#parser.add_argument("--lr", "--lr", type=float, default=2e-3)
parser.add_argument("--gamma", "--gamma", type=float, default=0.99)
parser.add_argument("--epochs", "--epochs", type=int, default=200)
parser.add_argument("--buffer_size", "--buffer_size", type=int, default=1e5)
parser.add_argument("--sample_size", "--sample_size", type=int, default=64)
parser.add_argument("--eps_start", "--eps_start", type=float, default=0.8)
parser.add_argument("--eps_end", "--eps_end", type=float, default=0.05)
parser.add_argument("--eps_decay", "--eps_decay", type=float, default=0.95)
parser.add_argument("--backtrack_steps", "--backtrack_steps", type=int, default=3)
parser.add_argument("--use_prioritized_buffer", "--use_prioritized_buffer", action="store_true")
parser.add_argument("--alpha", "--alpha", type=float, default=0.5)
parser.add_argument("--beta", "--beta", type=float, default=1e-2)
parser.add_argument("--num_workers", "--num_workers", type=int, default=6)
args = parser.parse_args()

ALGOS = [DQN, BacktrackDQN, MultiBatchDQN, BacktrackSarsaDQN]
ALGO_NAMES = [clz.__name__ for clz in ALGOS]
algo = BacktrackSarsaDQN
algo_name = 'BacktrackSarsaDQN'

if args.env == 'mountaincar':
    ENV_NAME = 'MountainCar-v0'
elif args.env == 'cartpole':
    ENV_NAME = 'CartPole-v0'
else:
    ENV_NAME = 'Acrobot-v1'

PARAMS = vars(args)

LOG_INTERVAL = 5
MAX_HORIZON = 10000
RUNNING_AVG_WEIGHT = 0.3
USE_EVAL_REWARDS = True
USE_RUNNING_AVG = True
SET_VERBOSE = False
NUM_EPOCHS = args.epochs
SEED_LIST = [227, 222, 1003, 1123]#, 101]
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
LEARNING_RATES_LABELS = ["lr = 1e-4", "lr = 5e-4", "lr = 1e-3", "lr = 5e-3", "lr = 1e-2", "lr = 5e-2"]

PLOT_NAME = 'pri={}_lr=na_buffer={}_bstep={}_eps={}_env={}.svg'.format(
    PARAMS['use_prioritized_buffer'],
    #PARAMS['lr'],
    PARAMS['buffer_size'],
    PARAMS['backtrack_steps'],
    '(' + str(PARAMS['eps_start']) + '|' + str(PARAMS['eps_end']) + '|' + str(PARAMS['eps_decay']) + ')',
    PARAMS['env']
)

def set_seed(env, seed):
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def worker(seed, algo, algo_name, lr):
    """ Run experiments for a specific algorithm with PARAMS with seed """
    print('lr: {}, seed: {} has started.'.format(lr, seed))
    st = time.time()

    env = gym.make(ENV_NAME)
    set_seed(env, seed)

    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.n

    records = []
    model = algo(STATE_DIM, ACTION_DIM, lr=lr, **PARAMS)
    training_running_reward = 0
    eval_running_reward = 0

    for i_episode in range(NUM_EPOCHS):
        state = env.reset()
        ep_reward = 0
        for t in range(1, MAX_HORIZON + 1):
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action)
            model.save_transition(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            model.update()
            if done:
                break

        model.end_episode()
        training_running_reward = RUNNING_AVG_WEIGHT * ep_reward + (1 - RUNNING_AVG_WEIGHT) * training_running_reward
        if i_episode % LOG_INTERVAL == 0:
            evaluation_reward = evaluate_model(env, model)
            if i_episode == 0:
                eval_running_reward = evaluation_reward
            else:
                eval_running_reward = RUNNING_AVG_WEIGHT * evaluation_reward + (1 - RUNNING_AVG_WEIGHT) * eval_running_reward
            records.append(eval_running_reward if USE_RUNNING_AVG else evaluation_reward if USE_EVAL_REWARDS else training_running_reward)
            if SET_VERBOSE:
                print('Algo: {}, Seed: {}, Episode {}\tLast reward: {:.2f}\tRunning training reward: {:.2f}\tEvaluation reward: {:.2f}\tRunning eval reward: '.format(
                    algo_name, seed, i_episode, ep_reward, training_running_reward, evaluation_reward, eval_running_reward))
    
    et = time.time()
    print('lr: {}, seed: {} has finished, elapsed time: {:2.4f}s.'.format(lr, seed, et - st))
    return records

def main():
    print('Experiment hyperparameters: ', PARAMS)

    records = {}
    arguments = []

    for lr in LEARNING_RATES:
        for seed in SEED_LIST:
            arguments.append([seed, algo, algo_name, lr])

    with Pool(PARAMS['num_workers']) as p:
        return_results = p.starmap(worker, arguments)
        
    for (_, _, _, lr), record in zip(arguments, return_results):
        if lr not in records:
            records[lr] = []
        records[lr].append(record)

    for lr in LEARNING_RATES:
        data = np.array(records[lr])

        y_mean = np.mean(data, axis=0)
        y_std = np.std(data, axis=0)
        x = [int((epoch + 1) * LOG_INTERVAL) for epoch in range(len(y_mean))]

        plt.ylim([0,200])
        plt.plot(x, y_mean)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)

    plt.legend(LEARNING_RATES_LABELS, loc='upper left')
    plt.savefig(os.path.join('result/lr', PLOT_NAME))
    #plt.show()

def evaluate_model(env, model, episodes=5, gamma=0.999):
    total_rewards = 0
    for _ in range(episodes):
        state = env.reset()
        for t in range(MAX_HORIZON):
            action = model.select_action(state, greedy=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_rewards += reward * (gamma ** t)
            if done:
                break
    return total_rewards / episodes

if __name__ == '__main__':
    main()
