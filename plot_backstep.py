# coding=utf-8
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import os.path
import random
import torch
from core.algorithms import BacktrackDQN, BacktrackSarsaDQN, DQN, MultiBatchDQN

parser = argparse.ArgumentParser()
parser.add_argument("--env", "--env", type=str, default='cartpole', choices=['mountaincar', 'cartpole', 'acrobot'])
parser.add_argument("--lr", "--lr", type=float, default=2e-3)
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
args = parser.parse_args()

PARAMS = vars(args)
print('Experiment hyperparameters: ', PARAMS)

LOG_INTERVAL = 10
MAX_HORIZON = 10000
USE_EVAL_REWARDS = True
NUM_EPOCHS = args.epochs
SEED_LIST = [227, 222, 1003, 1123]

ENV_NAMES = ['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1']
BACKSTEPS = [1, 3, 5, 10]


PLOT_NAME = 'pri={}_lr={}_buffer={}_bstep={}_env={}.svg'.format(
	PARAMS['use_prioritized_buffer'],
	PARAMS['lr'],
	PARAMS['buffer_size'],
	PARAMS['backtrack_steps'],
	PARAMS['env']
	)


def set_seed(seed):
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


def main():

	for env_name in ENV_NAMES:
		records = {}
		env = gym.make(env_name)
		STATE_DIM = env.observation_space.shape[0]
		ACTION_DIM = env.action_space.n
		
		for back_step in BACKTRACK_STEPS:
			records[back_step] = [[] for _ in range(len(SEED_LIST))]
	
			for seed_idx, seed in enumerate(SEED_LIST):
				set_seed(seed)
				print(
					'Env Name: %s | Seed: %d | STATE_DIM: %d | ACTION_DIM: %d | Algo: %s '
					% (ENV_NAME, seed, STATE_DIM, ACTION_DIM, algo_name)
					)
	
				model = BacktrackSarsaDQN(STATE_DIM, ACTION_DIM, **PARAMS)
				running_reward = 0
				for i_episode in range(NUM_EPOCHS):
					state = env.reset()
					ep_reward = 0
					for _ in range(1, MAX_HORIZON + 1):
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
						evaluation_reward = evaluate_model(env, model)
						records[algo_name][seed_idx].append(evaluation_reward if USE_EVAL_REWARDS else running_reward)
						print(
							'Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tEvaluation reward: {:.2f}'.format(
								i_episode, ep_reward, running_reward, evaluation_reward
								)
							)
	
			records_file = open('record_{}.pickle'.format(env_name), 'wb')
			pickle.dump(records, records_file)
			records_file.close()


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


def plot_backstep_by_env():
	for env_name in ENV_NAMES:
		with open('records_{}.pickle'.format(env_name), 'rb') as file:
			records = pickle.load(file)

		for back_step in BACKTRACK_STEPS:
			data = np.array(records[back_step])

			y_mean = np.mean(data, axis=0)
			y_std = np.std(data, axis=0)
			x = [epoch * 10 for epoch in range(1, len(y_mean) + 1)]

			plt.plot(x, y_mean)
			plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)

		plt.legend(["BackStep: {}".format(back_step) for back_step in BACKTRACK_STEPS])
		plt.title("{}".format(env_name))
		plt.xlabel("Epoch")
		plt.ylabel("Average Reward")
		plt.savefig('Backstep_{}.svg'.format(env_name))
		plt.clf()


if __name__ == '__main__':
	main()
	plot_backstep_by_env()
