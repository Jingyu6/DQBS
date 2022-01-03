# coding=utf-8
import argparse
import os
import pickle
import random
import time
from multiprocessing import Pool

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.algorithms import BacktrackSarsaDQN

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
# parser.add_argument("--backtrack_steps", "--backtrack_steps", type=int, default=3)
parser.add_argument("--use_prioritized_buffer", "--use_prioritized_buffer", action="store_true")
parser.add_argument("--alpha", "--alpha", type=float, default=0.5)
parser.add_argument("--beta", "--beta", type=float, default=1e-2)
parser.add_argument("--num_workers", "--num_workers", type=int, default=5)
args = parser.parse_args()

PARAMS = vars(args)
ALGO_NAME = 'BacktrackSarsaDQN'
# ENV_NAMES = ['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1']
ENV_NAME = "MountainCar-v0"
# ENV_NAMES = ['MountainCar-v0']
LOG_INTERVAL = 10
MAX_HORIZON = 10000
RUNNING_AVG_WEIGHT = 0.3
BACKSTEPS = [1, 3, 5, 10]
SEED_LIST = [227, 222, 1003, 1123]
PARAMS['num_workers'] = 16

USE_EVAL_REWARDS = True
USE_RUNNING_AVG = True
SET_VERBOSE = True


def set_seed(env, seed):
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


def worker(seed, backstep):
	""" Run experiments for a specific algorithm with PARAMS with seed """
	st = time.time()

	env = gym.make(ENV_NAME)
	set_seed(env, seed)

	if ENV_NAME == "MountainCar-v0":
		PARAMS["epochs"] = 400
		PARAMS["buffer_size"] = 1e4
		PARAMS["lr"] = 5e-4

	NUM_EPOCHS = args.epochs
	STATE_DIM = env.observation_space.shape[0]
	ACTION_DIM = env.action_space.n

	PARAMS["backtrack_steps"] = backstep

	print(
		'\nEnv Name: %s | Seed: %d | State Dim: %d | Action Dim: %d | Back Step: %s has starts.'
		% (ENV_NAME, seed, STATE_DIM, ACTION_DIM, backstep)
		)

	records = []
	model = BacktrackSarsaDQN(STATE_DIM, ACTION_DIM, **PARAMS)
	training_running_reward = 0
	eval_running_reward = 0

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
		training_running_reward = RUNNING_AVG_WEIGHT * ep_reward + (1 - RUNNING_AVG_WEIGHT) * training_running_reward
		if i_episode % LOG_INTERVAL == 0:
			evaluation_reward = evaluate_model(env, model)
			if i_episode == 0:
				eval_running_reward = evaluation_reward
			else:
				eval_running_reward = RUNNING_AVG_WEIGHT * evaluation_reward + (
						1 - RUNNING_AVG_WEIGHT) * eval_running_reward
			records.append(
				eval_running_reward if USE_RUNNING_AVG else evaluation_reward if USE_EVAL_REWARDS else training_running_reward
				)
			if SET_VERBOSE:
				print(
					'Algo: {}, Seed: {}, Episode {}\tLast reward: {:.2f}\tRunning training reward: {:.2f}\tEvaluation reward: {:.2f}\tRunning eval reward: {:.2f}'.format(
						ALGO_NAME, seed, i_episode, ep_reward, training_running_reward, evaluation_reward,
						eval_running_reward
						)
					)

	et = time.time()
	print(
		'\nEnv Name: {} | Seed: {} | State Dim: {} | Action Dim: {} | Back Step: {} has finished, elapsed time: {:2.4f}s.'
			.format(ENV_NAME, seed, STATE_DIM, ACTION_DIM, backstep, et - st)
		)

	return records


def main():
	records = {}
	arguments = []

	for backstep in BACKSTEPS:
		for seed in SEED_LIST:
			arguments.append([seed, backstep])

	with Pool(PARAMS['num_workers']) as p:
		return_results = p.starmap(worker, arguments)

	for (_, backstep), record in zip(arguments, return_results):
		if backstep not in records:
			records[backstep] = []
		records[backstep].append(record)

		records_file = open('result/backstep/non-pri/record_backstep_{}.pickle'.format(ENV_NAME), 'wb')
		pickle.dump(records, records_file)
		records_file.close()


def plot_backstep_by_env():
	with open('result/backstep/non-pri/record_backstep_{}.pickle'.format(ENV_NAME), 'rb') as file:
		records = pickle.load(file)

	for backstep in BACKSTEPS:
		data = np.array(records[backstep])

		y_mean = np.mean(data, axis=0)
		y_std = np.std(data, axis=0)
		x = [(epoch + 1) * LOG_INTERVAL for epoch in range(len(y_mean))]

		plt.plot(x, y_mean)
		plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)

	plt.legend(["backstep = {}".format(backstep) for backstep in BACKSTEPS])
	plt.savefig(os.path.join('result/backstep/non-pri/', 'Backstep_{}.svg'.format(ENV_NAME)))
	plt.clf()


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
	plot_backstep_by_env()
