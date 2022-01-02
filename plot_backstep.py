# coding=utf-8
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.algorithms import BacktrackDQN

ALGO_NAME = 'BacktrackSarsaDQN'
ENV_NAMES = ['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1']
# ENV_NAMES = ['MountainCar-v0']
# ENV_NAME = 'CartPole-v0'
LOG_INTERVAL = 10
BACKTRACK_STEPS = [1, 3, 5, 10, 15, 20, 30]
# BACKTRACK_STEPS = [1, 3]

PARAMS = dict(
	lr=2e-3,
	gamma=0.99,
	buffer_size=1e5,
	sample_size=64,
	eps_start=0.8,
	eps_end=0.05,
	eps_decay=0.95,
	use_double_dqn=True
	)

MAX_HORIZON = 10000
NUM_EPOCHS = 200
SEED_LIST = [227, 222, 1003, 1123]


# SEED_LIST = [227, 222]


def main():
	for env_name in ENV_NAMES:
		records = {}

		for back_step in BACKTRACK_STEPS:
			env: gym.Env = gym.make(env_name)
			state_dim: int = env.observation_space.shape[0]
			action_dim: int = env.action_space.n

			records[back_step] = [[] for _ in range(len(SEED_LIST))]

			PARAMS["backtrack_steps"] = back_step

			for seed_idx, seed in enumerate(SEED_LIST):
				env.seed(seed)
				torch.manual_seed(seed)

				print(
					'\nEnv Name: %s | Seed: %d | State Dim: %d | Action Dim: %d | Algo: %s | Back Step: %s '
					% (env_name, seed, state_dim, action_dim, ALGO_NAME, back_step)
					)

				model = BacktrackDQN(state_dim, action_dim, **PARAMS)

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
						print(
							'Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
								i_episode, ep_reward, running_reward
								)
							)
						records[back_step][seed_idx].append(running_reward)

		records_file = open('records_{}.pickle'.format(env_name), 'wb')
		pickle.dump(records, records_file)
		records_file.close()


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
	# main()
	plot_backstep_by_env()
