import torch
import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, state_size, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.current_size = 0
        self.idx = -1

        self.batch_size = batch_size

        self.states = np.zeros((self.buffer_size, state_size))
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.next_states = np.zeros((self.buffer_size, state_size))
        self.rewards = np.zeros((self.buffer_size, 1))
        self.dones = np.zeros((self.buffer_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.idx = (self.idx + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

    def _get_batch(self, indices):
        states = torch.from_numpy(self.states[indices]).float()
        actions = torch.from_numpy(self.actions[indices])
        rewards = torch.from_numpy(self.rewards[indices]).float()
        next_states = torch.from_numpy(self.next_states[indices]).float()
        dones = torch.from_numpy(self.dones[indices]).float()
        return states, actions, rewards, next_states, dones

    def sample(self, start_indices=None):
        if start_indices is None:
            indices = np.random.choice(range(self.current_size), self.batch_size)
            return *self._get_batch(indices), indices
        else:
            indices = start_indices - 1
            _, next_actions, _, _, _ = self._get_batch(start_indices)
            return *self._get_batch(indices), next_actions, indices
    
    def __len__(self):
        return self.current_size

