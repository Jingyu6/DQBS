import torch
import random
import numpy as np
from collections import namedtuple, deque

_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = namedtuple("Experience", field_names=_field_names)


class PrioritizedExperienceReplayBuffer:
    def __init__(self, state_size, buffer_size, batch_size, alpha):
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        :param state_size: dimension of state space
        :param buffer_size: maximum size of buffer
        :param batch_size: size of training batch
        :param alpha: strength of prioritized sampling
        """
        self.buffer_size = buffer_size
        self.current_size = 0
        self.last = -1
        self.alpha = alpha
        self.batch_size = batch_size

        self.states = np.zeros((self.buffer_size, state_size))
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.next_states = np.zeros((self.buffer_size, state_size))
        self.rewards = np.zeros((self.buffer_size, 1))
        self.dones = np.zeros((self.buffer_size, 1))
        self.priority = np.zeros(self.buffer_size)
        self.last_idx = np.empty(self.buffer_size, dtype=np.int64)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        priority = 1.0 if self.__len__() == 0 else self.priority.max()
        if self.current_size == self.buffer_size:
            if priority > self.priority.min():
                cur_idx = self.priority.argmin()
            else:
                return
        else:
            cur_idx = self.current_size
            self.current_size += 1
        self.states[cur_idx] = state
        self.actions[cur_idx] = action
        self.rewards[cur_idx] = reward
        self.next_states[cur_idx] = next_state
        self.dones[cur_idx] = done
        self.priority[cur_idx] = priority
        self.last_idx[cur_idx] = self.last
        self.last = cur_idx

    def _get_batch(self, indices):
        states = torch.from_numpy(self.states[indices]).float()
        actions = torch.from_numpy(self.actions[indices])
        rewards = torch.from_numpy(self.rewards[indices]).float()
        next_states = torch.from_numpy(self.next_states[indices]).float()
        dones = torch.from_numpy(self.dones[indices]).float()
        return states, actions, rewards, next_states, dones

    def sample(self, beta, start_indices=None):
        if start_indices is None:
            ps = self.priority[:self.current_size]
            sampling_probs = ps ** self.alpha / np.sum(ps ** self.alpha)
            idxs = np.random.choice(np.arange(ps.size), self.batch_size, replace=True, p=sampling_probs)
            weights = (self.current_size * sampling_probs[idxs])**(-beta)
            normalized_weights = (torch.Tensor(weights / weights.max()).view((-1, 1)))
            return *self._get_batch(idxs), idxs, normalized_weights
        else:
            idxs = self.last_idx[start_indices]
            idxs = np.squeeze(idxs[idxs != -1])
            _, next_actions, _, _, _ = self._get_batch(idxs)
            return *self._get_batch(idxs), next_actions, idxs

    def update_priorities(self, idxs, priorities):
        """Update the priorities associated with particular experiences."""
        self.priority[idxs] = priorities

    def __len__(self):
        return self.current_size

