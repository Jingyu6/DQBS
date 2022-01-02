import torch
import random
import numpy as np
from collections import namedtuple, deque

class ExperienceReplayBuffer:
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
        self.previous_idx = np.empty(self.buffer_size, dtype=np.int64)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        priority = 1.0 if self.__len__() == 0 else self.priority.max()
        if self.current_size == self.buffer_size:
            if priority > self.priority.min():
                cur_idx = self.priority.argmin()
                """ clean all records whose last indices = cur_idx """
                self.previous_idx[self.previous_idx == cur_idx] = -1
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
        self.previous_idx[cur_idx] = self.last
        self.last = cur_idx

    def _get_batch(self, indices):
        states = torch.from_numpy(self.states[indices]).float()
        actions = torch.from_numpy(self.actions[indices])
        rewards = torch.from_numpy(self.rewards[indices]).float()
        next_states = torch.from_numpy(self.next_states[indices]).float()
        dones = torch.from_numpy(self.dones[indices]).float()
        return states, actions, rewards, next_states, dones

    def _get_sampling_probs(self, mask=None):
        cur_priorities = np.copy(self.priority[:self.current_size])
        if mask is not None and mask.shape == cur_priorities.shape:
            """ set masked elements to be 0 """
            cur_priorities[mask] = 0
        sampling_probs = cur_priorities ** self.alpha / np.sum(cur_priorities ** self.alpha)
        return sampling_probs

    def has_valid_sarsa_transitions(self):
        return np.sum(self.previous_idx[:self.current_size] != -1) > 0

    def sample(self, beta, start_indices=None):
        if start_indices is None:
            """ return q learning data: [S, A, R, S', dones, indices, importance weights] """
            sampling_probs = self._get_sampling_probs()
            indices = np.random.choice(np.arange(self.current_size), self.batch_size, replace=True, p=sampling_probs)
            weights = (self.current_size * sampling_probs[indices]) ** (-beta)
            importance_weights = (torch.Tensor(weights / weights.max()).view((-1, 1))).detach()
            return *self._get_batch(indices), indices, importance_weights
        else:
            """ return SARSA data: [S, A, R, S', dones, A', indices, importance weights] """

            """ This function can only be called when there is valid SARSA transitions """
            assert self.has_valid_sarsa_transitions(), 'We do not have any valid SARSA transitions'

            prev_indices = self.previous_idx[start_indices]
            resample_required_indices = prev_indices == -1
            resample_required_cnt = np.sum(resample_required_indices)

            if resample_required_cnt > 0:
                """ for those whose previous indices were removed, sample random transitions """
                candidate_mask = self.previous_idx[:self.current_size] != -1 # candidates for start_indices
                
                # for sampling only
                candidate_sampling_probs = self._get_sampling_probs(~candidate_mask)

                newly_chosen_indices = np.random.choice(
                    np.arange(self.current_size), resample_required_cnt, replace=True, p=candidate_sampling_probs)
                start_indices[resample_required_indices] = newly_chosen_indices
                prev_indices = self.previous_idx[start_indices]

            weights = (self.current_size * self._get_sampling_probs()[prev_indices]) ** (-beta)
            importance_weights = (torch.Tensor(weights / weights.max()).view(-1, 1)).detach()

            _, next_actions, _, _, _ = self._get_batch(start_indices)
            return *self._get_batch(prev_indices), next_actions, prev_indices, importance_weights

    def update_priorities(self, indices, priorities):
        """Update the priorities associated with particular experiences."""
        self.priority[indices] = priorities

    def __len__(self):
        return self.current_size
