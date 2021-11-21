import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from core.models import FCN
from core.replay_buffer import ReplayBuffer


class DQN:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        lr=1e-3,
        gamma=0.99,
        buffer_size=1e5,
        sample_size=64,
        eps_start=0.8,
        eps_end=0.05,
        eps_decay=0.95,
        **kwargs
    ):
        self.gamma = gamma
        self.action_dim = action_dim

        self.q_func = FCN(state_dim, action_dim)
        self.target_q_func = FCN(state_dim, action_dim)
        self.target_q_func.eval()
        self._update_target_q_func()
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)

        """ Hyperparameters """
        self.buffer_size = int(buffer_size)
        self.sample_size = sample_size

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = self.eps_start

        self.memory = ReplayBuffer(state_dim, self.buffer_size, self.sample_size)

    def _update_target_q_func(self):
        self.target_q_func.load_state_dict(self.q_func.state_dict())

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)        
        with torch.no_grad():
            action_values = self.q_func.forward(state)

        if random.random() > self.eps:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def save_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def end_episode(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        self._update_target_q_func()

    def _q_learning_loss(self, states, actions, rewards, next_states, dones):
        q_targets_next = self.target_q_func.forward(next_states).max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_predict = self.q_func.forward(states).gather(1, actions)
        loss = F.mse_loss(q_targets, q_predict)
        return loss

    def _sarsa_loss(self, states, actions, rewards, next_states, next_actions):
        next_q_predict = self.q_func.forward(next_states).gather(1, next_actions).detach()
        sarsa_target = rewards + (self.gamma * next_q_predict)

        q_predict = self.q_func.forward(states).gather(1, actions)
        loss = F.mse_loss(sarsa_target, q_predict)
        return loss

    def update(self):
        if len(self.memory) < self.sample_size:
            return

        states, actions, rewards, next_states, dones, _ = self.memory.sample()
        loss = self._q_learning_loss(states, actions, rewards, next_states, dones)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class BacktrackSarsaDQN(DQN):
    def __init__(
        self,
        state_dim, 
        action_dim, 
        lr=1e-3,
        gamma=0.99,
        buffer_size=1e5,
        sample_size=64,
        eps_start=0.8,
        eps_end=0.05,
        eps_decay=0.95,
        backtrack_steps=3,
        **kwargs
    ):
        self.backtrack_steps = backtrack_steps
        super(BacktrackSarsaDQN, self).__init__(
            state_dim,
            action_dim, 
            lr,
            gamma,
            buffer_size,
            sample_size // self.backtrack_steps,
            eps_start,
            eps_end,
            eps_decay,
            **kwargs
        )

    def update(self):
        if len(self.memory) < (self.sample_size * self.backtrack_steps):
            return

        starting_indices = None

        for i in range(self.backtrack_steps):
            if i == 0:
                states, actions, rewards, next_states, dones, indices = self.memory.sample()
                loss = self._q_learning_loss(states, actions, rewards, next_states, dones)
            else:
                states, actions, rewards, next_states, _, next_actions, indices = self.memory.sample(starting_indices)
                loss = self._sarsa_loss(states, actions, rewards, next_states, next_actions)

            starting_indices = indices
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()                  

class MultiBatchDQN(DQN):
    def __init__(
        self,
        state_dim, 
        action_dim, 
        lr=1e-3,
        gamma=0.99,
        buffer_size=1e5,
        sample_size=64,
        eps_start=0.8,
        eps_end=0.05,
        eps_decay=0.95,
        backtrack_steps=3,
        **kwargs
    ):
        self.backtrack_steps = 3
        super(MultiBatchDQN, self).__init__(
            state_dim,
            action_dim, 
            lr,
            gamma,
            buffer_size,
            sample_size // self.backtrack_steps,
            eps_start,
            eps_end,
            eps_decay
        )

    def update(self):
        if len(self.memory) < (self.sample_size * self.backtrack_steps):
            return

        for i in range(self.backtrack_steps):
            states, actions, rewards, next_states, dones, _ = self.memory.sample()
            loss = self._q_learning_loss(states, actions, rewards, next_states, dones)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            

class BacktrackDQN(DQN):
    def __init__(
        self,
        state_dim, 
        action_dim, 
        lr=1e-3,
        gamma=0.99,
        buffer_size=1e5,
        sample_size=64,
        eps_start=0.8,
        eps_end=0.05,
        eps_decay=0.95,
        backtrack_steps=3,
        **kwargs
    ):
        self.backtrack_steps = 3
        super(BacktrackDQN, self).__init__(
            state_dim,
            action_dim, 
            lr,
            gamma,
            buffer_size,
            sample_size // self.backtrack_steps,
            eps_start,
            eps_end,
            eps_decay,
            **kwargs
        )

    def update(self):
        if len(self.memory) < (self.sample_size * self.backtrack_steps):
            return

        starting_indices = None

        for i in range(self.backtrack_steps):
            if starting_indices is None:
                states, actions, rewards, next_states, dones, indices = self.memory.sample(None)
            else:
                states, actions, rewards, next_states, dones, _, indices = self.memory.sample(starting_indices)
            loss = self._q_learning_loss(states, actions, rewards, next_states, dones)

            starting_indices = indices
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  
