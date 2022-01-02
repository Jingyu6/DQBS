import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from core.models import FCN
from core.replay_buffer import ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer

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
        alpha=0.5,
        beta=1e-2,
        use_prioritized_buffer=True,
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

        self.use_prioritized_buffer = use_prioritized_buffer
        if self.use_prioritized_buffer:
            self.memory = PrioritizedExperienceReplayBuffer(state_dim, self.buffer_size, self.sample_size, alpha, kwargs.get('backtrack_steps', 0))
            self.beta = beta
            self.cur_beta = 1.0
        else:
            self.memory = ExperienceReplayBuffer(state_dim, self.buffer_size, self.sample_size)

    def _update_target_q_func(self):
        self.target_q_func.load_state_dict(self.q_func.state_dict())

    def select_action(self, state, greedy=False):
        state = torch.from_numpy(state).float().unsqueeze(0)        
        with torch.no_grad():
            action_values = self.q_func.forward(state)

        if greedy or random.random() > self.eps:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def save_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def end_episode(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        self._update_target_q_func()

    def _q_learning_loss(self, states, actions, rewards, next_states, dones):
        """ standard q learning with target network """
        q_targets_next = self.target_q_func.forward(next_states).max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_predict = self.q_func.forward(states).gather(1, actions)
        return q_targets - q_predict

    def _sarsa_loss(self, states, actions, rewards, next_states, next_actions):
        next_q_predict = self.q_func.forward(next_states).gather(1, next_actions).detach()
        sarsa_target = rewards + (self.gamma * next_q_predict)

        q_predict = self.q_func.forward(states).gather(1, actions)
        return sarsa_target - q_predict

    def update(self):
        if len(self.memory) < self.sample_size:
            return

        if self.use_prioritized_buffer:
            self.cur_beta *= np.exp(-self.beta)
            states, actions, rewards, next_states, dones, indices, importance_weights = self.memory.sample(1 - self.cur_beta)
            delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
            priorities = (delta.abs().detach().numpy().flatten())
            self.memory.update_priorities(indices, priorities + 1e-6)
            loss = torch.mean((delta * importance_weights) ** 2)
        else:
            states, actions, rewards, next_states, dones, _ = self.memory.sample()
            delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
            loss = torch.mean(delta ** 2)

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
        alpha=0.5,
        beta=1e-2,
        use_prioritized_buffer=True,
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
            alpha,
            beta,
            use_prioritized_buffer,
            **kwargs
        )

    def update(self):
        if len(self.memory) < (self.sample_size * self.backtrack_steps):
            return

        starting_indices = None

        if self.use_prioritized_buffer:
            self.cur_beta *= np.exp(-self.beta)

        for i in range(self.backtrack_steps):
            if i == 0:
                if self.use_prioritized_buffer:
                    states, actions, rewards, next_states, dones, indices, importance_weights = self.memory.sample(1 - self.cur_beta)
                    delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
                    self.memory.update_delta(indices, delta.abs().detach().numpy().flatten() + 1e-6)
                    loss = torch.mean((delta * importance_weights) ** 2)
                else:
                    states, actions, rewards, next_states, dones, indices = self.memory.sample()
                    delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
                    loss = torch.mean(delta ** 2)
            else:
                if self.use_prioritized_buffer:
                    if self.memory.has_valid_sarsa_transitions():
                        """ Do not update priorities due to the different scales of q learning and SARSA delta """
                        states, actions, rewards, next_states, _, next_actions, indices, importance_weights = self.memory.sample(1 - self.cur_beta, starting_indices)
                        delta = self._sarsa_loss(states, actions, rewards, next_states, next_actions)
                        self.memory.update_delta(indices, delta.abs().detach().numpy().flatten() + 1e-6)
                        loss = torch.mean((delta * importance_weights) ** 2)
                    else:
                        """ If there's no SARSA transition, stop updating """
                        return
                else:
                    states, actions, rewards, next_states, _, next_actions, indices = self.memory.sample(starting_indices)
                    delta = self._sarsa_loss(states, actions, rewards, next_states, next_actions)
                    loss = torch.mean(delta ** 2)

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
        alpha=0.5,
        beta=1e-2,
        use_prioritized_buffer=True,
        **kwargs
    ):
        self.backtrack_steps = backtrack_steps
        super(MultiBatchDQN, self).__init__(
            state_dim,
            action_dim, 
            lr,
            gamma,
            buffer_size,
            sample_size // self.backtrack_steps,
            eps_start,
            eps_end,
            eps_decay,
            alpha,
            beta,
            use_prioritized_buffer,
            **kwargs
        )

    def update(self):
        if len(self.memory) < (self.sample_size * self.backtrack_steps):
            return

        if self.use_prioritized_buffer:
            self.cur_beta *= np.exp(-self.beta)

        for i in range(self.backtrack_steps):
            """ only use standard q learning loss """
            if self.use_prioritized_buffer:
                states, actions, rewards, next_states, dones, indices, importance_weights = self.memory.sample(1 - self.cur_beta)
                delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
                priorities = (delta.abs().detach().numpy().flatten())
                self.memory.update_priorities(indices, priorities + 1e-6)
                loss = torch.mean((delta * importance_weights) ** 2)
            else:
                states, actions, rewards, next_states, dones, _ = self.memory.sample()
                delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
                loss = torch.mean(delta ** 2)

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
        alpha=0.5,
        beta=1e-2,
        use_prioritized_buffer=True,
        **kwargs
    ):
        self.backtrack_steps = backtrack_steps
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
            alpha,
            beta,
            use_prioritized_buffer,
            **kwargs
        )

    def update(self):
        if len(self.memory) < (self.sample_size * self.backtrack_steps):
            return

        starting_indices = None

        if self.use_prioritized_buffer:
            self.cur_beta *= np.exp(-self.beta)

        for _ in range(self.backtrack_steps):
            if self.use_prioritized_buffer:
                states, actions, rewards, next_states, dones, indices, importance_weights = self.memory.sample(1 - self.cur_beta, starting_indices)
                delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
                priorities = (delta.abs().detach().numpy().flatten())
                self.memory.update_priorities(indices, priorities + 1e-6)
                loss = torch.mean((delta * importance_weights) ** 2)
            else:
                states, actions, rewards, next_states, dones, indices = self.memory.sample(starting_indices)
                delta = self._q_learning_loss(states, actions, rewards, next_states, dones)
                loss = torch.mean(delta ** 2)

            starting_indices = indices
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  



