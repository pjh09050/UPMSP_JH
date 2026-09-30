import torch
import numpy as np
import collections
import random

class ReplayBuffer():  # buffer class
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst), dtype=torch.int64),
                torch.tensor(np.array(r_lst)), torch.tensor(np.array(s_prime_lst), dtype=torch.float), torch.tensor(np.array(done_mask_lst)))

    def size(self):
        return len(self.buffer)


class ReplayBuffer_n_step():
    def __init__(self, buffer_limit, n_step=3, gamma=0.99):
        self.buffer = []
        self.buffer_limit = buffer_limit
        self.n_step_buffer = []
        self.n_step = n_step
        self.gamma = gamma

    def put(self, transition):
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step reward
        R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
        state, action, _, _, done = self.n_step_buffer[0]
        _, _, _, next_state, _ = self.n_step_buffer[-1]

        # Add the n-step transition
        self.buffer.append((state, action, R, next_state, done))

        if len(self.buffer) > self.buffer_limit:
            self.buffer.pop(0)

        self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)

        # Convert list of numpy arrays to numpy.ndarray for efficient tensor conversion
        states = np.array([transition[0] for transition in mini_batch], dtype=np.float32)
        actions = np.array([transition[1] for transition in mini_batch], dtype=np.int64)
        rewards = np.array([transition[2] for transition in mini_batch], dtype=np.float32)
        next_states = np.array([transition[3] for transition in mini_batch], dtype=np.float32)
        dones = np.array([transition[4] for transition in mini_batch], dtype=np.float32)

        return (torch.tensor(states, dtype=torch.float),
                torch.tensor(actions, dtype=torch.long).unsqueeze(1),
                torch.tensor(rewards, dtype=torch.float).unsqueeze(1),
                torch.tensor(next_states, dtype=torch.float),
                torch.tensor(dones, dtype=torch.float).unsqueeze(1))

    def size(self):
        return len(self.buffer)
