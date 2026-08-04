# 保存和采样五元组
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device

        self.states = np.zeros(
            (capacity, state_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (capacity, action_dim), dtype=np.float32
        )
        self.rewards = np.zeros(
            (capacity, 1), dtype=np.float32
        )
        self.next_states = np.zeros(
            (capacity, state_dim), dtype=np.float32
        )
        self.dones = np.zeros(
            (capacity, 1), dtype=np.float32
        )
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        index = self.position

        self.states[index] = np.asarray(
            state, dtype=np.float32
        ).reshape(-1)

        self.actions[index] = np.asarray(
            action, dtype=np.float32
        ).reshape(-1)

        self.rewards[index, 0] = float(reward)

        self.next_states[index] = np.asarray(
            next_state, dtype=np.float32
        ).reshape(-1)

        self.dones[index, 0] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError(
                f"buffer size {self.size} < batch size {batch_size}"
            )

        indices = np.random.randint(
            low=0,
            high=self.size,
            size=batch_size,
        )

        states = torch.as_tensor(
            self.states[indices],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            self.actions[indices],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(
            self.rewards[indices],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            self.next_states[indices],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            self.dones[indices],
            dtype=torch.float32,
            device=self.device,
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size
