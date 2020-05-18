from collections import deque
import numpy as np
import random


class ReplayBuffer():
    def __init__(self):
        """ReplayBuffer class for training an off-policy RL model."""
        self.replay_buffer = deque()

    def add(self, state, action, reward, next_state, done):
        """Adds trajectory data to the replay buffer as an experience tuple."""
        self.replay_buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        """Samples batch_size number of trajectories."""
        batch = random.sample(self.replay_buffer, batch_size)

        states = np.array([_[0] for _ in batch])
        actions = np.array([_[1] for _ in batch])
        rewards = np.array([_[2] for _ in batch])
        next_states = np.array([_[3] for _ in batch])
        dones = np.array([_[4] for _ in batch])

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.replay_buffer)
