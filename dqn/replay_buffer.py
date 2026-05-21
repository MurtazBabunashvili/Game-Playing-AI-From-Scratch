import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Experience Replay buffer (Mnih et al.)

    Stores agent transitions (s, a, r, s', done)

    When buffer is full, oldest transitions are overwritten.

    Parameters:
        capacity : int. Maximum number of transitions to store
                When full, oldest transition is overwritten
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Store one transition in the buffer.
        Each tuple experience at timestep t consists of
        e_t = (s_t, a_t, r_t, s_{t+1}, done_t)

        Parameters:
            state : array-like. Current state s_t
            action : int. Action taken a_t
            next_state : array-like. Next state s_{t+1]
            done : bool. Whether s_{t+1} is terminal
        """
        self.buffer.append((np.array(state, dtype=np.float32), int(action), float(reward), np.array(next_state, dtype=np.float32), bool(done)))

    def sample(self, batch_size):
        """
        Sample random minibatch of transitions uniformly from buffer.

        Parameters:
            batch_size : int. Number of transitions to sample

        Returns:
            states : np.ndarray shape (batch_size, state_dim)
            actions : np.ndarray shape (batch_size,)
            reward : np.ndarray shape (batch_size,)
            next_states : np.ndarray shape (batch_size, state_dim)
            dones       : np.ndarray shape (batch_size,)  bool
        """

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool)
        )

    def __len__(self):
        #Current number of transitions stored
        return len(self.buffer)

    def is_ready(self, batch_size):
        #Retruns true only when buffer has enough transitions to sample full minibatch
        return len(self.buffer) >= batch_size