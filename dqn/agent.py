import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dqn.model import DQNNetwork
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQN Agent combines Q-network, target network and experience replay nito one agent
    The agent interacts with Gymnasium environment using the step() / reset() interface

    Parameters:
        state_dim : int. Dimensionality of observation vector
        action_dim : int. Number of discrete actions
        hidden_dim : int. Hidden layer size
        lr : float. learning rate forr Adam optimizer
        gamma : float. Discount factor γ
        epsilon_start : float. Initial exploration rate
        epsilon_end : float. Minimum exploration rate
        epsilon_decay : float. multiplicative decay applied to every step
        buffer_capacity : int.
        batch_size : int
        target_update_freq : int. how many steps between target network syncs
        device : str. 'cpu' or 'cuda'
    """

    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon_start, epsilon_end, epsilon_decay,
                 buffer_capacity, batch_size, target_update_freq, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        #Q-Network
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)

        #Target network
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.steps_done = 0

    def select_action(self, state):
        """
        ε-greedy action selection

        Parameters:
            state : array-like. Current observation
        Returns:
            action :  int
        """

        if np.random.random() < self.epsilon:
            #Exploration: random action uniformly from action space
            return np.random.randint(self.action_dim)

        #Exploitation
        return self.q_network.get_action(state, self.device)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store one transition (s, a, r, s', done) in replay buffer

        Parameters:
            state      : array-like
            action     : int
            reward     : float
            next_state : array-like
            done       : bool
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Sample random minibatch from replay buffer and perform
        gradient descent step on DQN loss function:
            L(θ) = ( r + γ max_a' Q(s', a'; θ⁻) - Q(s, a; θ) )²

        Returns:
            loss_value: float.
        """

        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Target Q values: r + γ max_a' Q(s', a'; θ⁻)
            max_next_q_values = self.target_network(next_states).max(dim=1)[0]
            #Terminal states has no future rewards so only r
            td_targets = rewards + self.gamma * max_next_q_values * (~dones)

        #Mean squared TD error of  L(θ) = E[(td_target - Q(s,a;θ))²]
        loss = nn.MSELoss()(current_q_values, td_targets)

        #Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #Increment step counter and sync target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.sync_target_network()

        #Decay epsilon after every update step
        self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)

        return loss.item()

    def sync_target_network(self):
        """
        copy q_network weights into target_network
                θ⁻ ← θ
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        #Save q_network weights to disk
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        #Load q_network weights from disk and sync target network
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.sync_target_network()


