import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Actor: parametrized policy π(a|s, θ)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, state, device):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits = self.forward(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class ValueNetwork(nn.Module):
    """
    Critic: state-value function approximator v̂(s, w)
    """

    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def get_value(self, state, device):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            return self.forward(state_tensor).item()

class ActorCriticAgent:
    """
    One-step Actor-Critic

    Update rules per step:
        δ  = R + γv̂(S', w) - v̂(S, w)
        w  ← w  + α^w δ ∇v̂(S, w)
        θ  ← θ  + α^θ I δ ∇ ln π(A|S, θ)
        I  ← γI

    Parameters:
        state_dim   : int
        action_dim  : int
        hidden_dim  : int
        actor_lr    : float. Step size α^θ
        critic_lr   : float. Step size α^w
        gamma       : float. Discount factor γ
        device      : str
    """

    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, device):
        self.gamma = gamma
        self.device = device

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        return self.actor.get_action(state, self.device)

    def update(self, state, log_prob, reward, next_state, done, I):
        """
        One step actor-critic update

        Parameters:
            state: nd.ndarray
            log_prob   : torch.Tensor — ln π(A|S, θ)
            reward: float
            next_state: np.ndarray
            done: bool
            I: float - current discount accumulator

        Returns:
            new_I: float - updated I for next step
        """

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        value = self.critic(state_tensor)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor).detach()

        # If S' is terminal then v̂(S', w) = 0
        if done:
            next_value = torch.zeros(1).to(self.device)

        # δ = R + γv̂(S', w) - v̂(S, w)
        target = reward + self.gamma * next_value
        delta = (target - value).item()


        critic_loss = nn.MSELoss()(value, target.detach() if not done else torch.FloatTensor([reward]).to(self.device))
        actor_loss = -I * delta * log_prob


        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return self.gamma * I

