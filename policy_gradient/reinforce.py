import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Parametrized policy π(a|s, θ)

    Architecture:
        Input -> state vector
        Hidden -> fully connected layers with ReLU
        Output -> soft-max applied by Categorical

    Parameter:
        state_dim: int. Dimensionality of observation vector
        action_dim: int. Number of discrete actions
        hidden_dim : int. Hidden layer size
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                     nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        return self.network(x)

    def get_action(self, state, device):
        """
        Sample action from π(·|s, θ) and return log probability.

        Parameters:
            state : array-like. Current observation
            device : str

        Returns:
            action : int
            log_prob: torch. Tensor scalar ln π(At|St, θ) (eligibility vector)
        """

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits = self.forward(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class REINFORCEAgent:
    """
    REINFORCE: Monte Carlo Policy Gradient

    Update rule:
        θ ← θ + α γ^t Gt ∇ ln π(At|St, θ)

        Parameters:
            state_dim: int
            action_dim : int
            hidden_dim : int
            lr : float. Step size alpha
            gamma : float. Discount factor
            device : str
    """

    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, device):
        self.gamma = gamma
        self.device = device
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        #Sample At ~ π(·|St, θ)
        # return action and log_prob

        return self.policy.get_action(state, self.device)

    def update(self, log_probs, rewards):
        #REINFORCE update
        T = len(rewards)

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Build loss: L = -Σ_t γ^t Gt ln π(At|St, θ)
        loss = 0.0
        for t, (log_prob, G_t) in enumerate(zip(log_probs, returns)):
            gamma_t = self.gamma ** t  # γ^t
            loss += -gamma_t * G_t * log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
