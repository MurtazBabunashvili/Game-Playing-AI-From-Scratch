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


class ValueNetwork(nn.Module):
    """
    State-value function approximator v̂(s, w)

    Parameters:
        state_dim: int
        hidden_dim : int
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


class REINFORCEAgent:
    """
    REINFORCE: Monte Carlo Policy Gradient

    Update rules:
        δ  = Gt - v̂(St, w)
        w  ← w  + α^w γ^t δ ∇v̂(St, w)
        θ  ← θ  + α^θ γ^t δ ∇ ln π(At|St, θ)

    Parameters:
        state_dim: int
        action_dim : int
        hidden_dim : int
        lr : float. Step size α^θ for policy
        baseline_lr: float. Step size α^w for value network.
        gamma : float. Discount factor
        device : str
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, baseline_lr = 1e-3, gamma=0.99, device="cpu"):
        self.gamma = gamma
        self.device = device
        self.use_baseline = baseline_lr > 0

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        if self.use_baseline:
            self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=baseline_lr)

    def select_action(self, state):
        #Sample At ~ π(·|St, θ)
        # return action and log_prob

        return self.policy.get_action(state, self.device)

    def update(self, log_probs, rewards, states):
        #REINFORCE with baseline update


        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        #Compute baseline for each t
        if self.use_baseline:
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            values = self.value_net(states_tensor)
            deltas = returns - values.detach() # δ = Gt - v̂(St, w)
        else:
            deltas = returns


        # Build loss: L = -Σ_t γ^t Gt ln π(At|St, θ)
        policy_loss = 0.0
        for t, (log_prob, delta) in enumerate(zip(log_probs, deltas)):
            gamma_t = self.gamma ** t
            policy_loss += -gamma_t * delta * log_prob

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        if self.use_baseline:
            value_loss = nn.MSELoss()(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        return policy_loss.item()
