import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
    """
    Actor: parametrized policy π(a|s, θ_A)

    Parameters:
        state_dim : int. Dimensionality of observation vector
        action_dim: int. Number of discrete actions
        hidden_dim: int. Hidden layer size
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, state, device):
        """
        Sample action from π(·|s, θ_A) and return log probability and entropy

        Parameters:
            state  : array-like. Current observation
            device : str

        Returns:
            action   : int
            log_prob : torch.Tensor scalar  ln π(At|St, θ_A)
            entropy  : torch.Tensor scalar  H(π(·|St, θ_A))
        """

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits = self.forward(state_tensor)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        entropy = dist.entropy()

        return action.item(), log_prob, entropy

    def evaluate_actions(self, states, actions):
        """
        Evaluate log probabilities and entropies for batch of (state, action) pairs
        Parameters:
            states  : torch.Tensor shape (batch, state_dim)
            actions : torch.Tensor shape (batch,)

        Returns:
            log_probs : torch.Tensor shape (batch,)
            entropies : torch.Tensor shape (batch,)
        """

        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, entropies

class CriticNetwork(nn.Module):
    """
    Critic: state-value function approximator v̂(s, θ_C)

    Parameters:
        state_dim: int
        hidden_dim: int
    """

    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.network(x).unsqueeze(-1)