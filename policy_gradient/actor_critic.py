import numpy as np
import torch
import torch.nn as nn
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


class ActorCriticAgent:
    """
    Actor critic with Eligibility Traces

        δ       = R + γ v̂(S', w) − v̂(S, w)          [v̂(terminal) ≡ 0]
        z^w    ← γλ^w z^w + ∇v̂(S, w)
        z^θ    ← γλ^θ z^θ + I ∇ln π(A|S, θ)
        w      ← w  + α^w  δ z^w
        θ      ← θ  + α^θ  δ z^θ
        I      ← γ I

    Parameters:
        state_dim      : int
        action_dim     : int
        hidden_dim     : int
        actor_lr       : float  α^θ
        critic_lr      : float  α^w
        gamma          : float  γ
        lambda_actor   : float  λ^θ ∈ [0, 1]  (0 → one-step, 1 → Monte Carlo)
        lambda_critic  : float  λ^w ∈ [0, 1]
        device         : str
    """

    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, lambda_actor=0.9, lambda_critic=0.9, device='cpu'):
        self.gamma = gamma
        self.lambda_actor = lambda_actor
        self.lambda_critic = lambda_critic
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)

        self.z_actor = [torch.zeros_like(p, device=device) for p in self.actor.parameters()]
        self.z_critic = [torch.zeros_like(p, device=device) for p in self.critic.parameters()]


    def reset_traces(self):
        for z in self.z_actor:
            z.zero_()
        for z in self.z_critic:
            z.zero_()

    def select_action(self, state):
        return self.actor.get_action(state, self.device)

    def update(self, state, log_prob, reward, next_state, terminated, I):
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

        value = self.critic(state_tensor).squeeze()

        with torch.no_grad():
            next_value = self.critic(next_state_tensor).squeeze()

            if terminated:
                next_value = torch.tensor(0.0, device=self.device)

            target = reward + self.gamma * next_value.item()

        delta = target - value.item()
        delta = float(np.clip(delta, -10.0, 10.0))

        self.critic.zero_grad()
        value.backward()

        for z, p in zip(self.z_critic, self.critic.parameters()):
            # z^w ← γλ^w z^w + ∇v̂(S, w)
            z.mul_(self.gamma * self.lambda_critic)
            if p.grad is not None:
                z.add_(p.grad)
            norm = z.norm()
            if norm > 1.0:
                z.mul_(1.0 / norm)
            # w ← w + α^w δ z^w
            p.data.add_(self.critic_lr * delta * z)

        self.actor.zero_grad()
        log_prob.backward()

        for z, p in zip(self.z_actor, self.actor.parameters()):
            # z^θ ← γλ^θ z^θ + I ∇ln π(A|S, θ)
            z.mul_(self.gamma * self.lambda_actor)
            if p.grad is not None:
                z.add_(I * p.grad)

            norm = z.norm()
            if norm > 1.0:
                z.mul_(1.0 / norm)

            # θ ← θ + α^θ δ z^θ
            p.data.add_(self.actor_lr * delta * z)

        return self.gamma * I
