import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo.model import ActorNetwork, CriticNetwork

class PPOAgent:
    """
    PPO agent with clipped surrogate objective

    Parameters:
        state_dim          : int
        action_dim         : int
        hidden_dim         : int
        actor_lr           : float. α_A
        critic_lr          : float. α_C
        gamma              : float. Discount factor γ
        clip_epsilon       : float. ε for clipping neighborhood |r_t(θ) - 1| ≤ ε
        n_epochs           : int.   K — number of update epochs per batch
        batch_size         : int.   M — minibatch size
        entropy_coef       : float. β — entropy regularization weight
        gae_lambda         : float. λ for Generalized Advantage Estimation
        device             : str
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99,
                 clip_epsilon=0.2, n_epochs=10, batch_size=64, entropy_coef=0.01, gae_lambda=0.95, device='cpu'):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size= batch_size
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.device = device

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.reset_buffer()
    def reset_buffer(self):
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_log_probabilities = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_values = []

    def select_action(self, state):
        """
        Sample At ~ π(·|St, θ_A).

        Parameters:
            state : array-like

        Returns:
            action   : int
            log_prob : torch.Tensor scalar
            value    : float
        """

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state, self.device)
            value = self.critic(state_tensor).item()

        return action, log_prob, value

    def store_transition(self, state, action, log_prob, reward, done, value):
        """
        Store one transition in the trajectory buffer

        Parameters:
            state    : array-like
            action   : int
            log_prob : torch.Tensor scalar
            reward   : float
            done     : bool
            value    : float  V̂(St) from critic at collection time
        """
        self.buffer_states.append(np.array(state, dtype=np.float32))
        self.buffer_actions.append(action)
        self.buffer_log_probabilities.append(log_prob.items())
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)
        self.buffer_values.append(value)

    def compute_gae(self, next_value):
        """
        Parameters:
            next_value : float. V̂(S_{T+1}) from critic — 0 if terminal

        Returns:
            advantages : np.ndarray shape (T,)
            returns    : np.ndarray shape (T,)  — used as V-targets
        """
        rewards = self.buffer_rewards
        dones = self.buffer_dones
        values = self.buffer_values
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t+1]
            next_done = dones[t]

            #TD error δ_t = r_t + γ V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]

            #GAE: A_t = δ_t + γλ * (1 - done) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)

        return advantages, returns

    def update(self, next_state, done):
        """
        Run K epochs of minibatch PPO updates over the collected buffer

        Parameters:
            next_state : array-like. S_{T+1} — state after last stored transition
            done       : bool. Whether S_{T+1} is terminal

        Returns:
            mean_policy_loss : float
            mean_value_loss  : float
        """

        if done:
            next_value = 0.0
        else:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_value = self.critic(next_state_tensor).item()

        advantages, returns = self.compute_gae(next_value)

        states = torch.FloatTensor(np.array(self.buffer_states)).to(self.device)
        actions = torch.LongTensor(self.buffer_actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer_log_probabilities).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        T = len(self.buffer_states)
        indices = np.arange(T)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_updates = 0
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, T, self.batch_size):
                mb_index = indices[start: start + self.batch_size]

                mb_states = states[mb_index]
                mb_actions = actions[mb_index]
                mb_old_lp = old_log_probs[mb_index]
                mb_advantages = advantages_t[mb_index]
                mb_returns = returns_t[mb_index]

                new_log_probs, entropies = self.actor.evaluate_actions(mb_states, mb_actions)

                #probability ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratios = torch.exp(new_log_probs - mb_old_lp)

                #J^CLIP = E_t[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
                surr1 = ratios * mb_advantages * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)*mb_advantages

                j_clip = torch.min(surr1, surr2).mean()

                #Policy loss: L_pol = -J^CLIP - β*H
                entropy_bonus = entropies.mean()
                policy_loss = -j_clip - self.entropy_coef * entropy_bonus

                #Value loss: L_val = MSE(V̂(s), V^π_tar(s))
                values_pred = self.critic(mb_states)
                value_loss = nn.MSELoss()(values_pred, mb_returns)

                #gradient steps
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_updates += 1
        self.reset_buffer()

        return total_policy_loss/n_updates, total_value_loss/n_updates

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])



