import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from policy_gradient.reinforce import REINFORCEAgent
from policy_gradient.actor_critic import ActorCriticAgent

def train(env_id="CartPole-v1", n_episodes=1000, hidden_dim=128, lr=1e-3, baseline_lr=1e-3, gamma=0.99, print_every=50, save_path=None):
    """
    REINFORCE training loop

    Parameters:
        env_id: str.
        n_episodes: int. number of episodes
        hidden-dim : int. Policy network hidden layer size
        lr : float. learning rate alpha
        gamma : float. Discount factor
        print_every: int. prints progress every N episodes
        save_path: str or None. Save final policy weights

    Returns:
        episode_rewards : list of floats. Total reward per episode
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, hidden_dim, lr, baseline_lr, gamma, device)

    episode_rewards = []

    for episode_i in range(n_episodes):
        #Generate full episodes
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)

        log_probs = []
        rewards = []

        states = []

        while True:

            states.append(state)

            action, log_prob = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = np.array(next_obs, dtype=np.float32)

            if terminated or truncated:
                break
        agent.update(log_probs, rewards, states)

        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)

        if (episode_i + 1) % print_every == 0:
            avg = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode_i + 1:4d} / {n_episodes}. Average reward (for last {print_every}): {avg:7.2f}")

    env.close()

    if save_path is not None:
        torch.save(agent.policy.state_dict(), save_path)
        print(f"\nPolicy saved to: {save_path}")
    return episode_rewards


def train_actor_critic(env_id="CartPole-v1", n_episodes=600, hidden_dim=128,
                       actor_lr = 1e-3, critic_lr=5e-3, gamma=0.99,
                       print_every=50, save_path=None):
    #One step Actor-Critic training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, device)

    episode_rewards = []

    for episode_i in range(n_episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)

        I = 1.0
        episode_reward = 0.0

        while True:
            action, log_prob = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            next_state = np.array(next_obs, dtype=np.float32)

            I = agent.update(state, log_prob, reward, next_state, done, I)

            episode_reward += reward
            state = next_state

            if done:
                break
        episode_rewards.append(episode_reward)

        if (episode_i + 1) % print_every == 0:
            avg = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode_i + 1:4d} / {n_episodes}. Average reward (for last {print_every} episodes): {avg:7.2f}")
    env.close()

    if save_path is not None:
        torch.save(agent.actor.state_dict(), save_path)
        print(f"\nActor saved to {save_path}")

    return episode_rewards



def plot_training_curve(episode_rewards, title="REINFORCE Training Curve", window=50):
    """
    Plot total reward per episode with smoothed moving average overlay.

    Parameters:
        episode_rewards : list of floats
        title           : str
        window          : int. Moving average window
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episode_rewards, alpha=0.2, color="steelblue", label="raw reward")

    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), smoothed,
                color="steelblue", linewidth=2, label=f"avg ({window} episodes)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reinforce_curve.png", dpi=150)
    plt.show()
    print("Plot saved to reinforce_curve.png")

