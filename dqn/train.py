import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from dqn.agent import DQNAgent

def train(env_id="CartPole-v1", n_episodes=600, hidden_dim=128, lr=3e-4, gamma=0.99,
          epsilon_start=0.9, epsilon_end = 0.01, epsilon_decay = 0.995, buffer_capacity=10000,
          batch_size=128, target_update_freq=100, print_every=10, save_path=None):
    """
    Creates Gymnasium environment, implements DQNAgent and runs standard interaction loop
    reset -> select_action -> step -> store_transition -> update -> repeat

    Parameters:
        env_id             : str.   Gymnasium environment id
        n_episodes         : int.   Total number of training episodes
        hidden_dim         : int.   Hidden layer size passed to DQNAgent
        lr                 : float. Learning rate
        gamma              : float. Discount factor γ
        epsilon_start      : float. Initial exploration rate
        epsilon_end        : float. Minimum exploration rate
        epsilon_decay      : float. Multiplicative epsilon decay per update
        buffer_capacity    : int.   Replay buffer size
        batch_size         : int.   Minibatch size
        target_update_freq : int.   Steps between target network hard updates
        print_every        : int.   Print progress every N episodes
        save_path          : str or None. If given, save final model weights here

    Returns:
        episode_rewards : list of floats
                          Total reward accumulated in each episode
    """

    device= "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon_start, epsilon_end, epsilon_decay, buffer_capacity, batch_size, target_update_freq, device)

    episode_rewards = []

    for episode_i in range(n_episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)

        episode_reward = 0.0

        while True:
            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            next_state = np.array(next_obs, dtype=np.float32)
            episode_reward += reward

            agent.store_transition(state, action, reward, next_state, done)

            agent.update()

            state = next_state

            if done:
                break
        episode_rewards.append(episode_reward)


        if (episode_i + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            print(
                f"Episode {episode_i + 1:4d} / {n_episodes} | "
                f"Avg reward (last {print_every}): {avg_reward:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
    env.close()

    if save_path is not None:
        agent.save(save_path)
        print(f"\nModel saved to: {save_path}")

    return episode_rewards


def plot_training_curve(episode_rewards, title="DQN Training Curve", window=20):
    """
    Plot total reward per episode with a smoothed moving average overlay.

    Parameters:
        episode_rewards : list of floats. One value per episode
        title           : str. Plot title
        window          : int. Moving average window size
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(episode_rewards, alpha=0.3, color="steelblue", label="raw reward")

    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), smoothed,
                color="steelblue", linewidth=2, label=f"avg ({window} episodes)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()
