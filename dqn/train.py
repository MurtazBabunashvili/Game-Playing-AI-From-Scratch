import gymnasium as gym
import numpy as np
import torch

from dqn.agent import DQNAgent
from utils.logger import TrainingLogger


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

    logger = TrainingLogger(run_name=f"dqn_{env_id}", print_every=print_every, log_dir="runs")

    if isinstance(env_id, str):
        env = gym.make(env_id)
    else:
        env = env_id

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

            loss = agent.update()

            state = next_state

            if done:
                break
        episode_rewards.append(episode_reward)


        logger.log(episode_i, episode_reward, epsilon=agent.epsilon, loss=loss or 0.0)

    env.close()

    if save_path is not None:
        agent.save(save_path)
        print(f"\nModel saved to: {save_path}")

    logger.print_summary()
    logger.close()

    return logger.rewards
