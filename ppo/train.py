import numpy as np
import gymnasium as gym
import torch
from ppo.agent import PPOAgent
from utils.logger import TrainingLogger

def train(env_id="CartPole-v1", n_episodes=1000, hidden_dim=128, actor_lr=3e-4, critic_lr=1e-3,
          gamma=0.99, clip_epsilon=0.2, n_epochs=10, batch_size=64, entropy_coef=0.01,
          gae_lambda=0.95, update_interval=2048, print_every=50, save_path=None):
    """
    PPO training loop.

    Parameters:
        env_id          : str.   Gymnasium environment id
        n_episodes      : int.   Total training episodes
        hidden_dim      : int.   Hidden layer size
        actor_lr        : float. α_A — actor learning rate
        critic_lr       : float. α_C — critic learning rate
        gamma           : float. Discount factor γ
        clip_epsilon    : float. ε — clipping range
        n_epochs        : int.   K — update epochs per batch
        batch_size      : int.   M — minibatch size
        entropy_coef    : float. β — entropy regularization weight
        gae_lambda      : float. λ — GAE trace decay
        update_interval : int.   T — timesteps collected before each update
        print_every     : int.   Print progress every N episodes
        save_path       : str or None. Save final actor weights here
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    logger = TrainingLogger(run_name=f"ppo_{env_id}", print_every=print_every, window=50, log_dir="runs")

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, clip_epsilon, n_epochs, batch_size, entropy_coef, gae_lambda, device)

    episode_rewards = []
    timesteps_since_update = 0
    last_policy_loss = 0.0
    last_value_loss = 0.0

    for episode_i in range(n_episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)

        episode_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(state)
            done = terminated or truncated
            next_state = np.array(next_obs, dtype=np.float32)

            agent.store_transition(state, action, log_prob, reward, done, value)

            episode_reward += reward
            state = next_state
            timesteps_since_update += 1

            if timesteps_since_update >= update_interval:
                last_policy_loss, last_value_loss = agent.update(next_state, done)
                timesteps_since_update = 0

        episode_rewards.append(episode_reward)

        logger.log(episode_i, episode_reward, policy_loss=last_policy_loss, value_loss=last_value_loss)

    if len(agent.buffer_states) > 0:
        agent.update(state, done)

    env.close()

    if save_path is not None:
        agent.save(save_path)
        print(f"\n Model saved to: {save_path}")

    logger.print_summary()
    logger.close()

    return logger.rewards