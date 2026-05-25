import numpy as np
import gymnasium as gym
import torch

from policy_gradient.reinforce import REINFORCEAgent
from policy_gradient.actor_critic import ActorCriticAgent
from utils.logger import TrainingLogger


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

    logger = TrainingLogger(run_name=f"dqn_{env_id}", print_every=print_every, log_dir="runs")

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

        logger.log(episode_i, episode_reward)

    env.close()

    if save_path is not None:
        torch.save(agent.policy.state_dict(), save_path)
        print(f"\nPolicy saved to: {save_path}")

    logger.print_summary()
    logger.close()
    return logger.rewards

def train_actor_critic(env_id="CartPole-v1", n_episodes=600, hidden_dim=128,
                       actor_lr=1e-3, critic_lr=5e-3, gamma=0.99,
                       lambda_actor=0.9, lambda_critic=0.9,
                       print_every=50, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    logger = TrainingLogger(run_name=f"actor_critic_{env_id}", print_every=print_every, log_dir="runs")

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(state_dim, action_dim, hidden_dim,
                             actor_lr, critic_lr, gamma,
                             lambda_actor, lambda_critic, device)

    episode_rewards = []

    for episode_i in range(n_episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)

        agent.reset_traces()   # ← z^θ ← 0, z^w ← 0  at episode start
        I = 1.0
        episode_reward = 0.0

        while True:
            action, log_prob = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            next_state = np.array(next_obs, dtype=np.float32)

            I = agent.update(state, log_prob, reward, next_state, terminated, I)

            episode_reward += reward
            state = next_state

            if done:
                break
        episode_rewards.append(episode_reward)

        logger.log(episode_i, episode_reward)

    env.close()

    if save_path is not None:
        torch.save(agent.actor.state_dict(), save_path)
        print(f"\nActor saved to {save_path}")

    logger.print_summary()
    logger.close()
    return logger.rewards

