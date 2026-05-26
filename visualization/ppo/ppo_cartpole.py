import os
import time
import numpy as np
import gymnasium as gym

from ppo.agent import PPOAgent


def visualize(
    model_path: str = "ppo_cartpole.pth",
    n_episodes: int = 5,
    fps: int = 60,
    hidden_dim: int = 128,
):
    """
    Load saved PPO weights and render CartPole-v1 episodes.

    Parameters:
        model_path : str.  Path to the .pth saved by PPOAgent.save()
        n_episodes : int.  Number of episodes to render
        fps        : int.  Frames per second (controls playback speed)
        hidden_dim : int.  Must match the hidden_dim used during training
    """

    env = gym.make("CartPole-v1", render_mode="human")

    state_dim  = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device="cpu",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    agent.load(model_path)
    agent.actor.eval()
    agent.critic.eval()
    print(f"Loaded weights from: {model_path}\n")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)

        total_reward = 0.0
        steps = 0

        while True:
            action, _, _ = agent.select_action(state)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            state = np.array(obs, dtype=np.float32)

            time.sleep(1.0 / fps)

            if terminated or truncated:
                break

        print(f"Episode {ep:2d} | Steps: {steps:4d} | Reward: {total_reward:.1f}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "experiments", "ppo_cartpole.pth")
    visualize(model_path=MODEL_PATH, n_episodes=10, fps=120)