from ppo.train import train
from utils.config import PPOConfig
from utils.plotting import plot_training_curve

if __name__ == "__main__":
    cfg = PPOConfig(
        env_id="CartPole-v1",
        n_episodes=500,
        save_path="ppo_cartpole.pth"
    )

    rewards = train(**cfg.to_dict())
    plot_training_curve(rewards, title="PPO on CartPole-v1", window=50)