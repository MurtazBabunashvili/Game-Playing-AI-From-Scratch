from dqn.train import train
from utils.plotting import  plot_training_curve
from utils.config import DQNConfig


if __name__ == "__main__":
    cfg = DQNConfig(save_path="cartpole_dqn.pth")
    rewards = train(**cfg.to_dict())
    plot_training_curve(rewards, title="DQN on CartPole-v1", window=20)