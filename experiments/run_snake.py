from environments.snake import SnakeEnv
from utils.config import SnakeConfig
from dqn.train import train
from utils.plotting import  plot_training_curve

if __name__ == "__main__":
    cfg = SnakeConfig()
    env = SnakeEnv(grid_size=10, max_steps=1000)
    rewards = train(env_id=env, **{k: v for k, v in cfg.to_dict().items() if k != "env_id"})
    plot_training_curve(rewards, title="DQN on Snake", window=50)