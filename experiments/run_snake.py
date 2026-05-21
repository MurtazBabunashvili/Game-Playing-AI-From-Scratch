from environments.snake import SnakeEnv
from dqn.train import train, plot_training_curve

if __name__ == "__main__":
    env = SnakeEnv(grid_size=20)

    rewards = train(
        env_id=env,
        n_episodes=1000,
        hidden_dim=256,
        lr=1e-3,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9997,
        buffer_capacity=100_000,
        batch_size=1000,
        target_update_freq=100,
        print_every=50,
        save_path="snake_dqn.pth"
    )
    plot_training_curve(rewards, title="DQN on Snake", window=50)