from dqn.train import train, plot_training_curve

if __name__ == "__main__":
    rewards = train(
        env_id="CartPole-v1",
        n_episodes=600,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=128,
        target_update_freq=100,
        print_every=10,
        save_path="cartpole_dqn.pth"
    )
    plot_training_curve(rewards, title="DQN on CartPole-v1", window=20)