from policy_gradient.train import train, plot_training_curve

if __name__ == "__main__":
    rewards = train(
        env_id="CartPole-v1",
        n_episodes=600,
        hidden_dim=128,
        lr=1e-3,
        baseline_lr=5e-3,
        gamma=1.0,
        print_every=50,
        save_path="reinforce_cartpole.pth"
    )

    plot_training_curve(rewards, title="REINFORCE on CartPole-v1", window=50)