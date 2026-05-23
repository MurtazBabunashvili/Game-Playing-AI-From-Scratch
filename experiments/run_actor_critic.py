from policy_gradient.train import train_actor_critic, plot_training_curve

if __name__ == "__main__":

    rewards = train_actor_critic(
        env_id="Acrobot-v1",
        n_episodes=2000,
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
        lambda_actor=0.9,
        lambda_critic=0.9,
        print_every=50,
        save_path="actor_critic_acrobot.pth"
    )

    plot_training_curve(rewards, title="Acrobot: (Eligibility Traces) on Acrobot-v1", window=50)
