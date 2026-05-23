from policy_gradient.train import train_actor_critic, plot_training_curve

if __name__ == "__main__":
    rewards = train_actor_critic(
        env_id="CartPole-v1",
        n_episodes=600,
        hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        print_every=50,
        save_path="actor_critic_cartpole.pth"
    )

    plot_training_curve(rewards, title="Actor-Critic on CartPole-v1", window=50)