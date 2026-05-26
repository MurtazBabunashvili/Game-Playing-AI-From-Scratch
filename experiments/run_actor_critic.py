from policy_gradient.train import train_actor_critic
from utils.plotting import plot_training_curve
from utils.config import  ActorCriticConfig

if __name__ == "__main__":

    cfg = ActorCriticConfig(
        env_id="Acrobot-v1",
        n_episodes=2000,
        save_path="run_actor_acrobot.pth"
    )

    rewards = train_actor_critic(**cfg.to_dict())

    plot_training_curve(rewards, title="Acrobot: (Eligibility Traces) on Acrobot-v1", window=50)
