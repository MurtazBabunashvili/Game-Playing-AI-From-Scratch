from dataclasses import dataclass, asdict
from typing import Optional
import json
import os


@dataclass
class BaseConfig:
    env_id: str = "CartPole-v1"
    n_episodes: int = 500
    hidden_dim: int = 128
    gamma: float = 0.99
    print_every: int = 10
    save_path: Optional[str] = None


    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to: {path}")

    @classmethod
    def load(cls, path:str) -> "BaseConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__}"]
        for k, v in self.to_dict().items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)



@dataclass
class DQNConfig(BaseConfig):
    """
        Hyperparameters for DQNAgent

        lr                learning rate for Adam
        epsilon_start     initial exploration rate ε
        epsilon_end       minimum exploration rate ε
        epsilon_decay     multiplicative decay applied after every update step
        buffer_capacity   replay buffer size (oldest transitions overwritten)
        batch_size        minibatch size sampled from replay buffer
        target_update_freq  steps between hard target-network syncs θ⁻ ← θ
    """

    lr: float = 3e-4

    epsilon_start: float = 0.9
    epsilon_end: float=0.01
    epsilon_decay: float = 0.995

    buffer_capacity: int = 10_000
    batch_size: int = 128

    target_update_freq: int = 100


@dataclass
class REINFORCEConfig(BaseConfig):
    """
    Hyperparameters for REINFORCEAgent + train().

        lr           step size α^θ for the policy network
        baseline_lr  step size α^w for the value baseline network
                     set to 0.0 to disable the baseline entirely
    """

    lr: float=1e-3
    baseline_lr: float = 5e-3
    print_every: int = 50

@dataclass
class ActorCriticConfig(BaseConfig):
    """
    Hyperparameters for ActorCriticAgent + train_actor_critic().

        actor_lr      step size α^θ
        critic_lr     step size α^w
        lambda_actor  eligibility trace decay λ^θ  (0 → one-step, 1 → MC)
        lambda_critic eligibility trace decay λ^w
    """

    actor_lr: float = 1e-4
    critic_lr: float = 5e-4
    lambda_actor: float= 0.9
    lambda_critic: float = 0.9
    print_every: int = 50


@dataclass
class SnakeConfig(DQNConfig):
    """
    DQN config for SnakeEnv
    """
    env_id: str = "snake"
    n_episodes: int = 2_000
    hidden_dim: int = 256
    lr: float = 1e-4
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.9998
    buffer_capacity: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 200
    print_every: int = 50
    save_path: Optional[str] = "snake_10x10_dqn.pth"



REGISTRY = {
    "dqn":          DQNConfig,
    "reinforce":    REINFORCEConfig,
    "actor_critic": ActorCriticConfig,
    "snake":        SnakeConfig
}

def get_config(name: str, **overrides) -> BaseConfig:
    #To instantiate config, use format e.g: cfg = get_config("dqn", lr=1e-3, n_episodes=1000)

    if name not in REGISTRY:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(REGISTRY)}")
    return REGISTRY[name](**overrides)