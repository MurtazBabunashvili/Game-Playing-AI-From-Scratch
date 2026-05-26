import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.pong import PongEnv
from ppo.agent import PPOAgent


MODEL_NAMES = [
    "solved_rule_defense_ppo_pong.pth",
    "best_rule_survival_ppo_pong.pth",
    "best_rule_window_ppo_pong.pth",
    "ppo_pong_selfplay.pth",
    "latest_checkpoint_ppo_pong.pth",
    "best_mixed_ppo_pong.pth",
    "best_eval_ppo_pong.pth",
]


def greedy_action(agent, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        logits = agent.actor(state_tensor)
        return int(torch.argmax(logits, dim=-1).item())


def rule_based_action(obs):
    paddle_y = obs[0]
    ball_y = obs[3]

    if ball_y < paddle_y - 0.03:
        return 1
    if ball_y > paddle_y + 0.03:
        return 2
    return 0


class PongRenderer:
    def __init__(self, env, scale=5):
        pygame.init()
        pygame.font.init()

        self.env = env
        self.scale = scale
        self.width = int(env.width * scale)
        self.height = int(env.height * scale)
        self.info_h = 70

        self.screen = pygame.display.set_mode((self.width, self.height + self.info_h))
        pygame.display.set_caption("PPO Pong Model Tester")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18)

    def draw(self, model_name, episode, score_left, score_right, hits):
        self.screen.fill((20, 20, 20))

        pygame.draw.line(
            self.screen,
            (80, 80, 80),
            (self.width // 2, 0),
            (self.width // 2, self.height),
            2,
        )

        paddle_w = int(self.env.paddle_w * self.scale)
        paddle_h = int(self.env.paddle_h * self.scale)

        player_rect = pygame.Rect(
            0,
            int((self.env.player_y - self.env.paddle_h / 2) * self.scale),
            paddle_w,
            paddle_h,
        )

        opponent_rect = pygame.Rect(
            self.width - paddle_w,
            int((self.env.opponent_y - self.env.paddle_h / 2) * self.scale),
            paddle_w,
            paddle_h,
        )

        pygame.draw.rect(self.screen, (240, 240, 240), player_rect)
        pygame.draw.rect(self.screen, (240, 240, 240), opponent_rect)

        pygame.draw.circle(
            self.screen,
            (240, 240, 240),
            (int(self.env.ball_x * self.scale), int(self.env.ball_y * self.scale)),
            max(3, int(2 * self.scale)),
        )

        line1 = self.font.render(
            f"Model: {model_name}",
            True,
            (240, 240, 240),
        )
        line2 = self.font.render(
            f"Episode {episode} | Left PPO {score_left} : {score_right} Rule Bot | Hits: {hits}",
            True,
            (240, 240, 240),
        )
        line3 = self.font.render(
            "Keys: N=next model | R=restart model | Q/ESC=quit",
            True,
            (180, 180, 180),
        )

        self.screen.blit(line1, (12, self.height + 5))
        self.screen.blit(line2, (12, self.height + 27))
        self.screen.blit(line3, (12, self.height + 49))

        pygame.display.flip()

    def close(self):
        pygame.quit()


def load_agent(model_path, env, hidden_dim=128):
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
        device="cpu",
    )

    agent.load(str(model_path))
    agent.actor.eval()
    agent.critic.eval()

    return agent


def existing_models():
    paths = []

    for name in MODEL_NAMES:
        path = ROOT_DIR / "experiments" / name
        if path.exists():
            paths.append(path)
        else:
            print(f"Skipping missing model: {name}")

    return paths


def test_model_against_rule(
    model_path,
    renderer,
    env,
    n_episodes=10,
    fps=60,
    hidden_dim=128,
    deterministic=True,
):
    agent = load_agent(model_path, env, hidden_dim=hidden_dim)

    score_left = 0
    score_right = 0
    episode = 1

    while episode <= n_episodes:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        done = False
        hits = 0
        reward = 0.0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "quit"
                    if event.key == pygame.K_n:
                        return "next"
                    if event.key == pygame.K_r:
                        score_left = 0
                        score_right = 0
                        episode = 1
                        return "restart"

            opponent_state = env.get_obs_opponent()

            if deterministic:
                player_action = greedy_action(agent, state)
            else:
                player_action, _, _ = agent.select_action(state)

            opponent_action = rule_based_action(opponent_state)

            prev_ball_vx = env.ball_vx

            next_state, reward, terminated, truncated, _ = env.step(
                (player_action, opponent_action)
            )

            if not terminated and prev_ball_vx < 0 and env.ball_vx > 0:
                hits += 1

            state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated

            renderer.draw(
                model_name=model_path.name,
                episode=episode,
                score_left=score_left,
                score_right=score_right,
                hits=hits,
            )
            renderer.clock.tick(fps)

        if reward > 0:
            score_left += 1
        elif reward < 0:
            score_right += 1

        print(
            f"{model_path.name} | Episode {episode:2d} | "
            f"reward={reward:+.1f} | hits={hits:4d} | "
            f"score {score_left}:{score_right}"
        )

        episode += 1
        time.sleep(0.35)

    return "next"


def visualize_all(
    n_episodes_per_model=10,
    fps=60,
    hidden_dim=128,
    deterministic=True,
):
    model_paths = existing_models()

    if not model_paths:
        raise FileNotFoundError("No Pong model files found in experiments/")

    env = PongEnv(max_steps=2000)
    renderer = PongRenderer(env)

    try:
        index = 0

        while index < len(model_paths):
            model_path = model_paths[index]
            print(f"\nTesting model: {model_path.name}")

            result = test_model_against_rule(
                model_path=model_path,
                renderer=renderer,
                env=env,
                n_episodes=n_episodes_per_model,
                fps=fps,
                hidden_dim=hidden_dim,
                deterministic=deterministic,
            )

            if result == "quit":
                break

            if result == "restart":
                continue

            index += 1

    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    visualize_all(
        n_episodes_per_model=10,
        fps=60,
        hidden_dim=128,
        deterministic=True,
    )