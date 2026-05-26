import os
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


def greedy_action(agent, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        logits = agent.actor(state_tensor)
        return int(torch.argmax(logits, dim=-1).item())


class PongRenderer:
    def __init__(self, env, scale=5):
        pygame.init()
        pygame.font.init()

        self.env = env
        self.scale = scale
        self.width = int(env.width * scale)
        self.height = int(env.height * scale)
        self.info_h = 40

        self.screen = pygame.display.set_mode((self.width, self.height + self.info_h))
        pygame.display.set_caption("PPO Self-play Pong")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 22)

    def draw(self, episode, score_left, score_right):
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

        text = self.font.render(
            f"Episode {episode}   Left {score_left} : {score_right} Right",
            True,
            (240, 240, 240),
        )
        self.screen.blit(text, (12, self.height + 8))
        pygame.display.flip()

    def close(self):
        pygame.quit()


def visualize(
    model_path=None,
    n_episodes=10,
    fps=60,
    hidden_dim=128,
    deterministic=True,
):
    if model_path is None:
        model_path = ROOT_DIR / "experiments" / "ppo_pong_selfplay.pth"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    env = PongEnv(max_steps=2000)
    renderer = PongRenderer(env)

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
        device="cpu",
    )
    agent.load(str(model_path))
    agent.actor.eval()
    agent.critic.eval()

    score_left = 0
    score_right = 0

    try:
        for episode in range(1, n_episodes + 1):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            done = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                opponent_state = env.get_obs_opponent()

                if deterministic:
                    player_action = greedy_action(agent, state)
                    opponent_action = greedy_action(agent, opponent_state)
                else:
                    player_action, _, _ = agent.select_action(state)
                    opponent_action, _, _ = agent.select_action(opponent_state)

                next_state, reward, terminated, truncated, _ = env.step((player_action, opponent_action))
                state = np.array(next_state, dtype=np.float32)
                done = terminated or truncated

                renderer.draw(episode, score_left, score_right)
                renderer.clock.tick(fps)

            if reward > 0:
                score_left += 1
            elif reward < 0:
                score_right += 1

            time.sleep(0.4)

    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    MODEL_PATH = ROOT_DIR / "experiments" / "ppo_pong_selfplay.pth"
    visualize(model_path=MODEL_PATH, n_episodes=10, fps=60, deterministic=True)
