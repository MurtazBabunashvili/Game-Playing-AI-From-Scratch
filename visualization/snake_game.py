import pygame
from environments.snake import SnakeEnv
from dqn.agent import DQNAgent
import os


def visualize(model_path=None, grid_size=10, cell_size=40, fps=10):
    pygame.init()
    pygame.font.init()
    width = height = grid_size * cell_size
    screen = pygame.display.set_mode((width, height + 40))  # +40 for score bar
    pygame.display.set_caption("Snake DQN")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20)

    env = SnakeEnv(grid_size=grid_size, max_steps=500)

    # Load trained agent or use random
    if model_path:
        agent = DQNAgent(
            state_dim=23, action_dim=3, hidden_dim=256,
            lr=3e-4, gamma=0.95,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
            buffer_capacity=1, batch_size=1,
            target_update_freq=999999, device="cpu"
        )
        agent.load(model_path)
        agent.epsilon = 0.0
    else:
        agent = None  # Will use random actions

    COLORS = {
        "bg":    (30,  30,  30),
        "grid":  (45,  45,  45),
        "head":  (0,   200, 100),
        "body":  (0,   140, 70),
        "food":  (220, 60,  60),
        "score": (255, 255, 255),
    }

    running = True
    while running:
        obs, _ = env.reset()
        episode_done = False

        while not episode_done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            # Agent picks action
            if agent:
                action = agent.select_action(obs)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated

            screen.fill(COLORS["bg"])

            # Grid lines
            for i in range(grid_size + 1):
                pygame.draw.line(screen, COLORS["grid"],
                                 (i * cell_size, 0), (i * cell_size, height))
                pygame.draw.line(screen, COLORS["grid"],
                                 (0, i * cell_size), (width, i * cell_size))

            # Food
            fr, fc = env.food
            pygame.draw.rect(screen, COLORS["food"],
                             (fc * cell_size + 2, fr * cell_size + 2,
                              cell_size - 4, cell_size - 4), border_radius=4)

            # Snake
            for i, (r, c) in enumerate(env.snake):
                color = COLORS["head"] if i == 0 else COLORS["body"]
                pygame.draw.rect(screen, color,
                                 (c * cell_size + 2, r * cell_size + 2,
                                  cell_size - 4, cell_size - 4), border_radius=3)

            # Score bar
            pygame.draw.rect(screen, (20, 20, 20), (0, height, width, 40))
            txt = font.render(
                f"Score: {info['score']}   Steps: {env.steps_done}   "
                f"Length: {len(env.snake)}",
                True, COLORS["score"]
            )
            screen.blit(txt, (10, height + 10))

            pygame.display.flip()
            clock.tick(fps)
    pygame.quit()




if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "experiments", "snake_10x10_dqn.pth")
    print(f"Loading model from: {MODEL_PATH}")
    visualize(model_path=MODEL_PATH, grid_size=10, fps=12)