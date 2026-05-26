import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PongEnv(gym.Env):
    """
    1v1 Pong environment for self-play PPO training.

    S - State: [player_y, opponent_y, ball_x, ball_y, ball_vx, ball_vy]
                all values normalized to [-1, 1]
    A - Actions: {0: stay, 1: move up, 2: move down}
    R - Reward:
        +1  for scoring (ball passes opponent's side)
        -1  for conceding (ball passes player's side)
         0  otherwise

    Parameters:
        width      : int. Field width in abstract units
        height     : int. Field height in abstract units
        paddle_h   : int. Paddle height
        paddle_w   : int. Paddle width
        ball_speed : float. Initial ball speed
        paddle_speed: float. Paddle movement speed per step
        max_steps  : int. Max steps per episode before truncation
    """

    STAY      = 0
    MOVE_UP   = 1
    MOVE_DOWN = 2

    def __init__(self, width=160, height=120, paddle_h=20, paddle_w=5,
                 ball_speed=3.0, paddle_speed=4.0, max_steps=2000):

        super().__init__()

        self.width        = width
        self.height       = height
        self.paddle_h     = paddle_h
        self.paddle_w     = paddle_w
        self.ball_speed   = ball_speed
        self.paddle_speed = paddle_speed
        self.max_steps    = max_steps

        # 6-dim normalized observation
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # stay, up, down

        # game state
        self.player_y   = None
        self.opponent_y = None
        self.ball_x     = None
        self.ball_y     = None
        self.ball_vx    = None
        self.ball_vy    = None
        self.steps_done = 0
        self.score_player   = 0
        self.score_opponent = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_y   = self.height / 2.0
        self.opponent_y = self.height / 2.0

        self.ball_x = self.width  / 2.0
        self.ball_y = self.height / 2.0

        # random launch angle, always moves horizontally
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        direction = np.random.choice([-1.0, 1.0])
        self.ball_vx = direction * self.ball_speed * np.cos(angle)
        self.ball_vy = self.ball_speed * np.sin(angle)

        self.steps_done = 0

        return self._get_obs_player(), {}

    def step(self, actions):
        """
        Advance one timestep.

        Parameters:
            actions : tuple (player_action, opponent_action)
                      or single int (player_action) — opponent stays

        Returns:
            obs        : np.ndarray shape (6,)  — player perspective
            reward     : float                  — player reward
            terminated : bool
            truncated  : bool
            info       : dict with 'score_player', 'score_opponent'
        """

        if isinstance(actions, (tuple, list)):
            player_action, opponent_action = actions
        else:
            player_action  = actions
            opponent_action = self.STAY

        # Move paddles
        self.player_y   = self._move_paddle(self.player_y,   player_action)
        self.opponent_y = self._move_paddle(self.opponent_y, opponent_action)

        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Top / bottom wall bounce
        half = self.paddle_h / 2.0
        if self.ball_y <= 1.0:
            self.ball_y  = 1.0
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= self.height - 1.0:
            self.ball_y  = self.height - 1.0
            self.ball_vy = -abs(self.ball_vy)

        reward     = 0.0
        terminated = False

        # Left paddle collision (player)
        if self.ball_x <= self.paddle_w + 1.0:
            if abs(self.ball_y - self.player_y) <= self.paddle_h / 2.0:
                self.ball_x  = self.paddle_w + 1.0
                self.ball_vx = abs(self.ball_vx) * 1.02   # slight speed-up
                # deflect based on hit position
                offset = (self.ball_y - self.player_y) / (self.paddle_h / 2.0)
                self.ball_vy = offset * self.ball_speed
            else:
                # player missed — opponent scores
                reward = -1.0
                self.score_opponent += 1
                terminated = True

        # Right paddle collision (opponent)
        elif self.ball_x >= self.width - self.paddle_w - 1.0:
            if abs(self.ball_y - self.opponent_y) <= self.paddle_h / 2.0:
                self.ball_x  = self.width - self.paddle_w - 1.0
                self.ball_vx = -abs(self.ball_vx) * 1.02
                offset = (self.ball_y - self.opponent_y) / (self.paddle_h / 2.0)
                self.ball_vy = offset * self.ball_speed
            else:
                # opponent missed — player scores
                reward = +1.0
                self.score_player += 1
                terminated = True

        self.steps_done += 1
        truncated = self.steps_done >= self.max_steps

        info = {
            "score_player":   self.score_player,
            "score_opponent": self.score_opponent,
        }

        return self._get_obs_player(), reward, terminated, truncated, info

    def _get_obs_player(self):
        """
        Observation from LEFT paddle (player) perspective.
        All values normalized to [-1, 1].
        """
        return np.array([
            self._norm_y(self.player_y),
            self._norm_y(self.opponent_y),
            self._norm_x(self.ball_x),
            self._norm_y(self.ball_y),
            self.ball_vx / self.ball_speed,
            self.ball_vy / self.ball_speed,
        ], dtype=np.float32)

    def get_obs_opponent(self):
        """
        Mirrored observation from RIGHT paddle (opponent) perspective.
        Flip x-axis so the opponent always sees itself on the left.
        """
        return np.array([
            self._norm_y(self.opponent_y),
            self._norm_y(self.player_y),
            -self._norm_x(self.ball_x),   # mirror x
            self._norm_y(self.ball_y),
            -self.ball_vx / self.ball_speed,  # mirror vx
            self.ball_vy  / self.ball_speed,
        ], dtype=np.float32)

    def _move_paddle(self, y, action):
        if action == self.MOVE_UP:
            y -= self.paddle_speed
        elif action == self.MOVE_DOWN:
            y += self.paddle_speed
        return np.clip(y, self.paddle_h / 2.0, self.height - self.paddle_h / 2.0)

    def _norm_x(self, x):
        return (x / (self.width  / 2.0)) - 1.0

    def _norm_y(self, y):
        return (y / (self.height / 2.0)) - 1.0

    def render(self):
        grid = [["." for _ in range(self.width // 4)] for _ in range(self.height // 4)]

        def _row(y):
            return int(np.clip(y / 4, 0, self.height // 4 - 1))

        def _col(x):
            return int(np.clip(x / 4, 0, self.width // 4 - 1))

        # paddles
        for dy in range(-self.paddle_h // 8, self.paddle_h // 8 + 1):
            r = _row(self.player_y) + dy
            if 0 <= r < self.height // 4:
                grid[r][0] = "|"

            r = _row(self.opponent_y) + dy
            if 0 <= r < self.height // 4:
                grid[r][self.width // 4 - 1] = "|"

        # ball
        grid[_row(self.ball_y)][_col(self.ball_x)] = "O"

        print("\n+" + "-" * (self.width // 4) + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * (self.width // 4) + "+")
        print(f"Score  Player: {self.score_player}  Opponent: {self.score_opponent}\n")