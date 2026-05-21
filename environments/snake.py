import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SnakeEnv(gym.Env):
    """
    Snake game.

    S - Set of states: 11 dimensional binary vector (danger, direction, food)
    A - Set of actions: {straight, turn right, turn left}
    R - reward function:
        +10 for eating food
        -10 for dying (wall or self collision)
        0 otherwise

    State vector (11 values, all binary):
    Danger:    [danger_straight, danger_right, danger_left]
    Direction: [moving_left, moving_right, moving_up, moving_down]
    Food:      [food_left, food_right, food_up, food_down]

    Parameters:
        grid_size : int. Width and height of the square grid
        max_steps : int. Max steps per episode before truncation
    """

    STRAIGHT = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

    def __init__(self, grid_size=20, max_steps=1000):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(23,), dtype=np.float32)

        self.action_space = spaces.Discrete(3) #straight left right

        self.snake = None
        self.direction = None
        self.food = None
        self.steps_done = 0
        self.score = 0

    def reset(self, seed=None, options=None):
        """
        Start new episode. Begin at center, length 3, moving right

        Returns:
            obs : np.ndarray shape (11,)
            info : dict
        """

        super().reset(seed=seed)

        center = self.grid_size // 2

        self.snake = deque([
            (center, center),
            (center, center - 1),
            (center, center - 2)
        ])

        self.direction = self.RIGHT
        self.steps_done = 0
        self.score = 0

        self.place_food()

        return self.get_obs(), {}

    def step(self, action):
        """
        Apply action and advance one timestep

        Parameters:
            action : int (STRAIGHT=0, TURN_RIGHT=1, TURN_LEFT=2)

        Returns:
            obs        : np.ndarray shape (11,)
            reward     : float
            terminated : bool   — True if snake died
            truncated  : bool   — True if max_steps exceeded
            info       : dict   with 'score'
        """
        self.steps_done += 1

        self.direction = self.resolve_direction(action)

        head_row, head_col = self.snake[0]

        dr, dc = self.direction
        new_head = (head_row + dr, head_col + dc)

        terminated = self.is_collision(new_head)

        if terminated:
            reward = -10.0
            return self.get_obs(), reward, terminated, False, {"score": self.score}

        self.snake.appendleft(new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.place_food()
        else:
            self.snake.pop()
            reward = 0.0

        truncated = self.steps_done >= self.max_steps

        return self.get_obs(), reward, False, truncated, {"score": self.score}


    def get_obs(self):
        """
        Build 23-dimensional binary state vector

        Returns:
            obs : np.ndarray shape (11,) dtype float32
        """

        head = self.snake[0]
        dir_idx = self.DIRECTIONS.index(self.direction)

        straight_dir = self.DIRECTIONS[dir_idx]
        right_dir = self.DIRECTIONS[(dir_idx + 1) % 4]
        left_dir = self.DIRECTIONS[(dir_idx - 1) % 4]



        # Danger lookahead (6 values)
        danger = []
        for direction in [straight_dir, right_dir, left_dir]:
            cell = head
            for _ in range(2):
                cell = self.cell_in_direction(cell, direction)
                danger.append(float(self.is_collision(cell)))

        #Wall distances (4 values)
        head_row, head_col = head
        wall_up = head_row / self.grid_size
        wall_down = (self.grid_size - 1 - head_row) / self.grid_size
        wall_left = head_col / self.grid_size
        wall_right = (self.grid_size -1 - head_col) / self.grid_size


        #Tail distance
        tail_row, tail_col = self.snake[-1]
        tail_dist = (abs(head_row - tail_row) + abs(head_col - tail_col)) / (2 * self.grid_size)

        #Direction (one ahead)  4 values
        dir_one = [
            float(self.direction == self.LEFT),
            float(self.direction == self.RIGHT),
            float(self.direction == self.UP),
            float(self.direction == self.DOWN),
        ]

        #Food direction (4 values)
        food_row, food_col = self.food
        head_row, head_col = head

        food_dir = np.array([
            #Food direction relative to head
            float(food_col < head_col), #food is left
            float(food_col > head_col), #food is right
            float(food_row < head_row), #Food is up
            float(food_row > head_row) #Food is down
        ], dtype=np.float32)

        #Food normalized absolute distances 2 values
        food_dist_row = abs(food_row - head_row) / self.grid_size
        food_dist_col = abs(food_col - head_col) / self.grid_size

        #Snake length  normalized
        snake_length = len(self.snake) / (self.grid_size ** 2)

        #Steps without food normalized
        steps_no_food = self.steps_done / self.max_steps

        obs = np.array([
            *danger,
            *dir_one,
            *food_dir,
            wall_up,
            wall_down,
            wall_left,
            wall_right,
            tail_dist,
            food_dist_row,
            food_dist_col,
            snake_length,
            steps_no_food
        ], dtype=np.float32)

        return obs

    def resolve_direction(self, action):
        """
        Convert relative action into absolute direction
            STRAIGHT -> keep current direction
            TURN_RIGHT -> rotate 90 degree clockwise
            TURN_LEFT -> rotate 90 degree counter-clockwise
        """

        idx = self.DIRECTIONS.index(self.direction)

        if action == self.STRAIGHT:
            return self.DIRECTIONS[idx]
        elif action == self.TURN_RIGHT:
            return self.DIRECTIONS[(idx + 1) % 4]
        else: #Turn left
            return self.DIRECTIONS[(idx - 1) % 4]


    def is_collision(self, cell):
        """
        Returns True if cell is outside the grid

        Parameters:
            cell : (row, col)
        """

        row, col = cell
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return True
        if cell in list(self.snake)[1:]:
            return True
        return False

    def cell_in_direction(self, from_cell, direction):
        """
        Returns the cell one step in a given direction from from_cell
        """

        row, col = from_cell
        dr, dc = direction
        return (row + dr, col + dc)

    def place_food(self):
        """
        Places food at random cell (not occupied by snake).
        """

        snake_set = set(self.snake)

        while True:
            row = np.random.randint(0, self.grid_size)
            col = np.random.randint(0, self.grid_size)
            cell = (row, col)
            if cell not in snake_set:
                self.food = cell
                break


    def render(self):
        """
        Print grid to console
        H = snake head
        # = snake body
        F = food
        . = empty cell
        """
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for i, (r, c) in enumerate(self.snake):
            grid[r][c] = "H" if i == 0 else "#"

        fr, fc = self.food
        grid[fr][fc] = "F"

        print("\n" + "─" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("─" * (self.grid_size * 2 + 1))
        print(f"Score: {self.score} | Steps: {self.steps_done}\n")
