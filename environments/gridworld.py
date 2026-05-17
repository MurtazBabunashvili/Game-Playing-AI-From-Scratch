import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridWorld:
    """
    S - set of states: every (row, col) cell on grid
    A - set of actions: {up, down, left, right}
    p(s', r | s, a) dynamics function (transition and reward)
    gamma - discount rate: set by user


    Special states:
    e.g State A -> any action gives reward +10, transition into A'
    Off-grid -> reward -1, position unchanged
    Normall -> reward 0
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    MOVES = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1)
    }

    def __init__(self, rows=5, cols=5, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma

        self.n_states = rows*cols

        self.n_actions = 4
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]

        self.state_A = (0, 1)
        self.state_A_ = (4, 1)
        self.state_B = (0, 3)
        self.state_B_ = (2, 3)

        self.agent_pos = (0, 0) #Agent's current position

    def step(self, action):
        """
        Given state s and action a, environment produces
        next states s' and reward r.

        In deterministic environment since we have exactly one (s', r) pair
        p(s', r | s, a) = 1

        action : int (UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3)

        next_state : (row, col) tuple
        reward : float
        done : bool
        """

        row, col = self.agent_pos

        #Special state A
        if self.agent_pos == self.state_A:
            self.agent_pos = self.state_A_
            return self.state_A_, 10.0, False

        #Special state B
        if self.agent_pos == self.state_B:
            self.agent_pos = self.state_B_
            return self.state_B_, 5.0, False

        #Normal movement
        dr, dc = self.MOVES[action]
        new_row = row + dr
        new_col = col + dc

        #Off-grid check. Gives -1 reward and position stays unchanged
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            return self.agent_pos, -1.0, False

        #Valid move. reward = 0, new position is the next state
        self.agent_pos = (new_row, new_col)
        return self.agent_pos, 0.0, False


    #Reset helper function. Starts new episode. Returns initial state S_0
    def reset(self, start_pos=None):
        """
        Resets environment to a starting state
        Parameters:
            start_pos : (row, col) or None
                        If none, starts at (0,0)
                        Can pass any valid grid position
        Returns
        state : (row, col) - the initial state S_0
        """
        
        if start_pos is None:
            self.agent_pos = (0, 0)
        else:
            self.agent_pos = start_pos
        return self.agent_pos

    #state_to_index and _index_to_state helper functions

    def state_to_index(self, state):
        """
        Converts (row, col) into flat integer index

        Example on 5x5 grid:
            (0, 0) -> 0
            (0, 1) -> 1
            (1, 0) -> 5
            (4, 4) -> 24
        """

        row, col = state
        return row * self.cols + col

    def index_to_state(self, index):
        """
        Converts flat integer index into (row, col)
        Example on 5x5 grid:
            0 -> (0, 0)
            1 -> (0, 1)
            5 -> (1, 0)
            24 -> (4, 4)
        """

        row = index // self.cols
        col = index % self.cols
        return (row, col)

    #get_all_states helper function, returns every possible state.
    #Used when computing value functions
    def get_all_states(self):
        """
        Returns a list of all states in S
        """
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    #is_valid_state helper function
    def is_valid_state(self, state):
        row, col = state
        return 0 <= row < self.rows and 0 <= col < self.cols

    #Print grid to console, showing agent's current position
    def render(self):
        """
        A, B are special states (high-reward transitions)
        * is agent's current position
        . means empty cell
        """
        print("\n" + "─" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.agent_pos:
                    cell = " * "
                elif pos == self.state_A:
                    cell = " A "
                elif pos == self.state_B:
                    cell = " B "
                elif pos == self.state_A_:
                    cell = "A' "
                elif pos == self.state_B_:
                    cell = "B' "
                else:
                    cell = " . "
                row_str += cell + "|"
            print(row_str)
        print("-"*(self.cols*4 + 1))
        print(f"Agent at: {self.agent_pos}\n")


    #Shows V(s) as heatmap
    def plot_value_function(self, V, title="State-Value Function V(s)"):
        """
        V(s) tells us how good each state is under a given policy
        Brighter cells = higher long-run expected return

        Parameters:
        V : dict mapping (row, col) -> float
            (e.g. {(0,0): 3.3, (0,1): 8.8, ...}
        title : string label for the plot
        """
        #Build 2D numpy array from value dictionary
        grid = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                grid[r,c] = V.get((r, c), 0.0)

        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")
        plt.colorbar(im, ax=ax, label="V(s)")

        #Annotate each cell with its value
        for r in range(self.rows):
            for c in range(self.cols):
                val = grid[r, c]
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=10, fontweight="bold", color="black")

        #Mark special states
        for (r, c), label in [(self.state_A, "A"),
                              (self.state_B, "B"),
                              (self.state_A_, "A'"),
                              (self.state_B_, "B'")]:
            ax.text(c, r - 0.35, label, ha="center", va="center", fontsize=8, color="navy")

            ax.set_xticks(range(self.cols))
            ax.set_yticks(range(self.rows))

            ax.set_title(title, fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.show()

    #plot_policy shows the policy as arrows
    def plot_policy(self, policy, title="Policy π"):
        """
        Plot the policy as directional arrows on the grid

        Parameters:
             policy : dict mapping (row, col) -> action (int 0-3)
                For example, {(0,0): UP, (0,1): RIGHT, ...}
             title: string label for the plot
        """

        arrow_map = {
            self.UP: (0, 0.3),
            self.DOWN: (0, -0.3),
            self.LEFT: (-0.3, 0),
            self.RIGHT: (0.3, 0)
        }

        fix, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis() # like matrix, row 0 at top

        #Draw grid lines
        for r in range(self.rows):
            for c in range(self.cols):
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor="gray", facecolor="white")
                ax.add_patch(rect)

        #Draw arrows
        for (r, c,), action in policy.items():
            dx, dy = arrow_map[action]
            ax.annotate("", xy=(c + dx, r - dy), xytext=(c, r), arrowprops=dict(arrowstyle="->", color="steelblue", lw=2))

        #Label special states
        for (r, c), label in [(self.state_A, "A"),
                              (self.state_B, "B"),
                              (self.state_A_, "A'"),
                              (self.state_B_, "B'")]:
            ax.text(c, r, label, ha="center", va="center", fontsize=11, fontweight="bold", color="darkred")

        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()