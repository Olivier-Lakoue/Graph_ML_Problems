import numpy as np
import random

class GridEnv():
    def __init__(self):
        self.done = False
        self.start = (0,0)
        self.goal = (6,6)
        self.win = False
        self.position = self.start
        self.grid = np.array([
                    [1, 1, 0, 1, 1, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1]
                ])
        self.moves = np.array([[-1, 0],
                               [1, 0],
                               [0, -1],
                               [0, 1]])

    def observe(self):
        idx = np.where((self.moves + self.position).min(axis=1) >= 0)[0]
        return [(i, j) for i, j in self.moves[idx, :] + self.position
                if np.subtract(self.grid.shape, (i, j)).all()
                and self.grid[i, j] >= 0]

    def action(self, next_position):
        if self.grid[next_position] == 0:
            self.done = True
        elif next_position == self.goal:
            self.win = True
            self.done = True
        else:
            self.position = next_position

