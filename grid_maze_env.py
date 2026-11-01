import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=5):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(8,),
            dtype=np.int32
        )

        self.window = None
        self.clock = None
        self.cell_size = 100
        self.agent_pos = None
        self.goal_pos = None
        self.bad_cells = None

    def _get_obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            self.bad_cells[0][0], self.bad_cells[0][1],
            self.bad_cells[1][0], self.bad_cells[1][1]
        ], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "layout" in options:
            layout = options["layout"]
            self.agent_pos = tuple(layout["agent"])
            self.goal_pos = tuple(layout["goal"])
            self.bad_cells = [tuple(layout["bad1"]), tuple(layout["bad2"])]
        else:
            cells = self.np_random.choice(
                self.grid_size * self.grid_size,
                size=4,
                replace=False
            )

            positions = [
                (int(cell // self.grid_size), int(cell % self.grid_size))
                for cell in cells
            ]

            self.agent_pos = positions[0]
            self.goal_pos = positions[1]
            self.bad_cells = [positions[2], positions[3]]

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        row, col = self.agent_pos

        if action == 0:
            col += 1
        elif action == 1:
            row -= 1
        elif action == 2:
            col -= 1
        elif action == 3:
            row += 1

        row = int(np.clip(row, 0, self.grid_size - 1))
        col = int(np.clip(col, 0, self.grid_size - 1))
        self.agent_pos = (row, col)

        reward = 0.0
        terminated = False

        if self.agent_pos == self.goal_pos:
            reward = 1.0
            terminated = True
        elif self.agent_pos in self.bad_cells:
            reward = -1.0
            terminated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        width = self.grid_size * self.cell_size
        height = self.grid_size * self.cell_size

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill((255, 255, 255))

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        def draw_cell(position, color):
            row, col = position
            rect = pygame.Rect(
                col * self.cell_size + 5,
                row * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10
            )
            pygame.draw.rect(canvas, color, rect)

        draw_cell(self.goal_pos, (60, 180, 75))

        for bad_cell in self.bad_cells:
            draw_cell(bad_cell, (220, 60, 60))

        draw_cell(self.agent_pos, (65, 105, 225))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None