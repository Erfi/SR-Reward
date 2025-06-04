from typing import Tuple, Any, Union, List
import math
import numpy as np
import pygame
from omegaconf.listconfig import ListConfig

import gymnasium as gym
from gymnasium import spaces

from IRL.environments.wrappers import GymHistoryWrapper


class NavMap(gym.Wrapper):
    def __init__(
        self,
        size: int,
        start: Union[None, Tuple[float, float]],
        goal: Tuple[float, float],
        goal_radius: float,
        action_scale: float,
        render_mode=None,
        walls=None,
        max_episode_steps=500,
        history_len=1,
        seed=0,
        **kwargs,
    ):
        env = NavMapBase(
            size=size,
            start=start,
            goal=goal,
            goal_radius=goal_radius,
            action_scale=action_scale,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            seed=seed,
            **kwargs,
        )
        env = GymHistoryWrapper(env, history_len=history_len, seed=seed)
        if walls is not None:
            env.unwrapped.add_wall(walls)
        super().__init__(env)


class NavMapBase(gym.Env):
    """
    A simple 2D navigation environment with continuous state and action spaces.
    Action space represents the angle of the agent's movement.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 24,
        "max_steps": 500,
    }

    def __init__(
        self,
        size: int = 10,
        start: Union[None, Tuple[float, float], List[Tuple[float, float]]] = None,
        goal: Tuple[float, float] = (9, 9),
        goal_radius: float = 0.5,
        action_scale: float = 1.0,
        render_mode: str = None,
        max_episode_steps: int = 500,
        seed: int | None = None,
    ):
        """
        Initialize the environment.
        """
        self.goal_radius = goal_radius
        self.action_scale = action_scale
        self.window_size = 512  # The size of the PyGame window
        self.size = size  # The size of the grid
        self.start = start
        self.goal = np.array(goal)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=self.size,
            shape=(2,),
            dtype=np.float32,
        )
        self.metadata["max_steps"] = max_episode_steps
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.current = None
        self.walls = []
        self.arrows = []
        self.window = None
        self.clock = None
        self.reset(seed=seed)
        self._mayby_init_pygame()

    def _get_info(self):
        return {"distance": np.linalg.norm(self.current - np.array(self.goal), ord=1)}

    def _get_obs(self):
        return self.current

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> np.ndarray:
        """
        Reset the environment to a random state.
        """
        super().reset(seed=seed)
        self.step_count = 0
        if self.start and (
            isinstance(self.start[0], list) or isinstance(self.start[0], ListConfig)
        ):
            # choose randomly from the start positions
            self.current = np.array(self.start[np.random.choice(len(self.start))])
        elif self.start:
            self.current = np.array(self.start)
        else:
            self.current = np.random.uniform(low=0.0, high=self.size, size=(2,))
        while self._crashed_into_wall(self.current):
            self.current = np.random.uniform(low=0.0, high=self.size, size=(2,))
        return self._get_obs(), self._get_info()

    def add_wall(self, walls: list[list[float, float, float, float]]) -> None:
        """
        Add a rectangle to the map.
        Params:
            walls: a list of tuples (x1, y1, x2, y2) representing the rectangle's corners.
        """
        self.walls.extend(walls)

    def _is_out_of_bounds(self, candidate: np.ndarray) -> bool:
        """
        Check if the candidate position is out of bounds.
        """
        return np.any(candidate < 0) or np.any(candidate > self.size)

    def _crashed_into_wall(self, candidate: np.ndarray) -> bool:
        """
        Check if the candidate position is out of bounds.
        """
        for wall in self.walls:
            if (
                (candidate[0] >= wall[0])
                and (candidate[0] <= wall[2])
                and (candidate[1] >= wall[1])
                and (candidate[1] <= wall[3])
            ):
                return True
        return False

    def step(self, action):  # -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment. using a continuous action space.
        Param:
            action: displacement in x and y (unscaled)
        """
        self.step_count += 1
        self.candidate = self.current + (action * self.action_scale).squeeze()
        self.current = self.candidate
        reward = -0.001
        terminated = False
        truncated = False
        if np.allclose(self.current, self.goal, atol=self.goal_radius):
            reward = 1.0
            terminated = True
        if self._is_out_of_bounds(self.current) or self._crashed_into_wall(
            self.current
        ):
            terminated = True
        if self.step_count >= self.metadata["max_steps"]:
            truncated = True
        if self.render_mode == "human":
            self._render_frame()
        return self.current, reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode in ["rgb_array", "human"]:
            return self._render_frame()

    def _mayby_init_pygame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def _render_frame(self):
        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                self.canvas,
                pygame.Color(150, 150, 150),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.canvas,
                pygame.Color(150, 150, 150),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        # Now we draw the walls
        for wall in self.walls:
            pygame.draw.rect(
                self.canvas,
                pygame.Color(50, 50, 50, 128),
                pygame.Rect(
                    pix_square_size * np.array(wall[0]),
                    pix_square_size * np.array(wall[1]),
                    pix_square_size * np.array(wall[2] - wall[0]),
                    pix_square_size * np.array(wall[3] - wall[1]),
                ),
            )
        # Draw the target
        pygame.draw.circle(
            self.canvas,
            (255, 0, 0),
            (self.goal) * pix_square_size,
            self.goal_radius * pix_square_size,
        )
        # Now we draw the agent
        pygame.draw.circle(
            self.canvas,
            pygame.Color(0, 0, 255, 128),
            (self.current) * pix_square_size,
            pix_square_size / 3,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

        pygame.image.save(self.window, "screenshot.png")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    from pygame.locals import *

    env = NavMap(
        size=20,
        goal=(10, 10),
        start=None,
        goal_radius=1.0,
        action_scale=0.1,
        render_mode="human",
    )
    env.add_wall([(7, 8, 8, 12)])
    env.add_wall([(12, 8, 13, 12)])
    env.add_wall([(7, 12, 13, 13)])
    env.reset()
    env.render()

    def greedy_policy(state, goal):
        diff = goal - state
        action_rad = math.atan2(diff[1], diff[0])

        return action_rad

    def manual_policy(*args):
        """
        Ask the user for an action to take using pygame keys.
        Returns: ndarray, action plus guassian noise
        """
        while True:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LEFT]:
                        theta = -np.pi
                    elif keys[pygame.K_RIGHT]:
                        theta = 0
                    elif keys[pygame.K_UP]:
                        theta = -np.pi / 2
                    elif keys[pygame.K_DOWN]:
                        theta = np.pi / 2
                    action = np.array([np.cos(theta), np.sin(theta)])
                    return action, None

    while not pygame.key.get_pressed()[pygame.K_q]:
        action, _ = manual_policy(env.current, env.goal)
        print(action)
        # noise = np.random.normal(scale=0.2)
        # if action is not None:
        # action += noise
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()
