import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None

ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

_ACTION_TO_DELTA = {
    ACTION_STAY: (0.0, 0.0),
    ACTION_UP: (0.0, -1.0),
    ACTION_DOWN: (0.0, 1.0),
    ACTION_LEFT: (-1.0, 0.0),
    ACTION_RIGHT: (1.0, 0.0),
}

_EPS = 1e-7


@dataclass
class GameConfig:
    width: int = 800
    height: int = 600
    player_radius: int = 14
    cursor_radius: int = 10
    player_speed: float = 7.0
    cursor_speed: float = 5.2
    cursor_noise_std: float = 0.0
    max_life: int = 1
    max_steps: Optional[int] = 1200
    hit_cooldown_steps: int = 12
    seed: int = 42


class CursorAvoidEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    SURVIVAL_REWARD = 0.04
    HIT_PENALTY = 1.0
    DEATH_PENALTY = 4.0

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        render_mode: Optional[str] = None,
        use_mouse_cursor: bool = False,
        use_external_cursor: bool = False,
    ):
        super().__init__()
        self.cfg = config or GameConfig()
        self.render_mode = render_mode
        self.use_mouse_cursor = use_mouse_cursor
        self.use_external_cursor = use_external_cursor

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(self.cfg.seed)
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.cursor_pos = np.zeros(2, dtype=np.float32)
        self.life = self.cfg.max_life
        self.steps = 0
        self.deaths = 0
        self.hit_cooldown = 0

        self.window = None
        self.clock = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.player_pos = self._rand_pos(self.cfg.player_radius)
        self.cursor_pos = self._rand_pos(self.cfg.cursor_radius)
        self.life = self.cfg.max_life
        self.steps = 0
        self.deaths = 0
        self.hit_cooldown = 0

        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        self._move_player(action)
        if not self.use_external_cursor:
            self._update_cursor()

        reward = self.SURVIVAL_REWARD + self._resolve_collision()

        terminated = False
        truncated = False
        if self.cfg.max_steps is not None:
            truncated = self.steps >= self.cfg.max_steps

        if self.render_mode == "human":
            self._render_frame()

        info = {"life": self.life, "deaths": self.deaths}
        return self._obs(), reward, terminated, truncated, info

    def set_cursor_pos(self, x: float, y: float):
        self.cursor_pos[0] = float(x)
        self.cursor_pos[1] = float(y)
        self._clip_cursor_pos()

    def _obs(self):
        return np.array(
            [
                self.player_pos[0] / self.cfg.width,
                self.player_pos[1] / self.cfg.height,
                self.cursor_pos[0] / self.cfg.width,
                self.cursor_pos[1] / self.cfg.height,
                self.life / self.cfg.max_life,
            ],
            dtype=np.float32,
        )

    def _move_player(self, action: int):
        dx_unit, dy_unit = _ACTION_TO_DELTA.get(action, (0.0, 0.0))
        dx = dx_unit * self.cfg.player_speed
        dy = dy_unit * self.cfg.player_speed

        self.player_pos[0] = float(
            np.clip(
                self.player_pos[0] + dx,
                self.cfg.player_radius,
                self.cfg.width - self.cfg.player_radius,
            )
        )
        self.player_pos[1] = float(
            np.clip(
                self.player_pos[1] + dy,
                self.cfg.player_radius,
                self.cfg.height - self.cfg.player_radius,
            )
        )

    def _resolve_collision(self) -> float:
        if self._is_collision() and self.hit_cooldown <= 0:
            self.life -= 1
            self.hit_cooldown = self.cfg.hit_cooldown_steps
            reward_delta = -self.HIT_PENALTY
            if self.life <= 0:
                self.deaths += 1
                reward_delta -= self.DEATH_PENALTY
                self._respawn()
            return reward_delta

        self.hit_cooldown = max(0, self.hit_cooldown - 1)
        return 0.0

    def _update_cursor(self):
        if self.use_mouse_cursor:
            if pygame is None:
                return
            if self.window is None:
                return
            mouse_x, mouse_y = pygame.mouse.get_pos()
            self.cursor_pos[0] = float(mouse_x)
            self.cursor_pos[1] = float(mouse_y)
            self._clip_cursor_pos()
            return

        # Non-learning pointer: directly tracks the player position.
        vec = self.player_pos - self.cursor_pos
        dist = np.linalg.norm(vec) + _EPS
        unit = vec / dist
        noise = self.rng.normal(0, self.cfg.cursor_noise_std, size=2).astype(np.float32)
        direction = unit + noise
        norm = np.linalg.norm(direction) + _EPS
        direction /= norm
        self.cursor_pos += direction * self.cfg.cursor_speed
        self._clip_cursor_pos()

    def _clip_cursor_pos(self):
        self.cursor_pos[0] = float(np.clip(self.cursor_pos[0], 0, self.cfg.width))
        self.cursor_pos[1] = float(np.clip(self.cursor_pos[1], 0, self.cfg.height))

    def _is_collision(self) -> bool:
        d = math.dist(self.player_pos.tolist(), self.cursor_pos.tolist())
        return d <= (self.cfg.player_radius + self.cfg.cursor_radius)

    def _respawn(self):
        self.player_pos = self._rand_pos(self.cfg.player_radius)
        self.life = self.cfg.max_life
        self.hit_cooldown = self.cfg.hit_cooldown_steps

    def _rand_pos(self, radius: int) -> np.ndarray:
        x = self.rng.uniform(radius, self.cfg.width - radius)
        y = self.rng.uniform(radius, self.cfg.height - radius)
        return np.array([x, y], dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(to_array=True)
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self, to_array: bool = False):
        if pygame is None:
            raise RuntimeError("pygame is required for rendering. Install pygame first.")

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.cfg.width, self.cfg.height))
            else:
                self.window = pygame.Surface((self.cfg.width, self.cfg.height))
            self.clock = pygame.time.Clock()

        canvas = self.window
        canvas.fill((248, 249, 251))

        pygame.draw.circle(
            canvas,
            (43, 117, 214),
            (int(self.player_pos[0]), int(self.player_pos[1])),
            self.cfg.player_radius,
        )
        pygame.draw.circle(
            canvas,
            (220, 58, 58),
            (int(self.cursor_pos[0]), int(self.cursor_pos[1])),
            self.cfg.cursor_radius,
        )

        font = pygame.font.SysFont("monospace", 24)
        text = font.render(f"Life: {self.life}  Deaths: {self.deaths}", True, (22, 25, 35))
        canvas.blit(text, (16, 14))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        if to_array:
            arr = pygame.surfarray.array3d(canvas)
            return np.transpose(arr, (1, 0, 2))
        return None

    def close(self):
        if pygame and self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
