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

        reward = 0.04
        hit = self._is_collision()
        if hit and self.hit_cooldown <= 0:
            self.life -= 1
            self.hit_cooldown = self.cfg.hit_cooldown_steps
            reward -= 1.0
            if self.life <= 0:
                self.deaths += 1
                reward -= 4.0
                self._respawn()
        else:
            self.hit_cooldown = max(0, self.hit_cooldown - 1)

        terminated = False
        truncated = False
        if self.cfg.max_steps is not None:
            truncated = self.steps >= self.cfg.max_steps

        if self.render_mode == "human":
            self._render_frame()

        info = {"life": self.life, "deaths": self.deaths}
        return self._obs(), reward, terminated, truncated, info

    def set_cursor_pos(self, x: float, y: float):
        self.cursor_pos[0] = float(np.clip(x, 0, self.cfg.width))
        self.cursor_pos[1] = float(np.clip(y, 0, self.cfg.height))

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
        dx, dy = 0.0, 0.0
        if action == 1:
            dy = -self.cfg.player_speed
        elif action == 2:
            dy = self.cfg.player_speed
        elif action == 3:
            dx = -self.cfg.player_speed
        elif action == 4:
            dx = self.cfg.player_speed

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

    def _update_cursor(self):
        if self.use_mouse_cursor:
            if pygame is None:
                return
            if self.window is None:
                return
            mouse_x, mouse_y = pygame.mouse.get_pos()
            self.cursor_pos[0] = float(np.clip(mouse_x, 0, self.cfg.width))
            self.cursor_pos[1] = float(np.clip(mouse_y, 0, self.cfg.height))
            return

        # Non-learning pointer: directly tracks the player position.
        vec = self.player_pos - self.cursor_pos
        dist = np.linalg.norm(vec) + 1e-7
        unit = vec / dist
        noise = self.rng.normal(0, self.cfg.cursor_noise_std, size=2).astype(np.float32)
        direction = unit + noise
        norm = np.linalg.norm(direction) + 1e-7
        direction /= norm
        self.cursor_pos += direction * self.cfg.cursor_speed
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
