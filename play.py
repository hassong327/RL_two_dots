import argparse
import random
import re
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import CursorAvoidEnv, GameConfig


def _next_plot_path() -> Path:
    pattern = re.compile(r"^plot(\d+)\.png$")
    max_idx = 0
    for file in Path.cwd().iterdir():
        if not file.is_file():
            continue
        m = pattern.match(file.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return Path.cwd() / f"plot{max_idx + 1}.png"


def _save_life_time_plot(life_times_sec):
    path = _next_plot_path()
    x = np.arange(1, len(life_times_sec) + 1)
    plt.figure(figsize=(10, 4.8))
    plt.plot(x, life_times_sec, marker="o", linewidth=2)
    plt.title("Survival Time Per Life")
    plt.xlabel("Life Index")
    plt.ylabel("Survival Time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"Saved plot: {path}")
    return path


def run_human_mode():
    env = CursorAvoidEnv(
        config=GameConfig(max_steps=None, max_life=1),
        render_mode="human",
        use_mouse_cursor=False,
    )
    obs, _ = env.reset()
    life_times_sec = []
    life_start = time.perf_counter()
    prev_deaths = 0
    done = False
    while not done:
        action = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 4

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        obs, reward, terminated, truncated, info = env.step(action)
        if info["deaths"] > prev_deaths:
            now = time.perf_counter()
            life_times_sec.append(now - life_start)
            life_start = now
            prev_deaths = info["deaths"]
        done = done or terminated or truncated

    tail = time.perf_counter() - life_start
    if tail > 0.02:
        life_times_sec.append(tail)
    env.close()
    _save_life_time_plot(life_times_sec)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnlineDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_size: int = 80_000,
        min_buffer: int = 2_000,
        updates_per_step: int = 6,
        target_update_interval: int = 250,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9992,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.updates_per_step = updates_per_step
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cpu")
        self.online_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optim = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.steps = 0
        self.rng = random.Random(42)

    def act(self, obs):
        self.steps += 1
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, obs, action: int, reward: float, next_obs, done: bool):
        self.buffer.append((obs.copy(), action, reward, next_obs.copy(), done))
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self):
        if len(self.buffer) < self.min_buffer:
            return
        for _ in range(self.updates_per_step):
            batch = self.rng.sample(self.buffer, self.batch_size)
            obs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            next_obs = torch.tensor(
                np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device
            )
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

            q = self.online_net(obs).gather(1, actions)
            with torch.no_grad():
                next_q = self.target_net(next_obs).max(dim=1, keepdim=True).values
                target = rewards + self.gamma * (1.0 - dones) * next_q
            loss = F.smooth_l1_loss(q, target)

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
            self.optim.step()

        if self.steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())


def run_online_rl_mode(train_updates_per_step: int = 6):
    env = CursorAvoidEnv(
        config=GameConfig(max_steps=None, max_life=1),
        render_mode="human",
        use_mouse_cursor=False,
    )
    obs_dim = int(env.observation_space.shape[0])
    agent = OnlineDQNAgent(obs_dim=obs_dim, n_actions=env.action_space.n, updates_per_step=train_updates_per_step)
    obs, _ = env.reset()
    life_times_sec = []
    life_start = time.perf_counter()
    prev_deaths = 0
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if done:
            break
        prev_obs = obs
        action = agent.act(prev_obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["deaths"] > prev_deaths:
            now = time.perf_counter()
            life_times_sec.append(now - life_start)
            life_start = now
            prev_deaths = info["deaths"]
        transition_done = bool(terminated or truncated)
        agent.store(prev_obs, action, reward, obs, transition_done)
        agent.learn()
        done = done or terminated or truncated

    tail = time.perf_counter() - life_start
    if tail > 0.02:
        life_times_sec.append(tail)
    env.close()
    _save_life_time_plot(life_times_sec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "online"], default="online")
    parser.add_argument("--train-updates-per-step", type=int, default=6)
    args = parser.parse_args()

    if args.mode == "human":
        run_human_mode()
    else:
        run_online_rl_mode(train_updates_per_step=args.train_updates_per_step)


if __name__ == "__main__":
    main()
