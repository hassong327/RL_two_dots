import argparse
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import CursorAvoidEnv, GameConfig

HAND_WINDOW_NAME = "Right Index Tracker"


def run_human_mode():
    env = CursorAvoidEnv(
        config=GameConfig(max_steps=None, max_life=1),
        render_mode="human",
        use_mouse_cursor=False,
    )
    obs, _ = env.reset()
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
        done = done or terminated or truncated

    env.close()


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


class RightIndexTracker:
    def __init__(
        self,
        camera: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        mirror: bool = True,
        show_preview: bool = True,
    ):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe build has no mp.solutions. "
                "Use Python 3.11 + mediapipe==0.10.21."
            )
        self.mirror = mirror
        self.show_preview = show_preview
        self.cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_hands = mp_hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        if self.show_preview:
            cv2.namedWindow(HAND_WINDOW_NAME, cv2.WINDOW_NORMAL)

    def read_right_index(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, False, True

        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        right_tip = None
        preview_alive = True

        if res.multi_hand_landmarks:
            right_idx = None
            if res.multi_handedness:
                for i, handed in enumerate(res.multi_handedness):
                    label = handed.classification[0].label
                    if label == "Right":
                        right_idx = i
                        break
            if right_idx is None:
                right_idx = 0

            lm = res.multi_hand_landmarks[right_idx]
            self._mp_draw.draw_landmarks(
                frame,
                lm,
                self._mp_hands.HAND_CONNECTIONS,
            )
            tip = lm.landmark[8]
            right_tip = (float(tip.x), float(tip.y), float(tip.z))

        if self.show_preview:
            if cv2.getWindowProperty(HAND_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                preview_alive = False
            else:
                text = "Right index: not found"
                if right_tip is not None:
                    text = (
                        f"Right index norm=({right_tip[0]:.3f}, "
                        f"{right_tip[1]:.3f}, {right_tip[2]:.3f})"
                    )
                cv2.putText(
                    frame,
                    text,
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(HAND_WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    preview_alive = False

        return right_tip, preview_alive, False

    def close(self):
        self._hands.close()
        self.cap.release()
        if self.show_preview:
            cv2.destroyWindow(HAND_WINDOW_NAME)


def run_online_rl_mode(
    train_updates_per_step: int = 6,
    camera: int = 0,
    cam_width: int = 1280,
    cam_height: int = 720,
    cam_fps: int = 60,
    mirror_camera: bool = True,
    show_hand_preview: bool = True,
):
    env = CursorAvoidEnv(
        config=GameConfig(max_steps=None, max_life=1),
        render_mode="human",
        use_mouse_cursor=False,
        use_external_cursor=True,
    )
    tracker = RightIndexTracker(
        camera=camera,
        width=cam_width,
        height=cam_height,
        fps=cam_fps,
        mirror=mirror_camera,
        show_preview=show_hand_preview,
    )
    obs_dim = int(env.observation_space.shape[0])
    agent = OnlineDQNAgent(obs_dim=obs_dim, n_actions=env.action_space.n, updates_per_step=train_updates_per_step)
    obs, _ = env.reset()
    done = False

    try:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            if done:
                break

            tip, preview_alive, camera_failed = tracker.read_right_index()
            if camera_failed:
                print("Camera frame read failed. Stopping.")
                break
            if not preview_alive:
                done = True
                break
            if tip is not None:
                env.set_cursor_pos(tip[0] * env.cfg.width, tip[1] * env.cfg.height)

            prev_obs = obs
            action = agent.act(prev_obs)
            obs, reward, terminated, truncated, info = env.step(action)
            transition_done = bool(terminated or truncated)
            agent.store(prev_obs, action, reward, obs, transition_done)
            agent.learn()
            done = done or terminated or truncated
    finally:
        tracker.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "online"], default="online")
    parser.add_argument("--train-updates-per-step", type=int, default=6)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--cam-fps", type=int, default=60)
    parser.add_argument("--mirror-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-hand-preview", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.mode == "human":
        run_human_mode()
    else:
        run_online_rl_mode(
            train_updates_per_step=args.train_updates_per_step,
            camera=args.camera,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
            cam_fps=args.cam_fps,
            mirror_camera=args.mirror_camera,
            show_hand_preview=args.show_hand_preview,
        )


if __name__ == "__main__":
    main()
