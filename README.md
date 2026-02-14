# Finger-Controlled RL Escape Dots

This project lets you control the red dot with your **right index finger** (camera input), while the blue dot is controlled by an **online-learning DQN agent** that keeps learning to escape in real time.

## Features
- Red dot (cursor): tracked from your right index fingertip using MediaPipe Hands
- Blue dot (player): controlled by a DQN agent that keeps training every step
- Real-time interaction: finger control + RL learning at the same time

## Environment
- OS: Linux (Ubuntu recommended)
- Python: **3.11 recommended**
  - Some Python 3.12 + mediapipe combinations may not expose `mp.solutions`.

## Installation
```bash
cd /home/hassong327/sim/tracking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run
### Online RL + finger control
```bash
python play.py --mode online --camera 0 --cam-width 1280 --cam-height 720 --cam-fps 60
```

### Manual keyboard mode (debug)
```bash
python play.py --mode human
```

## Controls
- `online` mode: move your right index finger to control the red dot
- `human` mode: `WASD` or arrow keys
- Exit: close the game window or press `ESC` in the hand-tracker preview window

## Main Options
- `--camera`: camera index (default `0`)
- `--cam-width`, `--cam-height`, `--cam-fps`: camera capture settings
- `--no-mirror-camera`: disable mirrored camera input
- `--no-show-hand-preview`: disable hand-tracker preview window
- `--train-updates-per-step`: adjust online learning intensity

## Troubleshooting
1. `ModuleNotFoundError: No module named 'torch'`
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. `mediapipe has no attribute solutions`
```bash
source .venv/bin/activate
python -m pip install --no-cache-dir "mediapipe==0.10.21" "numpy<2" "protobuf<5"
```
Use a Python 3.11 virtual environment if possible.

3. Camera cannot be opened
- Try a different index: `--camera 1` or `--camera 2`
- Check available video devices:
```bash
ls -l /dev/video*
```
