# 🎯 Finger-Controlled RL Escape Dots

이 프로젝트는 **카메라 입력을 통해 사용자의 오른손 검지 손가락**으로 🔴 빨간 점을 조작하고, 🔵 파란 점은 **실시간으로 계속 학습하는 온라인 DQN 에이전트**가 제어하여 도망가도록 설계된 시스템입니다.

## ✨ 주요 기능
- 🔴 빨간 점 (커서): MediaPipe Hands를 사용해 **오른손 검지 끝 좌표**를 추적하여 제어
- 🔵 파란 점 (플레이어): **DQN 에이전트**가 매 스텝마다 학습하며 제어
- ⚡ 실시간 상호작용: 손가락 제어 + 강화학습이 동시에 동작

## 🖥️ 실행 환경
- OS: Linux (Ubuntu 권장)
- Python: **3.11 권장**
  - 일부 Python 3.12 + mediapipe 조합에서는 `mp.solutions`가 정상적으로 노출되지 않을 수 있습니다.

## 📦 설치 방법
```bash
cd /home/songha/RL_two_dots
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

이미 이 폴더에 가상환경이 있다면 다음처럼 바로 실행할 수 있습니다.
```bash
cd /home/songha/RL_two_dots
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## ▶️ 실행 방법

### Online RL + 손가락 제어 모드
```bash
./.venv/bin/python play.py --mode online --camera 0 --cam-width 1280 --cam-height 720 --cam-fps 60
```

### 수동 키보드 모드 (디버그용)
```bash
./.venv/bin/python play.py --mode human
```

## 🎮 조작 방법
- `online` 모드: **오른손 검지 손가락을 움직여** 빨간 점을 제어
- `human` 모드: `WASD` 또는 방향키 사용
- 종료: 게임 창을 닫거나, 손 추적 프리뷰 창에서 `ESC` 키 입력

## ⚙️ 주요 옵션
- `--camera`: 카메라 인덱스 (기본값 `0`)
- `--cam-width`, `--cam-height`, `--cam-fps`: 카메라 캡처 설정
- `--no-mirror-camera`: 카메라 좌우 반전 비활성화
- `--no-show-hand-preview`: 손 추적 프리뷰 창 비활성화
- `--train-updates-per-step`: 온라인 학습 강도 조절

## 🛠️ 문제 해결

### 1. `ModuleNotFoundError: No module named 'torch'`
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. `mediapipe has no attribute solutions`
```bash
source .venv/bin/activate
python -m pip install --no-cache-dir "mediapipe==0.10.21" "numpy<2" "protobuf<5"
```
가능하면 Python 3.11 가상 환경을 사용하세요.

### 3. 카메라를 열 수 없는 경우
- 다른 인덱스를 시도해 보세요: `--camera 1` 또는 `--camera 2`
- 사용 가능한 비디오 장치 확인:
```bash
ls -l /dev/video*
```
