# 🎯 Finger-Controlled RL Escape Dots

오른손 검지(카메라 입력)로 🔴 빨간 점을 조종하고, 🔵 파란 점은 RL(DQN)로 실시간 학습하며 도망가는 프로젝트입니다.

## ✨ 기능
- 🔴 빨간 점(cursor): MediaPipe Hands로 검지 끝 좌표 추적
- 🔵 파란 점(player): DQN 에이전트가 매 스텝 학습하며 회피
- ⚡ 실시간 플레이 + 온라인 학습 동시 진행

## 🖥️ 환경
- OS: Linux(Ubuntu 권장)
- Python: **3.11 권장**
  - 일부 Python 3.12 + mediapipe 조합에서 `mp.solutions` 미노출 이슈가 있습니다.

## 📦 설치
```bash
cd /home/hassong327/sim/tracking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## ▶️ 실행
### 온라인 학습 + 손가락 조종
```bash
python play.py --mode online --camera 0 --cam-width 1280 --cam-height 720 --cam-fps 60
```

### 키보드 수동 모드(디버그용)
```bash
python play.py --mode human
```

## 🎮 조작
- `online` 모드: 오른손 검지 위치로 빨간 점 제어
- `human` 모드: `WASD` 또는 방향키
- 종료: 게임 창 닫기 또는 트래커 창에서 `ESC`

## ⚙️ 주요 옵션
- `--camera`: 카메라 인덱스 (기본 `0`)
- `--cam-width`, `--cam-height`, `--cam-fps`: 카메라 캡처 설정
- `--no-mirror-camera`: 카메라 미러 모드 끄기
- `--no-show-hand-preview`: 손 추적 미리보기 창 끄기
- `--train-updates-per-step`: 온라인 학습 강도 조절

## 🛠️ 트러블슈팅
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
가능하면 Python 3.11 venv를 사용하세요.

3. 카메라가 안 열릴 때
- `--camera 1`, `--camera 2`로 변경 시도
- `/dev/video*` 장치 확인
```bash
ls -l /dev/video*
```
