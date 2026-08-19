# XLeRobot Teleoperation Setup

Remote teleoperation for XLeRobot — leader arms on desktop PC, follower arms + wheels + head on Jetson Orin Nano.

## Hardware

- **XLeRobot** with STS3215 motors (follower arms, wheels, head)
- **Jetson Orin Nano** (robot side)
- **Desktop PC** (operator side)
- **Intel RealSense D415** (head camera)
- 2× USB cameras (wrist cameras)
- SO-ARM100 leader arms connected to desktop PC

## Architecture

```
Desktop PC                        Jetson Orin Nano
─────────────────                 ─────────────────────────────
xlerobot_control.py               xlerobot_host.py
  - PyQt5 GUI                       - robot.connect()
  - Leader arms (BiSOLeader)        - RealSense D415 (head)
  - ZMQ PUSH → cmd (5555)    →      - USB wrist cameras
  - ZMQ PULL ← obs (5556)    ←      - camera frames (base64 JPEG)
  - Arrow keys → head               - arm state
  - i/j/k/l → wheels
```

## File Placement

### Jetson (`/home/sdlab/lerobot/src/lerobot/`)

| File | Destination |
|------|-------------|
| `jetson/xlerobot.py` | `robots/xlerobot/xlerobot.py` |
| `jetson/xlerobot_host.py` | `robots/xlerobot/xlerobot_host.py` |
| `jetson/config_xlerobot.py` | `robots/xlerobot/config_xlerobot.py` |
| `jetson/camera_opencv.py` | `cameras/opencv/camera_opencv.py` |

### Desktop

| File | Destination |
|------|-------------|
| `desktop/xlerobot_control.py` | anywhere (e.g. `~/xlerobot_control.py`) |

## Installation

### Jetson

**1. lerobot 설치** (공식 가이드 참고: https://github.com/huggingface/lerobot)

```bash
conda create -n lerobot python=3.10
conda activate lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

**2. 추가 패키지**

```bash
pip install pyrealsense2 pyzmq
```

**3. 수정 파일 적용** (이 레포의 `jetson/` 폴더 파일들을 lerobot 설치 경로에 복사)

```bash
LEROBOT=~/lerobot/src/lerobot   # lerobot 설치 경로

cp jetson/xlerobot.py        $LEROBOT/robots/xlerobot/xlerobot.py
cp jetson/xlerobot_host.py   $LEROBOT/robots/xlerobot/xlerobot_host.py
cp jetson/config_xlerobot.py $LEROBOT/robots/xlerobot/config_xlerobot.py
cp jetson/camera_opencv.py   $LEROBOT/cameras/opencv/camera_opencv.py
```

---

### Desktop

**1. 가상환경 생성**

```bash
python3 -m venv ~/lerobot_env
source ~/lerobot_env/bin/activate
```

**2. 기본 패키지 설치**

```bash
pip install pyzmq PyQt5 numpy opencv-python-headless
```

**3. lerobot 설치** (리더암 제어에 필요)

```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e .
```

**4. scservo_sdk 설치** (PyPI에 없어서 수동으로 복사해야 함)

scservo_sdk는 Feetech 모터 SDK로 lerobot 소스 안에 포함되어 있지만 별도 패키지로 설치되지 않습니다.
Jetson에서 데스크탑으로 복사:

```bash
# 데스크탑 터미널에서 실행
# Python 버전 확인 후 경로 조정 (python3.10, python3.11 등)
PYTHON_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PKG=~/lerobot_env/lib/$PYTHON_VER/site-packages

scp -r sdlab@<jetson_ip>:~/lerobot/src/lerobot/motors/scservo_sdk $SITE_PKG/
```

> **Jetson IP**: 같은 네트워크에서 `hostname -I` 로 확인

**5. 데스크탑 포트 권한 설정**

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
# 또는 영구 설정:
sudo usermod -aG dialout $USER  # 재로그인 필요
```

## Motor Port Mapping

| Port | Device | Motors |
|------|--------|--------|
| port1 | `/dev/ttyACM1` | Left follower arm (1–6) + Head (7–8) |
| port2 | `/dev/ttyACM0` | Right follower arm (1–6) + Wheels |

```bash
# Fix port permissions if needed
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

## Camera Config

Edit `config_xlerobot.py` to match your serial number and video devices:

```python
"head": RealSenseCameraConfig(serial_number_or_name="346522061393", fps=30, width=640, height=480),
"left_wrist":  OpenCVCameraConfig(index_or_path="/dev/video6", fps=30, width=320, height=240, backend=Cv2Backends.V4L2),
"right_wrist": OpenCVCameraConfig(index_or_path="/dev/video8", fps=30, width=320, height=240, backend=Cv2Backends.V4L2),
```

Check video device names: `v4l2-ctl --list-devices`

## Usage

### 1. Start host on Jetson

```bash
conda activate lerobot
python -m lerobot.robots.xlerobot.xlerobot_host
# Press ENTER to restore calibration from file
```

### 2. Start control GUI on Desktop

```bash
source ~/lerobot_env/bin/activate
python3 xlerobot_control.py
```

### 3. Connect

Fill in sidebar settings and click **연결**:

| Field | Default |
|-------|---------|
| Jetson IP | `192.168.0.31` |
| CMD 포트 | `5555` |
| OBS 포트 | `5556` |
| 왼쪽 리더암 | `/dev/ttyACM0` |
| 오른쪽 리더암 | `/dev/ttyACM1` |

## Controls

| Key | Action |
|-----|--------|
| ↑ ↓ ← → | Head tilt / pan |
| `i` / `k` | Forward / backward |
| `j` / `l` | Strafe left / right |
| `u` / `o` | Rotate left / right |
| `n` / `m` | Speed up / down |
| Leader arms | Direct arm control |

## Known Issues

- Motor 4 may show overheat warning after extended use — let it cool
- `base_right_wheel` (motor 9) not connected in current hardware
- USB wrist cameras share a hub — 320×240 resolution recommended to avoid bandwidth issues
