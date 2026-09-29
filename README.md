# XLeRobot Teleoperation Setup

Remote teleoperation for XLeRobot — leader arms on desktop PC, follower arms + wheels + head on Jetson Orin Nano. Includes IMU-based yaw tracking, a 3D pose viewer, and a built-in data-recording pipeline for training ACT / SmolVLA policies.

## Hardware

- **XLeRobot** with STS3215 motors (follower arms, wheels, head)
- **Jetson Orin Nano** (robot side)
- **Desktop PC** (operator side, GPU recommended for training)
- **Intel RealSense D415** (head camera)
- 2× USB cameras (wrist cameras)
- **MPU6050 IMU** (I2C, mounted on the base) for yaw tracking
- SO-ARM100 leader arms connected to desktop PC

## Features

- Real-time bimanual teleoperation over ZMQ (leader arms on desktop → follower arms/wheels/head on Jetson)
- Head + dual wrist camera streaming
- MPU6050 IMU with gyro-bias calibration and accumulated yaw angle (for precise 90° turns)
- Live per-motor temperature readout (14 motors: both arms + head)
- Embedded 3D robot pose visualization (matplotlib, simplified forward kinematics)
- One-click dataset recording (LeRobotDataset) for imitation learning — episode start/save/discard, resumable across sessions, background upload to Hugging Face
- Ready for **ACT** / **SmolVLA** training via `lerobot-train`
- Experimental **AI 모드**: run a trained checkpoint directly on the robot from the control GUI (no separate eval script needed)
- Standalone Whisper voice-command test script for mapping spoken tool names to canonical labels

## Architecture

```
Desktop PC                        Jetson Orin Nano
─────────────────                 ─────────────────────────────
xlerobot_control.py               xlerobot_host.py
  - PyQt5 GUI                       - robot.connect()
  - Leader arms (BiSOLeader)        - RealSense D415 (head)
  - ZMQ PUSH → cmd (5555)    →      - USB wrist cameras
  - ZMQ PULL ← obs (5556)    ←      - camera frames (base64 JPEG)
  - Arrow keys → head               - arm/base state
  - i/j/k/l/u/o → wheels            - MPU6050 IMU (I2C bus 7)
  - LeRobotDataset recording
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
| `desktop/whisper_test.py` | anywhere, standalone (no robot connection needed) |

## Installation

### Jetson

**1. lerobot 설치** (conda 환경, 공식 가이드 참고: https://github.com/huggingface/lerobot)

```bash
conda create -n lerobot python=3.10
conda activate lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

**2. 추가 패키지**

```bash
pip install pyrealsense2 pyzmq smbus2
```

**3. 수정 파일 적용** (이 레포의 `jetson/` 폴더 파일들을 lerobot 설치 경로에 복사)

```bash
LEROBOT=~/lerobot/src/lerobot   # lerobot 설치 경로

cp jetson/xlerobot.py        $LEROBOT/robots/xlerobot/xlerobot.py
cp jetson/xlerobot_host.py   $LEROBOT/robots/xlerobot/xlerobot_host.py
cp jetson/config_xlerobot.py $LEROBOT/robots/xlerobot/config_xlerobot.py
cp jetson/camera_opencv.py   $LEROBOT/cameras/opencv/camera_opencv.py
```

**4. MPU6050 IMU 연결**

Jetson Orin Nano의 40핀 헤더 기준:

| MPU6050 핀 | Jetson 핀 |
|------------|-----------|
| VCC | Pin 1 (3.3V) |
| GND | Pin 6 (GND) |
| SCL | Pin 5 (I2C) |
| SDA | Pin 3 (I2C) |

> Jetson Orin Nano에서 핀 3/5는 **I2C bus 7**에 매핑됩니다 (bus 1이 아님). 연결 확인:
> ```bash
> for i in 0 2 4 5 7; do echo "bus $i:"; i2cdetect -y -r $i 2>/dev/null | grep -q 68 && echo "  found at 0x68"; done
> ```

---

### Desktop

**1. 가상환경 생성** (Python 3.12 이상 필요 — lerobot 최신 버전 요구사항)

```bash
python3.12 -m venv ~/lerobot_env312
source ~/lerobot_env312/bin/activate
```

**2. lerobot 설치** (리더암 제어 + 데이터셋 기록 + 학습에 필요)

```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e ".[feetech,dataset,training]"
```

> `training` extra는 `accelerate` 등 `lerobot-train` 실행에 필요한 패키지를 포함합니다. 빠뜨리면 학습이 끝나는 시점에 `ImportError: 'accelerate' is required` 에러가 납니다.

**3. GUI 패키지 설치**

```bash
pip install pyzmq PyQt5 matplotlib opencv-python-headless
```

> ⚠️ **`opencv-python`(일반판) 대신 반드시 `opencv-python-headless`를 설치하세요.** 일반판은 자체 Qt 플러그인을 포함하고 있어 PyQt5와 충돌해 `Could not load the Qt platform plugin "xcb"` 오류로 앱이 아예 실행되지 않습니다. 이미 잘못 설치했다면:
> ```bash
> pip uninstall -y opencv-python opencv-python-headless
> rm -rf ~/lerobot_env312/lib/python3.12/site-packages/cv2 ~/lerobot_env312/lib/python3.12/site-packages/opencv_python*
> pip install "opencv-python-headless<4.14,>=4.9.0"
> ```

**4. scservo_sdk 설치** (PyPI에 없어서 수동으로 복사해야 함)

scservo_sdk는 Feetech 모터 SDK로 lerobot 소스 안에 포함되어 있지만 별도 패키지로 설치되지 않습니다.
Jetson에서 데스크탑으로 복사:

```bash
# 데스크탑 터미널에서 실행
SITE_PKG=~/lerobot_env312/lib/python3.12/site-packages
scp -r sdlab@<jetson_ip>:~/lerobot/src/lerobot/motors/scservo_sdk $SITE_PKG/
```

> **Jetson IP**: 같은 네트워크에서 `hostname -I` 로 확인. 공유기 재시작 등으로 IP가 바뀌면 GUI의 "Jetson IP" 필드도 갱신해야 합니다.

**5. 데스크탑 포트 권한 설정**

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
# 또는 영구 설정:
sudo usermod -aG dialout $USER  # 재로그인 필요
```

**6. Hugging Face 로그인** (데이터셋/모델 업로드에 필요, 선택)

```bash
hf auth login
```

**7. Whisper 음성 인식 테스트 패키지** (선택, `whisper_test.py` 사용 시)

```bash
sudo apt-get install -y portaudio19-dev libportaudio2
pip install faster-whisper sounddevice
```

## Motor Port Mapping

| Port | Device | Motors |
|------|--------|--------|
| port1 | `/dev/ttyACM1` | Left follower arm (1–6) + Head (7–8) |
| port2 | `/dev/ttyACM0` | Right follower arm (1–6) + Wheels |

## Camera Config

Edit `config_xlerobot.py` to match your serial number and video devices:

```python
"head": RealSenseCameraConfig(serial_number_or_name="346522061393", fps=30, width=640, height=480),
"left_wrist":  OpenCVCameraConfig(index_or_path="/dev/video6", fps=30, width=320, height=240, backend=Cv2Backends.V4L2),
"right_wrist": OpenCVCameraConfig(index_or_path="/dev/video8", fps=30, width=320, height=240, backend=Cv2Backends.V4L2),
```

Check video device names: `v4l2-ctl --list-devices`

> USB wrist cameras share a hub — 320×240 resolution recommended to avoid bandwidth issues.

## Usage

### 1. Start host on Jetson

```bash
conda activate lerobot
python -m lerobot.robots.xlerobot.xlerobot_host
# Press ENTER to restore calibration from file
```

### 2. Start control GUI on Desktop

```bash
source ~/lerobot_env312/bin/activate
python3 xlerobot_control.py
```

### 3. Connect

Fill in sidebar settings and click **연결**:

| Field | Default |
|-------|---------|
| Jetson IP | (matches your Jetson's current IP) |
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

## IMU & Yaw Tracking

The right panel shows live gyro/accel bars and an accumulated **Yaw** angle (useful for precise 90° base turns). Gyro sensors drift over time, so:

1. Keep the robot completely still.
2. Click **초기화** (reset). The GUI samples ~1 second of gyro data to measure the static bias.
3. Once calibration finishes ("bias=X.XX°/s 보정됨"), Yaw tracks rotation accurately from that reference point.

## Motor Temperature

The right panel's **모터 온도** section polls `Present_Temperature` on all 14 arm/head motors (~once per second — polling every motor on every control tick would slow the loop down). Color-coded: green < 50°C, orange 50–65°C, red ≥ 65°C (rule-of-thumb thresholds for STS3215, adjust as needed).

## Data Recording (for ACT / SmolVLA training)

The right panel's **데이터 녹화** section records demonstrations directly into a [LeRobotDataset](https://github.com/huggingface/lerobot):

1. Set **Repo ID** (e.g. `<hf_user>/<task_name>`) and **Task 설명**.
2. Click **데이터셋 생성** — creates a new dataset, or resumes an existing one if the Repo ID already has data (safe to stop and continue later, e.g. after a motor cooldown).
3. Click **● 에피소드 녹화 시작**, perform the demonstration with the leader arms, then **■ 에피소드 저장**. Use **현재 에피소드 폐기** to discard a bad take before saving.
4. Repeat for ~50+ episodes (see [LeRobot's data collection guide](https://github.com/huggingface/lerobot) for tips: vary object position/color, keep demonstrations consistent).
5. Click **데이터셋 종료** when done for the session, then **허깅페이스 업로드** to push to the Hub (optional, runs in the background).

> After recording a batch, it's worth sanity-checking that the saved `action` values actually vary across a trajectory (`ds[i]['action']` for a few frames of one episode) before spending an hour training on them — a silent all-zero `action` column (frozen policy at eval time) is the single easiest way to waste a training run.

### Training

```bash
lerobot-train \
  --dataset.repo_id=<hf_user>/<task_name> \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/<task_name> \
  --job_name=<task_name> \
  --batch_size=8 \
  --steps=30000 \
  --policy.repo_id=<hf_user>/act_<task_name>
```

Swap `--policy.type=act` for `smolvla` to try a language-conditioned VLA instead. See `lerobot-train --help` and the [LeRobot docs](https://github.com/huggingface/lerobot) for more options.

## AI Inference Mode (run a trained policy on the robot)

The left sidebar's **AI 추론 (실험적)** section loads a trained checkpoint and lets it drive both arms directly from the GUI — no separate `lerobot-eval`/`lerobot-record` process needed (XLeRobot's ZMQ host/client split isn't a standard registered lerobot `Robot`, so the usual real-robot eval CLIs don't apply here; this reuses the exact same ZMQ pipeline as teleoperation).

1. Set **체크포인트 경로** to a `pretrained_model` directory (e.g. `outputs/train/<task_name>/checkpoints/last/pretrained_model`) and **Task 설명** (should match what was used at training time).
2. Click **정책 로드** (loads in the background; watch the status line below the button).
3. Once loaded, **AI 모드 시작** hands both arms to the policy — leader-arm input is ignored while active. **AI 모드 중지** returns control to the leader arms.
4. Stay ready to stop the robot (button, or just close the GUI) — an undertrained policy can produce unexpected motion.

## Voice Command Testing (Whisper)

`desktop/whisper_test.py` is a standalone mic → [faster-whisper](https://github.com/SYSTRAN/faster-whisper) → canonical-tool-name script, useful for prototyping the "의사 음성 요청 → 도구 선택" pipeline before wiring it into the robot.

```bash
python3 desktop/whisper_test.py            # "small" model, CPU
python3 desktop/whisper_test.py medium     # bigger model = better accuracy, slower
```

Press `ENTER`, say a tool name (Korean or English), press `ENTER` again to stop, and it prints the transcription plus the matched canonical label. Edit `TOOL_ALIASES` in the script to add more phrases per tool.

> Runs on CPU (`compute_type="int8"`) by default — `faster-whisper`/`ctranslate2` needs its own cuBLAS libraries to use CUDA, which commonly aren't on the library path even when PyTorch's CUDA works fine. CPU is plenty fast for `small`/`medium` on short clips; only bother chasing GPU support if latency becomes a real bottleneck.

## Known Issues

- Motor 4 may show overheat warning after extended use — let it cool, then resume recording with the same Repo ID
- `base_right_wheel` (motor 9) not connected in current hardware
- USB wrist cameras share a hub — 320×240 resolution recommended to avoid bandwidth issues
- Never mix `opencv-python` and `opencv-python-headless` in the same environment — see the Desktop install section above
- `BiSOLeader.get_action()` returns keys like `left_shoulder_pan.pos` (no `arm_`); the robot's `send_action()` remaps this to `left_arm_shoulder_pan.pos` server-side, so teleoperation itself always worked, but any code capturing the *raw* teleop action on the desktop (dataset recording, AI-mode logging) must apply the same remap or every recorded arm action silently ends up `0.0`. `ControlThread.run()` now does this remap before storing `last_action` — if you fork this code, keep that remap in place.
- `torchcodec` usually fails to load its CUDA shared libraries and falls back to `pyav` — harmless, just noisy in the logs
- `faster-whisper` needs its own cuBLAS to use `device="cuda"`; `whisper_test.py` defaults to CPU to sidestep this
