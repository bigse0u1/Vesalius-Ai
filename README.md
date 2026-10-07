# XLeRobot Teleoperation Setup

Remote teleoperation for XLeRobot — leader arms on desktop PC, follower arms + wheels + head on Jetson Orin Nano. Includes IMU-based yaw tracking, a 3D pose viewer, a data-recording pipeline for training ACT / SmolVLA policies, and an experimental end-to-end "pick → rotate → handoff" demo sequence (voice-triggerable).

This is the manipulation/teleop layer for the **Y-MAS 베살리우스 팀** laparoscopic instrument-handoff robot project — see [`docs/research_plan.md`](docs/research_plan.md) for the full research plan (Task 1: tray-prior + handle-based visual verification + learning-based pick + vision-guided handover) and the [Research Plan Progress](#research-plan-progress) section below for what's implemented so far.

## Hardware

- **XLeRobot** with STS3215 motors (follower arms, wheels, head)
- **Jetson Orin Nano** (robot side)
- **Desktop PC** (operator side, GPU recommended for training)
- **Intel RealSense D415** (head camera)
- 2× USB cameras (wrist cameras)
- **MPU6050 IMU** (I2C, mounted on the base) for yaw tracking
- SO-ARM100 leader arms connected to desktop PC

## Features

- Real-time bimanual teleoperation over ZMQ (leader arms on desktop → follower arms/wheels/head on Jetson); leader arms are optional and hot-pluggable
- Head + dual wrist camera streaming
- MPU6050 IMU with gyro-bias calibration and accumulated yaw angle (for precise 90° turns)
- Live per-motor temperature readout (14 motors: both arms + head)
- Embedded 3D robot pose visualization (matplotlib, simplified forward kinematics)
- Tabbed control GUI (조종 / AI 모드 on the left, 데이터 / 상태 on the right) instead of one long scrolling sidebar
- One-click dataset recording (LeRobotDataset) for imitation learning — separate Pick and Place (tray return) datasets, episode start/save/discard, resumable across sessions, background upload to Hugging Face
- Ready for **ACT** / **SmolVLA** training via `lerobot-train`
- Experimental **AI 모드**: run a trained checkpoint directly on the robot from the control GUI (no separate eval script needed)
- **Voice-triggered demo sequence**: Whisper recognizes a spoken tool name → auto-fills the task prompt → runs the full pick → grip-lock → auto-rotate → handoff sequence, with a manual "▶ 전체 데모 시작" button and a "🛑 긴급 정지" kill switch
- Standalone Whisper voice-command test script (`desktop/whisper_test.py`) for mapping spoken tool names to canonical labels

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
| 왼쪽 리더암 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6081776-if00` |
| 오른쪽 리더암 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083489-if00` |

**Leader arms are optional.** Connecting only needs the Jetson; the leader arms are attached by a background watcher that retries every 2 s:

- Without leader arms, AI 모드, head (arrow keys) and wheels (i/j/k/l) all work; no arm keys are sent, so the follower arms simply hold their last pose.
- Plug the leader arms in at any time and they connect automatically; unplug them and the GUI keeps running (arms hold), then reconnects when they're plugged back in. The status line under **연결** shows the current leader state.
- Leader connection runs off the control loop, so a calibration prompt in the terminal (`Press ENTER to use provided calibration file…`) won't freeze head/wheel/AI commands.
- Dataset recording skips frames that have no arm action (no leader and no AI), so a missing leader can't silently record all-zero arm actions.

Use the `/dev/serial/by-id/...` paths rather than `/dev/ttyACM*`: ACM numbers are assigned in plug-in order, so after a replug the left leader could end up driving the right follower. Find your own IDs with `ls -l /dev/serial/by-id/`.

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

The right panel's **데이터** tab has two independent recording sections, each writing to its own [LeRobotDataset](https://github.com/huggingface/lerobot):

| Section | Default Repo ID | Default Task | What to demonstrate |
|---|---|---|---|
| **데이터 녹화 — Pick** | `bigse0u1/xlerobot_scrub_7tool` | `Pick up the grasper` | Home pose → grasp the tool from its tray slot → lift → ready pose |
| **데이터 녹화 — Place (반납)** | `bigse0u1/xlerobot_scrub_7tool_place` | `Place the grasper back in the tray` | Start from the ready pose *already holding* the tool → put it back in its slot → release → home pose |

Each section works the same way:

1. Set **Repo ID** (e.g. `<hf_user>/<task_name>`) and **Task 설명**.
2. Click **데이터셋 생성** — creates a new dataset, or resumes an existing one if the Repo ID already has data (safe to stop and continue later, e.g. after a motor cooldown).
3. Click **● 에피소드 녹화 시작**, perform the demonstration with the leader arms, then **■ 에피소드 저장**. Use **현재 에피소드 폐기** to discard a bad take before saving.
4. Repeat for ~50+ episodes (see [LeRobot's data collection guide](https://github.com/huggingface/lerobot) for tips: vary object position/color, keep demonstrations consistent).
5. When done for the session, click **허깅페이스 업로드** to push to the Hub (optional, runs in the background). Upload finalizes the dataset, so recording is disabled afterwards — reopen the GUI and click **데이터셋 생성** with the same Repo ID to continue. Use **데이터셋 종료** to close a session without uploading.

Both datasets can be open at the same time, but only **one episode records at a time** — starting an episode in one section while the other is recording is refused. This makes the natural loop *Pick 저장 → Place 녹화 시작 → put it back → Place 저장* quick, since the tool is already in the gripper at the end of each Pick episode.

Tips for Place data:

- Keep the starting state consistent (ready pose, tool held) — at run time Place starts right after retrieval, holding the tool.
- Use one fixed phrasing per tool (e.g. `Place the scissors back in the tray`). With the fixed tray layout, the tool name implies its home slot, matching the tray-return step in the research plan (§26).
- Use the same arm split as Pick (Slot 1–4 → left arm, Slot 5–7 → right arm).

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

See `lerobot-train --help` and the [LeRobot docs](https://github.com/huggingface/lerobot) for more options.

#### SmolVLA (7-tool, cloud GPU)

For SmolVLA, **always fine-tune from the pretrained `lerobot/smolvla_base`** with `--policy.path`. Using `--policy.type=smolvla` instead leaves `load_vlm_weights=False` together with `train_expert_only=True`, which means the vision-language backbone is randomly initialized *and* frozen — the policy can't really see or read the instruction (this is how the 2-tool model was trained; see Known Issues).

`smolvla_base` expects camera inputs named `camera1/2/3`, so map ours with `--rename_map`. The map is saved into the checkpoint's `policy_preprocessor.json`, so the GUI's AI 모드 can keep feeding `observation.images.head` etc. without code changes.

Command used for the 7-tool Pick dataset (700 episodes, 246,851 frames) on a RunPod RTX 5090 (32GB):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=bigse0u1/xlerobot_scrub_7tool \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=bigse0u1/smolvla_7tool \
  --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
  --batch_size=64 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=/workspace/outputs/smolvla_7tool \
  --job_name=smolvla_7tool
```

- batch 64 × 30,000 steps ≈ 8 epochs over 246k frames.
- RunPod setup: pick a PyTorch template with **CUDA 12.8+** (required for RTX 50-series / Blackwell), put `HF_HOME=/workspace/hf` on the persistent volume, `apt-get install -y ffmpeg tmux`, `pip install -e ".[smolvla]"` in a lerobot clone, then `hf auth login`. Check `torch.cuda.is_available()` after installing lerobot — pip can swap in a PyTorch build without Blackwell support.
- Run inside `tmux` so a dropped SSH session doesn't kill training. If the `data_s` time in the logs is large, raise `--num_workers` (video decoding is CPU-bound).
- Confirm the model is on the Hub before terminating the pod, then on the desktop: `hf download bigse0u1/smolvla_7tool --local-dir ~/lerobot/outputs/train/smolvla_7tool/pretrained_model`.

## AI Inference Mode (run a trained policy on the robot)

The **AI 모드** tab (left side) loads a trained checkpoint and lets it drive both arms directly from the GUI — no separate `lerobot-eval`/`lerobot-record` process needed (XLeRobot's ZMQ host/client split isn't a standard registered lerobot `Robot`, so the usual real-robot eval CLIs don't apply here; this reuses the exact same ZMQ pipeline as teleoperation).

1. Set **체크포인트 경로** to a `pretrained_model` directory (defaults to the current SmolVLA 2-tool checkpoint, `outputs/train/smolvla_2tool/checkpoints/last/pretrained_model`) and **Task 설명** (should match a phrasing used at training time, e.g. `Pick up the grasper`).
2. Click **정책 로드** (loads in the background; watch the status line below the button — SmolVLA loading is noticeably slower than ACT).
3. Once loaded, **AI 모드 시작** hands both arms to the policy — leader-arm input is ignored while active. **AI 모드 중지** returns control to the leader arms.
4. Stay ready to stop the robot (🛑 긴급 정지, or just close the GUI) — an undertrained policy can produce unexpected motion.

### Demo Sequence (pick → rotate → handoff)

Below AI 추론, the **데모 시퀀스** group chains a full "grab the tool and hand it to the surgeon" run:

1. **🎤 음성 명령 듣기** — click to start recording, click again to stop; Whisper transcribes and matches it against `TOOL_ALIASES`, auto-filling **Task 설명** (e.g. `Pick up the grasper`) and immediately kicking off the full sequence below. The tool name in the prompt comes from `TOOL_TASK_NAMES`, which must match the recorded task strings exactly (e.g. Clipper was recorded as `Pick up the clippers`) — update it if you record with different wording. Or trigger it manually with **▶ 전체 데모 시작** after setting Task 설명 yourself.
2. **Pick** — runs AI 모드 and waits for the arm to *settle* (joint positions stop changing for ~1.2s, with a 3s minimum before it's allowed to declare "done" and a 20s timeout) rather than using a fixed timer, so it doesn't cut the policy off mid-grasp.
3. **Grip lock** — freezes the arm at its current (just-settled) pose as a `scripted_action`, overriding both AI mode and leader-arm input, so the tool can't be dropped or bumped by stray teleop input during the turn.
4. **Rotate** — spins the base to **목표 각도** (default `-90`) using IMU yaw feedback, with the grip lock held throughout.
5. **Handoff** — moves to a pre-recorded **핸드오프 자세** (see below), waits 3s once arrived, opens the gripper, holds 1s, then releases control.

Before using the full sequence, record a handoff pose once: teleop the arm to a natural "present the tool to the surgeon's hand" position while actually holding something, then click **핸드오프 자세 저장**. **핸드오프 동작 실행** replays just that step standalone (useful for tuning release timing without redoing the whole pick). **자동 회전 시작** likewise works standalone and also grip-locks the current pose before turning.

> ⚠️ `observation.*.gripper.pos` is a raw motor reading, but `xlerobot.py`'s `send_action()` inverts gripper *actions* (`100 - value`) before writing to the motor. Any code that reads a gripper position from an observation and replays it as an action (grip-lock, saved handoff pose) must re-invert it (`100 - value`) first, or the gripper does the opposite of what was intended — see `_obs_to_locked_action()` in `xlerobot_control.py`. If you ever see the gripper open right when it should be holding the tool, this is the first thing to check.

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
- `faster-whisper` needs its own cuBLAS to use `device="cuda"`; `whisper_test.py` and the GUI's voice command both default to CPU to sidestep this
- Gripper state↔action inversion (see the ⚠️ note in Demo Sequence above) — any new code that round-trips an observed gripper position back into an action must re-apply `100 - value`
- `lerobot-calibrate` calls `robot.connect()` (which starts the background raw-tty keyboard listener for i/j/k/l wheel teleop) before calling `robot.calibrate()`, leaving the terminal in non-canonical mode and making `calibrate()`'s `input()` prompts raise `EOFError` immediately — happens even with a direct keyboard/monitor on the Jetson, not just over SSH. Fixed in `xlerobot.py`'s `calibrate()`, which now stops the keyboard listener for the duration of calibration and restarts it afterward.
- The 2-tool SmolVLA checkpoint (`smolvla_2tool`) was trained with `--policy.type=smolvla`, i.e. `load_vlm_weights=False` + `train_expert_only=True` — a randomly initialized, frozen vision-language backbone. This is a likely major cause of it ignoring the language instruction (next item). The 7-tool run fine-tunes from `lerobot/smolvla_base` instead (see Training)
- SmolVLA fine-tuned on a small (~100 episodes/tool), visually-cluttered scene (7 similar tools together) tends to ignore the language instruction and just grab whatever's nearest rather than the requested tool — language grounding needs either much more data, or (for now) a **fixed tool layout per recording session** rather than randomizing position every episode, since with limited data the model can't reliably learn both position-invariance and language-based tool selection at once
- Same-color gripper and tool (e.g. both white) removes a cheap visual cue the policy could otherwise use for fine alignment — no software fix for this (augmentation can't invent contrast that isn't in the pixels); mark the gripper fingers with contrasting tape before the next recording round
- The settle-detection used to decide when AI 모드 has "finished" picking (`_wait_for_settle`) can mistake a brief mid-sequence pause (e.g. before the gripper closes) for completion; a `min_wait_s=3.0` floor guards against the most obvious case, but this is a timing heuristic, not a grasp-success check — there's no classifier yet confirming the grasp actually succeeded (see Research Plan Progress below)

## Research Plan Progress

Status against the STEP 1–20 development order in [`docs/research_plan.md`](docs/research_plan.md) (§39):

| Step | Description | Status |
|---|---|---|
| 1 | 3색 블록 Pick | ✅ Done |
| 2 | ACT / SmolVLA 학습 pipeline 확인 | ✅ Done — both trained end-to-end via `lerobot-train` |
| 3 | 3색 블록 Pick → 90° Rotate → Place | ✅ Rotate/grip-lock/release mechanism built generically (works for any tool, not block-specific) |
| 4 | 실제 수술도구 2개 Pick | ✅ Done — Grasper + Scissors, SmolVLA, 110 episodes |
| 5 | 수술도구 4개 | ⏭️ Skipped — went straight to 7 tools |
| 6 | 수술도구 7개 | 🟡 Pick data done — 7 tools × 100 episodes on a fixed tray layout (`bigse0u1/xlerobot_scrub_7tool`); SmolVLA fine-tuning from `smolvla_base` on RunPod RTX 5090 in progress. Place (tray return) data also being recorded (`bigse0u1/xlerobot_scrub_7tool_place`, 567 episodes so far) |
| 7 | 왼팔 4개 / 오른팔 3개 역할 분담 | ⬜ Not implemented (`active_arm`/`held_tool` state not yet tracked) |
| 8 | Whisper 연결 | 🟡 Mic → canonical tool label → GUI task prompt is wired and triggers the full demo; no Tray Slot lookup or Command Parser state machine yet |
| 9 | Fixed Tray Slot 연결 | ⬜ Not started |
| 10 | Handle Classifier 학습 | ⬜ Not started |
| 11 | Visual Verification 연결 | ⬜ Not started |
| 12 | Pick → 90° Rotation | ✅ Done — settle-detection + grip-lock + IMU-feedback rotation |
| 13 | D415 Hand Detection | ⬜ Not started |
| 14 | Depth → 3D Hand Position | ⬜ Not started |
| 15 | Vision-Guided Handover | ⬜ Not started — current handoff replays a single pre-recorded fixed pose, not hand-position-driven |
| 16 | Return Zone | ⬜ Not started |
| 17 | Tool Retrieval | ⬜ Not started |
| 18 | Tray Return | 🟡 Place demonstration data being recorded (separate Place recorder in the GUI); no policy trained yet |
| 19 | Wrong-Slot Disturbance Experiment | ⬜ Not started |
| 20 | End-to-End Evaluation | ⬜ Not started |

**Takeaway**: the manipulation core (pick, grip-lock, rotate, scripted handoff, voice trigger, emergency stop) works end-to-end for a 2-tool case on a fixed layout, and the 7-tool Pick dataset (700 episodes) is complete with SmolVLA fine-tuning from `smolvla_base` underway. The next structural pieces — tray-slot state + handle classifier with tray scan and instruction remapping (STEP 9–11, the paper's Verify & Recover method), and real hand detection to replace the scripted handoff pose (STEP 13–15) — are what's needed to match the paper's "Proposed" method (§31, Method C) instead of today's "SmolVLA-only on a fixed layout" approximation.
