#!/usr/bin/env python3
import sys, os, json, time, base64, threading
import numpy as np
import cv2
import zmq
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QPushButton, QLineEdit, QFormLayout, QScrollArea,
    QProgressBar, QTabWidget, QComboBox, QSpinBox, QCheckBox, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    from lerobot.teleoperators.bi_so_leader import BiSOLeader
    from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    import torch
    from lerobot.common.control_utils import predict_action
    from lerobot.policies import prepare_observation_for_inference
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.device_utils import get_safe_torch_device
    LEROBOT_OK = True
except ImportError:
    LEROBOT_OK = False

try:
    from faster_whisper import WhisperModel
    import sounddevice as sd
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

TOOL_ALIASES = {
    "GRASPER": ["grasper", "그라스퍼", "그래스퍼", "그리스포","Grispo"],
    "BIPOLAR": ["bipolar", "바이폴라", "바이플라"],
    "HOOK": ["hook", "훅"],
    "CLIPPER": ["clipper", "클리퍼"],
    "SCISSORS": ["scissors", "scissor", "시저", "가위"],
    "IRRIGATOR": ["irrigator", "이리게이터", "이리게이터", "석션"],
    "SPECIMEN_BAG": ["specimen bag", "스페시먼백", "스페시먼 백", "백", "bag", "back"]
}

# Whisper: 모델 선택지 (첫 항목이 기본값) + 도구 이름 힌트. initial_prompt에 도구 이름을 넣으면
# 비슷한 발음을 그 단어 쪽으로 인식한다. CPU(i7-13700F) 기준 3초 음성: small 0.8s, medium 2.2s,
# large-v3-turbo 3.5s — turbo는 인코더가 large-v3 그대로라 CPU에서는 medium보다 느리다.
WHISPER_MODELS = ["medium", "large-v3-turbo", "small"]
# 짧은 명령용: 타임스탬프·이전 문맥·온도 재시도 끔, 출력 길이 제한 → 잡음에서 길게 늘어지는 것 방지
WHISPER_OPTS = dict(beam_size=5, initial_prompt=None, vad_filter=True, without_timestamps=True,
                    condition_on_previous_text=False, temperature=0.0, max_new_tokens=24)
WHISPER_PROMPT = ("수술 도구 요청. 그라스퍼, 바이폴라, 시저, 가위, 클리퍼, 훅, 이리게이터, 스페시먼 백. "
                  "Grasper, bipolar, scissors, clipper, hook, irrigator, specimen bag.")

# Canonical label -> tool name exactly as used in the recorded task strings
# ("Pick up the <name>"). Must match the dataset, e.g. Clipper was recorded as "clippers".
TOOL_TASK_NAMES = {
    "GRASPER": "grasper",
    "BIPOLAR": "bipolar",
    "HOOK": "hook",
    "CLIPPER": "clippers",
    "SCISSORS": "scissors",
    "IRRIGATOR": "irrigator",
    "SPECIMEN_BAG": "specimen bag",
}

# Standard tray layout, slot 1 → 7 (left → right), as used for the fixed-layout Pick dataset
TRAY_LAYOUT = ["GRASPER", "BIPOLAR", "SCISSORS", "CLIPPER", "HOOK", "IRRIGATOR", "SPECIMEN_BAG"]
EMPTY_SLOT = "EMPTY"


def verify_and_recover(requested, preds, method="proposed", conf_thr=0.8):
    """Decide which slot to pick and which instruction to give the (fixed-layout) policy.

    requested: canonical tool name, e.g. "SCISSORS"
    preds:     [(label, confidence)] for slot 1..7 from the handle classifier (ignored for baseline)
    The policy learned "Pick up the <tool>" as "go to that tool's home slot", so the instruction
    names the *original owner* of the slot where the requested tool actually is.
    Returns dict(status, slot, instruction, arm, reason). status ∈
      baseline  — no verification, requested name as instruction
      verified  — requested tool found in its home slot
      recovered — found in another slot → instruction remapped to that slot
      low_conf  — requested tool only seen below conf_thr → stop
      not_found — requested tool not on the tray → stop
    """
    home = TRAY_LAYOUT.index(requested) + 1
    def result(status, slot, reason):
        if slot is None:
            return {"status": status, "slot": None, "instruction": None, "arm": None, "reason": reason}
        return {"status": status, "slot": slot,
                "instruction": f"Pick up the {TOOL_TASK_NAMES[TRAY_LAYOUT[slot - 1]]}",
                "arm": "left" if slot <= 4 else "right", "reason": reason}
    if method == "baseline":
        return result("baseline", home, "요청 도구 이름 그대로 (검증 없음)")
    label, conf = preds[home - 1]
    if label == requested and conf >= conf_thr:
        return result("verified", home, f"Slot {home}에 {requested} 확인 ({conf:.2f})")
    hits = [(c, i + 1) for i, (l, c) in enumerate(preds) if l == requested]
    sure = [h for h in hits if h[0] >= conf_thr]
    if sure:
        c, slot = max(sure)
        return result("recovered", slot,
                      f"Slot {home}에는 {label} ({conf:.2f}) → {requested}는 Slot {slot} ({c:.2f})")
    if hits:
        c, slot = max(hits)
        return result("low_conf", None, f"{requested} 후보 Slot {slot} 신뢰도 {c:.2f} < {conf_thr}")
    return result("not_found", None, f"트레이에서 {requested}를 찾지 못함")


def _to_canonical_tool(text):
    t = text.lower()
    for canonical, aliases in TOOL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in t:
                return canonical
    return None


ARM_KEYS_L = ["left_arm_shoulder_pan", "left_arm_shoulder_lift", "left_arm_elbow_flex",
              "left_arm_wrist_flex", "left_arm_wrist_roll", "left_arm_gripper"]
ARM_KEYS_R = ["right_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex",
              "right_arm_wrist_flex", "right_arm_wrist_roll", "right_arm_gripper"]
TEMP_MOTOR_PAIRS = [
    ("L 숄더팬", "left_arm_shoulder_pan"), ("L 숄더리프트", "left_arm_shoulder_lift"),
    ("L 엘보", "left_arm_elbow_flex"), ("L 손목F", "left_arm_wrist_flex"),
    ("L 손목R", "left_arm_wrist_roll"), ("L 그리퍼", "left_arm_gripper"),
    ("R 숄더팬", "right_arm_shoulder_pan"), ("R 숄더리프트", "right_arm_shoulder_lift"),
    ("R 엘보", "right_arm_elbow_flex"), ("R 손목F", "right_arm_wrist_flex"),
    ("R 손목R", "right_arm_wrist_roll"), ("R 그리퍼", "right_arm_gripper"),
    ("헤드 팬", "head_motor_1"), ("헤드 틸트", "head_motor_2"),
]
STATE_KEYS = [f"{k}.pos" for k in ARM_KEYS_L + ARM_KEYS_R] + ["head_motor_1.pos", "head_motor_2.pos"]
ACTION_KEYS = [f"{k}.pos" for k in ARM_KEYS_L + ARM_KEYS_R] + [
    "head_motor_1.pos", "head_motor_2.pos", "x.vel", "y.vel", "theta.vel"
]
DATASET_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (len(STATE_KEYS),), "names": STATE_KEYS},
    "action": {"dtype": "float32", "shape": (len(ACTION_KEYS),), "names": ACTION_KEYS},
    "observation.images.head": {"dtype": "video", "shape": (480, 640, 3),
                                 "names": ["height", "width", "channels"]},
    "observation.images.left_wrist": {"dtype": "video", "shape": (240, 320, 3),
                                       "names": ["height", "width", "channels"]},
    "observation.images.right_wrist": {"dtype": "video", "shape": (240, 320, 3),
                                        "names": ["height", "width", "channels"]},
}


def _obs_to_locked_action(obs):
    """Build a holdable scripted action from an observation's arm state.

    observation.*.gripper.pos is the raw motor reading, but xlerobot.py's
    send_action() inverts gripper actions (100 - value) before writing to the
    motor. So replaying a state reading verbatim as an action flips the
    gripper open/closed; it must be re-inverted here to actually hold position.
    """
    pose = {k: obs.get(k, 0.0) or 0.0 for k in STATE_KEYS if k.startswith(("left_arm", "right_arm"))}
    for gkey in ("left_arm_gripper.pos", "right_arm_gripper.pos"):
        if gkey in pose:
            pose[gkey] = 100.0 - pose[gkey]
    return pose


def _decode_b64_image(b64):
    if not b64:
        return None
    try:
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return f  # already RGB-ordered after the encode/decode round trip (see CamLabel.show_frame)
    except Exception:
        return None


class Signals(QObject):
    obs_received   = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    policy_loaded  = pyqtSignal(bool, str)
    voice_done     = pyqtSignal(bool, str, str)
    cls_loaded     = pyqtSignal(bool, str)


class ControlThread(threading.Thread):
    def __init__(self, signals, cfg):
        super().__init__(daemon=True)
        self.signals = signals
        self.cfg = cfg
        self.running = False
        self.teleop = self.cmd_sock = self.obs_sock = None
        self._lock = threading.Lock()
        self.head_pan = self.head_tilt = 0.0
        self.wx = self.wy = self.wt = 0.0
        self.last_action = {}
        self._latest_obs = None
        self.ai_mode = False
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        self.ai_task = ""
        self.scripted_action = None
        # AI 실행 옵션 (둘 다 끄면 기존 동작과 동일: 1/60 s sleep, 50-step chunk를 다 쓴 뒤 동기 추론)
        self.ai_30hz = False     # 학습 데이터 fps(30)로 action 재생
        self.ai_smooth = False   # 다음 chunk를 미리(백그라운드) 계산하고 경계를 crossfade
        self.reset_ai_state()

    def set_head(self, pan, tilt):
        with self._lock:
            self.head_pan, self.head_tilt = pan, tilt

    def set_wheels(self, x, y, theta):
        with self._lock:
            self.wx, self.wy, self.wt = x, y, theta

    SMOOTH_PREFETCH = 20   # 남은 step이 이만큼이면 다음 chunk 계산 시작 (추론 ~145 ms ≈ 60Hz 9 step / 30Hz 5 step)
    SMOOTH_BLEND = 10      # 이전 chunk 남은 부분과 새 chunk를 섞는 step 수

    def reset_ai_state(self):
        self._chunk = None
        self._chunk_i = 0
        self._step = 0
        self._last_vec = None
        self._prefetch_busy = False
        self._prefetch_result = None  # (chunk ndarray, request step)

    def _inference_obs(self, obs):
        imgs = [_decode_b64_image(obs.get(k, "")) for k in ("head", "left_wrist", "right_wrist")]
        if obs is None or any(i is None for i in imgs):
            return None
        return {
            "observation.images.head": imgs[0],
            "observation.images.left_wrist": imgs[1],
            "observation.images.right_wrist": imgs[2],
            "observation.state": np.array([obs.get(k, 0.0) or 0.0 for k in STATE_KEYS], dtype=np.float32),
        }

    def _compute_chunk(self, obs):
        """관측 1개 → 정책의 action chunk 전체 (n, 17), 단위는 select_action 경로와 동일."""
        io = self._inference_obs(obs)
        if io is None:
            return None
        import torch
        with torch.inference_mode():
            b = self.preprocessor(prepare_observation_for_inference(io, self.device, self.ai_task, None))
            ch = self.policy.predict_action_chunk(b)
            out = self.postprocessor(ch)
        out = out.detach().to("cpu")
        return out.squeeze(0).numpy() if out.dim() == 3 else out.numpy()

    def _prefetch(self, obs, req_step):
        try:
            c = self._compute_chunk(obs)
            if c is not None:
                self._prefetch_result = (c, req_step)
        finally:
            self._prefetch_busy = False

    def _smooth_next_action(self):
        """현재 chunk를 재생하면서 다음 chunk를 미리 계산 → 멈춤 없이 이어 붙임 (경계는 crossfade)."""
        if self._chunk is None:
            c = self._compute_chunk(self._latest_obs)
            if c is None:
                return None
            self._chunk, self._chunk_i = c, 0
        if self._prefetch_result is not None:
            new, req = self._prefetch_result
            self._prefetch_result = None
            new = new[max(0, self._step - req):]      # 계산하는 동안 이미 지나간 step은 건너뜀
            old = self._chunk[self._chunk_i:]
            n = min(len(old), len(new), self.SMOOTH_BLEND)
            if len(new):
                w = (np.arange(n) + 1)[:, None] / (n + 1)
                head = (1 - w) * old[:n] + w * new[:n] if n else new[:0]
                self._chunk, self._chunk_i = np.concatenate([head, new[n:]]), 0
        remaining = len(self._chunk) - self._chunk_i
        if remaining <= self.SMOOTH_PREFETCH and not self._prefetch_busy and self._prefetch_result is None:
            self._prefetch_busy = True
            threading.Thread(target=self._prefetch, args=(self._latest_obs, self._step), daemon=True).start()
        if remaining <= 0:
            vec = self._last_vec  # 다음 chunk가 아직 없으면 마지막 자세 유지
            if vec is None:
                return None
        else:
            vec = self._chunk[self._chunk_i]; self._chunk_i += 1
        self._step += 1
        self._last_vec = vec
        return {k: float(v) for k, v in zip(ACTION_KEYS, vec)}

    def _infer_action(self):
        obs = self._latest_obs
        if obs is None:
            return None
        head_img = _decode_b64_image(obs.get("head", ""))
        left_img = _decode_b64_image(obs.get("left_wrist", ""))
        right_img = _decode_b64_image(obs.get("right_wrist", ""))
        if head_img is None or left_img is None or right_img is None:
            return None
        state = np.array([obs.get(k, 0.0) or 0.0 for k in STATE_KEYS], dtype=np.float32)
        inference_obs = {
            "observation.images.head": head_img,
            "observation.images.left_wrist": left_img,
            "observation.images.right_wrist": right_img,
            "observation.state": state,
        }
        action_out = predict_action(
            inference_obs, self.policy, self.device,
            self.preprocessor, self.postprocessor,
            use_amp=False, task=self.ai_task,
        )
        if isinstance(action_out, dict):
            action_out = action_out.get("action", next(iter(action_out.values())))
        action_out = action_out.detach().to("cpu")
        if action_out.dim() > 1:
            action_out = action_out.squeeze(0)
        action_vec = action_out.numpy()
        return {k: float(v) for k, v in zip(ACTION_KEYS, action_vec)}

    def run(self):
        try:
            leader_cfg = BiSOLeaderConfig(
                id=self.cfg["teleop_id"],
                left_arm_config=SOLeaderTeleopConfig(port=self.cfg["left_port"]),
                right_arm_config=SOLeaderTeleopConfig(port=self.cfg["right_port"]),
            )
            self.teleop = BiSOLeader(leader_cfg)
            self.teleop.connect()

            ctx = zmq.Context()
            self.cmd_sock = ctx.socket(zmq.PUSH)
            self.cmd_sock.setsockopt(zmq.CONFLATE, 1)
            self.cmd_sock.connect(f"tcp://{self.cfg['ip']}:{self.cfg['cmd_port']}")

            self.obs_sock = ctx.socket(zmq.PULL)
            self.obs_sock.setsockopt(zmq.CONFLATE, 1)
            self.obs_sock.connect(f"tcp://{self.cfg['ip']}:{self.cfg['obs_port']}")

            self.running = True
            self.signals.status_changed.emit("연결됨 ✓")

            next_t = time.perf_counter()
            while self.running:
                try:
                    try:
                        msg = self.obs_sock.recv_string(flags=zmq.NOBLOCK)
                        self._latest_obs = json.loads(msg)
                        self.signals.obs_received.emit(self._latest_obs)
                    except zmq.Again:
                        pass

                    if self.scripted_action is not None:
                        action = dict(self.scripted_action)
                    elif self.ai_mode and self.policy is not None:
                        action = self._smooth_next_action() if self.ai_smooth else self._infer_action()
                        if action is None:
                            time.sleep(1/60)
                            continue
                    else:
                        action = self.teleop.get_action()
                        # BiSOLeader returns "left_shoulder_pan.pos" etc; the dataset/policy
                        # convention (and the robot's own remap) expects "left_arm_shoulder_pan.pos"
                        action = {
                            ("left_arm_" + k[len("left_"):]) if k.startswith("left_") and not k.startswith("left_arm_")
                            else ("right_arm_" + k[len("right_"):]) if k.startswith("right_") and not k.startswith("right_arm_")
                            else k: v
                            for k, v in action.items()
                        }

                    with self._lock:
                        action.update({
                            "head_motor_1.pos": self.head_pan,
                            "head_motor_2.pos": self.head_tilt,
                            "x.vel": self.wx, "y.vel": self.wy, "theta.vel": self.wt,
                        })
                    self.last_action = dict(action)
                    self.cmd_sock.send_string(json.dumps(action))
                    if self.ai_30hz and self.ai_mode and self.policy is not None:
                        next_t += 1 / 30  # 처리 시간과 무관하게 정확히 30 Hz
                        d = next_t - time.perf_counter()
                        if d > 0:
                            time.sleep(d)
                        else:
                            next_t = time.perf_counter()
                    else:
                        time.sleep(1/60)  # 기존 동작
                        next_t = time.perf_counter()
                except Exception as e:
                    self.signals.status_changed.emit(f"오류: {e}")
                    time.sleep(0.1)
        except Exception as e:
            self.signals.status_changed.emit(f"연결 실패: {e}")

    def stop(self):
        self.running = False
        for obj in [self.teleop, self.cmd_sock, self.obs_sock]:
            try:
                if obj: obj.disconnect() if hasattr(obj, 'disconnect') else obj.close()
            except: pass


class ImuBar(QWidget):
    def __init__(self, label, color, mn=-250, mx=250):
        super().__init__()
        self.mn, self.mx = mn, mx
        self.setFixedHeight(22)
        layout = QHBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(4)
        lbl = QLabel(label); lbl.setFixedWidth(24)
        lbl.setStyleSheet(f"color:{color};font-family:monospace;font-size:11px;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000); self.bar.setValue(500); self.bar.setTextVisible(False)
        self.bar.setStyleSheet(f"QProgressBar{{background:#0d1117;border:1px solid #30363d;border-radius:3px;}}"
                               f"QProgressBar::chunk{{background:{color};border-radius:2px;}}")
        self.num = QLabel("  0.0"); self.num.setFixedWidth(52)
        self.num.setStyleSheet(f"color:{color};font-family:monospace;font-size:11px;")
        self.num.setAlignment(Qt.AlignRight)
        layout.addWidget(lbl); layout.addWidget(self.bar); layout.addWidget(self.num)

    def update_val(self, v):
        ratio = (v - self.mn) / (self.mx - self.mn)
        self.bar.setValue(int(max(0, min(1000, ratio * 1000))))
        self.num.setText(f"{v:+.1f}")


class TempPanel(QWidget):
    def __init__(self):
        super().__init__()
        fl = QFormLayout(self); fl.setSpacing(3); fl.setContentsMargins(0, 0, 0, 0)
        self.labels = {}
        for short, key in TEMP_MOTOR_PAIRS:
            l = QLabel("--°C")
            l.setStyleSheet("color:#7ee787;font-family:monospace;font-size:11px;font-weight:bold;")
            fl.addRow(short + ":", l)
            self.labels[key] = l

    def update_temps(self, obs):
        for key, lbl in self.labels.items():
            v = obs.get(f"{key}.temp")
            if v is None:
                continue
            if v >= 65:
                color = "#ff7b72"
            elif v >= 50:
                color = "#f0883e"
            else:
                color = "#7ee787"
            lbl.setText(f"{v}°C")
            lbl.setStyleSheet(f"color:{color};font-family:monospace;font-size:11px;font-weight:bold;")


class RobotVizWidget(FigureCanvasQTAgg if MPL_OK else QWidget):
    def __init__(self):
        if not MPL_OK:
            super().__init__()
            return
        fig = Figure(figsize=(2.8, 2.8), facecolor='#0d1117', tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection='3d')
        self._joints_l = [0.0]*6
        self._joints_r = [0.0]*6
        self._draw()

    def _setup(self):
        ax = self.ax; ax.cla()
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e', labelsize=6)
        for spine in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            spine.fill = False; spine.set_edgecolor('#30363d')
        ax.set_xlim(-0.35, 0.35); ax.set_ylim(-0.25, 0.25); ax.set_zlim(0, 0.55)
        ax.set_xlabel('X', color='#8b949e', fontsize=7)
        ax.set_ylabel('Y', color='#8b949e', fontsize=7)
        ax.set_zlabel('Z', color='#8b949e', fontsize=7)

    def _fk(self, joints, base):
        sp, sl, ef, wf = [np.radians(j) for j in joints[:4]]
        L1, L2, L3 = 0.11, 0.13, 0.09
        p0 = np.array(base)
        p1 = p0 + np.array([L1*np.cos(sp)*np.cos(sl), L1*np.sin(sp)*np.cos(sl), L1*np.sin(sl)])
        a2 = sl - ef
        p2 = p1 + np.array([L2*np.cos(sp)*np.cos(a2), L2*np.sin(sp)*np.cos(a2), L2*np.sin(a2)])
        a3 = a2 - wf
        p3 = p2 + np.array([L3*np.cos(sp)*np.cos(a3), L3*np.sin(sp)*np.cos(a3), L3*np.sin(a3)])
        return [p0, p1, p2, p3]

    def _draw(self):
        if not MPL_OK: return
        self._setup()
        for pts, color, label in [
            (self._fk(self._joints_l, [-0.18, 0, 0.28]), '#79c0ff', 'L'),
            (self._fk(self._joints_r, [ 0.18, 0, 0.28]), '#ff7b72', 'R'),
        ]:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            self.ax.plot(xs, ys, zs, 'o-', color=color, linewidth=2.5, markersize=5)
            self.ax.text(xs[0], ys[0], zs[0]+0.02, label, color=color, fontsize=8)
        # body
        self.ax.plot([-0.18, 0.18], [0, 0], [0.28, 0.28], color='#56d364', linewidth=3)
        self.ax.plot([0, 0], [0, 0], [0, 0.28], color='#56d364', linewidth=2, linestyle='--')
        self.draw()

    def update_pose(self, left_joints, right_joints):
        if not MPL_OK: return
        self._joints_l = left_joints
        self._joints_r = right_joints
        self._draw()


class CamLabel(QLabel):
    def __init__(self, title):
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0d1117; color:#58a6ff; border:1px solid #30363d; font-size:13px;")
        self.setMinimumSize(200, 150)
        self._img_size = None      # (w, h) of the last frame, for widget→image coordinate mapping
        self.on_roi_drawn = None   # callback(x1, y1, x2, y2) in image pixels; drag is enabled while set
        self.ref_overlay = None    # RGB reference image blended over the frame (alignment guide)
        self.ref_alpha = 0.5
        self.ref_diff = False      # True: show |frame - reference| instead of a blend
        self._drag_start = None
        self._drag_rect = None

    def _to_image_xy(self, pos):
        pm = self.pixmap()
        if pm is None or pm.isNull() or self._img_size is None:
            return None
        iw, ih = self._img_size
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        x = (pos.x() - ox) * iw / pm.width()
        y = (pos.y() - oy) * ih / pm.height()
        return int(min(max(x, 0), iw - 1)), int(min(max(y, 0), ih - 1))

    def mousePressEvent(self, e):
        if self.on_roi_drawn is None: return super().mousePressEvent(e)
        self._drag_start = self._to_image_xy(e.pos())

    def mouseMoveEvent(self, e):
        if self.on_roi_drawn is None or self._drag_start is None: return super().mouseMoveEvent(e)
        p = self._to_image_xy(e.pos())
        if p: self._drag_rect = (*self._drag_start, *p)

    def mouseReleaseEvent(self, e):
        if self.on_roi_drawn is None or self._drag_start is None: return super().mouseReleaseEvent(e)
        p = self._to_image_xy(e.pos())
        x0, y0 = self._drag_start
        self._drag_start = self._drag_rect = None
        if p and abs(p[0] - x0) > 5 and abs(p[1] - y0) > 5:
            self.on_roi_drawn(min(x0, p[0]), min(y0, p[1]), max(x0, p[0]), max(y0, p[1]))

    def show_frame(self, b64, boxes=None):
        if not b64: return
        try:
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is None: return
            ref = self.ref_overlay
            if ref is not None and ref.shape == f.shape:
                f = cv2.absdiff(f, ref) if self.ref_diff else cv2.addWeighted(f, 1 - self.ref_alpha, ref, self.ref_alpha, 0)
            if self._drag_rect:
                boxes = list(boxes or []) + [{"x1": self._drag_rect[0], "y1": self._drag_rect[1],
                                              "x2": self._drag_rect[2], "y2": self._drag_rect[3],
                                              "label": "", "color": (255, 255, 255)}]
            if boxes:
                for b in boxes:
                    color = b.get("color", (0, 255, 0))
                    cv2.rectangle(f, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
                    text = f"{b['label']} {b['conf']:.2f}" if "conf" in b else b["label"]
                    cv2.putText(f, text, (b["x1"], max(b["y1"]-6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            h, w, c = f.shape
            self._img_size = (w, h)
            qi = QImage(f.data, w, h, w*c, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qi).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except: pass


class DatasetRecorder(QGroupBox):
    """LeRobotDataset 녹화 패널 하나 (Repo ID / Task / 에피소드 / 업로드).

    Pick, Place 등 데이터셋마다 하나씩 만들어 독립적으로 녹화한다.
    is_busy()가 True면 (다른 패널이 녹화 중) 새 에피소드를 시작하지 않는다.
    """
    upload_done = pyqtSignal(bool, str)

    def __init__(self, title, repo_id, task, status, is_busy):
        super().__init__(title)
        self.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        self._status = status
        self._is_busy = is_busy
        self.dataset = None
        self.recording = False
        self.episode_count = 0
        self.upload_done.connect(self._on_upload_done)

        rfl = QVBoxLayout(self); rfl.setSpacing(4)
        self.f_repo = QLineEdit(repo_id)
        self.f_task = QLineEdit(task)
        rfl.addWidget(QLabel("Repo ID:")); rfl.addWidget(self.f_repo)
        rfl.addWidget(QLabel("Task 설명:")); rfl.addWidget(self.f_task)
        self.btn_dataset = QPushButton("데이터셋 생성")
        self.btn_dataset.setStyleSheet("QPushButton{background:#238636;color:white;padding:6px;border-radius:4px;}")
        self.btn_dataset.clicked.connect(self._toggle_dataset)
        rfl.addWidget(self.btn_dataset)
        self.btn_episode = QPushButton("● 에피소드 녹화 시작")
        self.btn_episode.setEnabled(False)
        self.btn_episode.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
        self.btn_episode.clicked.connect(self._toggle_episode)
        rfl.addWidget(self.btn_episode)
        self.btn_discard = QPushButton("현재 에피소드 폐기")
        self.btn_discard.setEnabled(False)
        self.btn_discard.setStyleSheet("QPushButton{background:#da3633;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
        self.btn_discard.clicked.connect(self._discard_episode)
        rfl.addWidget(self.btn_discard)
        self.l_episode_count = QLabel("녹화된 에피소드: 0")
        self.l_episode_count.setStyleSheet("color:#7ee787;font-family:monospace;")
        rfl.addWidget(self.l_episode_count)
        self.btn_upload = QPushButton("허깅페이스 업로드")
        self.btn_upload.setEnabled(False)
        self.btn_upload.setStyleSheet("QPushButton{background:#8957e5;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
        self.btn_upload.clicked.connect(self._upload_dataset)
        rfl.addWidget(self.btn_upload)

    def task(self):
        return self.f_task.text()

    def add_frame(self, frame):
        if self.dataset is None or not self.recording or frame is None:
            return
        try:
            self.dataset.add_frame(frame)
        except Exception as e:
            self._status(f"프레임 기록 실패: {e}")

    def _toggle_dataset(self):
        if self.dataset is None:
            if not LEROBOT_OK:
                self._status("lerobot 데이터셋 모듈을 불러올 수 없음")
                return
            repo_id = self.f_repo.text()
            existing = (HF_LEROBOT_HOME / repo_id).exists()
            try:
                if existing:
                    self.dataset = LeRobotDataset.resume(
                        repo_id=repo_id,
                        root=HF_LEROBOT_HOME / repo_id,
                        image_writer_threads=4,
                    )
                    self.episode_count = self.dataset.meta.total_episodes
                    msg = f"기존 데이터셋 이어서 녹화: {repo_id} (기존 {self.episode_count}개)"
                else:
                    self.dataset = LeRobotDataset.create(
                        repo_id=repo_id,
                        fps=30,
                        features=DATASET_FEATURES,
                        robot_type="xlerobot",
                        use_videos=True,
                        image_writer_threads=4,
                    )
                    self.episode_count = 0
                    msg = f"데이터셋 생성됨: {repo_id}"
                self.l_episode_count.setText(f"녹화된 에피소드: {self.episode_count}")
                self.btn_dataset.setText("데이터셋 종료")
                self.btn_episode.setEnabled(True)
                self.btn_upload.setEnabled(True)
                self.f_repo.setEnabled(False)
                self._status(msg)
            except Exception as e:
                self._status(f"데이터셋 열기 실패: {e}")
        else:
            if self.recording:
                self._toggle_episode()
            try:
                self.dataset.finalize()  # parquet footer 기록 — 안 하면 파일이 열리지 않음
            except Exception as e:
                self._status(f"데이터셋 마무리 실패: {e}")
            self.dataset = None
            self.btn_dataset.setText("데이터셋 생성")
            self.btn_episode.setEnabled(False)
            self.btn_discard.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.f_repo.setEnabled(True)
            self._status("데이터셋 세션 종료")

    def _toggle_episode(self):
        if self.dataset is None:
            return
        if not self.recording:
            if self._is_busy():
                self._status("다른 데이터셋이 녹화 중 — 먼저 저장/폐기하세요")
                return
            self.recording = True
            self.btn_episode.setText("■ 에피소드 저장")
            self.btn_discard.setEnabled(True)
            self._status(f"녹화 중... ({self.title()})")
        else:
            self.recording = False
            try:
                self.dataset.save_episode()
                self.episode_count += 1
                self.l_episode_count.setText(f"녹화된 에피소드: {self.episode_count}")
                self._status(f"에피소드 {self.episode_count} 저장됨 ({self.title()})")
            except Exception as e:
                self._status(f"저장 실패: {e}")
            self.btn_episode.setText("● 에피소드 녹화 시작")
            self.btn_discard.setEnabled(False)

    def _discard_episode(self):
        if self.dataset is None or not self.recording:
            return
        try:
            self.dataset.clear_episode_buffer()
        except Exception:
            pass
        self.recording = False
        self.btn_episode.setText("● 에피소드 녹화 시작")
        self.btn_discard.setEnabled(False)
        self._status("에피소드 폐기됨")

    def discard_on_close(self):
        if self.dataset is not None and self.recording:
            try: self.dataset.clear_episode_buffer()
            except Exception: pass
        if self.dataset is not None:
            try: self.dataset.finalize()
            except Exception: pass

    def _upload_dataset(self):
        if self.dataset is None or self.recording:
            return
        self.btn_upload.setEnabled(False)
        self.btn_episode.setEnabled(False)
        self.btn_discard.setEnabled(False)
        self._status("허깅페이스 업로드 중... (에피소드 수에 따라 수 분 소요)")
        dataset = self.dataset

        def _do_upload():
            try:
                dataset.finalize()
                dataset.push_to_hub()
                self.upload_done.emit(True, "허깅페이스 업로드 완료 ✓")
            except Exception as e:
                self.upload_done.emit(False, f"업로드 실패: {e}")

        threading.Thread(target=_do_upload, daemon=True).start()

    def _on_upload_done(self, ok, msg):
        self._status(msg)
        self.btn_upload.setEnabled(not ok)


class HandleClassifier:
    """ResNet18 손잡이 분류기 (classifier/train_classifier.py로 학습). 백그라운드에서 로드."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.tc = self.device = None
        self.error = None
        self.loading = False

    def load_async(self, on_done):
        if self.loading or self.model is not None:
            return
        self.loading = True

        def _work():
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "classifier"))
                import train_classifier as tc  # same preprocess() as training
                import torch
                ckpt = torch.load(self.model_path, map_location="cpu")
                model = tc.make_model()
                model.load_state_dict(ckpt["state_dict"])
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model, self.tc = model.to(self.device).eval(), tc
                on_done(True, f"분류기 로드 완료 ({self.device})")
            except Exception as e:
                self.error = str(e)
                on_done(False, f"분류기 로드 실패: {e}")
            finally:
                self.loading = False

        threading.Thread(target=_work, daemon=True).start()

    def classify(self, img_rgb, rois):
        """→ [(label, confidence)] per ROI"""
        import torch
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # preprocess() expects BGR like cv2.imread
        h, w = bgr.shape[:2]
        xs = []
        for x1, y1, x2, y2 in rois:
            crop = bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            xs.append(self.tc.preprocess(crop, gray=False, train=False))
        with torch.no_grad():
            p = torch.softmax(self.model(torch.stack(xs).to(self.device)), 1).cpu()
        conf, idx = p.max(1)
        return [(self.tc.CLASSES[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]


class HandDetector:
    """MediaPipe HandLandmarker (CPU, ~20 ms/frame). 손이 여럿이면 화면에서 가장 큰 손을 사용."""

    def __init__(self, model_path):
        self.model_path = model_path
        self._det = None
        self.error = None

    def _ensure(self):
        if self._det is None and self.error is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mpt
                from mediapipe.tasks.python import vision
                opt = vision.HandLandmarkerOptions(
                    base_options=mpt.BaseOptions(model_asset_path=self.model_path),
                    running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.5)
                self._mp, self._det = mp, vision.HandLandmarker.create_from_options(opt)
            except Exception as e:
                self.error = str(e)
        return self._det is not None

    def detect(self, img_rgb):
        """→ dict(cx, cy, box=(x1,y1,x2,y2), pts=[(x,y)...]) in pixels, or None"""
        if not self._ensure():
            return None
        h, w = img_rgb.shape[:2]
        r = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb)))
        best = None
        for lms in r.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in lms]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            palm = [pts[i] for i in (0, 5, 9, 13, 17)]  # 손목 + 손가락 뿌리 = 손바닥 중심
            cand = {"cx": int(np.mean([p[0] for p in palm])), "cy": int(np.mean([p[1] for p in palm])),
                    "box": (min(xs), min(ys), max(xs), max(ys)), "pts": pts, "area": area}
            if best is None or area > best["area"]:
                best = cand
        return best


def grid_cell(x, y, w, h, n=3):
    """픽셀 좌표 → (row, col), 0..n-1 (row: 위→아래, col: 왼쪽→오른쪽)"""
    return min(n - 1, int(y * n / h)), min(n - 1, int(x * n / w))


def nearest_saved_cell(saved, cell):
    """saved: {"r,c": pose}. 같은 칸이 없으면 가장 가까운 저장 칸 (맨해튼 거리)."""
    if not saved:
        return None
    key = f"{cell[0]},{cell[1]}"
    if key in saved:
        return key
    def dist(k):
        r, c = map(int, k.split(","))
        return abs(r - cell[0]) + abs(c - cell[1]), k
    return min(saved, key=dist)


def arm_from_instruction(text):
    """'Pick up the <tool>' → 그 도구의 home slot 기준 팔 ('left'/'right'), 모르면 None.
    고정 배치 정책은 도구 이름을 slot으로 해석하므로, remap된 instruction에도 그대로 맞다."""
    t = text.strip().lower().rstrip(".")
    for canon, name in TOOL_TASK_NAMES.items():
        if t == f"pick up the {name}":
            return "left" if TRAY_LAYOUT.index(canon) < 4 else "right"
    return None


class HandoverPanel(QGroupBox):
    """손 위치 기반 핸드오버 설정: 시선(헤드) 자세 + 팔별 3×3 칸 자세 저장/테스트, 손 검출 미리보기."""

    def __init__(self, config_path, get_obs, get_head, set_head, move_to, status):
        super().__init__("손 위치 기반 핸드오버 (3×3)")
        self.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        self._path, self._get_obs, self._get_head = config_path, get_obs, get_head
        self._set_head, self._move_to, self._status = set_head, move_to, status
        self.cfg = {"head": [-4.0, 40.0], "left": {}, "right": {}}
        try:
            with open(config_path) as f:
                self.cfg.update(json.load(f))
        except (OSError, ValueError):
            pass
        self.cell = (1, 1)
        self.last_hand = None      # 미리보기/진행 중 마지막 검출 결과
        self.last_hand_t = 0.0

        fl = QFormLayout(self); fl.setSpacing(4); fl.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.cb_preview = QCheckBox("손 검출 미리보기")
        self.cb_preview.setToolTip("헤드 화면에 9칸 격자와 검출된 손 표시")
        fl.addRow(self.cb_preview)
        self.l_head = QLabel(); self.l_head.setStyleSheet("color:#f0883e;font-family:monospace;")
        fl.addRow("시선 자세:", self.l_head)
        hb = QHBoxLayout()
        b1 = QPushButton("헤드 저장"); b1.clicked.connect(self._save_head)
        b1.setToolTip("현재 헤드 자세를 핸드오버 시선 자세로 저장")
        b2 = QPushButton("시선으로 이동"); b2.clicked.connect(lambda: self._set_head(*self.cfg["head"]))
        hb.addWidget(b1); hb.addWidget(b2); fl.addRow(hb)

        self.cb_arm = QComboBox(); self.cb_arm.addItems(["left", "right"])
        self.cb_arm.currentIndexChanged.connect(self._refresh)
        fl.addRow("팔:", self.cb_arm)
        grid = QWidget(); gl = QVBoxLayout(grid); gl.setSpacing(2); gl.setContentsMargins(0, 0, 0, 0)
        self.cell_btns = {}
        for r in range(3):  # 3×3 9칸 모두 사용
            row = QHBoxLayout(); row.setSpacing(2)
            for c in range(3):
                b = QPushButton(f"{r},{c}"); b.setFixedHeight(28)
                b.clicked.connect(lambda _, rc=(r, c): self._select(rc))
                row.addWidget(b); self.cell_btns[(r, c)] = b
            gl.addLayout(row)
        fl.addRow("칸 (화면 기준):", grid)
        hint = QLabel("미저장 칸 / 손 미검출 → 1,1 (없으면 가장 가까운 칸)"); hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b949e;font-size:11px;")
        fl.addRow(hint)
        bb = QHBoxLayout()
        b3 = QPushButton("현재 자세 저장"); b3.clicked.connect(self._save_pose)
        b3.setToolTip("선택한 팔·칸에 지금 팔 자세를 저장 (도구를 쥔 상태로)")
        b4 = QPushButton("삭제"); b4.clicked.connect(self._delete_pose)
        bb.addWidget(b3); bb.addWidget(b4); fl.addRow(bb)
        b5 = QPushButton("선택 칸으로 이동 (테스트)"); b5.clicked.connect(self._test_pose)
        fl.addRow(b5)
        self.l_info = QLabel(); self.l_info.setWordWrap(True)
        self.l_info.setStyleSheet("color:#8b949e;font-size:11px;")
        fl.addRow(self.l_info)
        self._refresh()

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self.cfg, f, indent=2)

    def _save_head(self):
        self.cfg["head"] = list(self._get_head()); self._save(); self._refresh()
        self._status(f"핸드오버 시선 자세 저장: pan {self.cfg['head'][0]:.1f}, tilt {self.cfg['head'][1]:.1f}")

    def _select(self, rc):
        self.cell = rc; self._refresh()

    def _save_pose(self):
        obs = self._get_obs()
        if not obs:
            self._status("관측 데이터 없음 (연결 확인)"); return
        arm, key = self.cb_arm.currentText(), f"{self.cell[0]},{self.cell[1]}"
        self.cfg[arm][key] = _obs_to_locked_action(obs); self._save(); self._refresh()
        self._status(f"핸드오버 자세 저장: {arm} 팔, 칸 {key}")

    def _delete_pose(self):
        arm, key = self.cb_arm.currentText(), f"{self.cell[0]},{self.cell[1]}"
        if self.cfg[arm].pop(key, None) is not None:
            self._save(); self._refresh(); self._status(f"삭제: {arm} 팔, 칸 {key}")

    def _test_pose(self):
        arm = self.cb_arm.currentText()
        key = nearest_saved_cell(self.cfg[arm], self.cell)
        if key is None:
            self._status(f"{arm} 팔에 저장된 자세 없음"); return
        self._move_to(self.cfg[arm][key])
        self._status(f"{arm} 팔 칸 {key} 자세로 이동 (🛑 긴급 정지로 해제)")

    DEFAULT_CELL = (1, 1)

    def pose_for(self, arm, cell):
        """저장된 칸이면 그 자세, 아니면(미저장·손 미검출 cell=None) 1,1 자세.
        1,1도 없으면 가장 가까운 저장 칸."""
        saved = self.cfg.get(arm, {})
        if cell is not None and f"{cell[0]},{cell[1]}" in saved:
            key = f"{cell[0]},{cell[1]}"
        else:
            key = nearest_saved_cell(saved, self.DEFAULT_CELL)
        return (key, saved[key]) if key else (None, None)

    def _refresh(self):
        arm = self.cb_arm.currentText()
        for rc, b in self.cell_btns.items():
            saved = f"{rc[0]},{rc[1]}" in self.cfg[arm]
            sel = rc == self.cell
            b.setStyleSheet(f"QPushButton{{background:{'#238636' if saved else '#30363d'};color:white;"
                            f"border:{'2px solid #f0883e' if sel else '1px solid #484f58'};border-radius:3px;}}")
        self.l_head.setText(f"pan={self.cfg['head'][0]:.1f}, tilt={self.cfg['head'][1]:.1f}")
        dk = f"{self.DEFAULT_CELL[0]},{self.DEFAULT_CELL[1]}"
        miss = [a for a in ("left", "right") if dk not in self.cfg[a]]
        self.l_info.setText(f"저장된 칸 — left {len(self.cfg['left'])}/9, right {len(self.cfg['right'])}/9 (초록 = 저장됨)"
                            + (f"\n⚠️ 기본 칸 {dk} 미저장: {', '.join(miss)}" if miss else ""))

    def overlay_boxes(self, w=640, h=480, target=None):
        boxes = []
        for r in range(3):
            for c in range(3):
                x1, y1 = c * w // 3, r * h // 3
                on = target == (r, c)
                boxes.append({"x1": x1 + 1, "y1": y1 + 1, "x2": x1 + w // 3 - 1, "y2": y1 + h // 3 - 1,
                              "label": f"{r},{c}", "color": (60, 140, 255) if on else (120, 120, 120)})
        hand = self.last_hand if time.time() - self.last_hand_t < 0.5 else None
        if hand:
            x1, y1, x2, y2 = hand["box"]
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": "hand", "color": (255, 220, 0)})
            boxes.append({"x1": hand["cx"] - 4, "y1": hand["cy"] - 4, "x2": hand["cx"] + 4, "y2": hand["cy"] + 4,
                          "label": "", "color": (255, 60, 60)})
        return boxes


class AlignGuide(QGroupBox):
    """정렬 가이드: 학습 데이터의 head camera 시작 화면을 현재 화면에 겹쳐 보여주고,
    트레이 영역의 어긋남(px)을 phase correlation으로 계산해 표시한다.

    기준 이미지는 Pick 데이터셋 여러 episode의 첫 프레임 중앙값(median)으로 만든다.
    모델이 위치를 외워서 동작하므로, 로봇/트레이 위치를 녹화 때와 맞추는 용도.
    """
    ref_ready = pyqtSignal(bool, str)
    TOL_PX = 5          # 이 이내면 "정렬됨"
    TRAY_ROWS = 0.6     # 비교에 쓰는 화면 위쪽 비율 (아래쪽 팔/그리퍼 제외)

    def __init__(self, ref_path, cam, get_repo, get_head, status):
        super().__init__("정렬 가이드")
        self.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        self._ref_path = ref_path
        self._cam = cam
        self._get_repo = get_repo
        self._get_head = get_head
        self._status = status
        self._ref = None
        self._ref_head_pose = (-4.0, 70.0)
        self._n = 0
        self.ref_ready.connect(self._on_ref_ready)

        fl = QFormLayout(self); fl.setSpacing(4)
        self.cb_on = QCheckBox("학습 화면 겹쳐 보기"); self.cb_on.toggled.connect(self._apply)
        fl.addRow(self.cb_on)
        self.cb_diff = QCheckBox("차이만 보기"); self.cb_diff.toggled.connect(self._apply)
        self.cb_diff.setToolTip("현재 화면과 학습 화면의 차이만 표시 (어긋난 곳이 밝게)")
        fl.addRow(self.cb_diff)
        self.sl_alpha = QSlider(Qt.Horizontal); self.sl_alpha.setRange(10, 90); self.sl_alpha.setValue(50)
        self.sl_alpha.valueChanged.connect(self._apply)
        fl.addRow("투명도:", self.sl_alpha)
        self.l_shift = QLabel("—"); self.l_shift.setWordWrap(True)
        self.l_shift.setStyleSheet("color:#8b949e;font-family:monospace;")
        fl.addRow(self.l_shift)
        self.btn_make = QPushButton("기준 화면 만들기 (데이터셋에서)")
        self.btn_make.clicked.connect(self._make_reference)
        fl.addRow(self.btn_make)
        self._load_reference()

    # ── reference image ──
    def _load_reference(self):
        img = cv2.imread(self._ref_path) if os.path.exists(self._ref_path) else None
        self._ref = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        self.btn_make.setText("기준 화면 다시 만들기" if self._ref is not None else "기준 화면 만들기 (데이터셋에서)")
        if self._ref is None:
            self.l_shift.setText("기준 화면 없음 — 아래 버튼으로 생성")
        self._apply()

    def _make_reference(self):
        if not LEROBOT_OK:
            self._status("lerobot 데이터셋 모듈을 불러올 수 없음"); return
        repo = self._get_repo()
        self.btn_make.setEnabled(False)
        self._status(f"기준 화면 생성 중... ({repo}, 수십 초 소요)")

        def _work():
            try:
                ds = LeRobotDataset(repo)
                n_ep = ds.meta.total_episodes
                eps = np.linspace(0, n_ep - 1, min(15, n_ep)).astype(int)
                frames = []
                for e in eps:
                    f = ds[int(ds.meta.episodes[int(e)]["dataset_from_index"])]
                    frames.append((f["observation.images.head"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                ref = np.median(np.stack(frames), axis=0).astype(np.uint8)  # RGB
                cv2.imwrite(self._ref_path, cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
                self.ref_ready.emit(True, f"기준 화면 저장: {len(frames)}개 episode 시작 프레임의 중앙값")
            except Exception as e:
                self.ref_ready.emit(False, f"기준 화면 생성 실패: {e}")

        threading.Thread(target=_work, daemon=True).start()

    def _on_ref_ready(self, ok, msg):
        self.btn_make.setEnabled(True)
        self._status(msg)
        if ok:
            self._load_reference()

    # ── overlay + shift ──
    def _apply(self):
        on = self.cb_on.isChecked() and self._ref is not None
        self._cam.ref_overlay = self._ref if on else None
        self._cam.ref_diff = self.cb_diff.isChecked()
        self._cam.ref_alpha = self.sl_alpha.value() / 100.0

    def on_frame(self, head_b64):
        """_on_obs에서 매 프레임 호출. 계산은 10프레임마다 (약 0.3초)."""
        if not self.cb_on.isChecked() or self._ref is None:
            return
        self._n += 1
        if self._n % 10:
            return
        live = _decode_b64_image(head_b64)
        if live is None or live.shape != self._ref.shape:
            return
        pan, tilt = self._get_head()
        if abs(pan - self._ref_head_pose[0]) > 0.5 or abs(tilt - self._ref_head_pose[1]) > 0.5:
            self.l_shift.setText(f"헤드 자세가 녹화 때와 다름 (pan {pan:.1f}, tilt {tilt:.1f})\n"
                                 f"→ pan {self._ref_head_pose[0]}, tilt {self._ref_head_pose[1]}로 맞춘 뒤 비교")
            self.l_shift.setStyleSheet("color:#d29922;font-family:monospace;")
            return
        h = int(live.shape[0] * self.TRAY_ROWS)
        g = lambda x: cv2.cvtColor(x[:h], cv2.COLOR_RGB2GRAY).astype(np.float32)
        win = cv2.createHanningWindow((live.shape[1], h), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(g(self._ref), g(live), win)  # +dx: 화면 속 트레이가 오른쪽, +dy: 아래
        ok = abs(dx) <= self.TOL_PX and abs(dy) <= self.TOL_PX
        lr = "오른쪽" if dx > 0 else "왼쪽"
        ud = "아래" if dy > 0 else "위"
        hint = "정렬됨 ✓" if ok else (f"화면 속 트레이가 기준보다 {lr} {abs(dx):.0f}px, {ud} {abs(dy):.0f}px\n"
                                      f"→ 트레이를 {'왼쪽' if dx > 0 else '오른쪽'}·{'뒤(로봇 반대쪽)' if dy > 0 else '앞(로봇 쪽)'}으로 옮기거나 로봇 위치 조정")
        self.l_shift.setText(f"dx {dx:+.1f}px  dy {dy:+.1f}px  (신뢰도 {resp:.2f})\n{hint}")
        self.l_shift.setStyleSheet(f"color:{'#7ee787' if ok else '#d29922'};font-family:monospace;")


class ClassifierCapture(QGroupBox):
    """손잡이 분류기 학습 데이터 수집 패널.

    고정된 스캔 헤드 자세에서 head camera 한 장을 찍고, 슬롯별 ROI로 잘라
    crops/<CLASS>/ 에 저장 (ImageFolder 구조). 원본과 배치 라벨은 raw/, labels.jsonl 에 기록.
    배치(arrangement)는 train/test 분할 시 그룹 단위로 쓰도록 arrangement_id로 남긴다.
    """
    N_SLOTS = 7

    def __init__(self, config_path, cam, get_head_b64, get_head, set_head, status):
        super().__init__("분류기 데이터 수집")
        self.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        self._config_path = config_path
        self._cam = cam
        self._get_head_b64 = get_head_b64
        self._get_head = get_head
        self._set_head = set_head
        self._status = status
        self._load_config()

        fl = QFormLayout(self); fl.setSpacing(4); fl.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.f_dir = QLineEdit(os.path.expanduser("~/xlerobot_classifier_data"))
        fl.addRow("저장 폴더:", self.f_dir)

        # 스캔 헤드 자세
        self.l_scan = QLabel(); self.l_scan.setStyleSheet("color:#f0883e;font-family:monospace;")
        fl.addRow("스캔 자세:", self.l_scan)
        hb = QHBoxLayout()
        b_save_pose = QPushButton("자세 저장"); b_save_pose.clicked.connect(self._save_scan_pose)
        b_goto_pose = QPushButton("스캔 자세로"); b_goto_pose.clicked.connect(self._goto_scan_pose)
        hb.addWidget(b_save_pose); hb.addWidget(b_goto_pose)
        fl.addRow(hb)

        # 슬롯 영역 (ROI)
        self.cb_show = QCheckBox("슬롯 영역 표시"); self.cb_show.setChecked(True)
        fl.addRow(self.cb_show)
        self.cb_edit = QCheckBox("영역 편집")
        self.cb_edit.setToolTip("체크 후 헤드 화면에서 선택한 슬롯의 손잡이 영역을 드래그")
        self.cb_edit.toggled.connect(self._toggle_edit)
        self.sp_slot = QSpinBox(); self.sp_slot.setRange(1, self.N_SLOTS)
        eh = QHBoxLayout(); eh.addWidget(self.cb_edit); eh.addWidget(QLabel("슬롯")); eh.addWidget(self.sp_slot)
        fl.addRow(eh)

        # 배치 입력
        self.combos = []
        options = TRAY_LAYOUT + [EMPTY_SLOT]
        for i in range(self.N_SLOTS):
            c = QComboBox(); c.addItems(options); c.setCurrentText(TRAY_LAYOUT[i])
            self.combos.append(c)
            fl.addRow(f"Slot {i+1}:", c)
        bb = QHBoxLayout()
        b_default = QPushButton("기본 배치"); b_default.clicked.connect(self._set_default)
        b_shuffle = QPushButton("무작위 섞기"); b_shuffle.clicked.connect(self._shuffle)
        bb.addWidget(b_default); bb.addWidget(b_shuffle)
        fl.addRow(bb)
        self.f_note = QLineEdit(); self.f_note.setPlaceholderText("예: 조명 어둡게, 도구 비틀림")
        fl.addRow("메모:", self.f_note)

        self.btn_capture = QPushButton("📷 촬영")
        self.btn_capture.setStyleSheet("QPushButton{background:#238636;color:white;padding:8px;border-radius:4px;font-weight:bold;}")
        self.btn_capture.clicked.connect(self._capture)
        fl.addRow(self.btn_capture)
        self.l_count = QLabel(); self.l_count.setWordWrap(True)
        self.l_count.setStyleSheet("color:#7ee787;font-family:monospace;")
        fl.addRow(self.l_count)
        self.f_dir.editingFinished.connect(self._refresh_counts)
        self._update_scan_label()
        self._refresh_counts()

    # ── config (scan pose + ROIs) ──
    def _load_config(self):
        self.scan_pan, self.scan_tilt = -4.0, 70.0
        w = 80  # default: 7 boxes spread across the 640×480 head image
        self.rois = [[20 + i * 88, 280, 20 + i * 88 + w, 420] for i in range(self.N_SLOTS)]
        try:
            with open(self._config_path) as f:
                cfg = json.load(f)
            self.scan_pan, self.scan_tilt = cfg["scan_pan"], cfg["scan_tilt"]
            if len(cfg["rois"]) == self.N_SLOTS:
                self.rois = cfg["rois"]
        except (OSError, ValueError, KeyError):
            pass

    def _save_config(self):
        with open(self._config_path, "w") as f:
            json.dump({"scan_pan": self.scan_pan, "scan_tilt": self.scan_tilt, "rois": self.rois}, f, indent=2)

    def _update_scan_label(self):
        self.l_scan.setText(f"pan={self.scan_pan:.1f}, tilt={self.scan_tilt:.1f}")

    def _save_scan_pose(self):
        self.scan_pan, self.scan_tilt = self._get_head()
        self._save_config(); self._update_scan_label()
        self._status(f"스캔 자세 저장: pan={self.scan_pan:.1f}, tilt={self.scan_tilt:.1f}")

    def _goto_scan_pose(self):
        self._set_head(self.scan_pan, self.scan_tilt)
        self._status("스캔 자세로 이동")

    def _at_scan_pose(self):
        pan, tilt = self._get_head()
        return abs(pan - self.scan_pan) < 0.5 and abs(tilt - self.scan_tilt) < 0.5

    # ── ROI editing / overlay ──
    def _toggle_edit(self, on):
        self._cam.on_roi_drawn = self._on_roi_drawn if on else None

    def _on_roi_drawn(self, x1, y1, x2, y2):
        i = self.sp_slot.value() - 1
        self.rois[i] = [x1, y1, x2, y2]
        self._save_config()
        self._status(f"Slot {i+1} 영역 저장: ({x1},{y1})–({x2},{y2})")
        if i + 1 < self.N_SLOTS:
            self.sp_slot.setValue(i + 2)  # 다음 슬롯으로 자동 이동

    def overlay_boxes(self):
        if not self.cb_show.isChecked():
            return None
        sel = self.sp_slot.value() - 1 if self.cb_edit.isChecked() else -1
        return [{"x1": r[0], "y1": r[1], "x2": r[2], "y2": r[3],
                 "label": f"{i+1}:{self.combos[i].currentText()[:4]}",
                 "color": (255, 200, 0) if i == sel else (0, 255, 0)}
                for i, r in enumerate(self.rois)]

    # ── arrangement ──
    def _set_default(self):
        for c, t in zip(self.combos, TRAY_LAYOUT):
            c.setCurrentText(t)

    def _shuffle(self):
        import random
        order = TRAY_LAYOUT[:]
        while order == TRAY_LAYOUT:
            random.shuffle(order)
        for c, t in zip(self.combos, order):
            c.setCurrentText(t)
        self._status("무작위 배치 제안 — 트레이를 이 순서대로 놓고 촬영하세요")

    def arrangement(self):
        return [c.currentText() for c in self.combos]

    # ── capture ──
    def _capture(self):
        arr = self.arrangement()
        tools = [t for t in arr if t != EMPTY_SLOT]
        if len(tools) != len(set(tools)):
            self._status("같은 도구가 두 슬롯에 지정됨 — 배치를 확인하세요"); return
        if not self._at_scan_pose():
            self._status("헤드가 스캔 자세가 아님 — '스캔 자세로 이동' 후 촬영하세요"); return
        img = _decode_b64_image(self._get_head_b64())  # RGB, same as dataset/inference pipeline
        if img is None:
            self._status("헤드 카메라 이미지 없음 — 로봇 연결을 확인하세요"); return
        root = self.f_dir.text()
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2.imwrite expects BGR → files look right in viewers
        try:
            os.makedirs(os.path.join(root, "raw"), exist_ok=True)
            raw_rel = f"raw/{ts}.jpg"
            cv2.imwrite(os.path.join(root, raw_rel), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = bgr.shape[:2]
            crops = []
            for i, (label, (x1, y1, x2, y2)) in enumerate(zip(arr, self.rois)):
                x1, x2 = max(0, x1), min(w, x2); y1, y2 = max(0, y1), min(h, y2)
                d = os.path.join(root, "crops", label); os.makedirs(d, exist_ok=True)
                rel = f"crops/{label}/{ts}_s{i+1}.png"
                cv2.imwrite(os.path.join(root, rel), bgr[y1:y2, x1:x2])
                crops.append(rel)
            pan, tilt = self._get_head()
            rec = {"time": ts, "image": raw_rel, "crops": crops, "arrangement": arr,
                   "arrangement_id": ",".join(arr), "rois": self.rois,
                   "head_pan": pan, "head_tilt": tilt, "note": self.f_note.text()}
            with open(os.path.join(root, "labels.jsonl"), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            self._status(f"저장 실패: {e}"); return
        self._refresh_counts()
        self._status(f"촬영 저장: {raw_rel} (crop {len(crops)}장)")

    def _refresh_counts(self):
        path = os.path.join(self.f_dir.text(), "labels.jsonl")
        n_img, arrs, per = 0, set(), {}
        try:
            with open(path) as f:
                for line in f:
                    r = json.loads(line); n_img += 1; arrs.add(r["arrangement_id"])
                    for t in r["arrangement"]:
                        per[t] = per.get(t, 0) + 1
        except (OSError, ValueError):
            pass
        cls = "  ".join(f"{t[:4]}:{per.get(t, 0)}" for t in TRAY_LAYOUT + [EMPTY_SLOT])
        self.l_count.setText(f"사진 {n_img}장 / 배치 {len(arrs)}종\n{cls}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XLeRobot Control Panel")
        self.resize(1600, 900)
        self.signals = Signals()
        self.signals.obs_received.connect(self._on_obs)
        self.signals.status_changed.connect(lambda m: self.statusBar().showMessage(m))
        self.signals.policy_loaded.connect(self._on_policy_loaded)
        self.signals.voice_done.connect(self._on_voice_done)
        self.signals.cls_loaded.connect(lambda ok, m: self.l_vr_cls.setText(m))
        self.ctrl = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        self.head_pan, self.head_tilt = -4.0, 70.0
        self.pressed = set()
        self.HEAD_STEP = 2.0
        self.SPD = 0.2
        self.THETA = 30.0
        self._yaw = 0.0
        self._last_imu_t = None
        self._gx_bias = 0.0
        self._calibrating = False
        self._calib_samples = []
        self._arm_joints = {'left': [0.0]*6, 'right': [0.0]*6}
        self._viz_counter = 0
        self.whisper_model = None
        self._whisper_name = None
        self._whisper_lock = threading.Lock()
        self._voice_recording = False
        self._voice_chunks = []
        self._voice_stream = None
        self.auto_rotate_active = False
        self.auto_rotate_target = 0.0
        self._demo_token = 0
        self._handoff_pose_path = os.path.expanduser("~/xlerobot-teleop/desktop/handoff_pose.json")
        self._latest_head_b64 = ""
        self._build_ui()
        t = QTimer(self); t.timeout.connect(self._tick); t.start(50)
        self.setFocusPolicy(Qt.StrongFocus)

    def _tab_page(self):
        """New scrollable tab-page (widget, layout) pair, styled to match the sidebar."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none; background:#161b22;}")
        page = QWidget(); page.setStyleSheet("background:#161b22;")
        layout = QVBoxLayout(page); layout.setContentsMargins(8,8,8,8); layout.setSpacing(8)
        scroll.setWidget(page)
        return scroll, layout

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QHBoxLayout(root); rl.setSpacing(6); rl.setContentsMargins(6,6,6,6)
        tab_style = ("QTabWidget::pane{border:none;background:#161b22;}"
                     "QTabBar::tab{background:#21262d;color:#8b949e;padding:6px 12px;}"
                     "QTabBar::tab:selected{background:#161b22;color:#58a6ff;font-weight:bold;}")

        # ── 왼쪽: 탭 (조종 / AI 모드) ───────────────
        tabs_l = QTabWidget(); tabs_l.setFixedWidth(280); tabs_l.setStyleSheet(tab_style)

        ctrl_scroll, sl = self._tab_page()

        # 연결 설정
        cg = QGroupBox("연결 설정"); cg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        fl = QFormLayout(cg); fl.setSpacing(4)
        self.f_ip    = QLineEdit("192.168.0.34")
        self.f_cmd   = QLineEdit("5555")
        self.f_obs   = QLineEdit("5556")
        self.f_lport = QLineEdit("/dev/ttyACM0")
        self.f_rport = QLineEdit("/dev/ttyACM1")
        self.f_tid   = QLineEdit("xlerobot_leader")
        for lbl, w in [("Jetson IP",self.f_ip),("CMD 포트",self.f_cmd),
                        ("OBS 포트",self.f_obs),("왼쪽 리더암",self.f_lport),
                        ("오른쪽 리더암",self.f_rport),("Teleop ID",self.f_tid)]:
            fl.addRow(lbl+":", w)
        self.btn = QPushButton("연결")
        self.btn.setStyleSheet("QPushButton{background:#238636;color:white;padding:8px;border-radius:4px;font-weight:bold;}QPushButton:hover{background:#2ea043;}")
        self.btn.clicked.connect(self._toggle)
        fl.addRow(self.btn)
        sl.addWidget(cg)

        # 조종 키
        kg = QGroupBox("조종 키"); kg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        kl = QVBoxLayout(kg)
        kl.addWidget(QLabel("↑↓←→  헤드 상하좌우\ni / k    전진 / 후진\nj / l    좌 / 우\nu / o    좌회전 / 우회전\nn / m    속도 +/-\n\n팔: 리더암으로 직접 제어"))
        kg.layout().itemAt(0).widget().setStyleSheet("color:#c9d1d9;font-family:monospace;font-size:12px;")
        sl.addWidget(kg)

        # 헤드 상태
        hg = QGroupBox("헤드 상태"); hg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        hfl = QFormLayout(hg)
        self.l_pan  = QLabel("0.0"); self.l_pan.setStyleSheet("color:#f0883e;")
        self.l_tilt = QLabel("0.0"); self.l_tilt.setStyleSheet("color:#f0883e;")
        self.l_spd  = QLabel("0.2"); self.l_spd.setStyleSheet("color:#f0883e;")
        hfl.addRow("Pan:", self.l_pan); hfl.addRow("Tilt:", self.l_tilt); hfl.addRow("속도:", self.l_spd)
        sl.addWidget(hg)

        # 팔 상태
        ag = QGroupBox("팔 상태"); ag.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        afl = QFormLayout(ag); afl.setSpacing(3)
        self.arm_lbls = {}
        for short, key in [
            ("L 숄더팬","left_arm_shoulder_pan"),("L 숄더리프트","left_arm_shoulder_lift"),
            ("L 엘보","left_arm_elbow_flex"),("L 손목F","left_arm_wrist_flex"),
            ("L 손목R","left_arm_wrist_roll"),("L 그리퍼","left_arm_gripper"),
            ("R 숄더팬","right_arm_shoulder_pan"),("R 숄더리프트","right_arm_shoulder_lift"),
            ("R 엘보","right_arm_elbow_flex"),("R 손목F","right_arm_wrist_flex"),
            ("R 손목R","right_arm_wrist_roll"),("R 그리퍼","right_arm_gripper"),
        ]:
            l = QLabel("—"); l.setStyleSheet("color:#7ee787;font-family:monospace;font-size:11px;")
            afl.addRow(short+":", l); self.arm_lbls[key] = l
        sl.addWidget(ag)

        # 정렬 가이드 (학습 데이터 head 화면 겹치기) — cam_head는 아래에서 만들어지므로 나중에 연결
        self._align_layout = sl
        sl.addStretch()
        tabs_l.addTab(ctrl_scroll, "조종")

        ai_scroll, aifl_outer = self._tab_page()
        sl = aifl_outer  # subsequent widgets below go into the AI 모드 tab

        # AI 추론
        aig = QGroupBox("AI 추론 (실험적)"); aig.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        aifl = QVBoxLayout(aig); aifl.setSpacing(4)
        default_ckpt = "/home/taekyeung/lerobot/outputs/train/smolvla_7tool/pretrained_model"
        self.f_policy_path = QLineEdit(default_ckpt)
        self.f_ai_task = QLineEdit("Pick up the grasper")
        aifl.addWidget(QLabel("체크포인트 경로:")); aifl.addWidget(self.f_policy_path)
        aifl.addWidget(QLabel("Task 설명:")); aifl.addWidget(self.f_ai_task)
        self.btn_load_policy = QPushButton("정책 로드")
        self.btn_load_policy.setStyleSheet("QPushButton{background:#8957e5;color:white;padding:6px;border-radius:4px;}")
        self.btn_load_policy.clicked.connect(self._load_policy)
        aifl.addWidget(self.btn_load_policy)
        self.btn_ai_mode = QPushButton("AI 모드 시작")
        self.btn_ai_mode.setEnabled(False)
        self.btn_ai_mode.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
        self.btn_ai_mode.clicked.connect(self._toggle_ai_mode)
        aifl.addWidget(self.btn_ai_mode)
        self.cb_ai_30hz = QCheckBox("30Hz (학습 속도)")
        self.cb_ai_30hz.setToolTip("action chunk를 학습 데이터와 같은 30 fps로 재생 (끄면 기존 60 Hz 루프)")
        self.cb_ai_smooth = QCheckBox("부드러운 연결")
        self.cb_ai_smooth.setToolTip("다음 동작 묶음을 미리 계산해 멈칫거림 없이 이어 붙임 (끄면 기존 방식)")
        for cb in (self.cb_ai_30hz, self.cb_ai_smooth):
            cb.toggled.connect(self._apply_ai_opts)
            aifl.addWidget(cb)
        self.l_ai_status = QLabel("정책 로드 안 됨")
        self.l_ai_status.setStyleSheet("color:#8b949e;font-size:11px;")
        self.l_ai_status.setWordWrap(True)
        aifl.addWidget(self.l_ai_status)
        wh_row = QHBoxLayout()
        wh_row.addWidget(QLabel("Whisper:"))
        self.cb_whisper = QComboBox(); self.cb_whisper.addItems(WHISPER_MODELS)
        self.cb_whisper.setToolTip("음성 인식 모델 (CPU, 3초 음성 기준 medium 2.2s / turbo 3.5s / small 0.8s)")
        self.cb_whisper.currentTextChanged.connect(self._on_whisper_model_changed)
        wh_row.addWidget(self.cb_whisper, 1)
        aifl.addLayout(wh_row)
        self.btn_voice = QPushButton("🎤 음성 명령 듣기")
        self.btn_voice.setStyleSheet("QPushButton{background:#9e6a03;color:white;padding:6px;border-radius:4px;}")
        self.btn_voice.clicked.connect(self._toggle_voice_record)
        aifl.addWidget(self.btn_voice)
        self.l_voice_status = QLabel("")
        self.l_voice_status.setStyleSheet("color:#8b949e;font-size:11px;")
        self.l_voice_status.setWordWrap(True)
        aifl.addWidget(self.l_voice_status)
        sl.addWidget(aig)

        # 데모 시퀀스
        dg = QGroupBox("데모 시퀀스"); dg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        dfl = QVBoxLayout(dg); dfl.setSpacing(4)
        # Verify & Recover: 요청 도구 → 트레이 스캔 → 실제 슬롯 → instruction remapping
        vr_form = QFormLayout(); vr_form.setSpacing(4); vr_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.cb_vr_method = QComboBox()
        self.cb_vr_method.addItems(["Proposed (V&R)", "Baseline", "Task 직접 입력"])
        self.cb_vr_method.setToolTip("Proposed: 트레이 스캔 → 실제 슬롯 → instruction remapping\nBaseline: 요청 도구 이름 그대로\nTask 직접 입력: AI Task 칸의 문장 사용")
        vr_form.addRow("방법:", self.cb_vr_method)
        self.cb_vr_tool = QComboBox(); self.cb_vr_tool.addItems(TRAY_LAYOUT)
        vr_form.addRow("요청 도구:", self.cb_vr_tool)
        dfl.addLayout(vr_form)
        self.btn_vr_scan = QPushButton("🔍 트레이 스캔")
        self.btn_vr_scan.setStyleSheet("QPushButton{background:#6e7681;color:white;padding:6px;border-radius:4px;}")
        self.btn_vr_scan.clicked.connect(lambda: self._vr_scan_then(None))
        dfl.addWidget(self.btn_vr_scan)
        self.l_vr_cls = QLabel("분류기: 로드 중...")
        self.l_vr_cls.setStyleSheet("color:#8b949e;font-size:11px;")
        dfl.addWidget(self.l_vr_cls)
        self.l_vr_result = QLabel("")
        self.l_vr_result.setWordWrap(True)
        self.l_vr_result.setStyleSheet("color:#8b949e;font-family:monospace;font-size:11px;")
        dfl.addWidget(self.l_vr_result)
        rot_row = QHBoxLayout()
        self.f_rotate_target = QLineEdit("-90")
        self.f_rotate_target.setFixedWidth(50)
        rot_row.addWidget(QLabel("목표 각도:")); rot_row.addWidget(self.f_rotate_target)
        dfl.addLayout(rot_row)
        self.btn_auto_rotate = QPushButton("자동 회전 시작")
        self.btn_auto_rotate.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}")
        self.btn_auto_rotate.clicked.connect(self._start_auto_rotate)
        dfl.addWidget(self.btn_auto_rotate)
        self.btn_save_handoff = QPushButton("핸드오프 자세 저장")
        self.btn_save_handoff.setStyleSheet("QPushButton{background:#6e7681;color:white;padding:6px;border-radius:4px;}")
        self.btn_save_handoff.clicked.connect(self._save_handoff_pose)
        dfl.addWidget(self.btn_save_handoff)
        self.btn_run_handoff = QPushButton("핸드오프 동작 실행")
        self.btn_run_handoff.setStyleSheet("QPushButton{background:#6e7681;color:white;padding:6px;border-radius:4px;}")
        self.btn_run_handoff.clicked.connect(lambda: self._run_handoff())
        dfl.addWidget(self.btn_run_handoff)
        self.cb_hand_guided = QCheckBox("손 위치 기반 핸드오버")
        self.cb_hand_guided.setToolTip("헤드 화면 9칸에서 손을 찾아 그 칸의 자세로 전달 (자세는 핸드오버 탭에서 설정)")
        self.cb_hand_guided.setChecked(True)
        dfl.addWidget(self.cb_hand_guided)
        self.cb_decide_only = QCheckBox("판별만 (집지 않음, 실험 04)")
        self.cb_decide_only.setToolTip("트레이 스캔 + 판단만 하고 로그에 기록. 로봇은 움직이지 않음")
        dfl.addWidget(self.cb_decide_only)
        self.f_vr_note = QLineEdit(); self.f_vr_note.setPlaceholderText("조건 메모 (예: A, 정상, 없음-grasper)")
        self.f_vr_note.setToolTip("로그에 함께 저장되는 실험 조건 표시")
        dfl.addWidget(self.f_vr_note)
        self.cb_pick_only = QCheckBox("집기까지만 (실험 04)")
        self.cb_pick_only.setToolTip("집기 + 자세 고정에서 멈춤 (회전·전달 생략). 🛑 긴급 정지로 해제")
        dfl.addWidget(self.cb_pick_only)
        self.btn_demo = QPushButton("▶ 전체 데모 시작")
        self.btn_demo.setStyleSheet("QPushButton{background:#da3633;color:white;padding:8px;border-radius:4px;font-weight:bold;}")
        self.btn_demo.clicked.connect(lambda: self._request_and_run(self.cb_vr_tool.currentText()))
        dfl.addWidget(self.btn_demo)
        self.btn_estop = QPushButton("🛑 긴급 정지")
        self.btn_estop.setStyleSheet("QPushButton{background:#67060c;color:white;padding:10px;border-radius:4px;font-weight:bold;font-size:13px;}QPushButton:hover{background:#8b1118;}")
        self.btn_estop.clicked.connect(self._emergency_stop)
        dfl.addWidget(self.btn_estop)
        self.l_demo_status = QLabel("")
        self.l_demo_status.setStyleSheet("color:#8b949e;font-size:11px;")
        self.l_demo_status.setWordWrap(True)
        dfl.addWidget(self.l_demo_status)
        sl.addWidget(dg)
        sl.addStretch()
        tabs_l.addTab(ai_scroll, "AI 모드")
        rl.addWidget(tabs_l)

        # ── 가운데 카메라 ──────────────────────────
        cp = QWidget(); cl = QVBoxLayout(cp); cl.setSpacing(6)
        self.cam_head  = CamLabel("헤드 카메라")
        cl.addWidget(self.cam_head, stretch=3)
        wr = QHBoxLayout()
        self.cam_left  = CamLabel("왼쪽 손목 카메라")
        self.cam_right = CamLabel("오른쪽 손목 카메라")
        wr.addWidget(self.cam_left); wr.addWidget(self.cam_right)
        cl.addLayout(wr, stretch=1)
        rl.addWidget(cp, stretch=1)

        # ── 오른쪽: 탭 (데이터 / 상태) ──────────────
        tabs_r = QTabWidget(); tabs_r.setFixedWidth(300); tabs_r.setStyleSheet(tab_style)

        data_scroll, rl2 = self._tab_page()

        # 데이터 녹화 (Pick / Place 데이터셋을 따로 녹화, 한 번에 하나만 녹화)
        status = lambda m: self.statusBar().showMessage(m)
        self.rec_pick = DatasetRecorder(
            "데이터 녹화 — Pick", "bigse0u1/xlerobot_scrub_7tool", "Pick up the specimen bag",
            status, lambda: self.rec_place.recording)
        self.rec_place = DatasetRecorder(
            "데이터 녹화 — Place (반납)", "bigse0u1/xlerobot_scrub_7tool_place",
            "Place the specimen bag back in the tray",
            status, lambda: self.rec_pick.recording)
        self.recorders = [self.rec_pick, self.rec_place]
        rl2.addWidget(self.rec_pick)
        rl2.addWidget(self.rec_place)
        rl2.addStretch()
        tabs_r.addTab(data_scroll, "데이터")

        cls_scroll, rl2 = self._tab_page()

        # 분류기 데이터 수집 (손잡이 ROI crop)
        self.cls_capture = ClassifierCapture(
            os.path.expanduser("~/xlerobot-teleop/desktop/classifier_config.json"),
            self.cam_head,
            lambda: self._latest_head_b64,
            lambda: (self.head_pan, self.head_tilt),
            self._set_head_pose,
            status)
        rl2.addWidget(self.cls_capture)

        self.align_guide = AlignGuide(
            os.path.expanduser("~/xlerobot-teleop/desktop/align_reference_head.png"),
            self.cam_head,
            lambda: self.rec_pick.f_repo.text(),
            lambda: (self.head_pan, self.head_tilt),
            status)
        self._align_layout.insertWidget(self._align_layout.count() - 1, self.align_guide)  # stretch 앞에

        self.handle_cls = HandleClassifier(os.path.expanduser(
            "~/xlerobot_classifier_data/models/handle_classifier_resnet18_color.pt"))
        self.handle_cls.load_async(lambda ok, m: self.signals.cls_loaded.emit(ok, m))
        rl2.addStretch()
        tabs_r.addTab(cls_scroll, "분류기")

        ho_scroll, rl2 = self._tab_page()
        self.hand_detector = HandDetector(os.path.expanduser("~/xlerobot-teleop/desktop/hand_landmarker.task"))
        self.handover_panel = HandoverPanel(
            os.path.expanduser("~/xlerobot-teleop/desktop/handover_poses.json"),
            lambda: self.ctrl._latest_obs if self.ctrl else None,
            lambda: (self.head_pan, self.head_tilt),
            self._set_head_pose,
            self._move_to_pose_test,
            status)
        rl2.addWidget(self.handover_panel)
        rl2.addStretch()
        tabs_r.addTab(ho_scroll, "핸드오버")
        self._ho_tab = ho_scroll
        self._handover_detecting = False
        self._handover_target = None
        self._hand_frame_n = 0
        self._cls_tab = cls_scroll
        self._tabs_r = tabs_r
        # 탭을 벗어나면 영역 편집(드래그)도 끔 — 다른 작업 중 헤드 화면 클릭으로 영역이 바뀌지 않도록
        tabs_r.currentChanged.connect(
            lambda _: tabs_r.currentWidget() is not cls_scroll and self.cls_capture.cb_edit.setChecked(False))

        status_scroll, rl2 = self._tab_page()

        # IMU
        ig = QGroupBox("IMU"); ig.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        il = QVBoxLayout(ig); il.setSpacing(4)
        lbl_g = QLabel("── 자이로 (deg/s) ──"); lbl_g.setStyleSheet("color:#8b949e;font-size:11px;")
        il.addWidget(lbl_g)
        self.imu_gx = ImuBar("GX", "#ff7b72")
        self.imu_gy = ImuBar("GY", "#79c0ff")
        self.imu_gz = ImuBar("GZ", "#56d364")
        il.addWidget(self.imu_gx); il.addWidget(self.imu_gy); il.addWidget(self.imu_gz)
        lbl_a = QLabel("── 가속도 (g) ──"); lbl_a.setStyleSheet("color:#8b949e;font-size:11px;")
        il.addWidget(lbl_a)
        self.imu_ax = ImuBar("AX", "#ff7b72", -2, 2)
        self.imu_ay = ImuBar("AY", "#79c0ff", -2, 2)
        self.imu_az = ImuBar("AZ", "#56d364", -2, 2)
        il.addWidget(self.imu_ax); il.addWidget(self.imu_ay); il.addWidget(self.imu_az)

        yaw_row = QHBoxLayout()
        self.imu_yaw = QLabel("Yaw: 0.0°")
        self.imu_yaw.setStyleSheet("color:#f0883e;font-family:monospace;font-size:13px;font-weight:bold;")
        yaw_reset = QPushButton("초기화")
        yaw_reset.setFixedWidth(60)
        yaw_reset.setStyleSheet("QPushButton{background:#6e7681;color:white;padding:3px;border-radius:3px;font-size:11px;}QPushButton:hover{background:#8b949e;}")
        yaw_reset.clicked.connect(self._reset_yaw)
        yaw_row.addWidget(self.imu_yaw); yaw_row.addWidget(yaw_reset)
        il.addLayout(yaw_row)
        rl2.addWidget(ig)

        # 모터 온도
        tg = QGroupBox("모터 온도"); tg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        tl = QVBoxLayout(tg)
        self.temp_panel = TempPanel()
        tl.addWidget(self.temp_panel)
        rl2.addWidget(tg)

        # 3D 로봇 포즈
        vg = QGroupBox("로봇 포즈 (3D)"); vg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        vl = QVBoxLayout(vg)
        if MPL_OK:
            self.robot_viz = RobotVizWidget()
            self.robot_viz.setMinimumHeight(260)
            vl.addWidget(self.robot_viz)
        else:
            vl.addWidget(QLabel("matplotlib 미설치\npip install matplotlib").setAlignment(Qt.AlignCenter) or QLabel("matplotlib 필요"))
            self.robot_viz = None
        rl2.addWidget(vg)
        rl2.addStretch()
        tabs_r.addTab(status_scroll, "상태")
        rl.addWidget(tabs_r)

        self.statusBar().showMessage("연결 안 됨")

    def _reset_yaw(self):
        self._yaw = 0.0
        self._last_imu_t = None
        self._calibrating = True
        self._calib_samples = []
        self.imu_yaw.setText("Yaw: 캘리브레이션 중... (가만히 있으세요)")

    def _toggle_voice_record(self):
        if not WHISPER_OK:
            self.l_voice_status.setText("faster-whisper/sounddevice 설치 필요")
            return
        if not self._voice_recording:
            self._voice_chunks = []
            self._voice_recording = True
            self.btn_voice.setText("■ 녹음 중지 (다시 클릭)")
            self.l_voice_status.setText("듣는 중...")
            self._ensure_whisper_async()  # 말하는 동안 모델 로드 (처음엔 다운로드)

            def _cb(indata, frames, time_info, status):
                self._voice_chunks.append(indata.copy())

            try:
                self._voice_stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=_cb)
                self._voice_stream.start()
            except Exception as e:
                self._voice_recording = False
                self.l_voice_status.setText(f"마이크 열기 실패: {e}")
                self.btn_voice.setText("🎤 음성 명령 듣기")
        else:
            self._voice_recording = False
            self.btn_voice.setText("🎤 음성 명령 듣기")
            self.l_voice_status.setText("인식 중...")
            try:
                self._voice_stream.stop(); self._voice_stream.close()
            except Exception:
                pass
            audio = (np.concatenate(self._voice_chunks, axis=0).flatten()
                     if self._voice_chunks else np.zeros(0, dtype=np.float32))
            threading.Thread(target=self._do_transcribe, args=(audio,), daemon=True).start()

    def _get_whisper(self):
        """선택된 Whisper 모델 (필요하면 로드, 스레드 안전)."""
        with self._whisper_lock:
            name = self.cb_whisper.currentText()
            if self.whisper_model is None or self._whisper_name != name:
                self.whisper_model = WhisperModel(name, device="cpu", compute_type="int8")
                self._whisper_name = name
            return self.whisper_model

    def _ensure_whisper_async(self):
        if self.whisper_model is None or self._whisper_name != self.cb_whisper.currentText():
            threading.Thread(target=self._get_whisper, daemon=True).start()

    def _on_whisper_model_changed(self, name):
        self.whisper_model = None  # 다음 녹음 때 새 모델 로드
        self.l_voice_status.setText(f"Whisper 모델: {name} (다음 녹음 때 로드)")

    def _do_transcribe(self, audio):
        try:
            if audio.size < 16000 * 0.3:
                self.signals.voice_done.emit(False, "너무 짧습니다", "")
                return
            model = self._get_whisper()
            t0 = time.time()
            segments, info = model.transcribe(audio, language=None, **{**WHISPER_OPTS, "initial_prompt": WHISPER_PROMPT})
            text = "".join(seg.text for seg in segments).strip()
            self._voice_elapsed = time.time() - t0
            canonical = _to_canonical_tool(text)
            self.signals.voice_done.emit(canonical is not None, text, canonical or "")
        except Exception as e:
            self.signals.voice_done.emit(False, f"인식 실패: {e}", "")

    def _on_voice_done(self, ok, text, canonical):
        if ok:
            self.l_voice_status.setText(f'"{text}" → {canonical} → 데모 시퀀스 시작 '
                                        f'({self._whisper_name}, {getattr(self, "_voice_elapsed", 0):.1f}s)')
            if self.cb_vr_method.currentIndex() == 2:  # Task 직접 입력 모드에서 음성 요청 → Verify & Recover 사용
                self.cb_vr_method.setCurrentIndex(0)
            method = "Proposed" if self.cb_vr_method.currentIndex() == 0 else "Baseline (스캔 없음)"
            self.l_voice_status.setText(self.l_voice_status.text() + f" [{method}]")
            self._request_and_run(canonical)
        else:
            self.l_voice_status.setText(f'매칭 실패: "{text}" ({self._whisper_name})')

    def _start_auto_rotate(self):
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요")
            return
        try:
            self.auto_rotate_target = float(self.f_rotate_target.text())
        except ValueError:
            self.statusBar().showMessage("목표 각도가 숫자가 아님")
            return
        self._demo_token += 1
        token = self._demo_token
        # 회전 중 팔이 흔들리지 않도록 지금 자세를 그대로 고정한 채 회전한다.
        obs = self.ctrl._latest_obs or {}
        lock_pose = _obs_to_locked_action(obs)
        self.ctrl.scripted_action = lock_pose
        self.auto_rotate_active = True
        self.l_demo_status.setText(f"자동 회전 중... 목표 {self.auto_rotate_target}° (현재 자세 고정)")

        def _wait_done():
            if token != self._demo_token:
                return
            if self.auto_rotate_active:
                QTimer.singleShot(200, _wait_done)
                return
            if self.ctrl:
                self.ctrl.scripted_action = None
            self.l_demo_status.setText(f"자동 회전 완료 (yaw={self._yaw:.1f}°)")

        QTimer.singleShot(200, _wait_done)

    def _save_handoff_pose(self):
        obs = self.ctrl._latest_obs if self.ctrl else None
        if not obs:
            self.statusBar().showMessage("관측 데이터 없음 (연결 확인)")
            return
        pose = _obs_to_locked_action(obs)
        try:
            with open(self._handoff_pose_path, "w") as f:
                json.dump(pose, f, indent=2)
            self.statusBar().showMessage(f"핸드오프 자세 저장됨: {self._handoff_pose_path}")
        except Exception as e:
            self.statusBar().showMessage(f"저장 실패: {e}")

    def _get_arm_pos_vec(self):
        obs = self.ctrl._latest_obs if self.ctrl else None
        if not obs:
            return None
        return [obs.get(k, 0.0) or 0.0 for k in STATE_KEYS if k.startswith(("left_arm", "right_arm"))]

    def _wait_for_settle(self, token, get_vec_fn, on_settled,
                          stable_checks=4, interval_ms=300, max_wait_s=20.0, eps=1.0, min_wait_s=3.0):
        """Polls get_vec_fn() until consecutive readings stop changing (or times out).

        min_wait_s guards against mistaking a brief mid-sequence pause (e.g. the
        policy pausing before closing the gripper) for the pick being finished.
        """
        state = {"prev": None, "stable": 0, "elapsed": 0.0}

        def _check():
            if token != self._demo_token:
                return  # sequence was cancelled/superseded
            state["elapsed"] += interval_ms / 1000.0
            cur = get_vec_fn()
            if cur is not None and state["prev"] is not None and len(cur) == len(state["prev"]):
                delta = max(abs(a - b) for a, b in zip(cur, state["prev"]))
                state["stable"] = state["stable"] + 1 if delta < eps else 0
            state["prev"] = cur
            past_min_wait = state["elapsed"] >= min_wait_s
            if (past_min_wait and state["stable"] >= stable_checks) or state["elapsed"] >= max_wait_s:
                on_settled()
                return
            QTimer.singleShot(interval_ms, _check)

        QTimer.singleShot(interval_ms, _check)

    def _move_scripted(self, target, duration_s, token, on_done, step_ms=33):
        """현재 scripted 자세(없으면 현재 관절 상태)에서 target까지 duration_s 동안 부드럽게 보간 이동."""
        obs = self.ctrl._latest_obs or {}
        start = dict(self.ctrl.scripted_action) if self.ctrl.scripted_action else _obs_to_locked_action(obs)
        keys = [k for k in target if k in start]
        n = max(1, int(duration_s * 1000 / step_ms))
        state = {"i": 0}

        def _step():
            if token is not None and token != self._demo_token:
                return
            state["i"] += 1
            u = min(1.0, state["i"] / n)
            u = u * u * (3 - 2 * u)  # smoothstep: 출발·도착을 천천히
            pose = dict(target)
            for k in keys:
                pose[k] = start[k] + (target[k] - start[k]) * u
            self.ctrl.scripted_action = pose
            if state["i"] < n:
                QTimer.singleShot(step_ms, _step)
            elif on_done:
                on_done()

        _step()

    def _run_handoff(self, move_duration_s=4.0, hold_s=1.0, release_s=1.0, return_s=4.0,
                     token=None, on_done=None, return_pose=None, pose=None, arm=None):
        """핸드오프 자세로 천천히 이동 → hold_s 후 그리퍼 열기 → release_s 대기 → return_pose로 천천히 복귀.
        pose가 없으면 고정 핸드오프 자세 파일을 사용. arm이 주어지면 그 팔의 그리퍼만 연다."""
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요")
            return
        if pose is None:
            try:
                with open(self._handoff_pose_path) as f:
                    pose = json.load(f)
            except Exception as e:
                self.statusBar().showMessage(f"핸드오프 자세 로드 실패: {e} (먼저 '핸드오프 자세 저장' 필요)")
                return
        if return_pose is None:  # 단독 실행: 시작했던 자세로 돌아감
            return_pose = _obs_to_locked_action(self.ctrl._latest_obs or {})
        if token is None:
            self._demo_token += 1
            token = self._demo_token
        self.l_demo_status.setText("핸드오프: 이동 중...")

        def _cancelled():
            return token != self._demo_token

        def _arrived():
            if _cancelled():
                return
            self.l_demo_status.setText("핸드오프: 도착, 대기 중...")
            QTimer.singleShot(int(hold_s * 1000), _release)

        def _release():
            if _cancelled():
                return
            release_pose = dict(pose)
            gkeys = [f"{arm}_arm_gripper.pos"] if arm else ["left_arm_gripper.pos", "right_arm_gripper.pos"]
            for gkey in gkeys:
                if gkey in release_pose:
                    release_pose[gkey] = 100.0 if release_pose[gkey] < 50.0 else 0.0
            self.ctrl.scripted_action = release_pose
            self.l_demo_status.setText("핸드오프: 놓는 중...")
            QTimer.singleShot(int(release_s * 1000), _go_back)

        def _go_back():
            if _cancelled():
                return
            self.l_demo_status.setText("핸드오프: 원래 자리로 복귀 중...")
            self._move_scripted(return_pose, return_s, token, _finish)

        def _finish():
            if _cancelled():
                return
            if self.ctrl:
                self.ctrl.scripted_action = None
            self.l_demo_status.setText("핸드오프 완료")
            if on_done:
                on_done()

        self._move_scripted(pose, move_duration_s, token, _arrived)

    # ── Verify & Recover ──────────────────────────────
    VR_CONF_THR = 0.8
    VR_LOG = os.path.expanduser("~/xlerobot_experiment_log.jsonl")

    def _vr_scan_then(self, on_preds, settle_s=0.8):
        """헤드를 스캔 자세로 보내고 잠시 기다린 뒤 head 이미지 1장을 분류. on_preds(preds) 콜백."""
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요"); return
        if self.handle_cls.model is None:
            self.statusBar().showMessage("분류기가 아직 로드되지 않음"); return
        cc = self.cls_capture
        moved = abs(self.head_pan - cc.scan_pan) > 0.5 or abs(self.head_tilt - cc.scan_tilt) > 0.5
        self._set_head_pose(cc.scan_pan, cc.scan_tilt)
        token = self._demo_token
        self.l_vr_result.setText("트레이 스캔 중...")

        def _do():
            if token != self._demo_token:
                return
            img = _decode_b64_image(self._latest_head_b64)
            if img is None:
                self.l_vr_result.setText("헤드 카메라 이미지 없음"); return
            preds = self.handle_cls.classify(img, cc.rois)
            self._vr_last = {"preds": preds, "time": time.time(), "target": None}
            self.l_vr_result.setText("트레이: " + "  ".join(
                f"{i+1}:{l[:4]}{'' if c >= self.VR_CONF_THR else '?'}" for i, (l, c) in enumerate(preds)))
            if on_preds:
                on_preds(preds)

        QTimer.singleShot(int((settle_s + (1.0 if moved else 0.0)) * 1000), _do)

    def _vr_overlay_boxes(self):
        """마지막 스캔 결과를 8초간 head 화면에 표시 (목표 슬롯은 파랑)."""
        last = getattr(self, "_vr_last", None)
        if not last or time.time() - last["time"] > 8:
            return None
        boxes = []
        for i, ((l, c), r) in enumerate(zip(last["preds"], self.cls_capture.rois)):
            if last["target"] == i + 1:
                color = (60, 140, 255)
            elif c < self.VR_CONF_THR:
                color = (255, 60, 60)
            elif l != TRAY_LAYOUT[i]:
                color = (255, 170, 0)   # 원래 자리가 아닌 도구
            else:
                color = (0, 255, 0)
            boxes.append({"x1": r[0], "y1": r[1], "x2": r[2], "y2": r[3], "label": f"{l[:4]} {c:.2f}", "color": color})
        return boxes

    def _vr_log(self, rec):
        try:
            with open(self.VR_LOG, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _request_and_run(self, requested):
        """요청 도구 → (방법에 따라) 스캔·판단 → instruction 설정 → 데모 시퀀스."""
        mode = self.cb_vr_method.currentIndex()
        if mode == 2:  # Task 직접 입력: 기존 동작
            self._start_demo_sequence(); return
        self.cb_vr_tool.setCurrentText(requested)
        method = "proposed" if mode == 0 else "baseline"
        decide_only = self.cb_decide_only.isChecked()
        base = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "method": method, "requested": requested,
                "decide_only": decide_only, "note": self.f_vr_note.text().strip()}

        def _decide(preds):
            d = verify_and_recover(requested, preds, method, self.VR_CONF_THR)
            self._vr_log({**base, "preds": preds, **d})
            if d["instruction"] is None:
                if decide_only:
                    self._vr_count = getattr(self, "_vr_count", 0) + 1
                    self.statusBar().showMessage(f"판별 기록됨 ({self._vr_count}회째): {requested} → {d['status']}")
                self.l_vr_result.setText(self.l_vr_result.text() + f"\n⚠️ 중단: {d['reason']}")
                self.l_demo_status.setText(f"⚠️ {d['reason']} — 의사에게 알림")
                self.statusBar().showMessage(f"Verify & Recover 중단: {d['reason']}")
                return
            if preds is not None:
                self._vr_last["target"] = d["slot"]
                self.l_vr_result.setText(self.l_vr_result.text() +
                                         f"\n{d['status']}: {d['reason']}\n→ \"{d['instruction']}\" ({d['arm']})")
            else:
                self.l_vr_result.setText(f"baseline → \"{d['instruction']}\"")
            if decide_only:
                self._vr_count = getattr(self, "_vr_count", 0) + 1
                self.statusBar().showMessage(f"판별 기록됨 ({self._vr_count}회째): {requested} → {d['status']} slot {d['slot']}")
                return
            self.f_ai_task.setText(d["instruction"])
            self._start_demo_sequence()

        if method == "baseline":
            _decide(None)
        else:
            self._vr_scan_then(_decide)

    HAND_STABLE_N = 5        # 같은 칸에서 연속 검출 횟수 (≈0.5 s)
    HAND_TIMEOUT_S = 5.0     # 이 시간 안에 손이 확정되지 않으면 기본 칸(1,1) 자세로 전달

    def _hand_guided_handoff(self, token, home_pose, arm, on_done):
        """시선 자세로 헤드 이동 → 손 검출 → 손이 있는 칸 확정 → 도구를 든 팔의 그 칸 자세로 핸드오버."""
        hp = self.handover_panel
        if arm is None:
            arm = "left"
        self._set_head_pose(*hp.cfg["head"])
        st = {"cell": None, "n": 0, "t0": time.time()}
        self._handover_detecting = True

        def _finish_all():
            self._handover_target = None
            cc = self.cls_capture
            self._set_head_pose(cc.scan_pan, cc.scan_tilt)  # 다음 집기를 위해 헤드를 원래 자세로
            on_done()

        def _handover_to(cell):
            self._handover_detecting = False
            key, pose = hp.pose_for(arm, cell)
            if pose is None:
                self.l_demo_status.setText(f"⚠️ {arm} 팔 핸드오버 자세가 없음 — 고정 핸드오프 자세 사용")
            else:
                self._handover_target = tuple(map(int, key.split(",")))
            where = f"손 위치 {cell[0]},{cell[1]}" if cell else "손 미검출"
            self.l_demo_status.setText(f"4/4 핸드오버: {arm} 팔 → 칸 {key or '고정'} ({where})")
            self._run_handoff(token=token, return_pose=home_pose, pose=pose, arm=arm, on_done=_finish_all)

        def _poll():
            if token != self._demo_token:
                self._handover_detecting = False; return
            elapsed = time.time() - st["t0"]  # 실제 경과 시간 (타이머 간격은 부하에 따라 늘어남)
            hand = hp.last_hand if time.time() - hp.last_hand_t < 0.3 else None
            if hand and elapsed > 1.0:  # 헤드가 시선 자세로 움직이는 동안은 무시
                cell = grid_cell(hand["cx"], hand["cy"], 640, 480)
                st["n"] = st["n"] + 1 if cell == st["cell"] else 1
                st["cell"] = cell
                self.l_demo_status.setText(f"4/4 손 검출: 칸 {cell[0]},{cell[1]} ({st['n']}/{self.HAND_STABLE_N})")
                if st["n"] >= self.HAND_STABLE_N:
                    _handover_to(cell); return
            if elapsed > 1.0 + self.HAND_TIMEOUT_S:
                _handover_to(None); return
            QTimer.singleShot(100, _poll)

        _poll()

    def _start_demo_sequence(self, grip_lock_s=1.0):
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요")
            return
        if not self.policy:
            self.statusBar().showMessage("먼저 정책을 로드하세요")
            return
        self._demo_token += 1
        token = self._demo_token
        home_pose = _obs_to_locked_action(self.ctrl._latest_obs or {})  # 핸드오프 후 돌아올 자세
        arm = arm_from_instruction(self.f_ai_task.text())  # 도구를 드는 팔 (remap된 instruction이면 실제 slot 기준)
        self.l_demo_status.setText("1/4 집는 중... (움직임이 멈출 때까지 대기)")
        if not self.ctrl.ai_mode:
            self._toggle_ai_mode()

        def _on_pick_settled():
            if token != self._demo_token:
                return
            self.ctrl.ai_mode = False
            self.btn_ai_mode.setText("AI 모드 시작")
            self.btn_ai_mode.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
            self.l_demo_status.setText("2/4 자세 고정 중...")
            # 현재(잡은 직후) 팔 자세를 그대로 고정해서 회전 중 도구를 놓치지 않게 한다.
            obs = self.ctrl._latest_obs or {}
            lock_pose = _obs_to_locked_action(obs)
            self.ctrl.scripted_action = lock_pose
            if self.cb_pick_only.isChecked():
                # 실험 04: 집은 자세를 고정한 채로 멈춤 (결과 확인용). 🛑 긴급 정지로 고정 해제.
                self.l_demo_status.setText("집기 완료 — 자세 고정 중. 결과 기록 후 🛑 긴급 정지로 해제")
                return
            QTimer.singleShot(int(grip_lock_s * 1000), _start_rotate)

        def _start_rotate():
            if token != self._demo_token:
                return
            self.l_demo_status.setText("3/4 회전 중...")
            self.auto_rotate_target = float(self.f_rotate_target.text() or -90)
            self.auto_rotate_active = True
            QTimer.singleShot(200, _wait_rotate)

        def _wait_rotate():
            if token != self._demo_token:
                return
            if self.auto_rotate_active:
                QTimer.singleShot(200, _wait_rotate)
                return
            done = lambda: self.l_demo_status.setText("데모 완료 ✓")
            if self.cb_hand_guided.isChecked():
                self.l_demo_status.setText("4/4 손 찾는 중...")
                self._hand_guided_handoff(token, home_pose, arm, done)
            else:
                self.l_demo_status.setText("4/4 핸드오프 중...")
                self._run_handoff(token=token, return_pose=home_pose, arm=arm, on_done=done)

        self._wait_for_settle(token, self._get_arm_pos_vec, _on_pick_settled)

    def _emergency_stop(self):
        self._demo_token += 1  # invalidate any pending demo-sequence callbacks
        self.auto_rotate_active = False
        self._handover_detecting = False
        self._handover_target = None
        if self.ctrl:
            self.ctrl.ai_mode = False
            self.ctrl.scripted_action = None
            self.ctrl.set_wheels(0.0, 0.0, 0.0)
        self.btn_ai_mode.setText("AI 모드 시작")
        self.btn_ai_mode.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
        self.l_demo_status.setText("🛑 긴급 정지됨")
        self.statusBar().showMessage("긴급 정지: AI/회전/스크립트 동작 모두 중단")

    def _load_policy(self):
        if not LEROBOT_OK:
            self.l_ai_status.setText("lerobot 정책 모듈을 불러올 수 없음")
            return
        path = self.f_policy_path.text()
        self.btn_load_policy.setEnabled(False)
        self.l_ai_status.setText("정책 로드 중...")

        def _do_load():
            try:
                policy_cfg = PreTrainedConfig.from_pretrained(path)
                policy_cls = get_policy_class(policy_cfg.type)
                policy = policy_cls.from_pretrained(path)
                device = get_safe_torch_device(policy_cfg.device if policy_cfg.device else "cuda")
                policy.to(device)
                policy.eval()
                preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=path)
                self.policy, self.preprocessor, self.postprocessor, self.device = (
                    policy, preprocessor, postprocessor, device
                )
                self.signals.policy_loaded.emit(True, f"정책 로드 완료 ({policy_cfg.type}, {device})")
            except Exception as e:
                self.signals.policy_loaded.emit(False, f"정책 로드 실패: {e}")

        threading.Thread(target=_do_load, daemon=True).start()

    def _on_policy_loaded(self, ok, msg):
        self.l_ai_status.setText(msg)
        self.btn_load_policy.setEnabled(True)
        self.btn_ai_mode.setEnabled(ok)

    def _move_to_pose_test(self, pose):
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요"); return
        self._demo_token += 1
        self._move_scripted(pose, 4.0, self._demo_token, None)

    def _set_head_pose(self, pan, tilt):
        # _tick이 다음 주기에 이 값을 ControlThread로 전송
        self.head_pan = max(-100, min(100, pan))
        self.head_tilt = max(-100, min(100, tilt))

    def _apply_ai_opts(self, *_):
        if self.ctrl:
            self.ctrl.ai_30hz = self.cb_ai_30hz.isChecked()
            self.ctrl.ai_smooth = self.cb_ai_smooth.isChecked()

    def _toggle_ai_mode(self):
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요")
            return
        if not self.ctrl.ai_mode:
            # SmolVLA/ACT keep a queue of up to n_action_steps (50) predicted actions;
            # clear it so a restart doesn't replay the previous run's leftover chunk
            # (stale motions from the old pose / old task) before re-inferring.
            self.policy.reset()
            self.ctrl.reset_ai_state()
            self._apply_ai_opts()
            self.ctrl.policy = self.policy
            self.ctrl.preprocessor = self.preprocessor
            self.ctrl.postprocessor = self.postprocessor
            self.ctrl.device = self.device
            self.ctrl.ai_task = self.f_ai_task.text()
            self.ctrl.ai_mode = True
            self.btn_ai_mode.setText("AI 모드 중지")
            self.btn_ai_mode.setStyleSheet("QPushButton{background:#da3633;color:white;padding:6px;border-radius:4px;}")
            self.statusBar().showMessage("AI 모드 시작 — 리더암 입력 무시됨")
        else:
            self.ctrl.ai_mode = False
            self.btn_ai_mode.setText("AI 모드 시작")
            self.btn_ai_mode.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
            self.statusBar().showMessage("AI 모드 중지, 리더암 제어로 복귀")

    def _capture_frame(self, obs, task):
        head_img = _decode_b64_image(obs.get("head", ""))
        left_img = _decode_b64_image(obs.get("left_wrist", ""))
        right_img = _decode_b64_image(obs.get("right_wrist", ""))
        if head_img is None or left_img is None or right_img is None:
            return None
        state = np.array([obs.get(k, 0.0) or 0.0 for k in STATE_KEYS], dtype=np.float32)
        action_src = self.ctrl.last_action if self.ctrl else {}
        action = np.array([action_src.get(k, 0.0) or 0.0 for k in ACTION_KEYS], dtype=np.float32)
        return {
            "observation.images.head": head_img,
            "observation.images.left_wrist": left_img,
            "observation.images.right_wrist": right_img,
            "observation.state": state,
            "action": action,
            "task": task,
        }

    def _toggle(self):
        if self.ctrl and self.ctrl.running:
            self.ctrl.stop()
            self.btn.setText("연결")
            self.btn.setStyleSheet("QPushButton{background:#238636;color:white;padding:8px;border-radius:4px;font-weight:bold;}")
            self.btn_ai_mode.setText("AI 모드 시작")
            self.btn_ai_mode.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
            self.statusBar().showMessage("연결 해제")
        else:
            cfg = dict(ip=self.f_ip.text(), cmd_port=self.f_cmd.text(),
                       obs_port=self.f_obs.text(), left_port=self.f_lport.text(),
                       right_port=self.f_rport.text(), teleop_id=self.f_tid.text())
            self.ctrl = ControlThread(self.signals, cfg)
            # 첫 명령부터 GUI의 헤드 기본값을 보내도록 미리 설정 (안 하면 _tick이 돌기 전
            # 몇 프레임 동안 (0, 0)이 전송되어 고개가 들렸다가 다시 내려감)
            self.ctrl.set_head(self.head_pan, self.head_tilt)
            self.ctrl.start()
            self.btn.setText("연결 끊기")
            self.btn.setStyleSheet("QPushButton{background:#da3633;color:white;padding:8px;border-radius:4px;font-weight:bold;}")
            self.statusBar().showMessage("연결 중...")

    def _on_obs(self, obs):
        self._latest_head_b64 = obs.get("head", "")
        cls_tab_open = self._tabs_r.currentWidget() is self._cls_tab
        ho_preview = self._tabs_r.currentWidget() is self._ho_tab and self.handover_panel.cb_preview.isChecked()
        if ho_preview or self._handover_detecting:
            self._hand_frame_n += 1
            if self._hand_frame_n % 3 == 0:  # ~10 Hz, 약 20 ms/회 (GUI 스레드)
                img = _decode_b64_image(self._latest_head_b64)
                if img is not None:
                    self.handover_panel.last_hand = self.hand_detector.detect(img)
                    self.handover_panel.last_hand_t = time.time()
        if cls_tab_open:
            boxes = self.cls_capture.overlay_boxes()
        elif ho_preview or self._handover_detecting or self._handover_target:
            boxes = self.handover_panel.overlay_boxes(target=self._handover_target)
        else:
            boxes = self._vr_overlay_boxes()
        self.cam_head.show_frame(self._latest_head_b64, boxes)
        self.align_guide.on_frame(self._latest_head_b64)
        self.cam_left.show_frame(obs.get("left_wrist", ""))
        self.cam_right.show_frame(obs.get("right_wrist", ""))

        # 팔 상태
        for key, lbl in self.arm_lbls.items():
            v = obs.get(f"{key}.pos")
            if v is not None: lbl.setText(f"{v:.1f}")

        # 3D 포즈 업데이트 (매 5프레임마다)
        self._viz_counter += 1
        if self._viz_counter % 5 == 0 and self.robot_viz:
            lj = [obs.get(f"{k}.pos", 0.0) or 0.0 for k in ARM_KEYS_L]
            rj = [obs.get(f"{k}.pos", 0.0) or 0.0 for k in ARM_KEYS_R]
            self.robot_viz.update_pose(lj, rj)

        # 데이터 녹화
        for rec in self.recorders:
            if rec.recording:
                rec.add_frame(self._capture_frame(obs, rec.task()))

        # 모터 온도
        self.temp_panel.update_temps(obs)

        # IMU
        gx = obs.get("imu_gx", 0.0); gy = obs.get("imu_gy", 0.0); gz = obs.get("imu_gz", 0.0)
        ax = obs.get("imu_ax", 0.0); ay = obs.get("imu_ay", 0.0); az = obs.get("imu_az", 0.0)
        self.imu_gx.update_val(gx); self.imu_gy.update_val(gy); self.imu_gz.update_val(gz)
        self.imu_ax.update_val(ax); self.imu_ay.update_val(ay); self.imu_az.update_val(az)
        now = time.time()
        if self._calibrating:
            # 1초(약 30샘플) 동안 정지 상태 측정 → 바이어스 계산
            self._calib_samples.append(gx)
            if len(self._calib_samples) >= 30:
                self._gx_bias = sum(self._calib_samples) / len(self._calib_samples)
                self._calibrating = False
                self._last_imu_t = now
                self.imu_yaw.setText(f"Yaw: 0.0° (bias={self._gx_bias:.2f}°/s 보정됨)")
        elif self._last_imu_t is not None:
            gx_corrected = gx - self._gx_bias
            if abs(gx_corrected) > 1.0:  # 바이어스 보정 후 1°/s 데드밴드
                self._yaw += gx_corrected * (now - self._last_imu_t)
            self.imu_yaw.setText(f"Yaw: {self._yaw:.1f}°")
        self._last_imu_t = now

    def keyPressEvent(self, e):
        if not e.isAutoRepeat(): self.pressed.add(e.key())

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat(): self.pressed.discard(e.key())

    def _tick(self):
        pk = self.pressed
        if Qt.Key_Up    in pk: self.head_tilt = max(-100, self.head_tilt - self.HEAD_STEP)
        if Qt.Key_Down  in pk: self.head_tilt = min( 100, self.head_tilt + self.HEAD_STEP)
        if Qt.Key_Left  in pk: self.head_pan  = max(-100, self.head_pan  - self.HEAD_STEP)
        if Qt.Key_Right in pk: self.head_pan  = min( 100, self.head_pan  + self.HEAD_STEP)
        if Qt.Key_N in pk: self.SPD = min(0.5, self.SPD + 0.01)
        if Qt.Key_M in pk: self.SPD = max(0.05, self.SPD - 0.01)
        self.l_pan.setText(f"{self.head_pan:.1f}")
        self.l_tilt.setText(f"{self.head_tilt:.1f}")
        self.l_spd.setText(f"{self.SPD:.2f}")
        x = y = t = 0.0
        if self.auto_rotate_active:
            err = self.auto_rotate_target - self._yaw
            if abs(err) < 2.0:
                self.auto_rotate_active = False
                self.l_demo_status.setText(f"자동 회전 완료 (yaw={self._yaw:.1f}°)")
            else:
                t = self.THETA if err > 0 else -self.THETA
        else:
            if Qt.Key_I in pk: x =  self.SPD
            if Qt.Key_K in pk: x = -self.SPD
            if Qt.Key_J in pk: y =  self.SPD
            if Qt.Key_L in pk: y = -self.SPD
            if Qt.Key_U in pk: t =  self.THETA
            if Qt.Key_O in pk: t = -self.THETA
        if self.ctrl and self.ctrl.running:
            self.ctrl.set_head(self.head_pan, self.head_tilt)
            self.ctrl.set_wheels(x, y, t)

    def closeEvent(self, e):
        if self.ctrl: self.ctrl.stop()
        for rec in self.recorders:
            rec.discard_on_close()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window,     QColor(22,27,34))
    p.setColor(QPalette.WindowText, QColor(201,209,217))
    p.setColor(QPalette.Base,       QColor(13,17,23))
    p.setColor(QPalette.Text,       QColor(201,209,217))
    p.setColor(QPalette.Button,     QColor(33,38,45))
    p.setColor(QPalette.ButtonText, QColor(201,209,217))
    app.setPalette(p)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
