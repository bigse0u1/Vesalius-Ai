#!/usr/bin/env python3
import sys, os, json, time, base64, threading
import numpy as np
import cv2
import zmq
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QPushButton, QLineEdit, QFormLayout, QScrollArea,
    QProgressBar, QTabWidget
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
    "GRASPER": ["grasper", "그라스퍼", "그래스퍼"],
    "BIPOLAR": ["bipolar", "바이폴라"],
    "HOOK": ["hook", "훅"],
    "CLIPPER": ["clipper", "클리퍼"],
    "SCISSORS": ["scissors", "scissor", "시저", "가위"],
    "IRRIGATOR": ["irrigator", "이리게이터", "석션"],
    "SPECIMEN_BAG": ["specimen bag", "스페시먼백", "스페시먼 백", "백"],
}


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
    upload_done    = pyqtSignal(bool, str)
    policy_loaded  = pyqtSignal(bool, str)
    voice_done     = pyqtSignal(bool, str, str)


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

    def set_head(self, pan, tilt):
        with self._lock:
            self.head_pan, self.head_tilt = pan, tilt

    def set_wheels(self, x, y, theta):
        with self._lock:
            self.wx, self.wy, self.wt = x, y, theta

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
                        action = self._infer_action()
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
                    time.sleep(1/60)
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

    def show_frame(self, b64, boxes=None):
        if not b64: return
        try:
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is None: return
            if boxes:
                for b in boxes:
                    cv2.rectangle(f, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
                    cv2.putText(f, f"{b['label']} {b['conf']:.2f}",
                                (b["x1"], max(b["y1"]-6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            h, w, c = f.shape
            qi = QImage(f.data, w, h, w*c, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qi).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except: pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XLeRobot Control Panel")
        self.resize(1600, 900)
        self.signals = Signals()
        self.signals.obs_received.connect(self._on_obs)
        self.signals.status_changed.connect(lambda m: self.statusBar().showMessage(m))
        self.signals.upload_done.connect(self._on_upload_done)
        self.signals.policy_loaded.connect(self._on_policy_loaded)
        self.signals.voice_done.connect(self._on_voice_done)
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
        self.dataset = None
        self.recording = False
        self.episode_count = 0
        self.whisper_model = None
        self._voice_recording = False
        self._voice_chunks = []
        self._voice_stream = None
        self.auto_rotate_active = False
        self.auto_rotate_target = 0.0
        self._demo_token = 0
        self._handoff_pose_path = os.path.expanduser("~/xlerobot-teleop/desktop/handoff_pose.json")
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
        sl.addStretch()
        tabs_l.addTab(ctrl_scroll, "조종")

        ai_scroll, aifl_outer = self._tab_page()
        sl = aifl_outer  # subsequent widgets below go into the AI 모드 tab

        # AI 추론
        aig = QGroupBox("AI 추론 (실험적)"); aig.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        aifl = QVBoxLayout(aig); aifl.setSpacing(4)
        default_ckpt = "/home/taekyeung/lerobot/outputs/train/smolvla_2tool/checkpoints/last/pretrained_model"
        self.f_policy_path = QLineEdit(default_ckpt)
        self.f_ai_task = QLineEdit("Pick up the block")
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
        self.l_ai_status = QLabel("정책 로드 안 됨")
        self.l_ai_status.setStyleSheet("color:#8b949e;font-size:11px;")
        self.l_ai_status.setWordWrap(True)
        aifl.addWidget(self.l_ai_status)
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
        self.btn_demo = QPushButton("▶ 전체 데모 시작")
        self.btn_demo.setStyleSheet("QPushButton{background:#da3633;color:white;padding:8px;border-radius:4px;font-weight:bold;}")
        self.btn_demo.clicked.connect(lambda: self._start_demo_sequence())
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

        # 데이터 녹화
        rg = QGroupBox("데이터 녹화"); rg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        rfl = QVBoxLayout(rg); rfl.setSpacing(4)
        self.f_repo = QLineEdit("bigse0u1/xlerobot_scrub_7tool")
        self.f_task = QLineEdit("Pick up the grasper")
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
        rl2.addWidget(rg)
        rl2.addStretch()
        tabs_r.addTab(data_scroll, "데이터")

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

    def _do_transcribe(self, audio):
        try:
            if audio.size < 16000 * 0.3:
                self.signals.voice_done.emit(False, "너무 짧습니다", "")
                return
            if self.whisper_model is None:
                self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, info = self.whisper_model.transcribe(audio, language=None, beam_size=5)
            text = "".join(seg.text for seg in segments).strip()
            canonical = _to_canonical_tool(text)
            self.signals.voice_done.emit(canonical is not None, text, canonical or "")
        except Exception as e:
            self.signals.voice_done.emit(False, f"인식 실패: {e}", "")

    def _on_voice_done(self, ok, text, canonical):
        if ok:
            task = f"Pick up the {canonical.lower().replace('_', ' ')}"
            self.f_ai_task.setText(task)
            self.l_voice_status.setText(f'"{text}" → {canonical} → 데모 시퀀스 시작')
            self._start_demo_sequence()
        else:
            self.l_voice_status.setText(f'매칭 실패: "{text}"')

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

    def _run_handoff(self, move_duration_s=2.5, hold_s=3.0, release_s=1.0, token=None, on_done=None):
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요")
            return
        try:
            with open(self._handoff_pose_path) as f:
                pose = json.load(f)
        except Exception as e:
            self.statusBar().showMessage(f"핸드오프 자세 로드 실패: {e} (먼저 '핸드오프 자세 저장' 필요)")
            return
        self.ctrl.scripted_action = dict(pose)
        self.l_demo_status.setText("핸드오프: 이동 중...")

        def _cancelled():
            return token is not None and token != self._demo_token

        def _arrived():
            if _cancelled():
                return
            self.l_demo_status.setText("핸드오프: 도착, 대기 중...")
            QTimer.singleShot(int(hold_s * 1000), _release)

        def _release():
            if _cancelled():
                return
            release_pose = dict(pose)
            for gkey in ("left_arm_gripper.pos", "right_arm_gripper.pos"):
                if gkey in release_pose:
                    release_pose[gkey] = 100.0 if release_pose[gkey] < 50.0 else 0.0
            self.ctrl.scripted_action = release_pose
            self.l_demo_status.setText("핸드오프: 놓는 중...")
            QTimer.singleShot(int(release_s * 1000), _finish)

        def _finish():
            if _cancelled():
                return
            if self.ctrl:
                self.ctrl.scripted_action = None
            self.l_demo_status.setText("핸드오프 완료")
            if on_done:
                on_done()

        QTimer.singleShot(int(move_duration_s * 1000), _arrived)

    def _start_demo_sequence(self, grip_lock_s=1.0):
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요")
            return
        if not self.policy:
            self.statusBar().showMessage("먼저 정책을 로드하세요")
            return
        self._demo_token += 1
        token = self._demo_token
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
            self.l_demo_status.setText("4/4 핸드오프 중...")
            self._run_handoff(token=token, on_done=lambda: self.l_demo_status.setText("데모 완료 ✓"))

        self._wait_for_settle(token, self._get_arm_pos_vec, _on_pick_settled)

    def _emergency_stop(self):
        self._demo_token += 1  # invalidate any pending demo-sequence callbacks
        self.auto_rotate_active = False
        if self.ctrl:
            self.ctrl.ai_mode = False
            self.ctrl.scripted_action = None
            self.ctrl.set_wheels(0.0, 0.0, 0.0)
        self.btn_ai_mode.setText("AI 모드 시작")
        self.btn_ai_mode.setStyleSheet("QPushButton{background:#1f6feb;color:white;padding:6px;border-radius:4px;}QPushButton:disabled{background:#30363d;color:#8b949e;}")
        self.l_demo_status.setText("🛑 긴급 정지됨")
        self.statusBar().showMessage("긴급 정지: AI/회전/스크립트 동작 모두 중단")

    def _toggle_dataset(self):
        if self.dataset is None:
            if not LEROBOT_OK:
                self.statusBar().showMessage("lerobot 데이터셋 모듈을 불러올 수 없음")
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
                self.statusBar().showMessage(msg)
            except Exception as e:
                self.statusBar().showMessage(f"데이터셋 열기 실패: {e}")
        else:
            if self.recording:
                self._toggle_episode()
            self.dataset = None
            self.btn_dataset.setText("데이터셋 생성")
            self.btn_episode.setEnabled(False)
            self.btn_discard.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.f_repo.setEnabled(True)
            self.statusBar().showMessage("데이터셋 세션 종료")

    def _toggle_episode(self):
        if self.dataset is None:
            return
        if not self.recording:
            self.recording = True
            self.btn_episode.setText("■ 에피소드 저장")
            self.btn_discard.setEnabled(True)
            self.statusBar().showMessage("녹화 중...")
        else:
            self.recording = False
            try:
                self.dataset.save_episode()
                self.episode_count += 1
                self.l_episode_count.setText(f"녹화된 에피소드: {self.episode_count}")
                self.statusBar().showMessage(f"에피소드 {self.episode_count} 저장됨")
            except Exception as e:
                self.statusBar().showMessage(f"저장 실패: {e}")
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
        self.statusBar().showMessage("에피소드 폐기됨")

    def _upload_dataset(self):
        if self.dataset is None or self.recording:
            return
        self.btn_upload.setEnabled(False)
        self.btn_episode.setEnabled(False)
        self.btn_discard.setEnabled(False)
        self.statusBar().showMessage("허깅페이스 업로드 중... (에피소드 수에 따라 수 분 소요)")
        dataset = self.dataset

        def _do_upload():
            try:
                dataset.finalize()
                dataset.push_to_hub()
                self.signals.upload_done.emit(True, "허깅페이스 업로드 완료 ✓")
            except Exception as e:
                self.signals.upload_done.emit(False, f"업로드 실패: {e}")

        threading.Thread(target=_do_upload, daemon=True).start()

    def _on_upload_done(self, ok, msg):
        self.statusBar().showMessage(msg)
        self.btn_upload.setEnabled(not ok)

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

    def _toggle_ai_mode(self):
        if not self.ctrl or not self.ctrl.running:
            self.statusBar().showMessage("먼저 로봇에 연결하세요")
            return
        if not self.ctrl.ai_mode:
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

    def _capture_frame(self, obs):
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
            "task": self.f_task.text(),
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
            self.ctrl.start()
            self.btn.setText("연결 끊기")
            self.btn.setStyleSheet("QPushButton{background:#da3633;color:white;padding:8px;border-radius:4px;font-weight:bold;}")
            self.statusBar().showMessage("연결 중...")

    def _on_obs(self, obs):
        self.cam_head.show_frame(obs.get("head", ""))
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
        if self.dataset is not None and self.recording:
            frame = self._capture_frame(obs)
            if frame is not None:
                try:
                    self.dataset.add_frame(frame)
                except Exception as e:
                    self.statusBar().showMessage(f"프레임 기록 실패: {e}")

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
        if self.dataset is not None:
            if self.recording:
                try: self.dataset.clear_episode_buffer()
                except Exception: pass
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
