#!/usr/bin/env python3
import sys, json, time, base64, threading
import numpy as np
import cv2
import zmq
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QPushButton, QLineEdit, QFormLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

try:
    from lerobot.teleoperators.bi_so_leader import BiSOLeader
    from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    LEROBOT_OK = True
except ImportError:
    LEROBOT_OK = False


class Signals(QObject):
    obs_received   = pyqtSignal(dict)
    status_changed = pyqtSignal(str)


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

    def set_head(self, pan, tilt):
        with self._lock:
            self.head_pan, self.head_tilt = pan, tilt

    def set_wheels(self, x, y, theta):
        with self._lock:
            self.wx, self.wy, self.wt = x, y, theta

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
                    action = self.teleop.get_action()
                    with self._lock:
                        action.update({
                            "head_motor_1.pos": self.head_pan,
                            "head_motor_2.pos": self.head_tilt,
                            "x.vel": self.wx, "y.vel": self.wy, "theta.vel": self.wt,
                        })
                    self.cmd_sock.send_string(json.dumps(action))
                    try:
                        msg = self.obs_sock.recv_string(flags=zmq.NOBLOCK)
                        self.signals.obs_received.emit(json.loads(msg))
                    except zmq.Again:
                        pass
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


class CamLabel(QLabel):
    def __init__(self, title):
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0d1117; color:#58a6ff; border:1px solid #30363d; font-size:13px;")
        self.setMinimumSize(200, 150)

    def show_frame(self, b64):
        if not b64: return
        try:
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is None: return
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w, c = f.shape
            qi = QImage(f.data, w, h, w*c, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qi).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except: pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XLeRobot Control Panel")
        self.resize(1400, 900)
        self.signals = Signals()
        self.signals.obs_received.connect(self._on_obs)
        self.signals.status_changed.connect(lambda m: self.statusBar().showMessage(m))
        self.ctrl = None
        self.head_pan = self.head_tilt = 0.0
        self.pressed = set()
        self.HEAD_STEP = 2.0
        self.SPD = 0.2
        self.THETA = 30.0
        self._build_ui()
        t = QTimer(self); t.timeout.connect(self._tick); t.start(50)
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QHBoxLayout(root); rl.setSpacing(8); rl.setContentsMargins(8,8,8,8)

        # ── sidebar (scrollable) ─────────────────
        scroll = QScrollArea()
        scroll.setFixedWidth(300)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none; background:#161b22;}")
        sb = QWidget()
        sb.setStyleSheet("background:#161b22;")
        sl = QVBoxLayout(sb); sl.setContentsMargins(10,10,10,10); sl.setSpacing(8)

        # connection
        cg = QGroupBox("연결 설정")
        cg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        fl = QFormLayout(cg); fl.setSpacing(5)
        self.f_ip    = QLineEdit("192.168.0.31")
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

        # controls guide
        kg = QGroupBox("조종 키")
        kg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        kl = QVBoxLayout(kg)
        kl.addWidget(QLabel(
            "↑↓←→  헤드 상하좌우\n"
            "i / k    전진 / 후진\n"
            "j / l    좌 / 우\n"
            "u / o    좌회전 / 우회전\n"
            "n / m    속도 +/-\n\n"
            "팔: 리더암으로 직접 제어"
        ))
        kg.layout().itemAt(0).widget().setStyleSheet("color:#c9d1d9;font-family:monospace;font-size:12px;")
        sl.addWidget(kg)

        # head state
        hg = QGroupBox("헤드 상태")
        hg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        hfl = QFormLayout(hg)
        self.l_pan  = QLabel("0.0"); self.l_pan.setStyleSheet("color:#f0883e;")
        self.l_tilt = QLabel("0.0"); self.l_tilt.setStyleSheet("color:#f0883e;")
        self.l_spd  = QLabel("0.2"); self.l_spd.setStyleSheet("color:#f0883e;")
        hfl.addRow("Pan:",  self.l_pan)
        hfl.addRow("Tilt:", self.l_tilt)
        hfl.addRow("속도:", self.l_spd)
        sl.addWidget(hg)

        # arm state
        ag = QGroupBox("팔 상태")
        ag.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
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
        scroll.setWidget(sb)
        rl.addWidget(scroll)

        # ── cameras ──────────────────────────────
        cp = QWidget(); cl = QVBoxLayout(cp); cl.setSpacing(6)
        self.cam_head  = CamLabel("헤드 카메라")
        cl.addWidget(self.cam_head, stretch=3)
        wr = QHBoxLayout()
        self.cam_left  = CamLabel("왼쪽 손목 카메라")
        self.cam_right = CamLabel("오른쪽 손목 카메라")
        wr.addWidget(self.cam_left); wr.addWidget(self.cam_right)
        cl.addLayout(wr, stretch=1)
        rl.addWidget(cp, stretch=1)

        self.statusBar().showMessage("연결 안 됨")

    def _toggle(self):
        if self.ctrl and self.ctrl.running:
            self.ctrl.stop()
            self.btn.setText("연결")
            self.btn.setStyleSheet("QPushButton{background:#238636;color:white;padding:8px;border-radius:4px;font-weight:bold;}")
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
        self.cam_head.show_frame(obs.get("head",""))
        self.cam_left.show_frame(obs.get("left_wrist",""))
        self.cam_right.show_frame(obs.get("right_wrist",""))
        for key, lbl in self.arm_lbls.items():
            v = obs.get(f"{key}.pos")
            if v is not None: lbl.setText(f"{v:.1f}")

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
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window,        QColor(22,27,34))
    p.setColor(QPalette.WindowText,    QColor(201,209,217))
    p.setColor(QPalette.Base,          QColor(13,17,23))
    p.setColor(QPalette.Text,          QColor(201,209,217))
    p.setColor(QPalette.Button,        QColor(33,38,45))
    p.setColor(QPalette.ButtonText,    QColor(201,209,217))
    app.setPalette(p)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
