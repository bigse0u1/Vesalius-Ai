#!/usr/bin/env python3
import sys, json, time, base64, threading
import numpy as np
import cv2
import zmq
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QPushButton, QLineEdit, QFormLayout, QScrollArea,
    QProgressBar
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
        self.resize(1600, 900)
        self.signals = Signals()
        self.signals.obs_received.connect(self._on_obs)
        self.signals.status_changed.connect(lambda m: self.statusBar().showMessage(m))
        self.ctrl = None
        self.head_pan = self.head_tilt = 0.0
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
        self._build_ui()
        t = QTimer(self); t.timeout.connect(self._tick); t.start(50)
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QHBoxLayout(root); rl.setSpacing(6); rl.setContentsMargins(6,6,6,6)

        # ── 왼쪽 사이드바 ─────────────────────────
        scroll_l = QScrollArea()
        scroll_l.setFixedWidth(270)
        scroll_l.setWidgetResizable(True)
        scroll_l.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_l.setStyleSheet("QScrollArea{border:none; background:#161b22;}")
        sb = QWidget(); sb.setStyleSheet("background:#161b22;")
        sl = QVBoxLayout(sb); sl.setContentsMargins(8,8,8,8); sl.setSpacing(8)

        # 연결 설정
        cg = QGroupBox("연결 설정"); cg.setStyleSheet("QGroupBox{color:#58a6ff;font-weight:bold;}")
        fl = QFormLayout(cg); fl.setSpacing(4)
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
        scroll_l.setWidget(sb)
        rl.addWidget(scroll_l)

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

        # ── 오른쪽 패널 ───────────────────────────
        scroll_r = QScrollArea()
        scroll_r.setFixedWidth(290)
        scroll_r.setWidgetResizable(True)
        scroll_r.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_r.setStyleSheet("QScrollArea{border:none; background:#161b22;}")
        rb = QWidget(); rb.setStyleSheet("background:#161b22;")
        rl2 = QVBoxLayout(rb); rl2.setContentsMargins(8,8,8,8); rl2.setSpacing(8)

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

        scroll_r.setWidget(rb)
        rl.addWidget(scroll_r)

        self.statusBar().showMessage("연결 안 됨")

    def _reset_yaw(self):
        self._yaw = 0.0
        self._last_imu_t = None
        self._calibrating = True
        self._calib_samples = []
        self.imu_yaw.setText("Yaw: 캘리브레이션 중... (가만히 있으세요)")

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
        self.cam_head.show_frame(obs.get("head", ""))
        self.cam_left.show_frame(obs.get("left_wrist", ""))
        self.cam_right.show_frame(obs.get("right_wrist", ""))

        # 팔 상태
        arm_keys_l = ["left_arm_shoulder_pan","left_arm_shoulder_lift","left_arm_elbow_flex",
                      "left_arm_wrist_flex","left_arm_wrist_roll","left_arm_gripper"]
        arm_keys_r = ["right_arm_shoulder_pan","right_arm_shoulder_lift","right_arm_elbow_flex",
                      "right_arm_wrist_flex","right_arm_wrist_roll","right_arm_gripper"]
        for key, lbl in self.arm_lbls.items():
            v = obs.get(f"{key}.pos")
            if v is not None: lbl.setText(f"{v:.1f}")

        # 3D 포즈 업데이트 (매 5프레임마다)
        self._viz_counter += 1
        if self._viz_counter % 5 == 0 and self.robot_viz:
            lj = [obs.get(f"{k}.pos", 0.0) or 0.0 for k in arm_keys_l]
            rj = [obs.get(f"{k}.pos", 0.0) or 0.0 for k in arm_keys_r]
            self.robot_viz.update_pose(lj, rj)

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
    p.setColor(QPalette.Window,     QColor(22,27,34))
    p.setColor(QPalette.WindowText, QColor(201,209,217))
    p.setColor(QPalette.Base,       QColor(13,17,23))
    p.setColor(QPalette.Text,       QColor(201,209,217))
    p.setColor(QPalette.Button,     QColor(33,38,45))
    p.setColor(QPalette.ButtonText, QColor(201,209,217))
    app.setPalette(p)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
