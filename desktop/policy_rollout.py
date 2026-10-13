"""Run the trained policy on the robot WITHOUT the GUI (diagnostic).

Same inputs/outputs as the GUI's AI mode (head+wrist images, 14-d state → 17-d action,
head fixed to the recording pose, base velocity 0), but nothing else running in the
process and an exact control rate. Compare with the GUI to tell GUI-side issues apart
from policy issues. Run with the GUI closed.

  python3 policy_rollout.py --task "Pick up the grasper"            # 30 Hz (dataset fps)
  python3 policy_rollout.py --task "Pick up the grasper" --hz 60    # like the GUI loop
  Ctrl+C stops; the arms then hold the last commanded pose.
"""
import argparse, base64, json, time
import numpy as np, cv2, zmq, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.common.control_utils import predict_action

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Pick up the grasper")
ap.add_argument("--policy", default="/home/taekyeung/lerobot/outputs/train/smolvla_7tool/pretrained_model")
ap.add_argument("--ip", default="192.168.0.34")
ap.add_argument("--hz", type=float, default=30.0)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--head", type=float, nargs=2, default=(-4.0, 70.0), metavar=("PAN", "TILT"))
args = ap.parse_args()

ARMS = [f"{s}_arm_{j}.pos" for s in ("left", "right")
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
STATE_KEYS = ARMS + ["head_motor_1.pos", "head_motor_2.pos"]
ACTION_KEYS = ARMS + ["head_motor_1.pos", "head_motor_2.pos", "x.vel", "y.vel", "theta.vel"]

def dec(b64):  # RGB-ordered after the Jetson encode / GUI decode round trip (same as the GUI)
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR) if b64 else None

print("loading policy ...")
cfg = PreTrainedConfig.from_pretrained(args.policy)
policy = get_policy_class(cfg.type).from_pretrained(args.policy)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device).eval(); policy.reset()
pre, post = make_pre_post_processors(cfg, pretrained_path=args.policy)

ctx = zmq.Context()
cmd = ctx.socket(zmq.PUSH); cmd.setsockopt(zmq.CONFLATE, 1); cmd.connect(f"tcp://{args.ip}:5555")
obs_s = ctx.socket(zmq.PULL); obs_s.setsockopt(zmq.CONFLATE, 1); obs_s.connect(f"tcp://{args.ip}:5556")
if not obs_s.poll(3000):
    raise SystemExit("no observation from host — is xlerobot_host running and the GUI closed?")

print(f"running '{args.task}' at {args.hz:.0f} Hz for {args.seconds:.0f}s (Ctrl+C to stop)")
latest, period, t_end = None, 1.0 / args.hz, time.time() + args.seconds
t_next, n, slow = time.perf_counter(), 0, 0
try:
    while time.time() < t_end:
        while obs_s.poll(0):
            latest = json.loads(obs_s.recv_string())
        if latest is None:
            continue
        imgs = [dec(latest.get(k, "")) for k in ("head", "left_wrist", "right_wrist")]
        if any(i is None for i in imgs):
            continue
        o = {"observation.images.head": imgs[0], "observation.images.left_wrist": imgs[1],
             "observation.images.right_wrist": imgs[2],
             "observation.state": np.array([latest.get(k, 0.0) or 0.0 for k in STATE_KEYS], np.float32)}
        a = predict_action(o, policy, device, pre, post, use_amp=False, task=args.task)
        a = a.detach().cpu().squeeze(0).numpy()
        msg = {k: float(v) for k, v in zip(ACTION_KEYS, a)}
        msg.update({"head_motor_1.pos": args.head[0], "head_motor_2.pos": args.head[1],
                    "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        cmd.send_string(json.dumps(msg)); n += 1
        t_next += period
        d = t_next - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            slow += 1; t_next = time.perf_counter()
except KeyboardInterrupt:
    print("\nstopped")
finally:
    print(f"sent {n} actions ({slow} late ticks) — arms hold the last pose")
    cmd.close(linger=0); obs_s.close(linger=0); ctx.term()
