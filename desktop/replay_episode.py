"""Replay a recorded demonstration on the robot (open loop, no policy).

Diagnostic: if a recorded *successful* episode no longer grasps the tool when its exact
joint commands are replayed, the robot/tool geometry changed — not the model.

Sends the episode's recorded actions to the Jetson host at the dataset fps, after a
smooth move from the current pose to the episode's first action. Run with the GUI closed.

  python3 replay_episode.py --task grasper              # median-length grasper episode
  python3 replay_episode.py --episode 5 --dry-run       # print what would be sent
  Ctrl+C stops streaming; the arms then hold the last commanded pose.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, zmq

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="bigse0u1/xlerobot_scrub_7tool")
ap.add_argument("--episode", type=int, default=None)
ap.add_argument("--task", default="grasper", help="tool name in 'Pick up the <tool>' (used if --episode not given)")
ap.add_argument("--ip", default="192.168.0.34")
ap.add_argument("--cmd-port", default="5555")
ap.add_argument("--obs-port", default="5556")
ap.add_argument("--approach-s", type=float, default=3.0, help="seconds to move from current pose to the first action")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo}")
info = json.load(open(f"{root}/meta/info.json"))
names, fps = info["features"]["action"]["names"], info["fps"]
eps = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{root}/meta/episodes/*/*.parquet")])
if args.episode is None:
    cand = eps[eps.tasks.map(lambda t: t[0].lower()) == f"pick up the {args.task}".lower()]
    if cand.empty:
        sys.exit(f"no episodes for task '{args.task}'")
    cand = cand.iloc[(cand.length - cand.length.median()).abs().argsort()]
    args.episode = int(cand.episode_index.iloc[0])
row = eps[eps.episode_index == args.episode].iloc[0]
data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
                  for f in glob.glob(f"{root}/data/*/*.parquet")])
ep = data[data.episode_index == args.episode].sort_values("frame_index")
A = np.stack(ep["action"].values)
print(f"episode {args.episode}: '{row.tasks[0]}', {len(A)} frames ({len(A) / fps:.1f}s @ {fps}fps)")

POS = [i for i, n in enumerate(names) if n.endswith(".pos")]          # arms + head
VEL = [i for i, n in enumerate(names) if n.endswith(".vel")]          # base: always 0

def to_msg(vec):
    msg = {names[i]: float(vec[i]) for i in POS}
    msg.update({names[i]: 0.0 for i in VEL})
    return msg

if args.dry_run:
    for t in (0, len(A) // 2, len(A) - 1):
        print(f"  frame {t}:", {k: round(v, 1) for k, v in to_msg(A[t]).items() if "left_arm" in k})
    sys.exit(0)

ctx = zmq.Context()
cmd = ctx.socket(zmq.PUSH); cmd.setsockopt(zmq.CONFLATE, 1); cmd.connect(f"tcp://{args.ip}:{args.cmd_port}")
obs = ctx.socket(zmq.PULL); obs.setsockopt(zmq.CONFLATE, 1); obs.connect(f"tcp://{args.ip}:{args.obs_port}")
if not obs.poll(3000):
    sys.exit("no observation from host — is xlerobot_host running and the GUI closed?")
o = json.loads(obs.recv_string())

# current pose in action space (gripper actions are inverted vs. observed position: action = 100 - obs)
cur = A[0].copy()
for i in POS:
    v = o.get(names[i])
    if v is not None:
        cur[i] = 100.0 - v if names[i].endswith("gripper.pos") else v
diff = np.abs(cur[POS] - A[0][POS]).max()
print(f"max joint distance from current pose to episode start: {diff:.1f}")

def stream(frames, period):
    t_next = time.perf_counter()
    for vec in frames:
        cmd.send_string(json.dumps(to_msg(vec)))
        t_next += period
        time.sleep(max(0.0, t_next - time.perf_counter()))

try:
    n = max(1, int(args.approach_s * fps))
    print(f"moving to episode start over {args.approach_s:.1f}s ...")
    stream([cur + (A[0] - cur) * (k + 1) / n for k in range(n)], 1.0 / fps)
    time.sleep(0.5)
    print("replaying demonstration ...")
    t0 = time.time()
    stream(A, 1.0 / fps)
    print(f"done in {time.time() - t0:.1f}s — holding last pose (send nothing more; arms keep the last goal)")
    stream([A[-1]] * fps, 1.0 / fps)
except KeyboardInterrupt:
    print("\nstopped — arms hold the last commanded pose")
finally:
    cmd.close(linger=0); obs.close(linger=0); ctx.term()
