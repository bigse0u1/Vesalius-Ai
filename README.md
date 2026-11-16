# Scrub Nurse Robot for Voice-Guided Surgical Instrument Handover

**A low-cost dual-arm robotic scrub assistant that hands laparoscopic instruments to the surgeon on a spoken request — and still delivers the right one when the tray is set up wrong.**

Target venue: ICEIC 2027 · Y-MAS 베살리우스 팀

## Demo

[![Demo video](https://img.youtube.com/vi/sg-hl-Vl0sA/hqdefault.jpg)](https://www.youtube.com/watch?v=sg-hl-Vl0sA)

*Click to watch on YouTube.*

---

## Abstract

Delivering the wrong laparoscopic instrument is a critical failure for a robotic scrub assistant, and instruments with similar shafts are easy to confuse. We present a low-cost dual-arm system that hands over instruments on a spoken request. A vision-language-action policy (SmolVLA) fine-tuned on 700 demonstrations picks instruments from a standardized tray, and we find that it interprets the instrument name as its tray slot rather than its appearance: when two instruments are swapped it picks the wrong one in all 60 trials. We therefore add a *Verify & Recover* step: a ResNet18 classifier on handle crops with low-cost color bands scans the tray, and if the requested instrument is misplaced the instruction is remapped to the slot where it actually is, without retraining the policy. On the real robot, verification was correct in 104/104 misplaced, 139/140 normal, and 140/140 missing-instrument trials; the full system delivered the requested instrument in 82.5% of misplaced trials, matching the 83.6% pick rate on the normal layout. An end-to-end evaluation from voice request to hand-guided handover succeeded in 17 of 20 randomized trials, with all failures in the grasp stage.

## Key Findings

| | Result |
|---|---|
| A fixed-layout VLA reads the tool name as a **slot**, not an appearance | Two tools swapped → baseline picks the correct tool **0/60** |
| Rewriting the instruction to the slot's original owner restores picking | Oracle remapping **50/60 (83.3%)** vs. normal layout 117/140 (83.6%) |
| Handle + color-band classifier knows the real tray layout | Offline 945/945; on robot: misplaced **104/104**, normal **139/140**, missing **140/140** |
| Verification costs nothing in pick success | Full system on swapped tools **33/40 (82.5%)** |
| The whole pipeline works end to end | Voice → handover, randomized **17/20 (85%)**; all 3 failures in the grasp |

## Method

```
Voice request ──► faster-whisper + alias parser ──► requested tool
        │
        ▼
┌──────────────────── Verify & Recover ────────────────────┐
│ head-camera tray scan → 7 slot crops → ResNet18 classifier│
│   home slot has tool   → verified   (keep instruction)    │
│   tool in other slot   → recovered  (remap instruction)   │
│   not seen / low conf  → stop and alert                   │
└───────────────────────────────────────────────────────────┘
        │  "Pick up the <original owner of the actual slot>"
        ▼
SmolVLA pick (arm chosen by slot) ──► grip lock ──► IMU 90° turn
        ──► MediaPipe hand detection (3×3 grid) ──► handover ──► return
```

**Instruction remapping.** The policy learned `Pick up the <tool>` as "go to that tool's home slot." So if Scissors (home slot 3) is found in slot 4, the system sends `Pick up the clippers` (slot 4's home tool) to the unchanged policy — no new data, no retraining.

## Results

| Method | Normal layout | Misplaced tool | Missing tool → stop |
|---|---|---|---|
| Baseline (SmolVLA, requested name) | 117/140 (83.6%) | **0/60 (0%)** | — (always picks) |
| Proposed — verification | 139/140 (99.3%) | **104/104 (100%)** | **140/140 (100%)** |
| Proposed — pick at verified slot | — | 50/60 (oracle) · **33/40 (82.5%, full system)** | — |
| Proposed — end-to-end | 12/14 | 5/6 | — |

Per-trial data for all five experiments: [`experiments/README.md`](experiments/README.md).

| # | Experiment | Result |
|---|---|---|
| 01 | Fixed-layout pick (7 tools × 20) | 117/140 = 83.6% |
| 02 | Swapped tools: baseline vs. oracle remapping | 0/60 vs. 50/60 |
| 03 | Handle classifier (color / grayscale, 5-fold by arrangement) | 100% / 100%; with 2 training arrangements 94% vs. 68% |
| 04 | Verify & Recover decision on robot (+ full-system pick) | 104/104 · 139/140 · 140/140 (+ 33/40) |
| 05 | End-to-end, voice → handover (20 randomized) | 17/20 = 85% |

## System

| Component | Spec |
|---|---|
| Robot | XLeRobot — two SO-ARM 6-DoF arms (STS3215), head pan/tilt, 3-wheel omni base |
| Robot computer | NVIDIA Jetson Orin Nano (cameras, motors, IMU) |
| Sensors | RealSense D415 head camera (RGB 640×480), 2× USB wrist cameras (320×240), MPU6050 IMU |
| Operator / inference PC | i7-13700F, RTX 4060 8 GB — policy, classifier, Whisper, GUI |
| Link | ZMQ over 5 GHz Wi-Fi (commands PUSH / observations PULL, JPEG frames) |
| Instruments | 7 laparoscopic tools on a fixed tray, each with a painted color band |

| Slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tool | Grasper | Bipolar | Scissors | Clipper | Hook | Irrigator | Specimen Bag |
| Band | Blue | Red | Yellow | Green | Brown | Purple | Black |
| Arm | Left | Left | Left | Left | Right | Right | Right |

**Policy:** SmolVLA fine-tuned from `lerobot/smolvla_base` on 700 teleoperated episodes (7 tools × 100, 246,851 frames), batch 64 × 30k steps on an RTX 5090.
**Classifier:** ResNet18 (ImageNet) on 945 handle crops from 54 arrangements, 8 classes (7 tools + EMPTY), ~9 ms per tray scan.

## Repository Structure

```
├── desktop/                    # operator PC
│   ├── xlerobot_control.py     #   PyQt5 GUI: teleop, recording, AI mode, Verify & Recover, handover
│   ├── policy_rollout.py       #   run the policy without the GUI (diagnostics)
│   ├── replay_episode.py       #   open-loop replay of a recorded episode (diagnostics)
│   ├── whisper_test.py         #   standalone voice → tool-name test
│   └── archive/                #   GUI snapshot used for experiment 01
├── jetson/                     # robot side — copied into the lerobot install
│   ├── xlerobot.py, xlerobot_host.py, config_xlerobot.py
│   └── camera_opencv.py
├── classifier/
│   └── train_classifier.py     # handle classifier: grouped 5-fold CV + final model
├── experiments/
│   └── README.md               # raw per-trial results (exp. 01–05)
└── docs/
    ├── setup.md                # installation, usage, GUI guide, known issues
    ├── paper_notes.md          # paper material: settings, numbers, discussion (KR)
    ├── figures_plan.md         # main figure plan (KR)
    └── research_plan.md        # full research plan (KR)
```

## Getting Started

Full instructions: [`docs/setup.md`](docs/setup.md).

```bash
# Jetson — robot host
python -m lerobot.robots.xlerobot.xlerobot_host

# Desktop — control GUI (teleop, data recording, AI mode, demo sequence)
python3 desktop/xlerobot_control.py

# Train the handle classifier
python classifier/train_classifier.py
```

## Resources

| | |
|---|---|
| Pick dataset | [`bigse0u1/xlerobot_scrub_7tool`](https://huggingface.co/datasets/bigse0u1/xlerobot_scrub_7tool) |
| Policy | [`bigse0u1/smolvla_7tool`](https://huggingface.co/bigse0u1/smolvla_7tool) |
| Built on | [LeRobot](https://github.com/huggingface/lerobot) · [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [MediaPipe](https://developers.google.com/mediapipe) |

## Limitations & Future Work

- The grasp is the bottleneck: verification erred once in 384 trials (a false alarm), and every other failure of the proposed method was a miss or drop during the pick.
- The pick policy is sensitive to the start pose and robot/tray placement (an alignment overlay is provided in the GUI).
- The classifier covers the same 7 instruments and environment; new tools need a few new arrangements (see the learning curve in exp. 03).
- Handover uses a 3×3 image grid with pre-recorded poses; depth-based 3D hand tracking and IK are future work.
- Tool retrieval and tray return after use are out of scope for this paper.
