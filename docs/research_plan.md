# Y-MAS 베살리우스 팀
## 복강경 수술도구 전달 로봇 Task 1 연구 계획

---

# 1. 연구 목표

본 연구의 목표는 **복강경 수술 환경에서 의사가 요청한 수술도구를 로봇이 정확하게 선택하고, 집어서 의사에게 전달한 뒤 사용이 끝난 도구를 다시 회수하여 제자리에 복귀시키는 robotic scrub assistant를 구현하는 것**이다.

현재 사용하는 플랫폼은 XLeRobot + SO-ARM 계열의 저비용 로봇 시스템이기 때문에, 숙련된 scrub nurse와 전달 속도를 경쟁하기보다는 다음을 우선 목표로 한다.

- 정확한 도구 선택
- 잘못된 도구 전달 최소화
- 안정적인 grasp
- 정확한 handover
- 사용한 도구의 정확한 회수
- 높은 End-to-End Success Rate

즉, 연구의 핵심은 **속도보다 정확성과 신뢰성**이다.

---

# 2. 대상 수술도구

현재 사용하는 복강경 수술도구는 총 7종이다.

1. Grasper
2. Bipolar
3. Scissors
4. Clipper
5. Hook
6. Irrigator
7. Specimen Bag

복강경 수술도구는 긴 shaft 부분이 서로 비슷하고, 실제 차이는 distal tip에 있는 경우가 많다. 하지만 로봇의 head camera에서는 tip 부분이 멀리 위치하기 때문에 작은 차이를 안정적으로 구별하기 어렵다.

따라서 본 연구에서는 tip 중심 인식보다는 다음 정보를 활용한다.

- Standardized Tray 위치
- 손잡이 형태
- 복강경 기구에 원래 존재하는 color band
- Robot system state

---

# 3. 전체 시스템 구조

```text
Surgeon Voice Request
        ↓
Speech Recognition
        ↓
Requested Tool
        ↓
Tray Slot Prior
        ↓
Handle Visual Verification
        ↓
Arm Selection
        ↓
ACT / SmolVLA Pick
        ↓
Grasp + Lift
        ↓
90° Robot Rotation
        ↓
Hand Detection
        ↓
Vision-Guided Handover
        ↓
Instrument Use
        ↓
Return Zone
        ↓
Retrieval
        ↓
Tray Return
```

---

# 4. 음성 인식

의사의 음성 요청은 Whisper 계열 모델을 이용해 처리한다.

초기에는 `faster-whisper`를 데스크탑에 설치하여 로컬에서 실행하며, 별도의 Whisper fine-tuning은 우선 수행하지 않는다.

```text
의사: "Scissors 주세요."
        ↓
USB Microphone
        ↓
faster-whisper
        ↓
"Scissors 주세요."
        ↓
Command Parser
        ↓
Requested Tool = SCISSORS
```

동일한 도구에 대해 여러 표현을 하나의 canonical label로 변환한다.

```text
"가위"
"시저"
"Scissors"
"Scissors 주세요"
→ SCISSORS
```

---

# 5. Standardized Tray

7개 도구는 정해진 tray 위치에 왼쪽부터 순서대로 배치한다.

```text
Slot 1 = Grasper
Slot 2 = Bipolar
Slot 3 = Scissors
Slot 4 = Clipper
Slot 5 = Hook
Slot 6 = Irrigator
Slot 7 = Specimen Bag
```

의사가 특정 도구를 요청하면 먼저 해당 도구의 expected slot을 조회한다.

```text
Requested Tool = Scissors
↓
Scissors = Slot 3
↓
Slot 3 접근
```

이 구조를 이용하면 매번 YOLO로 전체 tray에서 도구 위치를 탐색할 필요가 없다.

---

# 6. Visual Verification

Fixed Slot만 사용할 경우 도구가 잘못 배치되어 있으면 잘못된 도구를 집을 가능성이 있다.

```text
Database: Slot 3 = Scissors
실제:     Slot 3 = Clipper
```

이를 방지하기 위해 **손잡이 기반 Visual Verification**을 추가한다.

단, 잘못 배치된 것을 감지하고 **경고만 하는 것은 실패**로 본다. 의사 입장에서는 요청한 도구를 받지 못했기 때문이다. 따라서 본 연구의 Visual Verification은 **잘못 놓인 도구를 감지한 뒤, 요청한 도구의 실제 위치를 찾아 올바른 도구를 전달하는 것(Verify & Recover)**을 목표로 한다.

```text
요청: Scissors
Database: Scissors = Slot 3
↓
Tray 전체 스캔 (7개 slot 분류)
↓
실제: Slot 3 = Clipper, Slot 4 = Scissors
↓
Slot 4에서 Scissors Pick
↓
올바른 도구 전달
```

---

# 7. 손잡이 기반 도구 분류

복강경 수술도구는 tip보다 손잡이가 head camera에서 더 크게 보인다. 또한 다음 특징을 사용할 수 있다.

```text
Handle Shape
+
Existing Color Band
```

손잡이 영역을 crop한 뒤 Image Classifier에 입력한다.

```text
Handle ROI
↓
Image Classifier
↓
Grasper / Bipolar / Hook / Clipper / Scissors / Irrigator / Specimen Bag
```

예:

```text
Requested = Scissors
Predicted = Scissors
→ Pick 허용
```

```text
Requested = Scissors
Slot 3 Predicted = Clipper
→ 다른 slot에서 Scissors 탐색
→ Slot 4 Predicted = Scissors
→ Slot 4에서 Pick
```

```text
Requested = Scissors
어느 slot에서도 Scissors 미검출 (또는 confidence 낮음)
→ Pick 중단
→ 의사에게 알림 / Recheck
```

경고는 **요청한 도구가 tray 어디에도 없거나 분류 신뢰도가 낮을 때만** 발생한다.

## 7.1 Tray 전체 스캔

요청한 slot 하나만 확인하지 않고, 매 요청마다 head camera 이미지 한 장에서 **7개 slot의 손잡이 ROI를 모두 crop하여 분류**한다.

```text
Head Camera Image
↓
Slot 1~7 Handle ROI Crop
↓
Classifier × 7
↓
Actual Tray Map
{1: Grasper, 2: Bipolar, 3: Clipper, 4: Scissors, 5: Hook, 6: Irrigator, 7: Specimen Bag}
↓
Requested Tool의 실제 Slot 결정
```

이미지 한 장에서 crop 7개를 분류하므로 추가 시간은 매우 작다. 단, head camera에서 7개 slot의 손잡이가 모두 충분한 크기로 보이도록 scan 시 head pose를 고정한다.

## 7.2 Classifier 학습 데이터

- 고정 배치 Pick demonstration의 head camera 영상에서 slot ROI를 crop하면 자동으로 label을 얻을 수 있다.
- 단, 이 데이터만 사용하면 classifier가 **도구의 외형이 아니라 위치**를 학습할 수 있다.
- 따라서 **도구를 다른 slot에 섞어 배치한 이미지**를 별도로 수집하여 함께 학습한다.

---

# 8. Classifier 후보

초기에는 다음 모델을 비교할 수 있다.

- ResNet18
- MobileNetV3
- EfficientNet-B0

Classifier의 역할은 이미지 전체에서 **도구의 위치를 찾는 것**이 아니라 **각 slot에 있는 도구가 무엇인지 확인하는 것**이다. 7개 slot을 모두 분류한 결과로 요청한 도구의 실제 위치를 알 수 있다.

---

# 9. YOLO를 우선 사용하지 않는 이유

YOLO는 `무엇인가 + 어디 있는가`를 동시에 해결한다. 하지만 현재 시스템은 standardized tray를 사용하므로 도구가 놓일 수 있는 위치(7개 slot)를 이미 알고 있다. 도구가 잘못 놓이더라도 slot 사이에서 바뀌는 것이므로 slot별 분류만으로 실제 위치를 찾을 수 있다.

따라서 현재 단계에서는:

```text
Fixed Slot
+
Classifier Verification
```

이 더 단순하다.

향후 도구 위치를 랜덤하게 배치하는 실험으로 확장할 경우 YOLO 또는 YOLO-OBB를 사용할 수 있다.

---

# 10. Pick Manipulation

도구를 실제로 집는 동작은 imitation learning으로 학습한다.

후보:

- ACT
- SmolVLA

리더암을 이용해 사람이 demonstration을 수집한다.

---

# 11. Pick Demonstration

한 Pick episode는 다음과 같이 구성한다.

```text
Home Pose
↓
Target Tool 접근
↓
Gripper Open
↓
Grasp Position 접근
↓
Gripper Close
↓
Lift
↓
Ready Pose
```

저장 데이터:

```text
Head Camera RGB
Wrist Camera RGB
Robot Joint State
Gripper State
Robot Action
Task Instruction
Timestamp
```

SmolVLA 사용 시 instruction 예:

```text
"Pick up the scissors."
```

---

# 12. SmolVLA의 역할

```text
Camera
+
Robot State
+
Language Instruction
↓
SmolVLA
↓
Robot Action
```

SmolVLA의 주요 역할은 **도구에 어떻게 접근하고 잡을 것인가**이다.

본 연구에서는 도구 identity의 신뢰성을 높이기 위해 tray prior와 classifier를 추가로 사용한다.

## 12.1 Slot 기반 Instruction Remapping

Pick demonstration은 **고정 배치**(도구 위치 이동 없음, 도구당 100 episodes)로 수집한다. 이 경우 SmolVLA는 `"Pick up the scissors"`를 사실상 **"Slot 3 위치의 도구를 집어라"**로 학습할 가능성이 높다. 따라서 도구가 다른 slot으로 옮겨져 있으면 instruction을 그대로 넣었을 때 원래 slot으로 이동한다.

이를 해결하기 위해 SmolVLA에 넣는 instruction을 **실제 slot의 원래 도구 이름**으로 변환한다.

```text
Requested = Scissors
Classifier: Scissors는 실제로 Slot 4에 있음
Slot 4의 원래 도구 = Clipper
↓
SmolVLA Instruction = "Pick up the clipper."
↓
Arm이 Slot 4로 이동하여 Scissors Pick
```

고정 배치 데이터에서는 도구 이름과 slot이 1:1 대응하므로, instruction의 도구 이름을 **slot 지정자**로 사용하는 것이다.

장점:

- 고정 배치 데이터를 그대로 사용하며, 별도의 재수집이 필요 없다.
- **동일한 SmolVLA 모델**로 Method A(요청 도구 이름을 그대로 입력)와 Method C(remapping된 instruction 입력)를 비교할 수 있다.

주의:

- Slot 4에서 Clipper를 잡도록 학습된 동작으로 Scissors를 잡게 되므로, 손잡이 형태가 크게 다른 도구 조합에서는 grasp 실패가 발생할 수 있다. 실험 시 도구 조합별로 grasp 성공 여부를 기록한다.
- Arm Selection도 원래 도구가 아니라 **실제 slot 기준**으로 결정한다 (Slot 1~4 → LEFT, Slot 5~7 → RIGHT).

---

# 13. ACT와 SmolVLA 비교

실제 수술도구 데이터 수집 전에 3색 블록으로 pipeline을 먼저 검증한다.

```text
Red Block / Blue Block / Green Block
```

Task:

```text
Pick
↓
Lift
↓
90° Rotation
↓
Place
```

비교:

```text
ACT vs SmolVLA
```

평가:

- Pick Success
- Position Generalization
- End-to-End Success
- 안정성

---

# 14. 수술도구 Pick 데이터 수집량

## Pilot

```text
도구당 20~30 demonstrations
7개 → 총 140~210 episodes
```

## Main Dataset

```text
도구당 80~100 demonstrations
7개 → 총 560~700 episodes
```

## 실제 수집 현황

Pilot 단계(4개 도구)는 건너뛰고 바로 7개 도구 Main Dataset을 수집하였다.

```text
Pick:  7개 도구 × 100 episodes = 700 episodes (246,851 frames, 30 fps)
       고정 tray 배치 (도구 위치 이동 없음)
       Dataset: bigse0u1/xlerobot_scrub_7tool

Place: 도구를 든 상태에서 원래 slot에 내려놓는 demonstration (Tray Return용)
       Dataset: bigse0u1/xlerobot_scrub_7tool_place (수집 중)
```

Task instruction은 도구별로 하나의 문장으로 고정한다 (예: `"Pick up the grasper."`, Clipper는 `"Pick up the clippers."`). 추론 시 음성 명령으로 생성하는 instruction도 학습 문장과 정확히 일치해야 한다.

## 학습 설정

- SmolVLA는 사전학습된 `lerobot/smolvla_base`에서 fine-tuning한다. (2-tool pilot은 VLM 사전학습 가중치 없이 학습되어 language instruction을 거의 활용하지 못했다.)
- Head / Left Wrist / Right Wrist 카메라를 smolvla_base의 `camera1 / camera2 / camera3` 입력으로 매핑한다.
- Cloud GPU (RTX 5090 32GB), batch 64 × 30,000 steps (약 8 epoch).

---

# 15. 데이터 Variation

> 현재 Pick 데이터는 **고정 배치**로 수집하였으므로 아래의 위치/각도 variation은 적용하지 않았다. 고정 배치에서는 도구 이름과 slot이 1:1로 대응하며, 도구가 잘못 놓인 경우는 Tray Scan + Instruction Remapping(§12.1)으로 처리한다. 아래 variation은 향후 랜덤 배치로 확장할 때 적용한다.

같은 위치에서 반복만 하지 않고 다음 variation을 포함한다.

- 고정 위치: 약 20회
- 좌우/앞뒤 위치 변화: 약 20회
- 도구 각도 변화: 약 20회
- 조명/카메라 변화: 약 10~20회
- 접근 경로 및 grasp pose 변화: 약 10~20회

단, grasp strategy 자체는 지나치게 제각각이지 않도록 일관성을 유지한다.

---

# 16. Dual-Arm 역할 분담

```text
Left Arm  → 왼쪽 4개 도구
Right Arm → 오른쪽 3개 도구
```

현재 tray 배치 기준 mapping:

```text
Grasper       → LEFT
Bipolar       → LEFT
Scissors      → LEFT
Clipper       → LEFT
Hook          → RIGHT
Irrigator     → RIGHT
Specimen Bag  → RIGHT
```

실제 tray 배치에 따라 mapping은 변경 가능하다.

---

# 17. 어느 팔이 도구를 들고 있는지 관리

이 정보는 AI가 추론하게 하지 않고 시스템 state로 직접 관리한다.

```text
active_arm = LEFT
held_tool = CLIPPER
```

Pick 성공 후 상태를 저장한다.

```text
Pick Success
↓
active_arm 저장
held_tool 저장
↓
90° Rotation
↓
해당 arm으로 Handover
```

따라서 이를 별도로 학습할 필요는 없다.

---

# 18. 90° 회전

Pick 이후 의사 방향으로 약 90° 회전한다. 이 동작은 imitation learning으로 학습하지 않고 IMU와 base controller를 이용한다.

```text
Pick Success
↓
Current Yaw
↓
Target Yaw = Current + 90°
↓
Base Rotation
↓
IMU Feedback
↓
Target Angle 도달
↓
Stop
```

---

# 19. Hand Detection

90° 회전 후 의사가 손을 내밀면 D415로 손을 검출한다.

후보:

- MediaPipe Hands
- YOLO-based Hand Detector

```text
D415 RGB
↓
Hand Detector
↓
Hand Pixel Position (u, v)
```

---

# 20. D415 Depth를 이용한 3D 위치

```text
Hand Pixel (u, v)
+
Depth
↓
3D Hand Position (X, Y, Z)
```

이 정보를 robot coordinate system으로 변환한다.

---

# 21. Handover

현재 계획에서는 Handover를 별도로 imitation learning하지 않는다.

```text
Hand Detection
+
Depth
+
Geometry
+
Robot Control
```

전체 흐름:

```text
D415
↓
MediaPipe / YOLO
↓
Hand Position
↓
Depth
↓
3D Hand Position
↓
Safety Offset
↓
Handover Target Pose
↓
IK / Cartesian Control
↓
Robot Move
↓
Release
```

즉:

> **Pick은 learning-based manipulation**
> **Handover는 vision-guided control**

구조이다.

---

# 22. Handover Safety Offset

손 중심으로 직접 이동하지 않고 collision 방지를 위해 offset을 둔다.

```text
Handover Target
=
Hand Position
+
Safety Offset
```

실제 offset 값은 실험을 통해 결정한다.

---

# 23. Release

초기에는 rule-based 방식으로 구현한다.

```text
Handover Position 도달
↓
손이 일정 거리 안에 존재
↓
Robot Stop
↓
짧게 대기
↓
Gripper Open
```

---

# 24. Handover Policy는 현재 사용하지 않음

Handover Policy는 로봇이 도구를 잡은 상태에서 사람 손에 접근하고 전달하는 행동 자체를 demonstration으로 학습하는 방식이다.

현재 Task 1에서는 다음처럼 단순화한다.

```text
Pick      → ACT / SmolVLA
Handover  → Hand Detection + Depth + Control
```

---

# 25. 도구 회수

초기에는 의사 손에서 직접 다시 받지 않고 Return Zone을 사용한다.

```text
Surgeon Uses Instrument
↓
Return Zone에 도구 배치
↓
Robot Retrieval
```

---

# 26. Return 및 Tray 복귀

Robot은 이미 다음 상태를 알고 있다.

```text
held_tool
home_slot
active_arm
```

예:

```text
held_tool = Scissors
home_slot = Slot 3
```

따라서:

```text
Return Zone
↓
Tool Retrieval
↓
-90° Rotation
↓
Slot 3
↓
Place
```

가 가능하다.

Place 동작은 Pick과 마찬가지로 imitation learning으로 학습한다. 도구를 든 Ready Pose에서 시작하여 원래 slot에 내려놓고 gripper를 여는 demonstration을 별도 데이터셋으로 수집한다 (`"Place the scissors back in the tray."`). 고정 배치이므로 instruction의 도구 이름이 곧 home slot을 의미한다.

---

# 27. 반환 시 Visual Verification

정확성을 높이기 위해 반환된 도구를 classifier로 다시 확인하는 것을 optional verification layer로 둘 수 있다.

```text
Retrieved Tool
↓
Handle Camera Image
↓
Classifier
↓
Detected Tool
↓
Expected Tool과 비교
```

---

# 28. 최종 State Machine

```text
WAIT
↓
VOICE REQUEST
↓
SPEECH RECOGNITION
↓
TOOL SELECTION
↓
TRAY SLOT LOOKUP (expected slot)
↓
TRAY SCAN (7개 slot 분류 → actual tray map)
↓
VISUAL VERIFICATION
├─ expected slot = requested tool → 그대로 진행
├─ Mismatch → actual tray map에서 requested tool의 실제 slot으로 변경
└─ 어디에도 없음 / confidence 낮음 → STOP / 의사에게 알림
↓
INSTRUCTION REMAPPING (실제 slot의 원래 도구 이름)
↓
ARM SELECTION (실제 slot 기준)
↓
PICK
↓
GRASP CHECK
↓
SAVE active_arm / held_tool
↓
ROTATE +90°
↓
HAND DETECTION
↓
3D HAND POSITION
↓
VISION-GUIDED HANDOVER
↓
RELEASE
↓
WAIT FOR USE
↓
RETURN ZONE
↓
RETRIEVAL
↓
ROTATE -90°
↓
TRAY RETURN
↓
WAIT
```

---

# 29. 논문의 핵심 문제

단순히 **"SmolVLA로 수술도구를 집어서 전달한다."**를 contribution으로 하지 않는다.

핵심 문제는:

> **비슷하게 생긴 복강경 수술기구를 잘못 전달하는 오류를 어떻게 줄일 것인가?**
> **그리고 도구가 잘못 놓여 있어도 요청한 도구를 올바르게 전달할 수 있는가?**

이다. 오류를 감지하고 멈추는 것에 그치지 않고, **오류 상황에서도 올바른 도구를 전달하는 것**을 목표로 한다.

---

# 30. 제안하는 핵심 아이디어

```text
Standardized Tray Prior
+
Handle Appearance
+
Existing Color Band
+
Visual Verification & Recovery (Tray Scan + Instruction Remapping)
+
Learning-Based Manipulation
```

을 결합하여 Wrong Tool Selection 및 Wrong Tool Handover를 줄이고, 도구가 잘못 놓인 상황에서도 올바른 도구를 전달하는 것을 목표로 한다.

---

# 31. 논문용 비교 방법

## Method A — SmolVLA Only

```text
Camera + Language Instruction
↓
SmolVLA
↓
Direct Pick
```

## Method B — Fixed Slot Only

```text
Requested Tool
↓
Fixed Slot
↓
Pick
```

## Method C — Proposed

```text
Requested Tool
↓
Fixed Slot Prior
↓
Tray Scan + Handle Visual Verification
↓
Actual Slot 결정 (Mismatch 시 실제 위치로 변경)
↓
Instruction Remapping
↓
ACT / SmolVLA Pick
```

Method A와 C는 **동일한 SmolVLA 모델**을 사용하며, 차이는 입력 instruction뿐이다 (A: 요청 도구 이름 그대로, C: 실제 slot 기준으로 remapping된 이름).

---

# 32. 핵심 실험 1 — Tool Classification

질문:

> 손잡이와 기존 color band를 이용해 7개 도구를 얼마나 정확하게 구별할 수 있는가?

| Input | Accuracy |
|---|---:|
| Handle Shape | 측정 |
| Color Band | 측정 |
| Handle + Color Band | 측정 |

추가 평가:

- Confusion Matrix
- Precision
- Recall
- F1-score

---

# 33. 핵심 실험 2 — Wrong-Slot Disturbance

일부러 도구 위치를 바꿔놓는다.

```text
정상:       Slot 3 = Scissors
Disturbance: Slot 3 = Clipper
```

Fixed Slot 방식:

```text
Scissors 요청
↓
Slot 3
↓
Clipper Pick
↓
Wrong Tool
```

제안 방법:

```text
Scissors 요청
↓
Tray Scan
↓
Slot 3 = Clipper, Slot 4 = Scissors로 판단
↓
Instruction = "Pick up the clipper." (Slot 4의 원래 도구)
↓
Slot 4에서 Scissors Pick
↓
Correct Tool 전달
```

Disturbance 조건:

- 두 도구의 위치를 서로 바꾸는 swap (같은 arm 영역 내 / 다른 arm 영역 간)
- 요청한 도구를 tray에서 제거 → 올바르게 STOP/알림하는지 확인

비교 지표:

| Method | 정상 배치 | Disturbance |
|---|---|---|
| A. SmolVLA Only | Correct Tool | Wrong Tool 예상 |
| B. Fixed Slot | Correct Tool | Wrong Tool |
| **C. Proposed** | Correct Tool | **Correct Tool** |

---

# 34. 핵심 실험 3 — End-to-End Handover

```text
Voice
↓
Tool Selection
↓
Verification
↓
Pick
↓
90° Rotation
↓
Hand Detection
↓
Handover
```

7개 도구 × 20 trials = **140 autonomous trials**를 목표로 한다.

---

# 35. 평가 지표

- Speech Recognition Accuracy
- Tool Classification Accuracy
- Correct Tool Selection Rate
- Wrong Tool Selection Rate
- Grasp Success Rate
- Rotation Success Rate
- Hand Detection Success Rate
- Handover Success Rate
- Wrong Tool Handover Rate
- Disturbance Recovery Rate (잘못 배치된 상황에서 올바른 도구를 전달한 비율)
- Retrieval Success Rate
- Correct Return Rate
- End-to-End Success Rate

Response Time은 보조 지표로 기록한다.

---

# 36. 논문 핵심 결과표 예시

| Method | Tool Selection | Wrong Tool | Disturbance Recovery | Grasp | Handover | E2E |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLA Only | - | - | - | - | - | - |
| Fixed Slot | - | - | - | - | - | - |
| **Proposed** | **-** | **-** | **-** | **-** | **-** | **-** |

---

# 37. Figure 구성

## Figure 1 — Experimental Setup

- XLeRobot
- Instrument Tray
- D415
- Wrist Cameras
- 7 laparoscopic instruments
- Surgeon
- Handover Area
- Return Zone

## Figure 2 — Overall System Pipeline

```text
Voice → Whisper → Requested Tool → Tray Prior → Visual Verification
→ ACT / SmolVLA → Pick → 90° Rotation → Hand Detection → Handover → Return
```

## Figure 3 — Handle-Based Verification

7개 도구의 손잡이와 color band를 표시한다.

## Figure 4 — Method Comparison

```text
(a) SmolVLA Only
(b) Fixed Slot Only
(c) Proposed: Fixed Slot + Visual Verification & Recovery
```

---

# 38. ICEIC 논문 목차

```text
Abstract

I. Introduction

II. Related Work
   A. Robotic Scrub Nurse
   B. Surgical Instrument Recognition

III. Proposed System
   A. System Overview
   B. Voice Command Processing
   C. Tray-Prior Instrument Selection
   D. Handle-Based Visual Verification and Recovery
   E. Dual-Arm Manipulation
   F. Vision-Guided Handover

IV. Experiments
   A. Experimental Setup
   B. Instrument Classification
   C. Wrong-Slot Disturbance Test
   D. End-to-End Handover

V. Discussion

VI. Conclusion
```

---

# 39. 개발 순서

```text
STEP 1  3색 블록 Pick
STEP 2  ACT / SmolVLA 학습 pipeline 확인
STEP 3  3색 블록 Pick → 90° Rotate → Place
STEP 4  실제 수술도구 2개 Pick
STEP 5  수술도구 4개
STEP 6  수술도구 7개
STEP 7  왼팔 4개 / 오른팔 3개 역할 분담
STEP 8  Whisper 연결
STEP 9  Fixed Tray Slot 연결
STEP 10 Handle Classifier 학습
STEP 11 Visual Verification & Recovery 연결 (Tray Scan + Instruction Remapping)
STEP 12 Pick → 90° Rotation
STEP 13 D415 Hand Detection
STEP 14 Depth → 3D Hand Position
STEP 15 Vision-Guided Handover
STEP 16 Return Zone
STEP 17 Tool Retrieval
STEP 18 Tray Return
STEP 19 Wrong-Slot Disturbance Experiment
STEP 20 End-to-End Evaluation
```

---

# 40. 최종 연구 메시지

본 연구는 단순히 **"VLA를 이용해 수술도구를 집는 로봇"**을 목표로 하지 않는다.

최종적으로는:

> **Standardized tray의 위치 정보와 복강경 수술도구 손잡이의 시각적 특징 및 기존 color band를 이용한 visual verification을 결합하여, 잘못된 도구 전달을 줄이는 신뢰성 중심의 robotic scrub assistant**

를 목표로 한다.

핵심은 다음과 같다.

```text
빠른 수술도구 전달                 X

정확한 수술도구 선택               O
Wrong-Tool Handover 최소화         O
안정적인 Grasp                     O
Vision-Guided Handover             O
```

---

# 41. 향후 Task 2

Task 1 이후에는 현재 manipulation system을 그대로 사용하면서 상위-level intelligence를 추가한다.

```text
Surgical Video
↓
Surgical Phase Recognition
↓
Next Tool Prediction
↓
Predictive Preparation
↓
Task 1 Manipulation System
↓
Handover
```

최종적으로:

```text
Reactive Robotic Scrub Assistant
↓
Context-Aware Assistant
↓
Predictive Robotic Scrub Nurse
```

로 확장한다.
