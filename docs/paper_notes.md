# 논문 작성용 정리 (ICEIC 2027)

복강경 수술도구 전달 로봇 — **Verify & Recover**: 고정 배치로 학습한 VLA 정책에 손잡이 분류기 기반 트레이 검증과 instruction remapping을 결합해, 도구가 잘못 놓여도 요청한 도구를 전달한다.

실험 원자료: [`experiments/README.md`](../experiments/README.md). 비율 옆 괄호는 95% Wilson 신뢰구간.

---

## 1. 핵심 주장과 결과 요약

| 주장 | 근거 |
|---|---|
| 고정 배치로 학습한 VLA는 도구 이름을 **외형이 아니라 위치(home slot)로** 해석한다 | 두 도구를 바꿔 놓으면 Baseline은 **0/60** 올바른 도구 (0–6.0%) — 항상 원래 슬롯의 다른 도구를 집음 |
| 같은 정책도 **instruction을 실제 슬롯의 원래 도구 이름으로 바꾸면** 다른 도구를 정상 배치 수준으로 집는다 | Remapping(oracle slot) **50/60 = 83.3%** (72.0–90.7%) vs 정상 배치 117/140 = 83.6% (76.6–88.8%), Fisher p = 1.0 |
| 손잡이 + 색 띠 분류기로 트레이의 실제 배치를 정확히 알 수 있다 | 오프라인 945/945 (99.6–100%), 로봇 판별: 잘못 놓인 배치 **104/104**, 정상 **139/140**, 도구 없음 **140/140** |
| 분류기를 넣어도 Pick 성능이 떨어지지 않는다 | 통합 시스템(교환 배치 집기) **33/40 = 82.5%** (68.0–91.3%) ≈ 판별 100% × Pick 83.3% |
| 전체 파이프라인이 실제로 연결되어 동작한다 | End-to-End 무작위 20회 **17/20 = 85%** (64.0–94.8%), 실패 3회 모두 Pick 단계 |

**한 문장 요약**: 도구가 잘못 놓이면 Baseline은 0%, 제안 방법은 판별 100% · 집기 82.5%로 올바른 도구를 전달하며, 정상 배치에서 검증 단계로 인한 손실은 오경보 1/140뿐이다.

---

## 2. 시스템 구성

### 2.1 하드웨어

| 구성 | 사양 |
|---|---|
| 로봇 | XLeRobot (STS3215 서보), 양팔 SO-ARM (팔당 6 DoF: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper), 헤드 pan/tilt, 3륜 옴니 베이스 |
| 로봇 컴퓨터 | NVIDIA Jetson Orin Nano (15 W 모드) — 카메라·모터·IMU 담당 |
| 헤드 카메라 | Intel RealSense D415, RGB 640×480 @ 30 fps (depth 미사용) |
| 손목 카메라 | USB 카메라 ×2, 320×240 @ 30 fps |
| IMU | MPU6050 (베이스, I2C) — 90° 회전 yaw 피드백 |
| 텔레오퍼레이션 | SO-ARM100 리더암 ×2 |
| 데스크탑 (정책·분류기 추론) | Intel i7-13700F, NVIDIA RTX 4060 8 GB |
| 통신 | ZMQ (명령 PUSH/관측 PULL, JPEG 압축 영상), 5 GHz Wi-Fi |
| 학습 | RunPod RTX 5090 32 GB (정책), RTX 4060 (분류기) |

### 2.2 도구와 트레이

복강경 도구 7종, standardized tray 고정 배치 (왼쪽 → 오른쪽). 각 도구의 손잡이와 shaft에 **색 띠**를 칠함.

| Slot | 도구 | 색 띠 | 담당 팔 |
|---|---|---|---|
| 1 | Grasper | 파랑 | Left |
| 2 | Bipolar | 빨강 | Left |
| 3 | Scissors | 노랑 | Left |
| 4 | Clipper | 초록 | Left |
| 5 | Hook | 갈색 | Right |
| 6 | Irrigator | 보라 | Right |
| 7 | Specimen Bag | 검정 | Right |

### 2.3 파이프라인

```
음성 요청 → faster-whisper → 별칭 parser → 요청 도구
  → 트레이 스캔 (head camera 1장, 슬롯 7개 crop) → 손잡이 분류기
  → Verify & Recover 판단 → instruction remapping → 팔 선택
  → SmolVLA Pick → grip lock → IMU 90° 회전
  → 손 검출 (3×3 격자) → 저장 자세로 전달 → release → 복귀
```

---

## 3. Pick 정책 (SmolVLA)

### 3.1 데모 데이터

| 항목 | 값 |
|---|---|
| 데이터셋 | `bigse0u1/xlerobot_scrub_7tool` (LeRobotDataset v3) |
| 규모 | 7 도구 × 100 episodes = **700 episodes**, 246,851 frames, 30 fps |
| episode 길이 | 평균 11.8 s (9.3–15.4 s) |
| 수집 | 리더암 텔레오퍼레이션, 고정 배치, 하루 중 여러 시간대 |
| 동작 | Home pose → 접근 → 그리퍼 열기 → 정렬 → 닫기 → 들어올림 → Ready pose |
| Instruction | 도구별 1문장 고정: `Pick up the <tool>` (Clipper는 `clippers`) |
| 관측 | head RGB, 왼/오른 손목 RGB, state 14-d (양팔 12 + 헤드 2) |
| 행동 | 17-d (양팔 관절 12 + 헤드 2 + 베이스 속도 3) |

### 3.2 학습

| 항목 | 값 |
|---|---|
| 초기화 | `lerobot/smolvla_base` (사전학습 VLM 포함) fine-tune |
| 카메라 매핑 | head / left_wrist / right_wrist → camera1 / camera2 / camera3 |
| batch × steps | 64 × 30,000 (약 8 epoch) |
| action chunk | chunk 50, n_action_steps 50, flow matching 10 denoising steps |
| 하드웨어 | RTX 5090 (RunPod) |
| 추론 | RTX 4060에서 chunk 1개 약 145 ms, ZMQ로 로봇에 명령 |

참고: 2도구 파일럿은 `--policy.type=smolvla`(사전학습 VLM 미사용, VLM 고정)로 학습되어 language instruction을 거의 반영하지 못했음 → 7도구 모델은 `smolvla_base`에서 fine-tune.

### 3.3 Grip lock과 회전

- **완료 판단**: Pick 시작 후 3 s가 지나고 관절 변화가 1.2 s 동안 1 미만이면 완료 (최대 20 s).
- **Grip lock**: 완료 시점의 자세를 scripted action으로 고정해 회전 중 도구를 놓치지 않게 함.
- **회전**: IMU yaw(자이로 bias 보정 후 적분)로 −90°까지, 오차 2° 이내에서 정지.

---

## 4. 손잡이 분류기 (Verify 단계)

### 4.1 데이터 수집

| 항목 | 값 |
|---|---|
| 촬영 | 고정 스캔 헤드 자세(pan −4, tilt 70), head camera 640×480 |
| ROI | 슬롯별 손잡이·색 띠 영역 사각형 7개 (한 번 지정) |
| 배치 | 무작위로 섞은 **54개 배치**, 배치당 1–4장 → **135장** |
| crop | **945개**: 도구별 132–133, EMPTY 15 (도구를 뺀 슬롯) |
| 변화 | 슬롯 안에서 밀기·비틀기, 조명 어둡게 5장 |
| 위치·라벨 분리 | 각 도구가 7개 슬롯 전부에 고르게 등장 |
| 라벨 검수 | 색 띠 기반 자동 검사로 1개 배치의 Slot 4/7 라벨 뒤바뀜(6 crop) 수정 |

### 4.2 모델과 학습

| 항목 | 값 |
|---|---|
| 모델 | ResNet18 (ImageNet pretrained), 출력 8 클래스 (도구 7 + EMPTY) |
| 입력 | RGB crop → 224×112, ImageNet 정규화 |
| 증강 | 이동 ±8%, 회전 ±6°, 크기 ±8%, 밝기·대비 (색조는 바꾸지 않음 — 색 띠가 단서) |
| 학습 | AdamW (lr 3e-4, wd 1e-4), cosine, 25 epoch, batch 32, 클래스 가중 cross-entropy |
| 시간 | RTX 4060에서 모델당 약 20 s |
| 추론 | 트레이 스캔 1회(crop 7개) 약 9 ms |

### 4.3 평가 (실험 03)

- **배치 단위 5-fold 교차검증** (같은 배치가 학습·시험에 동시에 들어가지 않음)

| 입력 | 정확도 |
|---|---|
| 컬러 | **945/945 = 100%** (99.6–100%) |
| 흑백 | 945/945 = 100% |

- **Learning curve** — 학습에 쓴 배치 수별 정확도 (시험: 학습에 쓰지 않은 배치 20개, 도구 crop 331개, 3회 평균 ± SD)

| 학습 배치 수 | 컬러 | 흑백 |
|---:|---:|---:|
| 1 | 59.3 ± 9.1% | 30.9 ± 6.7% |
| 2 | **94.1 ± 3.3%** | 67.8 ± 8.5% |
| 4 | **99.6 ± 0.6%** | 89.5 ± 4.6% |
| 8 | 100.0 ± 0.0% | 98.9 ± 0.3% |

→ 충분한 데이터에서는 손잡이 모양만으로 구분 가능. **색 띠는 적은 수집량으로 분류기를 만들 때** 큰 차이 (배치 2개: 94% vs 68%). 논문의 color band 기여는 이 learning curve로 제시.

---

## 5. Verify & Recover

### 5.1 판단 규칙

입력: 요청 도구, 슬롯별 (예측 클래스, 신뢰도), 신뢰도 기준 τ = 0.8.

| 상황 | 판단 | 동작 |
|---|---|---|
| home slot에 요청 도구 (≥ τ) | `verified` | 그대로 진행 |
| 다른 슬롯에 요청 도구 (≥ τ) | `recovered` | 그 슬롯으로 목표 변경 |
| 요청 도구가 τ 미만으로만 보임 | `low_conf` | 멈추고 알림 |
| 트레이에 없음 | `not_found` | 멈추고 알림 |

### 5.2 Instruction remapping

고정 배치 정책은 `Pick up the <tool>`을 "그 도구의 home slot으로 가라"로 해석한다 (실험 02: 0/60). 따라서 **실제 슬롯의 원래 주인 도구 이름**을 instruction으로 넣는다.

```
요청 Scissors (home slot 3) → 스캔: Slot 3 = Clipper, Slot 4 = Scissors
→ recovered, slot 4 → instruction "Pick up the clippers" (slot 4의 원래 도구) → 왼팔
```

- 추가 데이터·재학습 없이 같은 정책을 그대로 사용.
- 팔도 실제 슬롯으로 선택 (Slot 1–4 Left, 5–7 Right) — 팔 영역이 바뀌는 교환에서도 동작.
- 고정 배치에서는 Method A (SmolVLA only)와 Method B (Fixed slot only)가 같은 동작이므로 하나의 Baseline으로 비교.

---

## 6. 음성 인식

| 항목 | 값 |
|---|---|
| 모델 | faster-whisper **medium**, CPU int8 |
| 설정 | beam 5, VAD, initial prompt에 도구 이름(한/영), 타임스탬프·이전 문맥 끔, temperature 0, 최대 24 토큰 |
| 지연 (3 s 음성, i7-13700F) | small 0.8 s · **medium 2.2 s** · large-v3-turbo 3.5 s |
| Parser | 도구별 별칭 사전(한/영) 부분 문자열 매칭 → 정규 이름 → instruction |

---

## 7. 손 위치 기반 핸드오버

| 항목 | 값 |
|---|---|
| 손 검출 | MediaPipe HandLandmarker (CPU, 약 20 ms/frame), 손바닥 중심 = 손목 + 손가락 뿌리 5점 평균, 여러 손이면 가장 큰 손 |
| 위치 → 자세 | head 화면 3×3 격자, 같은 칸 연속 5회(약 0.5 s) 검출 시 확정 → 해당 팔·칸에 저장된 전달 자세 |
| 기본값 | 미저장 칸 또는 5 s 안에 손 미검출 → 가운데 칸(1,1) |
| 동작 | 4 s 부드러운 접근 → 1 s 후 도구를 든 팔의 그리퍼만 열기 → 1 s 후 4 s 복귀 |
| 범위 | depth 기반 3D 위치·IK는 미사용 (future work) |

---

## 8. 실험

### 8.1 실험 01 — 고정 배치 Pick

7 도구 × 20회, 정상 배치, instruction = 요청 도구 이름.

| 도구 | Grasper | Bipolar | Scissors | Clipper | Hook | Irrigator | Specimen Bag | **전체** |
|---|---|---|---|---|---|---|---|---|
| 성공 | 18/20 | 16/20 | 17/20 | 16/20 | 15/20 | 17/20 | 18/20 | **117/140 = 83.6%** (76.6–88.8%) |

왼팔 67/80 = 83.8%, 오른팔 50/60 = 83.3%.

### 8.2 실험 02 — 잘못 놓인 배치: Baseline vs Remapping (oracle slot)

교환 3종 × 양방향 × 10회. A: Scissors↔Clipper (같은 팔, 모양 유사), B: Bipolar↔Hook (다른 팔), C: Grasper↔Irrigator (다른 팔, 모양 차이 큼).

| 방법 | 올바른 도구 |
|---|---|
| Baseline (요청 도구 이름) | **0/60 = 0%** (0–6.0%) |
| Remapping (실제 슬롯의 원래 도구 이름, 사람이 슬롯 지정) | **50/60 = 83.3%** (72.0–90.7%) |

- 교환별 Remapping: A 17/20, B 17/20, C 16/20
- Remapping 실패 10회: miss 6, drop 4, wrong 0

### 8.3 실험 03 — 분류기 정확도

§4.3 참고.

### 8.4 실험 04 — Verify & Recover 판별 정확도 (로봇, 집지 않음)

조건당 20회(교환 C는 12회), 시도마다 도구를 다시 놓음.

| 조건 | 정답 | 결과 |
|---|---|---|
| 잘못 놓인 배치 | 실제 슬롯으로 `recovered` | **104/104 = 100%** (96.4–100%) |
| 정상 배치 | home slot `verified` | **139/140 = 99.3%** (96.1–99.9%) — 실패 1회는 Irrigator를 EMPTY(0.51)로 보고 멈춤 (오경보) |
| 도구 없음 | `not_found` (멈춤) | **140/140 = 100%** (97.3–100%) |

교환 배치에서 집기까지 실행한 결과 (통합 시스템, B·C): **33/40 = 82.5%** (68.0–91.3%), 실패 7회 모두 Pick 단계 (miss 1, drop 6), 다른 도구 0회.

### 8.5 실험 05 — End-to-End

무작위 순서 20회 (정상 14회 = 도구당 2회, 교환 6회 = A·B·C 양방향 각 1회), 음성 요청부터 전달까지.

| | 결과 |
|---|---|
| 전체 | **17/20 = 85%** (64.0–94.8%) |
| 정상 배치 / 교환 배치 | 12/14 / 5/6 |
| 단계별 | 음성 20/20, 판별 20/20, Pick 17/20 (drop 2, miss 1), 회전 17/17, 전달 17/17 |

### 8.6 메인 결과표 (논문 Table I 초안)

| Method | Normal layout | Misplaced tool | Missing tool → stop |
|---|---|---|---|
| Baseline (SmolVLA, requested name) | 117/140 (83.6%) | **0/60 (0%)** | — (always picks) |
| Proposed — verification (exp. 04) | 139/140 (99.3%) | **104/104 (100%)** | **140/140 (100%)** |
| Proposed — pick at verified slot | — | 50/60 (83.3%, oracle) · **33/40 (82.5%, full system)** | — |
| Proposed — end-to-end | 12/14 | 5/6 | — |

### 8.7 통계

- 비율의 95% 신뢰구간: Wilson score interval.
- 두 비율 비교: Fisher exact test (Remapping 50/60 vs 정상 117/140: p = 1.0, 통합 33/40 vs oracle 50/60: p = 1.0, 통합 33/40 vs Baseline 0/60: p < 1e−12).

---

## 9. 관찰과 한계 (Discussion 재료)

- **위치 암기**: 고정 배치 VLA는 도구 이름을 위치로 학습 — 단점이지만 instruction을 슬롯 지정자로 쓸 수 있게 해 주는 성질이기도 하며, 이것이 remapping의 근거.
- **Pick 단계가 병목**: 판별은 사실상 완벽하고, 모든 실험에서 실패는 miss/drop. 성능 개선은 Pick 정책 쪽 (데이터 다양성, 증강, grasp 성공 판정).
- **시작 자세 민감도**: 시작 wrist_roll이 학습 범위를 벗어나면(약 10 차이) Pick이 거의 전부 실패.
- **간헐적 성능 저하**: 같은 모델·같은 배치에서 정책이 그리퍼를 연 채 맴도는 현상이 간헐적으로 발생. 모델 파일, 문장, 모터 추종, 카메라 지연, 통신, Jetson 상태, 조명은 원인이 아님을 확인했고 시연 재생(open loop)은 성공 — 폐루프 정책이 작은 입력 변화에 민감함을 시사.
- **분류기 범위**: 같은 7개 실물 도구, 같은 조명 환경, 고정 헤드 자세. 새 도구·환경에서는 재수집 필요 (learning curve상 배치 몇 개면 충분).
- **색 띠**: 원래 기구에 있는 band가 아니라 직접 칠한 표시 — 저비용 마킹으로 적은 데이터로 분류기를 만들 수 있다는 점을 기여로 제시.
- **핸드오버**: 3×3 격자 + 미리 녹화한 자세 선택 방식. depth·IK 기반 연속 위치 추종은 future work.
- **범위 밖 (future work)**: 사용 후 도구 회수, Return zone, 트레이 복귀, 수술 단계 인식 기반 다음 도구 예측 (Task 2).

---

## 10. 논문 구성 (ICEIC, IEEE 2단)

| 섹션 | 내용 | 재료 |
|---|---|---|
| I. Introduction | 수술 중 도구 전달, 잘못된 도구 전달 위험, 저가형 로봇 + VLA의 한계 | §1, §9 |
| II. Related Work | Robotic scrub nurse, 수술 도구 인식, VLA/imitation learning | — |
| III. Proposed System | A 개요 · B 음성 · C tray prior + 손잡이 분류기 · D Verify & Recover (remapping) · E Pick (SmolVLA) · F 핸드오버 | §2–§7 |
| IV. Experiments | A 환경 · B 분류기 (exp 03) · C Wrong-slot (exp 02, 04) · D End-to-End (exp 05) | §8 |
| V. Discussion | 위치 암기, Pick 병목, 한계 | §9 |
| VI. Conclusion | — | §1 |

Figure 계획: [`docs/figures_plan.md`](figures_plan.md) (메인 Figure 1장).

---

## 11. Abstract 초안 (영문)

> Delivering the wrong laparoscopic instrument is a critical failure for a robotic scrub assistant, and instruments with similar shafts are easy to confuse. We present a low-cost dual-arm system that hands over instruments on a spoken request. A vision-language-action policy (SmolVLA) fine-tuned on 700 demonstrations picks instruments from a standardized tray, and we find that it interprets the instrument name as its tray slot rather than its appearance: when two instruments are swapped it picks the wrong one in all 60 trials. We therefore add a *Verify & Recover* step: a ResNet18 classifier on handle crops with low-cost color bands scans the tray, and if the requested instrument is misplaced the instruction is remapped to the slot where it actually is, without retraining the policy. On the real robot, verification was correct in 104/104 misplaced, 139/140 normal, and 140/140 missing-instrument trials; the full system delivered the requested instrument in 82.5% of misplaced trials, matching the 83.6% pick rate on the normal layout. An end-to-end evaluation from voice request to hand-guided handover succeeded in 17 of 20 randomized trials, with all failures in the grasp stage.

---

## 12. 재현 정보

| 항목 | 위치 |
|---|---|
| Pick 데이터셋 | `bigse0u1/xlerobot_scrub_7tool` (Hugging Face) |
| 정책 | `bigse0u1/smolvla_7tool` (Hugging Face), 학습 명령은 README "Training" |
| 분류기 데이터·모델 | `~/xlerobot_classifier_data` (crops, labels.jsonl, models/) |
| 분류기 학습 | `python classifier/train_classifier.py` |
| GUI (V&R, 핸드오버, 수집 도구) | `desktop/xlerobot_control.py` |
| 실험 판별 로그 | `~/xlerobot_experiment_log.jsonl` |
| lerobot 버전 | 데스크탑 commit `3f2c29ef` (학습 환경도 같은 버전 권장) |
