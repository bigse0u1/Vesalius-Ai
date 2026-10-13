# 실험 기록

ICEIC 2027 논문용 실험 결과 모음.

| # | 실험 | 상태 | 핵심 결과 |
|---|---|---|---|
| 01 | 고정 배치 Pick 성공률 (7 tools × 20) | ✅ 완료 | **117/140 = 83.6%** (도구별 75–90%) |
| 02 | 잘못 놓인 배치 — Baseline vs Remapping (oracle slot) | ✅ 완료 | Baseline **0/60 (0%)** vs Remapping **50/60 (83.3%)** |
| 03 | 분류기 정확도 (컬러 / 흑백) | ✅ 완료 | **100%** (945/945, 배치 단위 5-fold). 적은 데이터에서 컬러가 우세 (배치 2개: 94% vs 68%) |
| 04 | 통합 시스템 (분류기 + Remapping 자동) — 잘못 놓인 60 / 정상 70 / 도구 없음 70 | 🟡 잘못 놓인 배치·도구 없음 완료 (실행 방식 확인 필요), 정상 배치 남음 | 잘못 놓인 배치 **53/60 (88.3%)**, 도구 없음 전부 stop |
| 05 | End-to-End (음성 → 전달, 7 tools × 10) | ⬜ 예정 | |

**기록 규칙**

- 시도 1회 = 1칸. `1` = 성공, `0` = 실패. 가능하면 실패 유형도 기록: `miss`(못 잡음), `drop`(놓침), `wrong`(다른 도구), `timeout`.
- 모델, 데이터셋, 배치, instruction, 날짜 등 **재현에 필요한 조건**을 적는다.
- n이 작으므로 성공률에 95% 신뢰구간(Wilson)을 함께 적는다.
- **head camera 방향을 확인한다.** 2026-10-12~13 AI 모드가 모든 도구에서 그리퍼를 연 채 맴도는 현상이 두 번 있었고, 두 번 모두 head camera를 만지거나 닦은 뒤 해결됨 (같은 시점에 시연 재생 `desktop/replay_episode.py`는 정상 → 로봇·도구 위치 관계는 정상, 카메라 방향/렌즈가 원인으로 추정). 로봇·트레이를 바닥 테이프 표시에 맞춘 상태에서 정렬 가이드가 어긋나면 트레이가 아니라 **카메라**를 조정한다.
- **시작 자세를 학습 데이터와 맞춘다.** 특히 손목 회전(wrist_roll)이 왼팔 약 -2, 오른팔 약 5 (학습 데이터 episode 시작 평균)인지 GUI "팔 상태"로 확인한 뒤 AI 모드를 시작한다. 2026-10-11 wrist_roll이 왼팔 -12.8 / 오른팔 11.1로 틀어진 채 시작해 Pick이 거의 전부 실패한 적이 있음 — 이 상태에서 나온 시도는 기록에서 제외.

---

## 01. 고정 배치 Pick 성공률

- **완료일**: 2026-10-09 이전 (기록 기준)
- **목적**: 데이터 수집 때와 같은 정상 배치에서 7개 도구 Pick 성공률. 논문 메인 비교표의 **정상 배치** 열 / Grasp Success Rate.

**조건**

| 항목 | 값 |
|---|---|
| 모델 | `bigse0u1/smolvla_7tool` (smolvla_base fine-tune, batch 64 × 30k steps) |
| 학습 데이터 | `bigse0u1/xlerobot_scrub_7tool` — 7 tools × 100 episodes, 고정 배치 |
| 배치 | 데이터 수집 때와 동일 (Slot 1→7: Grasper, Bipolar, Scissors, Clipper, Hook, Irrigator, Specimen Bag) |
| Instruction | `Pick up the <tool>` (Clipper는 `clippers`) |
| 실행 | GUI AI 모드, 도구당 20회 |

**원본 데이터**

| Slot | 도구 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Grasper | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| 2 | Bipolar | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| 3 | Scissors | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| 4 | Clipper | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 |
| 5 | Hook | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| 6 | Irrigator | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 7 | Specimen Bag | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |

**결과**

| Slot | 도구 | 팔 | 성공 | 성공률 | 95% CI |
|---|---|---|---:|---:|---|
| 1 | Grasper | L | 18/20 | 90% | 70–97% |
| 2 | Bipolar | L | 16/20 | 80% | 58–92% |
| 3 | Scissors | L | 17/20 | 85% | 64–95% |
| 4 | Clipper | L | 16/20 | 80% | 58–92% |
| 5 | Hook | R | 15/20 | 75% | 53–89% |
| 6 | Irrigator | R | 17/20 | 85% | 64–95% |
| 7 | Specimen Bag | R | 18/20 | 90% | 70–97% |
| | **전체** | | **117/140** | **83.6%** | 76.6–88.8% |

왼팔 (Slot 1–4) 67/80 = 83.8%, 오른팔 (Slot 5–7) 50/60 = 83.3%.

**메모**

- 도구당 n=20이라 신뢰구간이 넓다 (±15%p 내외). 도구 간 차이(75–90%)는 통계적으로 유의하다고 보기 어렵다 — 논문에서는 전체 83.6%를 주 수치로 쓰는 것이 안전.
- 연속 실패: Bipolar 14–16회, Hook 16–18회에서 3회 연속 실패. 조명·도구 위치·모터 온도 등 조건 변화 여부 확인 필요.
- 실패 유형은 이번 기록에 없음. 이후 실험부터 함께 기록.

---

## 02. 잘못 놓인 배치 (Wrong-Slot)

- **목적**: 도구가 다른 슬롯에 놓였을 때 Baseline과 Proposed(Verify & Recover)의 전달 결과 비교. 논문 메인 비교표의 **잘못 놓인 배치** 열.

### 02a. Baseline 예비 관찰 (완료)

| 항목 | 값 |
|---|---|
| 모델 | `bigse0u1/smolvla_7tool` |
| 조작 | 두 도구의 자리를 서로 바꿈 |
| Instruction | **요청한 도구 이름 그대로** (예: Scissors 요청 → `Pick up the scissors`) |
| 결과 | **모든 시도에서 요청한 도구의 원래 슬롯으로 이동해 그 자리의 다른 도구를 집음** — 올바른 도구 0회 |

해석: 고정 배치로 학습한 SmolVLA는 instruction의 도구 이름을 **외형이 아니라 위치**로 학습했다. 따라서 Baseline(SmolVLA only)과 Fixed Slot only는 고정 배치에서 사실상 같은 동작을 하며, 도구가 잘못 놓이면 항상 잘못된 도구를 전달한다 → Verify & Recover의 필요성 근거.

TODO:

- 시도 횟수 기록이 없어 정식 수치는 02b의 Baseline 60회로 대체

### 02b. 다른 자리에서 집기 — Baseline vs Remapping (✅ 완료)

자리를 바꾼 배치에서 두 가지 instruction을 비교. Remapping은 **실제 슬롯의 원래 도구 이름**을 사람이 직접 입력한 **oracle slot** 조건 → 논문에서는 Proposed의 **Pick 단계 성능**으로 보고 (분류기 정확도는 실험 03에서 별도 측정).

설계: 교환 3개 (같은 팔 / 다른 팔, 모양 유사 / 상이) × 양방향 × 10회. 결과 코드: `1` 성공, `miss` 못 잡음, `drop` 놓침, `wrong` 다른 슬롯의 도구를 집음, `timeout` 20초 초과.

**Baseline — 요청한 도구 이름 그대로**

| 교환 | 집을 도구 (현재 슬롯) | Instruction | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 올바른 도구 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|
| A: Scissors↔Clipper (3↔4) | Clipper (Slot 3) | `Pick up the clippers` | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | 0/10 |
| | Scissors (Slot 4) | `Pick up the scissors` | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | 0/10 |
| B: Bipolar↔Hook (2↔5) | Hook (Slot 2) | `Pick up the hook` | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | 0/10 |
| | Bipolar (Slot 5) | `Pick up the bipolar` | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | 0/10 |
| C: Grasper↔Irrigator (1↔6) | Irrigator (Slot 1) | `Pick up the irrigator` | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | 0/10 |
| | Grasper (Slot 6) | `Pick up the grasper` | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | wrong | 0/10 |
| | | | | | | | | | | | | | **0/60 (0%)** |

60회 모두 요청한 도구의 **원래 슬롯**으로 가서 그 자리에 놓인 다른 도구를 집음. 95% CI 0–6%.

**Remapping — 실제 슬롯의 원래 도구 이름 (oracle slot)**

| 교환 | 집을 도구 (현재 슬롯) | Instruction | 팔 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 성공 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|
| A: Scissors↔Clipper (3↔4) | Scissors (Slot 4) | `Pick up the clippers` | L | 1 | 1 | drop | 1 | miss | 1 | 1 | 1 | 1 | 1 | 8/10 |
| | Clipper (Slot 3) | `Pick up the scissors` | L | 1 | 1 | 1 | 1 | 1 | miss | 1 | 1 | 1 | 1 | 9/10 |
| B: Bipolar↔Hook (2↔5) | Bipolar (Slot 5) | `Pick up the hook` | R | 1 | 1 | 1 | 1 | 1 | drop | 1 | 1 | 1 | 1 | 9/10 |
| | Hook (Slot 2) | `Pick up the bipolar` | L | 1 | 1 | 1 | 1 | 1 | 1 | 1 | miss | 1 | miss | 8/10 |
| C: Grasper↔Irrigator (1↔6) | Grasper (Slot 6) | `Pick up the irrigator` | R | 1 | 1 | 1 | miss | 1 | 1 | 1 | 1 | miss | 1 | 8/10 |
| | Irrigator (Slot 1) | `Pick up the grasper` | L | 1 | 1 | 1 | 1 | drop | 1 | 1 | 1 | 1 | drop | 8/10 |
| | | | | | | | | | | | | | | **50/60 (83.3%)** |

**결과**

| | 시도 | 성공 | 성공률 | 95% CI |
|---|---:|---:|---:|---|
| Baseline (요청 도구 이름) | 60 | 0 | **0%** | 0–6% |
| Remapping (oracle slot) | 60 | 50 | **83.3%** | 72.0–90.7% |
| 참고: 정상 배치 (실험 01) | 140 | 117 | 83.6% | 76.6–88.8% |

- 교환별 Remapping: A (같은 팔, 모양 유사) 17/20 = 85% · B (다른 팔) 17/20 = 85% · C (다른 팔, 모양 차이 큼) 16/20 = 80%.
- 실패 10회: miss 6, drop 4. `wrong`(다른 도구)은 0회 — 팔은 항상 지정한 슬롯으로 감.
- **다른 도구가 놓인 슬롯에서 집어도 성공률이 정상 배치와 같다** (83.3% vs 83.6%, Fisher exact p = 1.0). 슬롯별로 학습된 grasp 동작이 그 자리의 다른 도구에도 일반화되며, 팔 영역이 바뀌는 경우(B, C)에도 유지된다.
- 결론: **Remapping 유효 → 분류기 + GUI 통합 진행.** 남은 변수는 분류기 정확도 (실험 03).

---

## 03. 분류기 정확도

- **완료일**: 2026-10-12
- **목적**: head camera의 슬롯별 손잡이 crop으로 7개 도구 + EMPTY를 구분하는 정확도. Verify & Recover의 Tray Scan 단계.

**조건**

| 항목 | 값 |
|---|---|
| 데이터 | GUI 분류기 탭으로 수집. 사진 135장 / 배치 54종 / crop 945개 (도구별 132–133, EMPTY 15). 조명 어둡게 5장 |
| 배치 | 무작위 섞기 — 각 도구가 7개 슬롯 전부에 고르게 등장 (위치와 라벨 분리) |
| 라벨 검수 | 색 띠 자동 검사로 1개 배치(3장)의 Slot 4/7 라벨 뒤바뀜 발견 → 수정 (원본 `labels.jsonl.bak_before_fix`) |
| 모델 | ResNet18 (ImageNet pretrained) fine-tune, 입력 224×112, 25 epoch, AdamW 3e-4, 증강: 이동·회전·밝기/대비 (색조 변화 없음) |
| 평가 | **배치 단위 5-fold 교차검증** — 같은 배치가 학습/시험에 동시에 들어가지 않음 |
| 코드 | [`classifier/train_classifier.py`](../classifier/train_classifier.py) |

**결과 — 전체 데이터 (5-fold CV)**

| 입력 | 정확도 | 95% CI |
|---|---:|---|
| 컬러 | **945/945 (100%)** | 99.6–100% |
| 흑백 | 945/945 (100%) | 99.6–100% |

모든 클래스 recall 100%, 어두운 조명 35/35. 혼동 행렬은 대각선만 존재.

**결과 — 학습 데이터 양에 따른 정확도 (learning curve)**

시험: 학습에 쓰지 않은 배치 20개 (도구 crop 331개, EMPTY 제외). 3회 반복 평균 ± 표준편차.

| 학습에 쓴 배치 수 | 컬러 | 흑백 |
|---:|---:|---:|
| 1 | 59.3 ± 9.1% | 30.9 ± 6.7% |
| 2 | **94.1 ± 3.3%** | 67.8 ± 8.5% |
| 4 | **99.6 ± 0.6%** | 89.5 ± 4.6% |
| 8 | 100.0 ± 0.0% | 98.9 ± 0.3% |

**해석**

- 충분한 데이터(배치 ~40개)에서는 손잡이 모양만으로도(흑백) 7개 도구를 완벽히 구분 → Tray Scan은 이 환경에서 병목이 아니다.
- 데이터가 적을 때는 **색 띠가 큰 차이**를 만든다: 배치 2개(약 35 crop)만으로 컬러 94% vs 흑백 68%, 배치 4개에서 컬러 99.6% vs 흑백 89.5%. → 색 띠는 새 도구 추가·환경 변경 시 **적은 수집량으로 분류기를 만들 수 있게** 한다. 논문의 color band 기여는 이 learning curve로 보여주는 것이 적절.
- 한계: 같은 7개 실물 도구, 같은 조명 환경, 고정 헤드 자세에서의 결과. 최종 모델: `~/xlerobot_classifier_data/models/handle_classifier_resnet18_color.pt` (전체 데이터 학습).

---

## 04. 통합 시스템 Wrong-Slot — Proposed (분류기 + Remapping 자동)

- **목적**: 실험 03의 분류기로 트레이를 스캔하고 instruction을 자동 remapping하는 **완전한 Proposed 시스템**의 성능. 실험 02 (Baseline / oracle remapping)와 같은 조건.
- **상태**: 진행 중 (잘못 놓인 배치, 도구 없음 완료 / 정상 배치 남음)

GUI 데모 시퀀스: 방법 = `Proposed (Verify & Recover)`, **"집기 + 자세 고정까지만"** 체크. 요청 도구 선택 → ▶ 전체 데모 시작 → 트레이 스캔 → 판단 → 집기 → 자세 고정에서 멈춤. 분류 결과·판단·instruction은 `~/xlerobot_experiment_log.jsonl`에 자동 기록되므로 표에는 집기 결과만 적는다. Baseline은 실험 02b (0/60)로 대체.

결과 코드: `1` 올바른 도구 집음, `miss`, `drop`, `wrong` 다른 도구, `scan_err` 분류 결과가 틀림, `stop` 멈추고 알림.

**잘못 놓인 배치 (조건당 10회, 60회 — 실험 02b와 같은 조건·횟수)**

| 교환 | 요청 도구 (현재 슬롯) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 성공 |
|---|---|---|---|---|---|---|---|---|---|---|---|---:|
| A: Scissors↔Clipper (3↔4) | Scissors (Slot 4) | 1 | 1 | 1 | 1 | 1 | 1 | drop | 1 | 1 | 1 | 9/10 |
| | Clipper (Slot 3) | 1 | 1 | 1 | 1 | miss | 1 | 1 | 1 | 1 | drop | 8/10 |
| B: Bipolar↔Hook (2↔5) | Bipolar (Slot 5) | 1 | 1 | miss | 1 | miss | 1 | 1 | 1 | 1 | 1 | 8/10 |
| | Hook (Slot 2) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10/10 |
| C: Grasper↔Irrigator (1↔6) | Grasper (Slot 6) | miss | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 9/10 |
| | Irrigator (Slot 1) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | miss | 1 | 9/10 |
| | | | | | | | | | | | | **53/60** |

**잘못 놓인 배치 결과**: **53/60 (88.3%, 95% CI 77.8–94.2%)**. 교환별 A 17/20 · B 18/20 · C 18/20. 실패 7회: miss 5, drop 2, `wrong` 0. Baseline 0/60 대비 p < 1e-12, oracle remapping 50/60 대비 차이 없음 (Fisher p = 0.60) → 분류기를 넣어도 Pick 단계 성능이 유지된다.

> ⚠️ **확인 필요 (2026-10-13)**: 분류 로그(`~/xlerobot_experiment_log.jsonl`)에는 Proposed 요청이 18건뿐이다 (교환 A recovered 13, 정상 배치 verified 5, 교환 B·C 및 도구 없음 0건). 기록된 스캔은 모두 정확했다 (모든 슬롯 신뢰도 1.00). 나머지 시도가 데모 시퀀스(Proposed)로 실행됐는지, AI 모드에 remap 문장을 직접 입력한 oracle 조건이었는지 확인 후 확정할 것.

**정상 배치 (도구당 10회, 70회)** — 검증 단계가 정상 배치 성능을 해치지 않는지 (오경보·잘못된 remapping 없이 실험 01 수준인지) 직접 측정

| 요청 도구 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 성공 |
|---|---|---|---|---|---|---|---|---|---|---|---:|
| Grasper | | | | | | | | | | | | /10 |
| Bipolar | | | | | | | | | | | | /10 |
| Scissors | | | | | | | | | | | | /10 |
| Clipper | | | | | | | | | | | | /10 |
| Hook | | | | | | | | | | | | /10 |
| Irrigator | | | | | | | | | | | | /10 |
| Specimen Bag | | | | | | | | | | | | /10 |
| | | | | | | | | | | | **/70** |

**도구 없음 (도구당 10회, 70회)** — 요청 도구를 트레이에서 빼둠 (그 슬롯은 비워둠). 집지 않고 `stop`(멈추고 알림)이면 성공. 팔이 움직이지 않으므로 1회 약 10초.

**결과 (사용자 보고, 2026-10-13)**: 모든 시도에서 빠진 도구를 인식하고 멈춤 (`stop`). 시도 횟수와 회차별 기록은 추가 필요 (분류 로그에 `not_found` 기록 없음 — 위 확인 사항 참고).

| 빠진 도구 = 요청 도구 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 성공 |
|---|---|---|---|---|---|---|---|---|---|---|---:|
| Grasper | | | | | | | | | | | | /10 |
| Bipolar | | | | | | | | | | | | /10 |
| Scissors | | | | | | | | | | | | /10 |
| Clipper | | | | | | | | | | | | /10 |
| Hook | | | | | | | | | | | | /10 |
| Irrigator | | | | | | | | | | | | /10 |
| Specimen Bag | | | | | | | | | | | | /10 |
| | | | | | | | | | | | **/70** |

**메인 비교표**

| | 정상 배치 | 잘못 놓인 배치 | 도구 없음 → 멈춤 |
|---|---|---|---|
| Baseline | 117/140 (83.6%, 실험 01) | **0/60 (0%)** — 실험 02b | — (검증 없음, 항상 집으러 감) |
| Proposed — Pick 단계 (oracle slot) | — | **50/60 (83.3%)** — 실험 02b | — |
| Proposed — 통합 시스템 (분류기 포함) | /70 | **53/60 (88.3%)** ⚠️ | 전부 stop (횟수 확인 필요) ⚠️ |

---

## 05. End-to-End

통합 후.
