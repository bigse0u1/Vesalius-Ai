# 논문 Figure 제작 계획 (ICEIC 2027)

Y-MAS 베살리우스 팀 — 복강경 수술도구 전달 로봇 Task 1

이 문서는 논문에 들어갈 Figure와 Table을 **무엇을, 어떻게, 어떤 순서로** 만들지 정리한 것이다. `📷 사진 자리`라고 표시된 부분은 직접 촬영한 사진으로 채운다. 실험 수치는 [`experiments/README.md`](../experiments/README.md) 기준.

---

## 0. 공통 규칙

### 0.1 분량과 우선순위

ICEIC는 IEEE 2단 양식이고 분량이 짧다 (투고 규정의 페이지 제한을 먼저 확인할 것). 4페이지 기준 **Figure 4–5개 + Table 1–2개**가 현실적인 최대치다.

| 우선순위 | 항목 | 내용 | 비고 |
|---|---|---|---|
| ⭐ 필수 | Fig. 1 | 실험 환경 (사진) | 리뷰어가 가장 먼저 보는 그림 |
| ⭐ 필수 | Fig. 2 | 전체 시스템 파이프라인 (다이어그램) | 제안 방법의 위치를 보여줌 |
| ⭐ 필수 | Fig. 4 | Wrong-Slot 정성 비교 (사진 시퀀스) | 핵심 주장을 한눈에 |
| ⭐ 필수 | Table I | 메인 결과표 | 핵심 수치 |
| 권장 | Fig. 3 | Tray + 손잡이 검증 (사진 + crop) | 분량 부족하면 Fig. 1에 합침 |
| 권장 | Fig. 5 | 결과 막대 그래프 | Table I과 중복되면 둘 중 하나만 |
| 선택 | Fig. 6 | 분류기 혼동 행렬 | 실험 03 완료 후 |
| 선택 | Table II | End-to-End 단계별 성공률 | 실험 05 완료 후 |

### 0.2 크기

| 배치 | 폭 | 용도 |
|---|---|---|
| 1단 (single column) | **88 mm (3.5 in)** | 대부분의 Figure |
| 2단 (double column) | **181 mm (7.16 in)** | Fig. 2 파이프라인처럼 가로로 긴 것 |

- 처음부터 **실제 크기로** 만든다. 크게 만들고 줄이면 글씨가 읽을 수 없게 작아진다.
- 그림 안의 글씨는 축소 후 **8 pt 이상**. 본문 글꼴(Times)과 맞추거나 전부 Helvetica/Arial로 통일.

### 0.3 파일 형식

| 종류 | 형식 | 해상도 |
|---|---|---|
| 다이어그램, 그래프 | **PDF (벡터)** | — |
| 사진 | PNG 또는 고품질 JPG | **300 dpi 이상** (1단 폭이면 가로 약 1050 px 이상) |

### 0.4 색

- **Baseline = 회색, Proposed = 파랑 계열 하나**로 모든 그림에서 통일.
- 흑백 인쇄해도 구분되도록 색 + 패턴(빗금) 또는 색 + 라벨을 함께 쓴다.
- 도구 색 띠(파랑·빨강·노랑·초록·갈색·보라·검정)는 사진 속 실제 색이므로 그대로 둔다. 다이어그램에서 도구를 표시할 때만 이 색을 범례로 재사용.

### 0.5 캡션

- 캡션만 읽어도 이해되게 쓴다: **무엇을 보여주는지 + 핵심 관찰 한 줄**.
- 예: *"Fig. 4. Wrong-slot trial (Scissors ↔ Clipper). The baseline follows the memorized slot and picks the Clipper, whereas the proposed method locates the Scissors in Slot 4 and picks it."*

### 0.6 사진 촬영 공통

- 삼각대 사용, **같은 조명**에서 한 번에 몰아서 촬영.
- 배경 정리 (케이블, 노트북, 개인 물품 치우기).
- 사람이 나오면 얼굴은 프레임 밖으로 (손과 팔만).
- 원본은 `docs/figures/raw/`에, 편집본은 `docs/figures/`에 저장.

---

## Fig. 1 — 실험 환경

**목적**: 시스템 구성 요소와 공간 배치(트레이 ↔ 로봇 ↔ 의사 위치, 90° 회전)를 한 장에.

**크기**: 1단 (88 mm), 가로:세로 약 4:3

```
┌─────────────────────────────────────────────┐
│                                             │
│            📷 사진 자리 (메인)               │
│     로봇 + 트레이 + 의사 위치 전체 샷          │
│     (45° 위쪽 비스듬한 각도)                  │
│                                             │
│   ① XLeRobot      ④ Wrist cameras          │
│   ② D415 (head)   ⑤ 7 instruments + tray   │
│   ③ SO-ARM ×2     ⑥ Surgeon / handover area │
│                                             │
│   ┌──────────┐                              │
│   │📷 inset  │  ← 트레이 확대 (색 띠가 보이게) │
│   └──────────┘                              │
└─────────────────────────────────────────────┘
```

> 📷 **사진 자리 1-a (메인)**: `docs/figures/fig1_setup.jpg`
> - 로봇 정면에서 약 45° 위쪽, 트레이와 의사 위치(로봇 기준 90°)가 모두 보이게.
> - 로봇은 Home Pose, 트레이는 기본 배치.
> - 의사 위치는 실제 사람 손이나 마네킹 손을 Handover 영역에 둠.

> 📷 **사진 자리 1-b (inset)**: `docs/figures/fig1_tray_inset.jpg`
> - 트레이 위에서 내려다본 샷, 7개 도구의 손잡이와 색 띠가 선명하게.

**편집 방법** (PowerPoint / Keynote / Inkscape)

1. 메인 사진 위에 번호 원(①–⑥)과 짧은 지시선을 얹는다. 흰 글씨 + 검은 테두리로 배경과 무관하게 보이게.
2. 90° 회전 방향을 곡선 화살표로 표시 (트레이 방향 → 의사 방향).
3. inset은 오른쪽 아래 모서리에 흰 테두리로 넣는다.
4. 번호 설명은 그림 안에 넣지 말고 **캡션에** 쓴다 (그림 안 글씨 최소화).

---

## Fig. 2 — 시스템 파이프라인

**목적**: 음성 요청부터 전달까지의 흐름과, 제안 방법(Verify & Recover)이 어디에 들어가는지.

**크기**: 2단 (181 mm), 가로로 긴 한 줄 흐름

```
 ┌───────┐   ┌─────────┐   ┌──────────┐   ╔═══════════════════════════════╗   ┌────────┐   ┌────────┐   ┌────────┐   ┌─────────┐
 │ Voice │ → │ Whisper │ → │Requested │ → ║ Verify & Recover (proposed)   ║ → │ Arm    │ → │SmolVLA │ → │ 90°    │ → │ Hand det│ → Handover
 │request│   │ + parser│   │  tool    │   ║ Tray scan → Handle classifier ║   │selection│   │ Pick   │   │ rotate │   │ + depth │
 └───────┘   └─────────┘   └──────────┘   ║ → Actual slot → Instr. remap  ║   └────────┘   └────────┘   └────────┘   └─────────┘
                                          ╚═══════════════════════════════╝
                                              ↓ (not found / low conf.)
                                            Alert surgeon
```

**구성**

| 블록 | 표시 | 메모 |
|---|---|---|
| Voice → Whisper → Requested tool | 회색 상자 | 기존 기술 |
| **Verify & Recover** | **파랑 테두리 굵게, 배경 연한 파랑** | 제안 방법. 안에 4단계를 작은 상자로 |
| Arm selection → SmolVLA Pick → Rotate → Hand detection → Handover | 회색 상자 | |
| Alert 분기 | 점선 화살표 | 도구 없음 / 신뢰도 낮음 |

**선택 사항 — 블록 아래 작은 썸네일**

각 블록 밑에 그 단계의 실제 화면을 작게 (높이 15 mm 정도) 넣으면 이해가 훨씬 빨라진다.

> 📷 **썸네일 자리 2-a**: Tray scan — head camera 화면 + 슬롯 ROI (GUI 분류기 탭 캡처) `docs/figures/fig2_thumb_scan.png`
> 📷 **썸네일 자리 2-b**: SmolVLA Pick — 손목 카메라 화면 `docs/figures/fig2_thumb_pick.png`
> 📷 **썸네일 자리 2-c**: Handover — 의사 손에 전달하는 순간 `docs/figures/fig2_thumb_handover.png`

**제작 도구**: draw.io (무료, PDF 내보내기 지원) 또는 PowerPoint → PDF로 저장. 상자 모서리 둥글기, 화살표 굵기를 모두 같게.

---

## Fig. 3 — Tray와 손잡이 기반 검증

**목적**: 왜 tip이 아니라 손잡이 + 색 띠를 보는지, 분류기 입력이 어떻게 생겼는지.

**크기**: 1단, 위아래 2단 구성

```
┌─────────────────────────────────────────┐
│ (a) 📷 head camera 화면 + 7개 슬롯 ROI     │
│     [S1][S2][S3][S4][S5][S6][S7]        │
├─────────────────────────────────────────┤
│ (b) 분류기 입력 crop 예시                  │
│  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐       │
│  │G ││B ││Sc││Cl││H ││Ir││SB││∅ │       │
│  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘       │
│  Grasp Bipol Sciss Clip Hook Irrig Bag Empty│
└─────────────────────────────────────────┘
```

> 📷 **사진 자리 3-a**: GUI 분류기 탭에서 ROI가 표시된 head camera 화면 캡처 `docs/figures/fig3_roi.png`
> - 화면 캡처는 화질이 낮으므로, `~/xlerobot_classifier_data/raw/`의 원본 사진에 ROI 사각형을 따로 그려 넣는 것을 권장 (스크립트로 생성 가능).

> 📷 **사진 자리 3-b**: `~/xlerobot_classifier_data/crops/<CLASS>/`에서 도구별 대표 crop 1장씩 + EMPTY 1장 `docs/figures/fig3_crops.png`
> - 같은 크기로 리사이즈해서 한 줄로 배치, 아래에 도구 이름.

**대안 (분량 부족 시)**: (a)를 Fig. 1의 inset으로 대체하고 (b)만 Fig. 6 혼동 행렬 옆에 붙인다.

---

## Fig. 4 — Wrong-Slot 정성 비교 ⭐

**목적**: 논문의 핵심 주장 — *"도구가 잘못 놓이면 Baseline은 틀린 도구를, Proposed는 맞는 도구를 집는다"* — 을 사진으로.

**크기**: 1단 또는 2단, **2행 × 3열** 사진 그리드

**시나리오**: 교환 A (Scissors ↔ Clipper). Slot 3에 Clipper, Slot 4에 Scissors. 요청 = Scissors.

```
              t = 0 (시작)          t = 접근              t = 집기 완료
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 Baseline  │📷 4-a        │ →  │📷 4-b        │ →  │📷 4-c     ✗  │
 "scissors"│ 배치 (교환됨) │    │ Slot 3로 이동 │    │ Clipper 집음  │
           └──────────────┘    └──────────────┘    └──────────────┘
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 Proposed  │📷 4-d        │ →  │📷 4-e        │ →  │📷 4-f     ✓  │
 "clippers"│ Tray scan    │    │ Slot 4로 이동 │    │ Scissors 집음 │
 (remapped)│ + ROI 결과    │    │              │    │              │
           └──────────────┘    └──────────────┘    └──────────────┘
```

> 📷 **사진 자리 4-a ~ 4-c (Baseline)**: `docs/figures/fig4_baseline_{0,1,2}.jpg`
> 📷 **사진 자리 4-d ~ 4-f (Proposed)**: `docs/figures/fig4_proposed_{0,1,2}.jpg`

**촬영 방법**

1. **head camera 영상**을 쓰는 것이 가장 깔끔하다 (같은 시점, 같은 구도). GUI에서 해당 시도를 데이터셋 녹화로 함께 기록하면 프레임을 바로 뽑을 수 있다.
   - 또는 외부 카메라(휴대폰 + 삼각대)로 위에서 촬영한 영상에서 프레임 추출.
2. 세 시점: 시작 / 팔이 목표 슬롯 바로 위 / 들어 올린 직후.
3. 각 사진에서 **목표 슬롯을 사각형으로 강조**: Baseline은 빨강 점선(Slot 3), Proposed는 파랑 실선(Slot 4).
4. 마지막 열 오른쪽 위에 ✗ / ✓ 표시.
5. 4-d에는 분류 결과를 작게 겹쳐 표시: `S3: Clipper  S4: Scissors`.

---

## Fig. 5 — 결과 그래프

**목적**: Table I의 핵심 대비를 시각적으로. (Table I과 중복이면 둘 중 하나만 싣는다 — 분량이 빠듯하면 Table만.)

**크기**: 1단, 높이 50–60 mm

**형태**: 그룹 막대 그래프 (x축: 배치 조건, 막대: 방법, 오차 막대: 95% Wilson CI)

```
 100% ┤
      │  ███ ▒▒▒                  ▒▒▒
  80% ┤  ███ ▒▒▒                  ▒▒▒
      │  ███ ▒▒▒                  ▒▒▒
      │  ███ ▒▒▒                  ▒▒▒
   0% ┼──███─▒▒▒──────────────▁───▒▒▒──
         Normal layout       Wrong-slot
      ███ Baseline (gray)   ▒▒▒ Proposed (blue)
```

**현재 확보된 수치** (실험 01, 02b)

| 조건 | 방법 | 성공 | 성공률 | 95% CI |
|---|---|---:|---:|---|
| Normal layout | Baseline | 117/140 | 83.6% | 76.6–88.8% |
| Wrong-slot | Baseline | 0/60 | 0% | 0–6% |
| Wrong-slot | Proposed — Pick 단계 (oracle slot) | 50/60 | 83.3% | 72.0–90.7% |
| Normal layout | Proposed (분류기 포함) | 실험 04 후 | | |
| Wrong-slot | Proposed (분류기 포함) | 실험 04 후 | | |

> ⚠️ 현재 Proposed 수치는 슬롯 정보를 사람이 준 **oracle** 조건이다. 최종 그래프에는 분류기를 포함한 통합 시스템 결과(실험 04)를 쓰고, oracle 결과는 별도 막대(연한 파랑, 빗금)로 두거나 본문에서만 언급한다.

**보조 그래프 (선택)**: 교환별 Remapping 성공률 — A 85% / B 85% / C 80% — 를 작은 막대로 붙이면 "팔 영역이 바뀌거나 모양이 달라도 유지된다"를 보여줄 수 있다.

**제작**: matplotlib → `savefig("fig5.pdf")`. 막대 폭·색·글꼴은 0장 규칙 따름. 0% 막대는 높이가 없으므로 막대 위에 `0%` 텍스트를 직접 표시.

---

## Fig. 6 — 분류기 혼동 행렬 (실험 03 후)

**목적**: 7개 도구 + EMPTY를 얼마나 정확히 구분하는지, 무엇끼리 헷갈리는지.

**크기**: 1단, 정사각형 또는 (a) 컬러 / (b) 흑백 두 개 나란히

```
            예측 →
         G  B  Sc Cl H  Ir SB ∅
 실제 G [■  .  .  .  .  .  .  .]
 ↓    B [.  ■  .  .  .  .  .  .]
      Sc[.  .  ■  ▪  .  .  .  .]   ← 헷갈린 칸만 진하게
      ...
```

- 칸 안에 비율(%) 표시, 색은 단일 계열(흰색 → 파랑).
- (a) 컬러 입력 vs (b) 흑백 입력을 나란히 두면 **색 띠의 기여**가 바로 보인다.
- 테스트 데이터는 **배치(arrangement) 단위로 분리**했음을 캡션에 명시.

---

## Table I — 메인 결과표

| Method | Normal layout | Wrong-slot | Wrong-tool handover |
|---|---:|---:|---:|
| Baseline (SmolVLA, requested name) | 83.6% (117/140) | 0% (0/60) | 100% |
| Proposed — oracle slot | — | 83.3% (50/60) | 0% |
| **Proposed (full system)** | **(실험 04)** | **(실험 04)** | **(실험 04)** |

- 각 칸에 성공 횟수/시도 횟수를 괄호로 함께 적는다 (n이 작으므로 필수).
- 본문에서 "Baseline과 Fixed-slot은 고정 배치 학습 시 동일하게 동작한다"는 점을 한 줄 설명하고 행을 하나로 합친다.

## Table II — End-to-End 단계별 성공률 (실험 05 후)

| 단계 | 성공 / 시도 | 성공률 |
|---|---:|---:|
| Speech recognition | | |
| Tool verification (classifier) | | |
| Pick | | |
| 90° rotation | | |
| Hand detection | | |
| Handover | | |
| **End-to-End** | | |

---

## 제작 순서 (추천)

1. **지금 바로**: Fig. 2 다이어그램 (실험 결과와 무관) → draw.io로 초안.
2. **분류기 데이터 수집할 때 같이**: Fig. 1, Fig. 3 사진 촬영 (같은 날, 같은 조명).
3. **실험 04 할 때 같이**: Fig. 4용 시도를 녹화 (Baseline 1회 + Proposed 1회만 있으면 됨).
4. **실험 03 후**: Fig. 6 혼동 행렬 (스크립트로 자동 생성).
5. **실험 04 / 05 후**: Fig. 5, Table I, II 최종 수치 채우기.

## 파일 정리

```
docs/figures/
├── raw/                    ← 원본 사진, 영상 프레임
├── fig1_setup.jpg
├── fig1_tray_inset.jpg
├── fig2_pipeline.pdf
├── fig3_roi.png
├── fig3_crops.png
├── fig4_baseline_{0,1,2}.jpg
├── fig4_proposed_{0,1,2}.jpg
├── fig5_results.pdf
└── fig6_confusion.pdf
```
