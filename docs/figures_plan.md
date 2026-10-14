# 메인 Figure 제작 계획 (ICEIC 2027)

논문 첫 페이지 상단에 들어갈 **메인 Figure 한 장**. 이 그림만 보고도 "무엇을 만들었고, 무엇이 새롭고, 효과가 있는지"가 보여야 한다. `📷`은 직접 찍은 사진으로 채운다.

## 구성

2단 폭 **181 mm × 약 60 mm**, 세 칸을 가로로 놓는다.

```
┌──────────────┬─────────────────────────────────────────┬────────────────────────┐
│ (a) Setup    │ (b) Pipeline                            │ (c) "Scissors, please" │
│              │                                         │    (Scissors↔Clipper)  │
│   📷 a        │ Voice → [Verify & Recover] → Pick       │  Baseline   Proposed   │
│ 로봇 + 트레이  │            (파랑 강조)       → Turn 90°  │   📷 c1       📷 c2      │
│  + 의사 손     │                            → Handover   │    ✗          ✓        │
│              │                                         │   0/60     (실험 04)    │
└──────────────┴─────────────────────────────────────────┴────────────────────────┘
```

### (a) Setup — 폭 약 45 mm
> 📷 **a**: 로봇, 트레이(7개 도구, 색 띠가 보이게), 의사 손이 한 장에 들어오는 45° 위쪽 샷 — `docs/figures/main_a.jpg`

글씨는 넣지 않거나 최소한으로 (Robot / Tray / Surgeon 정도).

### (b) Pipeline — 폭 약 90 mm
상자 5개를 화살표로 잇는다.

| 상자 | 표시 |
|---|---|
| Voice request | 회색 |
| **Verify & Recover** | **파란 굵은 테두리 + 연한 파란 배경** (제안 방법) |
| SmolVLA pick | 회색 |
| 90° turn | 회색 |
| Hand-guided handover | 회색 |

Verify & Recover 상자 아래에 한 줄만:
`tray scan → Scissors found in Slot 4 → "Pick up the clippers"`

### (c) Result — 폭 약 45 mm
같은 교환 배치(Slot 3 = Clipper, Slot 4 = Scissors)에서 같은 요청 "Scissors"에 대한 **집기 완료 순간**.

> 📷 **c1** Baseline: Slot 3의 Clipper를 집음, 빨간 ✗ — `docs/figures/main_c1.jpg`
> 📷 **c2** Proposed: Slot 4의 Scissors를 집음, 파란 ✓ — `docs/figures/main_c2.jpg`

사진 아래 숫자 한 줄: Baseline `0/60` · Proposed `실험 04 결과` (확정 전에는 oracle `50/60`을 임시로).

## 캡션 초안

*Fig. 1. (a) Setup. (b) Pipeline: the requested instrument is verified on the tray by a handle classifier; if misplaced, the instruction to the fixed-layout policy is remapped to its actual slot before picking, turning, and handing it over. (c) With two instruments swapped, the baseline picks the wrong one (0/60), while the proposed method picks the requested one.*

## 만드는 법

1. **(b)** draw.io로 상자 5개 + 화살표 → 지금 바로 가능.
2. **(a)** 실험 환경 사진 한 장 (삼각대, 배경 정리, 얼굴은 프레임 밖).
3. **(c)** 실험 04 진행 중에 Baseline 1회, Proposed 1회를 데이터셋으로 녹화 → head camera 영상에서 집기 완료 프레임을 뽑아 목표 슬롯에 박스(빨강 점선 / 파랑 실선)를 그린다.
4. PowerPoint나 Inkscape에서 181 mm 폭 캔버스에 세 칸을 배치 → **PDF로 저장**.

## 지킬 것

- 처음부터 실제 크기(181 mm)로 만든다. 축소 후 글씨 **8 pt 이상**.
- 색은 두 가지만: **Baseline = 회색/빨강, Proposed = 파랑**.
- 사진은 300 dpi 이상 (45 mm 폭이면 가로 약 550 px 이상).
