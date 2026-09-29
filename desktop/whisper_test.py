#!/usr/bin/env python3
"""Quick mic -> faster-whisper -> canonical tool-name test.

Press ENTER, speak the tool name, press ENTER again to stop recording.
Ctrl+C to quit.
"""
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

# Canonical label -> phrases that should map to it (lowercased substring match)
TOOL_ALIASES = {
    "GRASPER": ["grasper", "그라스퍼", "그래스퍼"],
    "BIPOLAR": ["bipolar", "바이폴라"],
    "HOOK": ["hook", "훅"],
    "CLIPPER": ["clipper", "클리퍼"],
    "SCISSORS": ["scissors", "scissor", "시저", "가위"],
    "IRRIGATOR": ["irrigator", "이리게이터", "이리게이터", "석션"],
    "SPECIMEN_BAG": ["specimen bag", "스페시먼백", "스페시먼 백", "백"],
}


def to_canonical(text: str) -> str | None:
    t = text.lower()
    for canonical, aliases in TOOL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in t:
                return canonical
    return None


def record_until_enter() -> np.ndarray:
    print("ENTER를 누르면 녹음을 시작합니다...")
    input()
    print("녹음 중... 말한 뒤 ENTER를 눌러 멈추세요.")
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).flatten()


def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "small"
    print(f"Whisper 모델 로드 중 ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("준비 완료.\n")

    while True:
        try:
            audio = record_until_enter()
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        if audio.size < SAMPLE_RATE * 0.3:
            print("너무 짧습니다. 다시 시도하세요.\n")
            continue

        segments, info = model.transcribe(audio, language=None, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        canonical = to_canonical(text)

        print(f"  인식된 언어: {info.language} (확률 {info.language_probability:.2f})")
        print(f"  인식된 텍스트: \"{text}\"")
        print(f"  매칭된 도구: {canonical if canonical else '❌ 매칭 실패'}")
        print()


if __name__ == "__main__":
    main()
