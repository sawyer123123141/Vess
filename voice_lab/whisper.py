"""Whisper accuracy and timing metrics for Voice Lab."""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return word-level edit distance divided by reference word count."""
    left = _tokens(reference)
    right = _tokens(hypothesis)
    if not left:
        return 0.0 if not right else 1.0

    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1] / len(left)


def measure_transcription(
    samples: np.ndarray,
    reference: str,
    transcribe: Callable[[np.ndarray], str],
    sample_rate: int = 16_000,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Measure one transcription without microphone or conversation overhead."""
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    started = now()
    transcript = transcribe(waveform)
    elapsed = max(now() - started, 0.0)
    duration = waveform.size / sample_rate if sample_rate > 0 else 0.0
    return {
        "reference": reference,
        "transcript": transcript,
        "transcription_ms": round(elapsed * 1000.0, 3),
        "utterance_seconds": round(duration, 6),
        "realtime_factor": round(elapsed / duration, 6) if duration > 0 else None,
        "word_error_rate": round(word_error_rate(reference, transcript), 6),
    }


def _tokens(value: str) -> list[str]:
    normalized = "".join(
        character.lower() if character.isalnum() else " "
        for character in value
    )
    return normalized.split()
