"""Offline endpoint replay using Vess's production utterance assembler."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np

from perception.audio import UtteranceAssembler


def replay_endpoint(
    samples: np.ndarray,
    audio_settings: dict[str, object],
    silence_seconds: float,
    expected_utterances: int,
    block_samples: int = 320,
) -> dict[str, object]:
    """Replay one waveform and report how one silence threshold segments it."""
    if silence_seconds <= 0.0:
        raise ValueError("silence_seconds must be positive")
    if expected_utterances < 1:
        raise ValueError("expected_utterances must be at least 1")
    if block_samples < 1:
        raise ValueError("block_samples must be at least 1")

    sample_rate = int(audio_settings.get("sample_rate", 16_000))
    assembler = UtteranceAssembler(
        sample_rate=sample_rate,
        threshold=float(audio_settings.get("vad_threshold", 0.015)),
        min_utterance_seconds=float(audio_settings.get("min_utterance_seconds", 0.25)),
        silence_seconds=float(silence_seconds),
        max_utterance_seconds=float(audio_settings.get("max_utterance_seconds", 15.0)),
        pre_roll_seconds=float(audio_settings.get("pre_roll_seconds", 0.25)),
    )

    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    flush_samples = ceil(sample_rate * (silence_seconds + 0.05))
    stream = np.concatenate(
        [waveform, np.zeros(flush_samples, dtype=np.float32)]
    )

    emitted = 0
    for start in range(0, stream.size, block_samples):
        block = stream[start : start + block_samples]
        if assembler.push_with_timing(block) is not None:
            emitted += 1

    return {
        "silence_seconds": round(float(silence_seconds), 4),
        "configured_endpoint_ms": round(float(silence_seconds) * 1000.0, 1),
        "expected_utterances": int(expected_utterances),
        "emitted_utterances": emitted,
        "premature_split": emitted > expected_utterances,
        "missed_split": emitted < expected_utterances,
    }
