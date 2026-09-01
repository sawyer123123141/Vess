"""Lightweight text-to-speech engine contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from performance import PerformanceCue


@dataclass(frozen=True)
class SynthesisResult:
    """One synthesized mono waveform and its authoritative sample rate."""

    audio: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.audio, np.ndarray):
            raise ValueError("audio must be a NumPy array")
        if self.audio.dtype != np.float32:
            raise ValueError("audio must use float32 dtype")
        if self.audio.ndim != 1:
            raise ValueError("audio must be one-dimensional")
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")


class TTSEngine(Protocol):
    """Model-specific synthesis behind a tiny production boundary."""

    def synthesize(
        self,
        text: str,
        performance: PerformanceCue,
    ) -> SynthesisResult:
        """Synthesize one clause using the supplied transient performance cue."""
        raise NotImplementedError
