"""Reusable fake TTS engines for model-free voice tests."""

from collections.abc import Callable

import numpy as np

from output.tts.base import SynthesisResult
from performance import PerformanceCue


class FakeTTSEngine:
    """Record synthesis calls and return deterministic model-free audio."""

    def __init__(
        self,
        synthesize: Callable[[str, PerformanceCue], SynthesisResult] | None = None,
        *,
        sample_rate: int = 24_000,
    ) -> None:
        self.calls: list[tuple[str, PerformanceCue]] = []
        self._sample_rate = sample_rate
        self._synthesize = synthesize

    def synthesize(self, text: str, performance: PerformanceCue) -> SynthesisResult:
        self.calls.append((text, performance))
        if self._synthesize is not None:
            return self._synthesize(text, performance)
        return SynthesisResult(np.ones(1, dtype=np.float32), self._sample_rate)
