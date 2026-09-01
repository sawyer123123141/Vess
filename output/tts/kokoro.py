"""Lazy Kokoro text-to-speech adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from output.tts.base import SynthesisResult
from performance import PerformanceCue


class KokoroEngine:
    """CPU Kokoro engine matching Vess's existing synthesis behavior."""

    SAMPLE_RATE = 24_000

    def __init__(self, config: dict[str, Any]) -> None:
        self._voice = str(config.get("voice", {}).get("name", "af_heart"))
        self._pipeline: Any | None = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code="a", device="cpu")
        return self._pipeline

    def synthesize(
        self,
        text: str,
        performance: PerformanceCue,
    ) -> SynthesisResult:
        parts: list[np.ndarray] = []
        for result in self._get_pipeline()(text, voice=self._voice):
            audio = result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        joined = (
            np.concatenate(parts)
            if parts
            else np.array([], dtype=np.float32)
        )
        return SynthesisResult(joined, self.SAMPLE_RATE)
