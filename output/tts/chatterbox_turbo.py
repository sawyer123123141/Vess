"""Optional lazy Chatterbox Turbo text-to-speech adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from output.tts.base import SynthesisResult
from performance import PerformanceCue


class ChatterboxTurboEngine:
    """Load Chatterbox Turbo only when synthesis is first requested."""

    def __init__(self, config: dict[str, Any]) -> None:
        chatterbox = config.get("voice", {}).get("chatterbox", {})
        self._device = str(chatterbox.get("device", "cuda"))
        self._reference_audio = str(chatterbox.get("reference_audio", "")).strip()
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
            except ImportError as error:
                raise RuntimeError(
                    "TTS engine 'chatterbox_turbo' requires the optional "
                    "chatterbox-tts package"
                ) from error

            self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)
        return self._model

    def synthesize(
        self,
        text: str,
        performance: PerformanceCue,
    ) -> SynthesisResult:
        model = self._get_model()
        kwargs: dict[str, object] = {}
        if self._reference_audio:
            kwargs["audio_prompt_path"] = self._reference_audio

        waveform = model.generate(text, **kwargs)
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        return SynthesisResult(audio, int(model.sr))
