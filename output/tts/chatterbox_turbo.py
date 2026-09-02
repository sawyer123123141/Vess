"""Optional lazy Chatterbox Turbo text-to-speech adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from output.tts.base import SynthesisCancelled, SynthesisResult
from performance import PerformanceCue


_PLAYFUL_CHUCKLE_MIN_INTENSITY = 0.60


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
            if self._reference_audio:
                self._model.prepare_conditionals(self._reference_audio)
        return self._model

    def synthesize(
        self,
        text: str,
        performance: PerformanceCue,
    ) -> SynthesisResult:
        model = self._get_model()
        prepared = self._prepare_text(text, performance)
        return self._result(model, model.generate(prepared))

    def synthesize_cancellable(
        self,
        text: str,
        performance: PerformanceCue,
        should_cancel: Callable[[], bool],
    ) -> SynthesisResult:
        """Abort obsolete Turbo work at safe Python boundaries inside generation."""
        self._raise_if_cancelled(should_cancel)
        model = self._get_model()
        transformer = getattr(getattr(model, "t3", None), "tfmr", None)
        hook_handle = None
        restored_methods: list[tuple[object, str, object]] = []

        if transformer is not None and hasattr(transformer, "register_forward_pre_hook"):
            def cancel_before_transformer(_module: object, _inputs: object) -> None:
                self._raise_if_cancelled(should_cancel)

            hook_handle = transformer.register_forward_pre_hook(
                cancel_before_transformer
            )

        s3gen = getattr(model, "s3gen", None)
        if s3gen is not None:
            for method_name in ("flow_inference", "hift_inference"):
                original = getattr(s3gen, method_name, None)
                if not callable(original):
                    continue

                def checked_stage(
                    *args: object,
                    _original: Callable[..., object] = original,
                    **kwargs: object,
                ) -> object:
                    self._raise_if_cancelled(should_cancel)
                    return _original(*args, **kwargs)

                setattr(s3gen, method_name, checked_stage)
                restored_methods.append((s3gen, method_name, original))

        try:
            prepared = self._prepare_text(text, performance)
            waveform = model.generate(prepared)
            self._raise_if_cancelled(should_cancel)
        finally:
            if hook_handle is not None:
                hook_handle.remove()
            for owner, method_name, original in restored_methods:
                setattr(owner, method_name, original)

        return self._result(model, waveform)

    @staticmethod
    def _prepare_text(text: str, performance: PerformanceCue) -> str:
        """Apply only expressive tokens that are safe enough for live speech."""
        if (
            text
            and performance.expression == "playful"
            and performance.intensity >= _PLAYFUL_CHUCKLE_MIN_INTENSITY
        ):
            return f"{text} [chuckle]"
        return text

    @staticmethod
    def _raise_if_cancelled(should_cancel: Callable[[], bool]) -> None:
        if should_cancel():
            raise SynthesisCancelled("synthesis cancelled")

    @staticmethod
    def _result(model: Any, waveform: object) -> SynthesisResult:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        return SynthesisResult(audio, int(model.sr))
