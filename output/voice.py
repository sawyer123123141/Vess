"""Serialized local text-to-speech output."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np


class VoiceOutput:
    """Synthesize and play one clause at a time on a dedicated worker."""

    def __init__(
        self,
        config: dict[str, Any],
        state: Any,
        event_log: Any,
        synthesize: Callable[[str], np.ndarray] | None = None,
        play: Callable[[np.ndarray, int], None] | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._event_log = event_log
        self._synthesize = synthesize
        self._play = play or _play_audio
        self._queue: queue.Queue[tuple[str, str | None] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._acknowledgement = "Yeah?"
        self._acknowledgement_audio: np.ndarray | None = None

    def start(self) -> None:
        """Start serial synthesis/playback."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="voice-output",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, text: str) -> None:
        """Speak text after every previously queued clause."""
        self._queue.put(("speak", text))

    def prepare_acknowledgement(self, text: str = "Yeah?") -> None:
        """Cache a short acknowledgement in the speech worker."""
        self._acknowledgement = text
        self._queue.put(("prepare", text))

    def enqueue_acknowledgement(self) -> None:
        """Play the prepared acknowledgement at the next queue position."""
        self._queue.put(("acknowledgement", None))

    def close(self) -> None:
        """Finish queued speech before stopping the worker."""
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, text = item
            if kind == "prepare":
                self._prepare(text or self._acknowledgement)
            elif kind == "acknowledgement":
                self._speak_acknowledgement()
            else:
                self._speak(text or "")

    def _prepare(self, text: str) -> None:
        try:
            self._acknowledgement_audio = self._synthesize_text(text)
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error)})

    def _speak_acknowledgement(self) -> None:
        if self._acknowledgement_audio is None:
            self._prepare(self._acknowledgement)
        if self._acknowledgement_audio is not None:
            self._play_waveform(self._acknowledgement_audio)

    def _speak(self, text: str) -> None:
        try:
            audio = self._synthesize_text(text)
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error), "text": text})
            return
        self._play_waveform(audio)

    def _play_waveform(self, audio: np.ndarray) -> None:
        with self._state.locked():
            self._state.speaking = True
        try:
            if audio.size:
                sample_rate = int(self._config.get("voice", {}).get("sample_rate", 24_000))
                self._play(audio, sample_rate)
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error)})
        finally:
            with self._state.locked():
                self._state.speaking = False

    def _synthesize_text(self, text: str) -> np.ndarray:
        if self._synthesize is None:
            self._synthesize = _make_synthesizer(self._config)
        return self._synthesize(text)


def _make_synthesizer(config: dict[str, Any]) -> Callable[[str], np.ndarray]:
    """Build the CPU Kokoro pipeline only inside the speech worker."""
    from kokoro import KPipeline

    voice = config.get("voice", {}).get("name", "af_heart")
    pipeline = KPipeline(lang_code="a", device="cpu")

    def synthesize(text: str) -> np.ndarray:
        parts: list[np.ndarray] = []
        for result in pipeline(text, voice=voice):
            audio = result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    return synthesize


def _play_audio(audio: np.ndarray, sample_rate: int) -> None:
    import sounddevice as sound_device

    sound_device.play(audio, sample_rate)
    sound_device.wait()
