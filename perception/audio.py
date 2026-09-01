"""Audio segmentation and wake-phrase matching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
import queue
import threading
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class WakeMatch:
    variant: str
    distance: int
    consumed_words: int


class UtteranceAssembler:
    """Collect audible microphone samples into bounded utterances."""

    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        min_utterance_seconds: float,
        silence_seconds: float,
        max_utterance_seconds: float,
        pre_roll_seconds: float = 0.25,
    ) -> None:
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_samples = ceil(sample_rate * min_utterance_seconds)
        self._silence_samples = ceil(sample_rate * silence_seconds)
        self._max_samples = ceil(sample_rate * max_utterance_seconds)
        self._pre_roll: deque[float] = deque(
            maxlen=max(ceil(sample_rate * pre_roll_seconds), 0)
        )
        self._samples: list[float] = []
        self._trailing_quiet = 0

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        """Return one finished utterance, if this block completes one."""
        for sample in np.asarray(samples).reshape(-1):
            amplitude = abs(float(sample))
            if not self._samples:
                if amplitude < self._threshold:
                    self._pre_roll.append(float(sample))
                    continue
                self._samples.extend(self._pre_roll)
                self._pre_roll.clear()
                self._samples.append(sample)
                self._trailing_quiet = 0
                continue

            self._samples.append(sample)
            if amplitude < self._threshold:
                self._trailing_quiet += 1
            else:
                self._trailing_quiet = 0

            if len(self._samples) >= self._max_samples:
                return self._finish()
            if self._trailing_quiet >= self._silence_samples:
                return self._finish()
        return None

    def status(self) -> dict[str, object]:
        """Expose current VAD assembly without exposing its sample buffer."""
        return {
            "vad_active": bool(self._samples),
            "vad_seconds": len(self._samples) / self._sample_rate,
        }

    def _finish(self) -> np.ndarray | None:
        final_index = len(self._samples) - self._trailing_quiet
        utterance = np.asarray(self._samples[:final_index])
        self._samples = []
        self._trailing_quiet = 0
        if len(utterance) < self._min_samples:
            return None
        return utterance


class AudioLoop:
    """Capture utterances and dispatch only fuzzy wake-phrase requests."""

    def __init__(
        self,
        config: dict[str, Any],
        state: Any,
        event_log: Any,
        on_request: Callable[[str], None],
        transcribe: Callable[[np.ndarray], str] | None = None,
    ) -> None:
        settings = config.get("audio", {})
        self._config = config
        self._state = state
        self._event_log = event_log
        self._on_request = on_request
        self._transcribe = transcribe
        self._variants = list(settings.get("wake_variants", ["hey vess"]))
        self._max_distance = int(settings.get("wake_max_distance", 2))
        self._blocks: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._assembler = UtteranceAssembler(
            int(settings.get("sample_rate", 16_000)),
            float(settings.get("vad_threshold", 0.015)),
            float(settings.get("min_utterance_seconds", 0.25)),
            float(settings.get("silence_seconds", 0.8)),
            float(settings.get("max_utterance_seconds", 15.0)),
            float(settings.get("pre_roll_seconds", 0.25)),
        )

    def handle_utterance(self, samples: np.ndarray) -> None:
        """Transcribe one utterance and either log rejection or dispatch it."""
        with self._state.locked():
            self._state.listening = True
        try:
            if self._transcribe is None:
                raise RuntimeError("audio loop was not started")
            transcript = self._transcribe(samples).strip()
        except Exception as error:
            self._event_log.append("audio_error", {"error": str(error)})
            self._state.record_debug("audio_error", error=str(error))
            return
        finally:
            with self._state.locked():
                self._state.listening = False

        self._state.record_debug("transcript", transcript=transcript)
        closest = match_wake_phrase(transcript, self._variants, 1_000_000)
        accepted = (
            closest if closest is not None and closest.distance <= self._max_distance else None
        )
        payload = _wake_payload(transcript, closest)
        if accepted is None:
            self._event_log.append("wake_rejected", payload)
            self._state.record_debug("wake_rejected", **payload)
            return

        self._event_log.append("wake_accepted", payload)
        self._state.record_debug("wake_accepted", **payload)
        request = " ".join(transcript.split()[accepted.consumed_words:])
        self._on_request(request)

    def start(self) -> None:
        """Open the microphone and begin processing queued audio blocks."""
        if self._thread is not None:
            return

        settings = self._config.get("audio", {})
        try:
            import sounddevice as sound_device

            self._stream = sound_device.InputStream(
                device=settings.get("device"),
                samplerate=int(settings.get("sample_rate", 16_000)),
                channels=int(settings.get("channels", 1)),
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as error:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            raise RuntimeError(f"cannot open audio device: {error}") from error

        self._thread = threading.Thread(
            target=self._run,
            name="audio-loop",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop capture and wait for the audio worker to exit."""
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            try:
                self._blocks.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join()
            self._thread = None

    def _on_audio(self, indata: np.ndarray, *_: object) -> None:
        try:
            self._blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _run(self) -> None:
        if self._transcribe is None:
            try:
                self._transcribe = _make_transcriber(self._config)
            except Exception as error:
                self._event_log.append("audio_error", {"error": str(error)})
                self._state.record_debug("audio_error", error=str(error))
                return
        while not self._stop.is_set():
            block = self._blocks.get()
            if block is None:
                return
            with self._state.locked():
                speaking = self._state.speaking
            if speaking:
                self._state.update_debug(audio_ignored=True)
                continue
            utterance = self._assembler.push(block)
            peak = float(np.max(np.abs(block))) if block.size else 0.0
            self._state.update_debug(
                audio_ignored=False,
                mic_peak=round(peak, 4),
                **self._assembler.status(),
            )
            if utterance is not None:
                self.handle_utterance(utterance)


def match_wake_phrase(
    transcript: str,
    variants: list[str],
    max_distance: int,
) -> WakeMatch | None:
    """Return the closest configured wake phrase at the utterance start."""
    words = _normalise(transcript).split()
    if not words or max_distance < 0:
        return None

    best_match: WakeMatch | None = None
    for variant in variants:
        normalised_variant = _normalise(variant)
        if not normalised_variant:
            continue
        for word_count in range(1, min(3, len(words)) + 1):
            prefix = " ".join(words[:word_count])
            distance = _levenshtein(prefix, normalised_variant)
            candidate = WakeMatch(variant, distance, word_count)
            if distance > max_distance:
                continue
            if best_match is None or (distance, word_count) < (
                best_match.distance,
                best_match.consumed_words,
            ):
                best_match = candidate
    return best_match


def _normalise(value: str) -> str:
    characters = (
        character.lower() if character.isalnum() else " "
        for character in value
    )
    return " ".join("".join(characters).split())


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _wake_payload(transcript: str, match: WakeMatch | None) -> dict[str, object]:
    word_count = match.consumed_words if match is not None else 3
    return {
        "transcript": transcript,
        "tested_prefix": " ".join(_normalise(transcript).split()[:word_count]),
        "closest_variant": match.variant if match is not None else None,
        "distance": match.distance if match is not None else None,
    }


def _make_transcriber(config: dict[str, Any]) -> Callable[[np.ndarray], str]:
    """Create the local CPU/int8 Whisper transcriber only for live capture."""
    from faster_whisper import WhisperModel

    settings = config.get("whisper", {})
    model = WhisperModel(
        settings.get("model", "small"),
        device=settings.get("device", "cpu"),
        compute_type=settings.get("compute_type", "int8"),
    )

    def transcribe(samples: np.ndarray) -> str:
        segments, _ = model.transcribe(samples)
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe
