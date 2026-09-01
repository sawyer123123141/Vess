"""Audio segmentation and wake-phrase matching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
import queue
import threading
import time
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
        self._pre_roll_floor = threshold * 0.25
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
                self._samples.extend(self._useful_pre_roll())
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

    def _useful_pre_roll(self) -> list[float]:
        """Keep contiguous pre-trigger audio once it rises above deep silence."""
        buffered = list(self._pre_roll)
        for index, sample in enumerate(buffered):
            if abs(sample) >= self._pre_roll_floor:
                return buffered[index:]
        return []

    def _finish(self) -> np.ndarray | None:
        final_index = len(self._samples) - self._trailing_quiet
        utterance = np.asarray(self._samples[:final_index])
        self._samples = []
        self._trailing_quiet = 0
        if len(utterance) < self._min_samples:
            return None
        return utterance


class AudioLoop:
    """Capture utterances and dispatch wake phrases or active-conversation followups."""

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
        self._conversation_timeout = float(
            settings.get("conversation_timeout_seconds", 30.0)
        )
        self._conversation_until = 0.0
        self._blocks: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=16)
        # One utterance may be in Whisper and only the newest may wait behind it.
        self._utterances: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue(
            maxsize=1
        )
        self._stop = threading.Event()
        self._transcriber_failed = threading.Event()
        self._thread: threading.Thread | None = None
        self._transcribe_thread: threading.Thread | None = None
        self._stream: Any = None
        self._assembler = UtteranceAssembler(
            int(settings.get("sample_rate", 16_000)),
            float(settings.get("vad_threshold", 0.015)),
            float(settings.get("min_utterance_seconds", 0.25)),
            float(settings.get("silence_seconds", 0.45)),
            float(settings.get("max_utterance_seconds", 15.0)),
            float(settings.get("pre_roll_seconds", 0.25)),
        )

    def handle_utterance(
        self,
        samples: np.ndarray,
        *,
        speech_ended_at: float | None = None,
    ) -> None:
        """Transcribe one utterance and dispatch it if the conversation gate allows."""
        transcription_started = time.perf_counter()
        try:
            if self._transcribe is None:
                raise RuntimeError("audio loop was not started")
            transcript = self._transcribe(samples).strip()
        except Exception as error:
            self._event_log.append("audio_error", {"error": str(error)})
            self._state.record_debug("audio_error", error=str(error))
            return

        transcription_finished = time.perf_counter()
        if speech_ended_at is not None:
            transcription_ms = (transcription_finished - transcription_started) * 1000.0
            speech_to_transcript_ms = (transcription_finished - speech_ended_at) * 1000.0
            self._state.update_debug(
                transcription_ms=round(transcription_ms, 1),
                speech_to_transcript_ms=round(speech_to_transcript_ms, 1),
            )

        self._state.record_debug("transcript", transcript=transcript)
        if not transcript:
            payload = {"transcript": "", "reason": "no_speech"}
            self._event_log.append("wake_rejected", payload)
            self._state.record_debug("no_speech_rejected", **payload)
            return

        closest = match_wake_phrase(transcript, self._variants, 1_000_000)
        accepted = (
            closest if closest is not None and closest.distance <= self._max_distance else None
        )
        payload = _wake_payload(transcript, closest)
        if accepted is not None:
            self._open_conversation()
            self._event_log.append("wake_accepted", payload)
            self._state.record_debug("wake_accepted", **payload)
            request = " ".join(transcript.split()[accepted.consumed_words:])
            self._on_request(request)
            return

        if self._conversation_is_active():
            self._open_conversation()
            followup_payload = {"transcript": transcript}
            self._event_log.append("followup_accepted", followup_payload)
            self._state.record_debug("followup_accepted", **followup_payload)
            self._on_request(transcript)
            return

        self._event_log.append("wake_rejected", payload)
        self._state.record_debug("wake_rejected", **payload)

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
        """Stop capture and wait for the audio workers to exit."""
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
        if self._transcribe_thread is not None:
            self._transcribe_thread.join()
            self._transcribe_thread = None

    def _on_audio(self, indata: np.ndarray, *_: object) -> None:
        try:
            self._blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _run(self) -> None:
        self._ensure_transcribe_thread()
        try:
            while not self._stop.is_set():
                block = self._blocks.get()
                if block is None:
                    return
                with self._state.locked():
                    speaking = self._state.speaking
                if speaking:
                    with self._state.locked():
                        self._state.listening = False
                    self._state.update_debug(audio_ignored=True)
                    continue

                utterance = self._assembler.push(block)
                status = self._assembler.status()
                with self._state.locked():
                    self._state.listening = bool(status["vad_active"])
                peak = float(np.max(np.abs(block))) if block.size else 0.0
                self._state.update_debug(
                    audio_ignored=False,
                    mic_peak=round(peak, 4),
                    transcription_queue=self._utterances.qsize(),
                    **status,
                )
                if utterance is not None:
                    if self._transcriber_failed.is_set():
                        self._state.record_debug(
                            "audio_dropped", reason="transcriber_unavailable"
                        )
                    else:
                        self._queue_latest_utterance(utterance, time.perf_counter())
        finally:
            with self._state.locked():
                self._state.listening = False
            self._stop_transcription_queue()

    def _queue_latest_utterance(
        self,
        utterance: np.ndarray,
        speech_ended_at: float,
    ) -> None:
        item = (utterance, speech_ended_at)
        while True:
            try:
                self._utterances.put_nowait(item)
                self._state.update_debug(transcription_queue=self._utterances.qsize())
                return
            except queue.Full:
                try:
                    replaced = self._utterances.get_nowait()
                except queue.Empty:
                    continue
                if replaced is None:
                    self._utterances.put_nowait(None)
                    return
                _, replaced_at = replaced
                age_ms = (time.perf_counter() - replaced_at) * 1000.0
                self._state.record_debug(
                    "pending_utterance_replaced",
                    age_ms=round(age_ms, 1),
                )

    def _stop_transcription_queue(self) -> None:
        while True:
            try:
                self._utterances.get_nowait()
            except queue.Empty:
                break
        try:
            self._utterances.put_nowait(None)
        except queue.Full:
            pass

    def _ensure_transcribe_thread(self) -> None:
        if self._transcribe_thread is not None:
            return
        self._transcribe_thread = threading.Thread(
            target=self._run_transcription,
            name="audio-transcribe",
            daemon=True,
        )
        self._transcribe_thread.start()

    def _run_transcription(self) -> None:
        if self._transcribe is None:
            try:
                self._transcribe = _make_transcriber(self._config)
            except Exception as error:
                self._transcriber_failed.set()
                self._event_log.append("audio_error", {"error": str(error)})
                self._state.record_debug("audio_error", error=str(error))
                return

        while True:
            item = self._utterances.get()
            if item is None:
                return
            utterance, speech_ended_at = item
            age_ms = (time.perf_counter() - speech_ended_at) * 1000.0
            self._state.update_debug(
                transcription_queue=self._utterances.qsize(),
                utterance_age_ms=round(age_ms, 1),
            )
            self.handle_utterance(utterance, speech_ended_at=speech_ended_at)

    def _open_conversation(self) -> None:
        if self._conversation_timeout <= 0.0:
            return
        self._conversation_until = time.time() + self._conversation_timeout
        self._state.update_debug(conversation_active=True)

    def _conversation_is_active(self) -> bool:
        if self._conversation_timeout <= 0.0:
            return False
        with self._state.locked():
            last_spoke = float(self._state.last_spoke)
        deadline = max(
            self._conversation_until,
            last_spoke + self._conversation_timeout,
        )
        active = time.time() <= deadline
        self._state.update_debug(conversation_active=active)
        return active


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
        segments, _ = model.transcribe(
            samples,
            language=settings.get("language", "en"),
            beam_size=int(settings.get("beam_size", 1)),
            condition_on_previous_text=bool(
                settings.get("condition_on_previous_text", False)
            ),
            vad_filter=bool(settings.get("vad_filter", True)),
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe
