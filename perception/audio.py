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

from perception.audio_preprocess import CapturePreprocessor, CapturedAudioBlock


@dataclass(frozen=True)
class WakeMatch:
    variant: str
    distance: int
    consumed_words: int


@dataclass(frozen=True)
class _QueuedUtterance:
    samples: np.ndarray
    speech_ended_at: float | None
    utterance_finalized_at: float
    barge_in: bool = False


@dataclass(frozen=True)
class CompletedUtterance:
    samples: np.ndarray
    speech_end_offset_samples: int | None


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
        self._timing_valid = True

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        """Return one finished utterance, if this block completes one."""
        completed = self.push_with_timing(samples)
        return completed.samples if completed is not None else None

    def push_with_timing(self, samples: np.ndarray) -> CompletedUtterance | None:
        """Return speech plus its last-voiced position within the input block."""
        block = np.asarray(samples).reshape(-1)
        for index, sample in enumerate(block):
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
                return self._finish(block.size - index - 1)
            if self._trailing_quiet >= self._silence_samples:
                return self._finish(block.size - index - 1)
        return None

    def status(self) -> dict[str, object]:
        """Expose current VAD assembly without exposing its sample buffer."""
        return {
            "vad_active": bool(self._samples),
            "vad_seconds": len(self._samples) / self._sample_rate,
        }

    def reset(self) -> None:
        """Forget any partial utterance without emitting it."""
        self._pre_roll.clear()
        self._samples = []
        self._trailing_quiet = 0
        self._timing_valid = True

    def invalidate_timing(self) -> None:
        """Mark an active utterance's sample-derived endpoint as unreliable."""
        if self._samples:
            self._timing_valid = False
        else:
            self._pre_roll.clear()

    def _useful_pre_roll(self) -> list[float]:
        """Keep contiguous pre-trigger audio once it rises above deep silence."""
        buffered = list(self._pre_roll)
        for index, sample in enumerate(buffered):
            if abs(sample) >= self._pre_roll_floor:
                return buffered[index:]
        return []

    def _finish(self, remaining_block_samples: int) -> CompletedUtterance | None:
        final_index = len(self._samples) - self._trailing_quiet
        utterance = np.asarray(self._samples[:final_index], dtype=np.float32)
        speech_end_offset_samples = None
        if self._timing_valid:
            speech_end_offset_samples = self._trailing_quiet + max(
                int(remaining_block_samples),
                0,
            )
        self._samples = []
        self._trailing_quiet = 0
        self._timing_valid = True
        if len(utterance) < self._min_samples:
            return None
        return CompletedUtterance(utterance, speech_end_offset_samples)


class AudioLoop:
    """Capture utterances and dispatch wake phrases or active-conversation followups."""

    def __init__(
        self,
        config: dict[str, Any],
        state: Any,
        event_log: Any,
        on_request: Callable[[str], None],
        transcribe: Callable[[np.ndarray], str] | None = None,
        *,
        on_timed_request: Callable[[str, dict[str, object]], None] | None = None,
        preprocessor: CapturePreprocessor | None = None,
        interruption_detector: Any | None = None,
        turn_coordinator: Any | None = None,
    ) -> None:
        settings = config.get("audio", {})
        barge_in_settings = config.get("barge_in", {})
        self._config = config
        self._state = state
        self._event_log = event_log
        self._on_request = on_request
        self._on_timed_request = on_timed_request
        self._transcribe = transcribe
        self._sample_rate = int(settings.get("sample_rate", 16_000))
        self._variants = list(settings.get("wake_variants", ["hey vess"]))
        self._max_distance = int(settings.get("wake_max_distance", 2))
        self._conversation_timeout = float(
            settings.get("conversation_timeout_seconds", 30.0)
        )
        self._conversation_until = 0.0

        self._preprocessor = preprocessor
        self._interruption_detector = interruption_detector
        self._turn_coordinator = turn_coordinator
        self._barge_in_enabled = bool(barge_in_settings.get("enabled", False))
        self._barge_in_available = self._barge_in_enabled and all(
            dependency is not None
            for dependency in (
                self._preprocessor,
                self._interruption_detector,
                self._turn_coordinator,
            )
        )
        self._disable_on_preprocessor_error = bool(
            barge_in_settings.get("disable_on_preprocessor_error", True)
        )
        self._barge_in_capture_active = False
        self._barge_in_decision_pending = threading.Event()
        self._capture_sequence = 0
        self._last_capture_sequence: int | None = None
        self._audio_blocks_dropped = 0

        self._blocks: queue.Queue[CapturedAudioBlock | np.ndarray | None] = queue.Queue(
            maxsize=16
        )
        # One utterance may be in Whisper and only the newest may wait behind it.
        self._utterances: queue.Queue[_QueuedUtterance | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._transcriber_failed = threading.Event()
        self._thread: threading.Thread | None = None
        self._transcribe_thread: threading.Thread | None = None
        self._stream: Any = None
        self._assembler = self._make_assembler(settings)
        self._barge_in_assembler = self._make_assembler(settings)

    def handle_utterance(
        self,
        samples: np.ndarray,
        *,
        speech_ended_at: float | None = None,
        utterance_finalized_at: float | None = None,
    ) -> None:
        """Transcribe one utterance and dispatch it if the conversation gate allows."""
        transcription_started = time.perf_counter()
        try:
            if self._transcribe is None:
                raise RuntimeError("audio loop was not started")
            transcript = self._transcribe(samples).strip()
        except Exception as error:
            self._record_audio_error(error)
            return

        transcription_finished = time.perf_counter()
        timing = self._transcription_timing(
            transcription_started,
            transcription_finished,
            speech_ended_at,
            utterance_finalized_at,
            int(np.asarray(samples).size),
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
            self._dispatch_request(request, timing)
            return

        if self._conversation_is_active():
            self._open_conversation()
            followup_payload = {"transcript": transcript}
            self._event_log.append("followup_accepted", followup_payload)
            self._state.record_debug("followup_accepted", **followup_payload)
            self._dispatch_request(transcript, timing)
            return

        self._event_log.append("wake_rejected", payload)
        self._state.record_debug("wake_rejected", **payload)

    def handle_barge_in_utterance(
        self,
        samples: np.ndarray,
        *,
        speech_ended_at: float | None = None,
        utterance_finalized_at: float | None = None,
    ) -> None:
        """Transcribe overlapping speech directly into the pending turn decision."""
        coordinator = self._turn_coordinator
        if coordinator is None:
            self._barge_in_decision_pending.clear()
            return

        transcription_started = time.perf_counter()
        try:
            if self._transcribe is None:
                raise RuntimeError("audio loop was not started")
            transcript = self._transcribe(samples).strip()
        except Exception as error:
            self._record_audio_error(error)
            try:
                coordinator.on_transcription_error(error)
            finally:
                self._barge_in_decision_pending.clear()
            return

        transcription_finished = time.perf_counter()
        timing = self._transcription_timing(
            transcription_started,
            transcription_finished,
            speech_ended_at,
            utterance_finalized_at,
            int(np.asarray(samples).size),
        )
        self._state.record_debug(
            "transcript",
            transcript=transcript,
            source="barge_in",
        )
        try:
            coordinator.on_transcript(
                transcript,
                timing=timing,
            )
        finally:
            self._barge_in_decision_pending.clear()

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

    def _on_audio(
        self,
        indata: np.ndarray,
        _frames: object = None,
        time_info: object = None,
        _status: object = None,
    ) -> None:
        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
        self._capture_sequence += 1
        block = CapturedAudioBlock(
            samples=samples,
            adc_time=_input_adc_time(time_info),
            received_at=time.perf_counter(),
            capture_sequence=self._capture_sequence,
        )
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            self._audio_blocks_dropped += 1

    def _run(self) -> None:
        self._ensure_transcribe_thread()
        try:
            while not self._stop.is_set():
                queued_block = self._blocks.get()
                if queued_block is None:
                    return
                block = self._coerce_capture_block(queued_block)
                self._observe_capture_sequence(block)
                with self._state.locked():
                    speaking = self._state.speaking

                if self._barge_in_decision_pending.is_set():
                    self._state.update_debug(
                        audio_ignored=True,
                        barge_in_decision_pending=True,
                    )
                    continue

                if speaking or self._barge_in_capture_active:
                    if self._barge_in_available:
                        self._handle_speaking_block(block)
                    else:
                        if self._barge_in_capture_active:
                            self._reset_barge_in_capture()
                        with self._state.locked():
                            self._state.listening = False
                        self._state.update_debug(audio_ignored=True)
                    continue

                utterance = self._assembler.push_with_timing(block.samples)
                status = self._assembler.status()
                with self._state.locked():
                    self._state.listening = bool(status["vad_active"])
                peak = float(np.max(np.abs(block.samples))) if block.samples.size else 0.0
                self._state.update_debug(
                    audio_ignored=False,
                    barge_in_decision_pending=False,
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
                        speech_ended_at = None
                        if utterance.speech_end_offset_samples is not None:
                            speech_ended_at = (
                                block.received_at
                                - utterance.speech_end_offset_samples / self._sample_rate
                            )
                        self._queue_latest_utterance(
                            utterance.samples,
                            speech_ended_at,
                            time.perf_counter(),
                            barge_in=False,
                        )
        finally:
            with self._state.locked():
                self._state.listening = False
            self._stop_transcription_queue()

    def _handle_speaking_block(self, block: CapturedAudioBlock) -> None:
        assert self._preprocessor is not None
        assert self._interruption_detector is not None
        assert self._turn_coordinator is not None

        try:
            processed = np.asarray(
                self._preprocessor.process_capture(block),
                dtype=np.float32,
            ).reshape(-1)
        except Exception as error:
            self._handle_preprocessor_error(error)
            return

        candidate = bool(self._interruption_detector.push(processed))
        if candidate and not self._barge_in_capture_active:
            accepted = bool(self._turn_coordinator.on_candidate())
            if accepted:
                self._barge_in_capture_active = True
            else:
                self._interruption_detector.reset()
                self._barge_in_assembler.reset()

        utterance = self._barge_in_assembler.push_with_timing(processed)
        status = self._barge_in_assembler.status()
        peak = float(np.max(np.abs(processed))) if processed.size else 0.0
        self._state.update_debug(
            audio_ignored=False,
            barge_in_capture_active=self._barge_in_capture_active,
            mic_peak=round(peak, 4),
            transcription_queue=self._utterances.qsize(),
            **status,
        )

        if utterance is None or not self._barge_in_capture_active:
            return

        self._barge_in_capture_active = False
        self._interruption_detector.reset()
        self._barge_in_decision_pending.set()
        self._turn_coordinator.on_utterance_queued_for_transcription()
        if self._transcriber_failed.is_set():
            error = RuntimeError("transcriber unavailable")
            self._turn_coordinator.on_transcription_error(error)
            self._barge_in_decision_pending.clear()
            return
        speech_ended_at = None
        if utterance.speech_end_offset_samples is not None:
            speech_ended_at = (
                block.received_at
                - utterance.speech_end_offset_samples / self._sample_rate
            )
        self._queue_latest_utterance(
            utterance.samples,
            speech_ended_at,
            time.perf_counter(),
            barge_in=True,
        )

    def _handle_preprocessor_error(self, error: Exception) -> None:
        had_active_candidate = self._barge_in_capture_active
        payload = {"error": str(error), "stage": "preprocessor"}
        self._event_log.append("barge_in_error", payload)
        self._state.record_debug("barge_in_preprocessor_error", **payload)
        if self._disable_on_preprocessor_error:
            self._barge_in_available = False
            self._state.update_debug(barge_in_available=False)
        self._reset_barge_in_capture()
        if had_active_candidate and self._turn_coordinator is not None:
            self._turn_coordinator.on_transcription_error(error)

    def _reset_barge_in_capture(self) -> None:
        self._barge_in_capture_active = False
        self._barge_in_decision_pending.clear()
        self._barge_in_assembler.reset()
        if self._interruption_detector is not None:
            self._interruption_detector.reset()

    def _coerce_capture_block(
        self,
        block: CapturedAudioBlock | np.ndarray,
    ) -> CapturedAudioBlock:
        if isinstance(block, CapturedAudioBlock):
            return block
        return CapturedAudioBlock(
            samples=np.asarray(block, dtype=np.float32).reshape(-1).copy(),
            adc_time=None,
            received_at=time.perf_counter(),
        )

    def _observe_capture_sequence(self, block: CapturedAudioBlock) -> None:
        sequence = block.capture_sequence
        if sequence is None:
            return
        previous = self._last_capture_sequence
        self._last_capture_sequence = sequence
        if previous is None or sequence == previous + 1:
            return
        self._assembler.invalidate_timing()
        self._barge_in_assembler.invalidate_timing()
        self._state.update_debug(audio_blocks_dropped=self._audio_blocks_dropped)

    def _queue_latest_utterance(
        self,
        utterance: np.ndarray,
        speech_ended_at: float | None,
        utterance_finalized_at: float,
        *,
        barge_in: bool,
    ) -> None:
        item = _QueuedUtterance(
            utterance,
            speech_ended_at,
            utterance_finalized_at,
            barge_in,
        )
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
                age_origin = (
                    replaced.speech_ended_at
                    if replaced.speech_ended_at is not None
                    else replaced.utterance_finalized_at
                )
                age_ms = (time.perf_counter() - age_origin) * 1000.0
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
            load_started = time.perf_counter()
            try:
                self._transcribe = _make_transcriber(self._config)
            except Exception as error:
                self._transcriber_failed.set()
                self._record_audio_error(error)
                return
            load_finished = time.perf_counter()
            settings = self._config.get("whisper", {})
            payload = {
                "whisper_model": str(settings.get("model", "small")),
                "whisper_device": str(settings.get("device", "cpu")),
                "whisper_compute_type": str(settings.get("compute_type", "int8")),
                "whisper_device_index": int(settings.get("device_index", 0)),
                "whisper_load_ms": round((load_finished - load_started) * 1000.0, 1),
            }
            self._state.update_debug(**payload)
            self._state.record_debug("whisper_ready", **payload)

        while True:
            item = self._utterances.get()
            if item is None:
                return
            age_origin = (
                item.speech_ended_at
                if item.speech_ended_at is not None
                else item.utterance_finalized_at
            )
            age_ms = (time.perf_counter() - age_origin) * 1000.0
            self._state.update_debug(
                transcription_queue=self._utterances.qsize(),
                utterance_age_ms=round(age_ms, 1),
            )
            if item.barge_in:
                self.handle_barge_in_utterance(
                    item.samples,
                    speech_ended_at=item.speech_ended_at,
                    utterance_finalized_at=item.utterance_finalized_at,
                )
            else:
                self.handle_utterance(
                    item.samples,
                    speech_ended_at=item.speech_ended_at,
                    utterance_finalized_at=item.utterance_finalized_at,
                )

    def _transcription_timing(
        self,
        transcription_started: float,
        transcription_finished: float,
        speech_ended_at: float | None,
        utterance_finalized_at: float | None,
        sample_count: int,
    ) -> dict[str, object]:
        transcription_seconds = max(transcription_finished - transcription_started, 0.0)
        utterance_seconds = (
            max(sample_count, 0) / self._sample_rate if self._sample_rate > 0 else 0.0
        )
        payload: dict[str, object] = {
            "speech_ended_at": speech_ended_at,
            "utterance_finalized_at": utterance_finalized_at,
            "transcription_started_at": transcription_started,
            "transcription_finished_at": transcription_finished,
            "latency_timing_valid": (
                speech_ended_at is not None and utterance_finalized_at is not None
            ),
            "transcription_ms": round(transcription_seconds * 1000.0, 1),
            "utterance_seconds": round(utterance_seconds, 3),
            "transcription_rtf": (
                round(transcription_seconds / utterance_seconds, 3)
                if utterance_seconds > 0.0
                else None
            ),
        }
        if speech_ended_at is not None:
            payload["speech_to_transcript_ms"] = round(
                (transcription_finished - speech_ended_at) * 1000.0,
                1,
            )
        if speech_ended_at is not None and utterance_finalized_at is not None:
            payload["endpoint_wait_ms"] = round(
                max(utterance_finalized_at - speech_ended_at, 0.0) * 1000.0,
                1,
            )
            payload["transcription_queue_ms"] = round(
                max(transcription_started - utterance_finalized_at, 0.0) * 1000.0,
                1,
            )
        return payload

    def _dispatch_request(self, request: str, timing: dict[str, object]) -> None:
        if self._on_timed_request is not None:
            self._on_timed_request(request, timing)
        else:
            self._on_request(request)

    def _record_audio_error(self, error: Exception) -> None:
        self._event_log.append("audio_error", {"error": str(error)})
        self._state.record_debug("audio_error", error=str(error))

    @staticmethod
    def _make_assembler(settings: dict[str, Any]) -> UtteranceAssembler:
        return UtteranceAssembler(
            int(settings.get("sample_rate", 16_000)),
            float(settings.get("vad_threshold", 0.015)),
            float(settings.get("min_utterance_seconds", 0.25)),
            float(settings.get("silence_seconds", 0.45)),
            float(settings.get("max_utterance_seconds", 15.0)),
            float(settings.get("pre_roll_seconds", 0.25)),
        )

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


def _input_adc_time(time_info: object) -> float | None:
    if time_info is None:
        return None
    value = getattr(time_info, "inputBufferAdcTime", None)
    if value is None:
        try:
            value = time_info["inputBufferAdcTime"]  # type: ignore[index]
        except (KeyError, TypeError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _make_transcriber(config: dict[str, Any]) -> Callable[[np.ndarray], str]:
    """Create the configured faster-whisper transcriber for live capture."""
    from faster_whisper import WhisperModel

    settings = config.get("whisper", {})
    model = WhisperModel(
        settings.get("model", "small"),
        device=settings.get("device", "cpu"),
        compute_type=settings.get("compute_type", "int8"),
        device_index=int(settings.get("device_index", 0)),
        cpu_threads=int(settings.get("cpu_threads", 0)),
        num_workers=int(settings.get("num_workers", 1)),
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
