"""Pipelined local text-to-speech output."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from output.audio_player import (
    AudioPlayer,
    CallbackAudioPlayer,
    PlaybackReceipt,
    SoundDeviceAudioPlayer,
)
from output.tts.base import SynthesisResult, TTSEngine
from performance import PerformanceCue


@dataclass(frozen=True)
class _PlaybackContext:
    kind: str
    text: str
    generation_id: int | None
    performance: PerformanceCue | None


class VoiceOutput:
    """Synthesize one clause ahead while skipping obsolete generations."""

    def __init__(
        self,
        config: dict[str, Any],
        state: Any,
        event_log: Any,
        synthesize: Callable[[str], np.ndarray] | None = None,
        play: Callable[[np.ndarray, int], None] | None = None,
        engine: TTSEngine | None = None,
        player: AudioPlayer | None = None,
        on_delivery: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._event_log = event_log
        self._legacy_synthesize = synthesize
        self._engine = engine
        if player is not None:
            self._player = player
        elif play is not None:
            self._player = CallbackAudioPlayer(play)
        else:
            self._player = SoundDeviceAudioPlayer()
        self._on_delivery = on_delivery

        self._queue: queue.Queue[
            tuple[
                str,
                str | None,
                int | None,
                PerformanceCue,
                float | None,
            ]
            | None
        ] = queue.Queue()
        self._ready_queue: queue.Queue[
            tuple[
                str,
                str,
                np.ndarray,
                int,
                int | None,
                float | None,
                tuple[float, float] | None,
                PerformanceCue | None,
            ]
            | None
        ] = queue.Queue(maxsize=1)
        self._ready_slots = threading.Semaphore(1)

        self._generation_lock = threading.Lock()
        self._active_generation = 0
        self._thread: threading.Thread | None = None
        self._playback_thread: threading.Thread | None = None

        self._interruption_condition = threading.Condition()
        self._current_playback: _PlaybackContext | None = None
        self._paused_playback: _PlaybackContext | None = None
        self._interruption_decision: str | None = None

        self._acknowledgement = "Yeah?"
        self._acknowledgement_audio: np.ndarray | None = None
        self._acknowledgement_sample_rate: int | None = None
        self._last_playback_ended_at: float | None = None
        self._last_playback_generation: int | None = None

    def start(self) -> None:
        """Start independent synthesis and playback workers."""
        if self._thread is not None:
            return
        self._playback_thread = threading.Thread(
            target=self._playback_run,
            name="voice-playback",
            daemon=True,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="voice-synthesis",
            daemon=True,
        )
        self._playback_thread.start()
        self._thread.start()

    def begin_generation(self, generation_id: int) -> None:
        """Mark every older queued or prepared response as obsolete."""
        with self._generation_lock:
            self._active_generation = generation_id
        self._state.update_debug(tts_generation=generation_id)

    def enqueue(
        self,
        text: str,
        generation_id: int | None = None,
        performance: PerformanceCue | None = None,
    ) -> None:
        """Queue text and its transient performance cue for synthesis."""
        queued_at = time.perf_counter()
        self._queue.put(
            (
                "speak",
                text,
                generation_id,
                performance or PerformanceCue(),
                queued_at,
            )
        )
        self._state.update_debug(tts_queue=self._queue.qsize())

    def prepare_acknowledgement(self, text: str = "Yeah?") -> None:
        """Cache a short acknowledgement in the synthesis worker."""
        self._acknowledgement = text
        self._queue.put(("prepare", text, None, PerformanceCue(), None))
        self._state.update_debug(tts_queue=self._queue.qsize())

    def enqueue_acknowledgement(self, generation_id: int | None = None) -> None:
        """Queue the prepared acknowledgement if it is still current."""
        self._queue.put(
            ("acknowledgement", None, generation_id, PerformanceCue(), None)
        )
        self._state.update_debug(tts_queue=self._queue.qsize())

    def finish_generation(self, generation_id: int) -> None:
        """Queue an ordered marker that fires after all preceding speech drains."""
        self._queue.put(("finish", None, generation_id, PerformanceCue(), None))
        self._state.update_debug(tts_queue=self._queue.qsize())

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        """Pause the currently audible waveform without cancelling its generation."""
        with self._interruption_condition:
            context = self._current_playback
        if context is None or context.kind != "speak":
            return None

        receipt = self._player.pause_for_interruption()
        if (
            receipt is None
            or receipt.status != "paused"
            or receipt.generation_id != context.generation_id
        ):
            return None

        with self._interruption_condition:
            if self._current_playback is not context:
                return None
            self._paused_playback = context
            self._interruption_decision = None
            self._interruption_condition.notify_all()

        self._deactivate_playback_state(context)
        self._mark_spoke_if_audible(receipt)
        self._emit_delivery(
            "clause_paused",
            generation_id=context.generation_id,
            text=context.text,
            frames_completed=receipt.frames_completed,
            total_frames=receipt.total_frames,
        )
        self._state.record_debug(
            "tts_paused",
            text=context.text,
            generation_id=context.generation_id,
            frames_completed=receipt.frames_completed,
            total_frames=receipt.total_frames,
        )
        return receipt

    def commit_interruption(self, generation_id: int) -> bool:
        """Permanently abandon the paused remainder for exactly one generation."""
        with self._interruption_condition:
            context = self._paused_playback
            if (
                context is None
                or context.generation_id != generation_id
                or self._interruption_decision is not None
            ):
                return False
            self._interruption_decision = "commit"
            self._interruption_condition.notify_all()
        return True

    def resume_after_false_interruption(self, generation_id: int) -> bool:
        """Resume the same paused waveform only if its generation is still current."""
        if self._is_stale(generation_id):
            return False
        with self._interruption_condition:
            context = self._paused_playback
            if (
                context is None
                or context.generation_id != generation_id
                or self._interruption_decision is not None
            ):
                return False
            self._interruption_decision = "resume"
            self._interruption_condition.notify_all()
        return True

    def close(self) -> None:
        """Finish queued synthesis and playback before stopping both workers."""
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join()
        self._ready_queue.put(None)
        if self._playback_thread is not None:
            self._playback_thread.join()
        self._thread = None
        self._playback_thread = None

    def _run(self) -> None:
        """Synthesize text while playback proceeds independently."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, text, generation_id, performance, queued_at = item
            self._state.update_debug(tts_queue=self._queue.qsize())
            if kind != "prepare" and self._is_stale(generation_id):
                self._record_stale_skip(generation_id, stage="queued")
                continue
            if kind == "prepare":
                self._prepare(text or self._acknowledgement)
            elif kind == "acknowledgement":
                self._speak_acknowledgement(generation_id)
            elif kind == "finish":
                self._queue_finish_marker(generation_id)
            else:
                self._speak(
                    text or "",
                    generation_id=generation_id,
                    performance=performance,
                    queued_at=queued_at,
                )

    def _playback_run(self) -> None:
        """Play prepared waveforms serially while synthesis stays one ahead."""
        while True:
            item = self._ready_queue.get()
            if item is None:
                return

            self._ready_slots.release()
            self._state.update_debug(tts_ready_queue=self._ready_queue.qsize())

            (
                kind,
                text,
                audio,
                sample_rate,
                generation_id,
                synthesis_ms,
                raw_edge_silence,
                performance,
            ) = item
            if self._is_stale(generation_id):
                self._record_stale_skip(generation_id, stage="prepared")
                continue

            if kind == "finish":
                self._emit_delivery(
                    "generation_playback_drained",
                    generation_id=generation_id,
                )
                self._state.record_debug(
                    "generation_playback_drained",
                    generation_id=generation_id,
                )
                continue

            if kind == "acknowledgement":
                self._state.record_debug("tts_started", text=text)
            self._play_waveform(
                audio,
                kind=kind,
                sample_rate=sample_rate,
                text=text,
                synthesis_ms=synthesis_ms,
                generation_id=generation_id,
                raw_edge_silence_ms=raw_edge_silence,
                performance=performance,
            )
            self._state.record_debug("tts_complete", text=text)

    def _prepare(self, text: str) -> None:
        try:
            result = self._synthesize_text(text, PerformanceCue())
            self._acknowledgement_audio = result.audio
            self._acknowledgement_sample_rate = result.sample_rate
            self._state.record_debug("tts_prepared", text=text)
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error)})
            self._state.record_debug("tts_error", error=str(error), text=text)

    def _speak_acknowledgement(self, generation_id: int | None = None) -> None:
        if not self._reserve_ready_slot(generation_id):
            return

        synthesis_ms: float | None = None
        if self._acknowledgement_audio is None or self._acknowledgement_sample_rate is None:
            synthesis_started = time.perf_counter()
            self._prepare(self._acknowledgement)
            synthesis_ms = (time.perf_counter() - synthesis_started) * 1000.0

        if self._acknowledgement_audio is None or self._acknowledgement_sample_rate is None:
            self._ready_slots.release()
            return
        if self._is_stale(generation_id):
            self._ready_slots.release()
            self._record_stale_skip(generation_id, stage="after_synthesis")
            return

        self._ready_queue.put(
            (
                "acknowledgement",
                self._acknowledgement,
                self._acknowledgement_audio,
                self._acknowledgement_sample_rate,
                generation_id,
                synthesis_ms,
                None,
                None,
            )
        )
        self._state.update_debug(tts_ready_queue=self._ready_queue.qsize())

    def _speak(
        self,
        text: str,
        generation_id: int | None = None,
        performance: PerformanceCue | None = None,
        *,
        queued_at: float | None = None,
    ) -> None:
        if not self._reserve_ready_slot(generation_id):
            return

        cue = performance or PerformanceCue()
        self._state.record_debug("tts_started", text=text)
        synthesis_started = time.perf_counter()
        worker_wait_ms = None
        if queued_at is not None:
            worker_wait_ms = max(synthesis_started - queued_at, 0.0) * 1000.0
        try:
            result = self._synthesize_text(text, cue)
        except Exception as error:
            self._ready_slots.release()
            self._event_log.append("voice_error", {"error": str(error), "text": text})
            self._state.record_debug("tts_error", error=str(error), text=text)
            return

        synthesis_finished = time.perf_counter()
        synthesis_ms = (synthesis_finished - synthesis_started) * 1000.0
        if self._is_stale(generation_id):
            self._ready_slots.release()
            self._record_stale_skip(generation_id, stage="after_synthesis")
            return

        if isinstance(generation_id, int) and worker_wait_ms is not None:
            self._emit_delivery(
                "clause_synthesized",
                generation_id=generation_id,
                text=text,
                worker_wait_ms=round(worker_wait_ms, 1),
                synthesis_ms=round(synthesis_ms, 1),
            )

        sample_rate = result.sample_rate
        raw_edge_silence = _waveform_edge_silence_ms(result.audio, sample_rate)
        audio = _trim_waveform_edges(result.audio, sample_rate)

        self._ready_queue.put(
            (
                "speak",
                text,
                audio,
                sample_rate,
                generation_id,
                synthesis_ms,
                raw_edge_silence,
                cue,
            )
        )
        self._state.update_debug(tts_ready_queue=self._ready_queue.qsize())

    def _queue_finish_marker(self, generation_id: int | None) -> None:
        if not self._reserve_ready_slot(generation_id):
            return
        sample_rate = int(self._config.get("voice", {}).get("sample_rate", 24_000))
        self._ready_queue.put(
            (
                "finish",
                "",
                np.asarray([], dtype=np.float32),
                sample_rate,
                generation_id,
                None,
                None,
                None,
            )
        )
        self._state.update_debug(tts_ready_queue=self._ready_queue.qsize())

    def _reserve_ready_slot(self, generation_id: int | None) -> bool:
        """Reserve the sole prepared-audio slot before doing synthesis work."""
        self._ready_slots.acquire()
        if self._is_stale(generation_id):
            self._ready_slots.release()
            self._record_stale_skip(generation_id, stage="before_synthesis")
            return False
        return True

    def _play_waveform(
        self,
        audio: np.ndarray,
        *,
        kind: str,
        sample_rate: int,
        text: str,
        synthesis_ms: float | None = None,
        generation_id: int | None = None,
        raw_edge_silence_ms: tuple[float, float] | None = None,
        performance: PerformanceCue | None = None,
    ) -> None:
        if self._is_stale(generation_id):
            self._record_stale_skip(generation_id, stage="before_playback")
            return

        context = _PlaybackContext(kind, text, generation_id, performance)
        if not audio.size:
            return

        payload: dict[str, object] = {"text": text}
        if synthesis_ms is not None:
            rounded = round(synthesis_ms, 1)
            payload["synthesis_ms"] = rounded
            self._state.update_debug(tts_synthesis_ms=rounded)

        leading_silence_ms, trailing_silence_ms = _waveform_edge_silence_ms(
            audio,
            sample_rate,
        )
        payload["leading_silence_ms"] = leading_silence_ms
        payload["trailing_silence_ms"] = trailing_silence_ms
        debug_values: dict[str, object] = {
            "tts_leading_silence_ms": leading_silence_ms,
            "tts_trailing_silence_ms": trailing_silence_ms,
        }
        if raw_edge_silence_ms is not None:
            raw_leading_ms, raw_trailing_ms = raw_edge_silence_ms
            payload["raw_leading_silence_ms"] = raw_leading_ms
            payload["raw_trailing_silence_ms"] = raw_trailing_ms
            debug_values["tts_raw_leading_silence_ms"] = raw_leading_ms
            debug_values["tts_raw_trailing_silence_ms"] = raw_trailing_ms
        self._state.update_debug(**debug_values)

        with self._interruption_condition:
            self._current_playback = context
            self._paused_playback = None
            self._interruption_decision = None

        self._activate_playback_state(context)
        # This marker is immediately before synchronous delivery/debug callbacks.
        # The player call follows those callbacks, so this is not DAC/audio onset.
        playback_delivery_started_at = time.perf_counter()
        if (
            self._last_playback_ended_at is not None
            and generation_id == self._last_playback_generation
        ):
            gap_ms = round(
                (playback_delivery_started_at - self._last_playback_ended_at) * 1000.0,
                1,
            )
            payload["gap_ms"] = gap_ms
            self._state.update_debug(tts_gap_ms=gap_ms)
        if kind == "speak":
            self._emit_delivery(
                "clause_started",
                generation_id=generation_id,
                text=text,
                playback_delivery_started_at=playback_delivery_started_at,
            )
        self._state.record_debug(
            "tts_playback_started",
            generation_id=generation_id,
            **payload,
        )

        try:
            receipt = self._player.play(audio, sample_rate, generation_id)
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error)})
            self._state.record_debug("tts_error", error=str(error))
            self._finish_playback_context(context, completed=False)
            return

        if receipt.status == "paused" and kind == "speak":
            self._handle_paused_playback(context, receipt)
            return

        if receipt.status == "error":
            error = "audio player failed during playback"
            self._event_log.append("voice_error", {"error": error})
            self._state.record_debug("tts_error", error=error)
            self._finish_playback_context(context, completed=False)
            return

        completed = receipt.status == "completed"
        if completed and kind == "speak":
            self._emit_delivery(
                "clause_completed",
                generation_id=generation_id,
                text=text,
            )
        self._mark_spoke_if_audible(receipt)
        self._finish_playback_context(context, completed=completed)

    def _handle_paused_playback(
        self,
        context: _PlaybackContext,
        receipt: PlaybackReceipt,
    ) -> None:
        with self._interruption_condition:
            while self._paused_playback is not context:
                self._interruption_condition.wait(timeout=0.05)
            while self._interruption_decision is None:
                self._interruption_condition.wait(timeout=0.05)
            decision = self._interruption_decision

        if decision == "commit":
            self._player.discard_paused()
            self._emit_delivery(
                "clause_abandoned",
                generation_id=context.generation_id,
                text=context.text,
                frames_completed=receipt.frames_completed,
                total_frames=receipt.total_frames,
            )
            self._finish_playback_context(context, completed=False)
            return

        if decision != "resume" or self._is_stale(context.generation_id):
            self._player.discard_paused()
            self._emit_delivery(
                "clause_abandoned",
                generation_id=context.generation_id,
                text=context.text,
                frames_completed=receipt.frames_completed,
                total_frames=receipt.total_frames,
            )
            self._finish_playback_context(context, completed=False)
            return

        self._activate_playback_state(context)
        self._emit_delivery(
            "clause_resumed",
            generation_id=context.generation_id,
            text=context.text,
            frames_started=receipt.frames_completed,
        )
        self._state.record_debug(
            "tts_resumed",
            text=context.text,
            generation_id=context.generation_id,
            frames_started=receipt.frames_completed,
        )
        try:
            resumed = self._player.resume()
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error)})
            self._state.record_debug("tts_error", error=str(error))
            self._finish_playback_context(context, completed=False)
            return

        completed = resumed.status == "completed"
        if completed:
            self._emit_delivery(
                "clause_completed",
                generation_id=context.generation_id,
                text=context.text,
            )
        elif resumed.status == "error":
            error = "audio player failed during resumed playback"
            self._event_log.append("voice_error", {"error": error})
            self._state.record_debug("tts_error", error=error)
        self._mark_spoke_if_audible(resumed)
        self._finish_playback_context(context, completed=completed)

    def _activate_playback_state(self, context: _PlaybackContext) -> None:
        with self._state.locked():
            self._state.speaking = True
            if context.performance is not None:
                self._state.performance = context.performance
        if context.performance is not None:
            self._state.update_debug(
                performance_expression=context.performance.expression,
                performance_intensity=context.performance.intensity,
            )
            self._state.record_debug(
                "performance_started",
                text=context.text,
                expression=context.performance.expression,
                intensity=context.performance.intensity,
                generation_id=context.generation_id,
            )

    def _deactivate_playback_state(self, context: _PlaybackContext) -> None:
        if context.performance is not None:
            with self._state.locked():
                self._state.performance = PerformanceCue()
            self._state.update_debug(
                performance_expression="neutral",
                performance_intensity=0.0,
            )
            self._state.record_debug(
                "performance_ended",
                expression=context.performance.expression,
                generation_id=context.generation_id,
            )
        with self._state.locked():
            self._state.speaking = False

    def _finish_playback_context(
        self,
        context: _PlaybackContext,
        *,
        completed: bool,
    ) -> None:
        self._deactivate_playback_state(context)
        with self._interruption_condition:
            if self._current_playback is context:
                self._current_playback = None
            if self._paused_playback is context:
                self._paused_playback = None
            self._interruption_decision = None
            self._interruption_condition.notify_all()
        if completed:
            self._last_playback_ended_at = time.perf_counter()
            self._last_playback_generation = context.generation_id

    def _mark_spoke_if_audible(self, receipt: PlaybackReceipt) -> None:
        if receipt.frames_completed <= receipt.frames_started:
            return
        with self._state.locked():
            self._state.last_spoke = time.time()

    def _emit_delivery(self, event_type: str, **payload: object) -> None:
        if self._on_delivery is None:
            return
        try:
            self._on_delivery(event_type, dict(payload))
        except Exception as error:
            self._state.record_debug(
                "delivery_callback_error",
                event_type=event_type,
                error=str(error),
            )

    def _is_stale(self, generation_id: int | None) -> bool:
        if generation_id is None:
            return False
        with self._generation_lock:
            return generation_id != self._active_generation

    def _record_stale_skip(self, generation_id: int | None, *, stage: str) -> None:
        self._state.record_debug(
            "stale_tts_skipped",
            generation_id=generation_id,
            stage=stage,
        )

    def _synthesize_text(
        self,
        text: str,
        performance: PerformanceCue,
    ) -> SynthesisResult:
        if self._engine is None and self._legacy_synthesize is None:
            from output.tts.factory import create_tts_engine

            self._engine = create_tts_engine(self._config)

        if self._engine is not None:
            return self._engine.synthesize(text, performance)

        audio = np.asarray(self._legacy_synthesize(text), dtype=np.float32).reshape(-1)
        sample_rate = int(self._config.get("voice", {}).get("sample_rate", 24_000))
        return SynthesisResult(audio, sample_rate)


def _waveform_edge_silence_ms(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.001,
) -> tuple[float, float]:
    """Estimate contiguous near-silence at the beginning and end of a waveform."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or sample_rate <= 0:
        return 0.0, 0.0

    audible = np.flatnonzero(np.abs(samples) > threshold)
    if audible.size == 0:
        duration_ms = round(samples.size / sample_rate * 1000.0, 1)
        return duration_ms, duration_ms

    first = int(audible[0])
    last = int(audible[-1])
    leading_ms = round(first / sample_rate * 1000.0, 1)
    trailing_ms = round((samples.size - last - 1) / sample_rate * 1000.0, 1)
    return leading_ms, trailing_ms


def _trim_waveform_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.001,
    keep_leading_ms: float = 100.0,
    keep_trailing_ms: float = 160.0,
) -> np.ndarray:
    """Trim only excess near-silence while preserving generous speech margins.

    The same conservative threshold used by diagnostics finds a confident speech
    edge, then extra audio is kept around that edge so quiet consonants or breathy
    starts are not cut merely because they stay below the threshold briefly.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or sample_rate <= 0:
        return samples

    audible = np.flatnonzero(np.abs(samples) > threshold)
    if audible.size == 0:
        return samples

    first = int(audible[0])
    last = int(audible[-1])
    leading_keep = max(0, round(sample_rate * keep_leading_ms / 1000.0))
    trailing_keep = max(0, round(sample_rate * keep_trailing_ms / 1000.0))
    start = max(0, first - leading_keep)
    end = min(samples.size, last + 1 + trailing_keep)
    return samples[start:end].copy()
