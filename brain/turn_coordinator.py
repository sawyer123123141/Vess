"""Two-phase reversible coordination for speech that overlaps Vess playback."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


IDLE = "IDLE"
PENDING_CAPTURE = "PENDING_CAPTURE"
PENDING_TRANSCRIBE = "PENDING_TRANSCRIBE"
CLOSED = "CLOSED"


TimerFactory = Callable[[float, Callable[[], None]], Any]


class TurnCoordinator:
    """Pause quickly, then commit or roll back only the exact paused generation."""

    def __init__(
        self,
        state: Any,
        event_log: Any,
        voice: Any,
        conversation: Any,
        transcript_submit: Callable[[str], None],
        *,
        false_timeout_seconds: float,
        decision_watchdog_seconds: float,
        timed_transcript_submit: Callable[[str, dict[str, object]], None] | None = None,
        timer_factory: TimerFactory | None = None,
    ) -> None:
        self._state = state
        self._event_log = event_log
        self._voice = voice
        self._conversation = conversation
        self._transcript_submit = transcript_submit
        self._timed_transcript_submit = timed_transcript_submit
        self._false_timeout_seconds = max(0.0, float(false_timeout_seconds))
        self._decision_watchdog_seconds = max(0.0, float(decision_watchdog_seconds))
        self._timer_factory = timer_factory or _threading_timer

        self._lock = threading.Lock()
        self._phase = IDLE
        self._paused_generation: int | None = None
        self._candidate_started_at: float | None = None
        self._false_timer: Any | None = None
        self._watchdog_timer: Any | None = None

    def on_candidate(self) -> bool:
        """Request one reversible pause for a credible overlapping-speech candidate."""
        with self._lock:
            if self._phase == CLOSED:
                return False
            if self._phase in (PENDING_CAPTURE, PENDING_TRANSCRIBE):
                return True
            self._phase = PENDING_CAPTURE
            self._paused_generation = None
            self._candidate_started_at = time.perf_counter()

        receipt = self._voice.pause_for_interruption()
        if (
            receipt is None
            or receipt.status != "paused"
            or not isinstance(receipt.generation_id, int)
        ):
            with self._lock:
                if self._phase == PENDING_CAPTURE and self._paused_generation is None:
                    self._phase = IDLE
                    self._candidate_started_at = None
            return False

        generation_id = receipt.generation_id
        false_timer = self._timer_factory(
            self._false_timeout_seconds,
            lambda: self._on_false_timeout(generation_id),
        )
        discard_after_close = False
        with self._lock:
            if self._phase == CLOSED:
                discard_after_close = True
            elif self._phase == PENDING_CAPTURE and self._paused_generation is None:
                self._paused_generation = generation_id
                self._false_timer = false_timer
            else:
                discard_after_close = True

        if discard_after_close:
            _cancel_timer(false_timer)
            self._voice.commit_interruption(generation_id)
            self._set_listening(False)
            return False

        self._set_listening(True)
        self._state.record_debug(
            "barge_in_candidate",
            generation_id=generation_id,
            frames_completed=receipt.frames_completed,
            total_frames=receipt.total_frames,
        )
        false_timer.start()

        with self._lock:
            closed_after_start = self._phase == CLOSED
        if closed_after_start:
            self._set_listening(False)
            return False
        return True

    def on_utterance_queued_for_transcription(self) -> None:
        """Suspend the ordinary false timeout while transcription is in flight."""
        with self._lock:
            if self._phase != PENDING_CAPTURE or self._paused_generation is None:
                return
            generation_id = self._paused_generation

        watchdog = self._timer_factory(
            self._decision_watchdog_seconds,
            lambda: self._on_watchdog(generation_id),
        )

        false_timer: Any | None = None
        with self._lock:
            if (
                self._phase != PENDING_CAPTURE
                or self._paused_generation != generation_id
            ):
                _cancel_timer(watchdog)
                return
            self._phase = PENDING_TRANSCRIBE
            false_timer = self._false_timer
            self._false_timer = None
            self._watchdog_timer = watchdog

        _cancel_timer(false_timer)
        self._state.record_debug(
            "barge_in_transcribing",
            generation_id=generation_id,
        )
        watchdog.start()

    def on_transcript(
        self,
        text: str,
        *,
        timing: dict[str, object] | None = None,
    ) -> None:
        """Commit a real turn for non-empty text, otherwise roll the pause back."""
        clean_text = text.strip()
        if not clean_text:
            self._rollback("empty_transcript")
            return

        pending = self._take_pending()
        if pending is None:
            return
        generation_id, false_timer, watchdog_timer = pending
        _cancel_timer(false_timer)
        _cancel_timer(watchdog_timer)
        self._set_listening(False)

        cancelled = self._conversation.cancel_generation(generation_id, "barge_in")
        self._voice.commit_interruption(generation_id)
        if timing is not None and self._timed_transcript_submit is not None:
            self._timed_transcript_submit(clean_text, timing)
        else:
            self._transcript_submit(clean_text)
        payload = {
            "generation_id": generation_id,
            "generation_cancelled": cancelled,
            "transcript": clean_text,
        }
        self._event_log.append("barge_in_committed", payload)
        self._state.record_debug("barge_in_committed", **payload)

    def on_transcription_error(self, error: Exception) -> None:
        """Treat a failed transcription as a reversible false interruption."""
        self._event_log.append("barge_in_transcription_error", {"error": str(error)})
        self._state.record_debug("barge_in_transcription_error", error=str(error))
        self._rollback("transcription_error")

    def close(self) -> None:
        """Cancel pending decisions and make future timer callbacks inert."""
        with self._lock:
            if self._phase == CLOSED:
                return
            generation_id = self._paused_generation
            false_timer = self._false_timer
            watchdog_timer = self._watchdog_timer
            self._phase = CLOSED
            self._paused_generation = None
            self._candidate_started_at = None
            self._false_timer = None
            self._watchdog_timer = None

        _cancel_timer(false_timer)
        _cancel_timer(watchdog_timer)
        self._set_listening(False)
        if generation_id is not None:
            self._voice.commit_interruption(generation_id)

    def _on_false_timeout(self, generation_id: int) -> None:
        self._rollback(
            "false_timeout",
            expected_generation=generation_id,
            expected_phase=PENDING_CAPTURE,
        )

    def _on_watchdog(self, generation_id: int) -> None:
        pending = self._take_pending(
            expected_generation=generation_id,
            expected_phase=PENDING_TRANSCRIBE,
        )
        if pending is None:
            return
        paused_generation, false_timer, watchdog_timer = pending
        _cancel_timer(false_timer)
        _cancel_timer(watchdog_timer)
        self._set_listening(False)

        payload = {"generation_id": paused_generation}
        self._event_log.append("barge_in_watchdog", payload)
        self._state.record_debug("barge_in_watchdog", **payload)
        self._resume_or_discard(paused_generation, reason="decision_watchdog")

    def _rollback(
        self,
        reason: str,
        *,
        expected_generation: int | None = None,
        expected_phase: str | None = None,
    ) -> None:
        pending = self._take_pending(
            expected_generation=expected_generation,
            expected_phase=expected_phase,
        )
        if pending is None:
            return
        generation_id, false_timer, watchdog_timer = pending
        _cancel_timer(false_timer)
        _cancel_timer(watchdog_timer)
        self._set_listening(False)
        self._resume_or_discard(generation_id, reason=reason)

    def _resume_or_discard(self, generation_id: int, *, reason: str) -> None:
        resumed = self._voice.resume_after_false_interruption(generation_id)
        if not resumed:
            self._voice.commit_interruption(generation_id)
        payload = {
            "generation_id": generation_id,
            "reason": reason,
            "resumed": resumed,
        }
        self._event_log.append("barge_in_false_interruption", payload)
        self._state.record_debug("barge_in_false_interruption", **payload)

    def _take_pending(
        self,
        *,
        expected_generation: int | None = None,
        expected_phase: str | None = None,
    ) -> tuple[int, Any | None, Any | None] | None:
        with self._lock:
            if self._phase not in (PENDING_CAPTURE, PENDING_TRANSCRIBE):
                return None
            if expected_phase is not None and self._phase != expected_phase:
                return None
            generation_id = self._paused_generation
            if generation_id is None:
                return None
            if expected_generation is not None and generation_id != expected_generation:
                return None

            false_timer = self._false_timer
            watchdog_timer = self._watchdog_timer
            self._phase = IDLE
            self._paused_generation = None
            self._candidate_started_at = None
            self._false_timer = None
            self._watchdog_timer = None
            return generation_id, false_timer, watchdog_timer

    def _set_listening(self, value: bool) -> None:
        with self._state.locked():
            self._state.listening = value


def _threading_timer(seconds: float, callback: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(seconds, callback)
    timer.daemon = True
    return timer


def _cancel_timer(timer: Any | None) -> None:
    if timer is not None:
        timer.cancel()
