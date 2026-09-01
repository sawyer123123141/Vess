"""Pipelined local text-to-speech output."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class VoiceOutput:
    """Synthesize one clause ahead while skipping obsolete generations."""

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

        self._queue: queue.Queue[
            tuple[str, str | None, int | None] | None
        ] = queue.Queue()
        self._ready_queue: queue.Queue[
            tuple[str, str, np.ndarray, int | None, float | None] | None
        ] = queue.Queue(maxsize=1)
        self._ready_slots = threading.Semaphore(1)

        self._generation_lock = threading.Lock()
        self._active_generation = 0
        self._thread: threading.Thread | None = None
        self._playback_thread: threading.Thread | None = None

        self._acknowledgement = "Yeah?"
        self._acknowledgement_audio: np.ndarray | None = None
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

    def enqueue(self, text: str, generation_id: int | None = None) -> None:
        """Queue text for synthesis unless a newer response supersedes it."""
        self._queue.put(("speak", text, generation_id))
        self._state.update_debug(tts_queue=self._queue.qsize())

    def prepare_acknowledgement(self, text: str = "Yeah?") -> None:
        """Cache a short acknowledgement in the synthesis worker."""
        self._acknowledgement = text
        self._queue.put(("prepare", text, None))
        self._state.update_debug(tts_queue=self._queue.qsize())

    def enqueue_acknowledgement(self, generation_id: int | None = None) -> None:
        """Queue the prepared acknowledgement if it is still current."""
        self._queue.put(("acknowledgement", None, generation_id))
        self._state.update_debug(tts_queue=self._queue.qsize())

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
            kind, text, generation_id = item
            self._state.update_debug(tts_queue=self._queue.qsize())
            if kind != "prepare" and self._is_stale(generation_id):
                self._record_stale_skip(generation_id, stage="queued")
                continue
            if kind == "prepare":
                self._prepare(text or self._acknowledgement)
            elif kind == "acknowledgement":
                self._speak_acknowledgement(generation_id)
            else:
                self._speak(text or "", generation_id=generation_id)

    def _playback_run(self) -> None:
        """Play prepared waveforms serially while synthesis stays one ahead."""
        while True:
            item = self._ready_queue.get()
            if item is None:
                return

            self._ready_slots.release()
            self._state.update_debug(tts_ready_queue=self._ready_queue.qsize())

            kind, text, audio, generation_id, synthesis_ms = item
            if self._is_stale(generation_id):
                self._record_stale_skip(generation_id, stage="prepared")
                continue

            if kind == "acknowledgement":
                self._state.record_debug("tts_started", text=text)
            self._play_waveform(
                audio,
                text=text,
                synthesis_ms=synthesis_ms,
                generation_id=generation_id,
            )
            self._state.record_debug("tts_complete", text=text)

    def _prepare(self, text: str) -> None:
        try:
            self._acknowledgement_audio = self._synthesize_text(text)
            self._state.record_debug("tts_prepared", text=text)
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error)})
            self._state.record_debug("tts_error", error=str(error), text=text)

    def _speak_acknowledgement(self, generation_id: int | None = None) -> None:
        if not self._reserve_ready_slot(generation_id):
            return

        synthesis_ms: float | None = None
        if self._acknowledgement_audio is None:
            synthesis_started = time.perf_counter()
            self._prepare(self._acknowledgement)
            synthesis_ms = (time.perf_counter() - synthesis_started) * 1000.0

        if self._acknowledgement_audio is None:
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
                generation_id,
                synthesis_ms,
            )
        )
        self._state.update_debug(tts_ready_queue=self._ready_queue.qsize())

    def _speak(
        self,
        text: str,
        generation_id: int | None = None,
    ) -> None:
        if not self._reserve_ready_slot(generation_id):
            return

        self._state.record_debug("tts_started", text=text)
        synthesis_started = time.perf_counter()
        try:
            audio = self._synthesize_text(text)
        except Exception as error:
            self._ready_slots.release()
            self._event_log.append("voice_error", {"error": str(error), "text": text})
            self._state.record_debug("tts_error", error=str(error), text=text)
            return

        synthesis_ms = (time.perf_counter() - synthesis_started) * 1000.0
        if self._is_stale(generation_id):
            self._ready_slots.release()
            self._record_stale_skip(generation_id, stage="after_synthesis")
            return

        self._ready_queue.put(("speak", text, audio, generation_id, synthesis_ms))
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
        text: str,
        synthesis_ms: float | None = None,
        generation_id: int | None = None,
    ) -> None:
        if self._is_stale(generation_id):
            self._record_stale_skip(generation_id, stage="before_playback")
            return

        played = False
        with self._state.locked():
            self._state.speaking = True
        try:
            if audio.size:
                sample_rate = int(self._config.get("voice", {}).get("sample_rate", 24_000))
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
                self._state.update_debug(
                    tts_leading_silence_ms=leading_silence_ms,
                    tts_trailing_silence_ms=trailing_silence_ms,
                )

                playback_started_at = time.perf_counter()
                if (
                    self._last_playback_ended_at is not None
                    and generation_id == self._last_playback_generation
                ):
                    gap_ms = round(
                        (playback_started_at - self._last_playback_ended_at) * 1000.0,
                        1,
                    )
                    payload["gap_ms"] = gap_ms
                    self._state.update_debug(tts_gap_ms=gap_ms)

                self._state.record_debug("tts_playback_started", **payload)
                self._play(audio, sample_rate)
                played = True
        except Exception as error:
            self._event_log.append("voice_error", {"error": str(error)})
            self._state.record_debug("tts_error", error=str(error))
        finally:
            with self._state.locked():
                self._state.speaking = False
                if played:
                    self._state.last_spoke = time.time()
            if played:
                self._last_playback_ended_at = time.perf_counter()
                self._last_playback_generation = generation_id

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

    def _synthesize_text(self, text: str) -> np.ndarray:
        if self._synthesize is None:
            self._synthesize = _make_synthesizer(self._config)
        return self._synthesize(text)


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


def _make_synthesizer(config: dict[str, Any]) -> Callable[[str], np.ndarray]:
    """Build the CPU Kokoro pipeline only inside the synthesis worker."""
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
