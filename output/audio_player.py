"""Cancellable, resumable audio playback with render-reference publication."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Callable, Protocol

import numpy as np

from perception.audio_preprocess import RenderedAudioBlock


@dataclass(frozen=True)
class PlaybackReceipt:
    status: str
    generation_id: int | None
    frames_started: int
    frames_completed: int
    total_frames: int
    sample_rate: int


class AudioPlayer(Protocol):
    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generation_id: int | None,
    ) -> PlaybackReceipt:
        raise NotImplementedError

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        raise NotImplementedError

    def resume(self) -> PlaybackReceipt:
        raise NotImplementedError

    def discard_paused(self) -> None:
        raise NotImplementedError


class _OutputBackend(Protocol):
    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        on_render: Callable[[np.ndarray, int], None],
    ) -> int:
        raise NotImplementedError

    def abort(self) -> None:
        raise NotImplementedError


@dataclass
class _CurrentPlayback:
    audio: np.ndarray
    sample_rate: int
    generation_id: int | None
    start_frame: int
    total_frames: int
    pause_requested: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    receipt: PlaybackReceipt | None = None


@dataclass(frozen=True)
class _PausedPlayback:
    audio: np.ndarray
    sample_rate: int
    generation_id: int | None
    completed_frames: int


class CallbackAudioPlayer:
    """Compatibility player for the existing blocking two-argument callbacks."""

    def __init__(self, callback: Callable[[np.ndarray, int], None]) -> None:
        self._callback = callback

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generation_id: int | None,
    ) -> PlaybackReceipt:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._callback(samples, sample_rate)
        return PlaybackReceipt(
            status="completed",
            generation_id=generation_id,
            frames_started=0,
            frames_completed=samples.size,
            total_frames=samples.size,
            sample_rate=sample_rate,
        )

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        return None

    def resume(self) -> PlaybackReceipt:
        raise RuntimeError("callback playback cannot resume an interrupted waveform")

    def discard_paused(self) -> None:
        return None


class SoundDeviceAudioPlayer:
    """Own one physical waveform and make its remainder resumable after abort."""

    def __init__(
        self,
        *,
        backend: _OutputBackend | None = None,
        render_callback: Callable[[RenderedAudioBlock], None] | None = None,
    ) -> None:
        self._backend = backend or _SoundDeviceOutputBackend()
        self._render_callback = render_callback
        self._lock = threading.Lock()
        self._current: _CurrentPlayback | None = None
        self._paused: _PausedPlayback | None = None

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generation_id: int | None,
    ) -> PlaybackReceipt:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        return self._play_segment(
            samples,
            sample_rate,
            generation_id,
            start_frame=0,
        )

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        with self._lock:
            if self._current is None:
                if self._paused is None:
                    return None
                paused = self._paused
                return PlaybackReceipt(
                    status="paused",
                    generation_id=paused.generation_id,
                    frames_started=0,
                    frames_completed=paused.completed_frames,
                    total_frames=paused.audio.size,
                    sample_rate=paused.sample_rate,
                )
            current = self._current
            current.pause_requested = True
            done = current.done

        self._backend.abort()
        if not done.wait(timeout=1.0):
            return None
        receipt = current.receipt
        if receipt is None or receipt.status != "paused":
            return None
        return receipt

    def resume(self) -> PlaybackReceipt:
        with self._lock:
            paused = self._paused
            if paused is None:
                raise RuntimeError("no paused waveform to resume")
            self._paused = None

        return self._play_segment(
            paused.audio,
            paused.sample_rate,
            paused.generation_id,
            start_frame=paused.completed_frames,
        )

    def discard_paused(self) -> None:
        with self._lock:
            self._paused = None

    def _play_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generation_id: int | None,
        *,
        start_frame: int,
    ) -> PlaybackReceipt:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        total_frames = int(audio.size)
        start = min(max(int(start_frame), 0), total_frames)
        current = _CurrentPlayback(
            audio=audio,
            sample_rate=sample_rate,
            generation_id=generation_id,
            start_frame=start,
            total_frames=total_frames,
        )

        with self._lock:
            if self._current is not None:
                raise RuntimeError("audio playback is already active")
            self._current = current

        completed_segment = 0
        backend_error = False
        try:
            completed_segment = int(
                self._backend.play(
                    audio[start:],
                    sample_rate,
                    self._publish_rendered,
                )
            )
        except Exception:
            backend_error = True

        completed_segment = min(max(completed_segment, 0), total_frames - start)
        completed = start + completed_segment

        with self._lock:
            if backend_error:
                status = "error"
            elif current.pause_requested and completed < total_frames:
                status = "paused"
            else:
                status = "completed"

            receipt = PlaybackReceipt(
                status=status,
                generation_id=generation_id,
                frames_started=start,
                frames_completed=completed,
                total_frames=total_frames,
                sample_rate=sample_rate,
            )
            current.receipt = receipt
            if status == "paused":
                self._paused = _PausedPlayback(
                    audio=audio,
                    sample_rate=sample_rate,
                    generation_id=generation_id,
                    completed_frames=completed,
                )
            if self._current is current:
                self._current = None
            current.done.set()

        return receipt

    def _publish_rendered(self, samples: np.ndarray, sample_rate: int) -> None:
        if self._render_callback is None:
            return
        block = RenderedAudioBlock(
            samples=np.asarray(samples, dtype=np.float32).reshape(-1).copy(),
            sample_rate=sample_rate,
            dac_time=None,
        )
        self._render_callback(block)


class _SoundDeviceOutputBackend:
    """Small stream-level sounddevice backend; imported only on actual playback."""

    def __init__(self, chunk_frames: int = 256) -> None:
        self._chunk_frames = max(1, int(chunk_frames))
        self._lock = threading.Lock()
        self._stream = None
        self._abort_requested = threading.Event()

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        on_render: Callable[[np.ndarray, int], None],
    ) -> int:
        import sounddevice as sound_device

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._abort_requested.clear()
        stream = sound_device.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        with self._lock:
            self._stream = stream

        completed = 0
        try:
            stream.start()
            while completed < samples.size and not self._abort_requested.is_set():
                end = min(completed + self._chunk_frames, samples.size)
                chunk = samples[completed:end]
                try:
                    stream.write(chunk.reshape(-1, 1))
                except Exception:
                    if self._abort_requested.is_set():
                        break
                    raise
                completed = end
                on_render(chunk, sample_rate)
            if not self._abort_requested.is_set():
                stream.stop()
        finally:
            try:
                stream.close()
            finally:
                with self._lock:
                    if self._stream is stream:
                        self._stream = None
        return completed

    def abort(self) -> None:
        self._abort_requested.set()
        with self._lock:
            stream = self._stream
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            return
