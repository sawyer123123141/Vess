"""Deterministic sustained-speech interruption detection."""

from __future__ import annotations

from math import ceil

import numpy as np


class InterruptionDetector:
    """Detect sustained near-end speech without treating waveform zero crossings as silence."""

    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        pause_after_speech_seconds: float,
    ) -> None:
        self._threshold = float(threshold)
        self._required = max(1, ceil(sample_rate * pause_after_speech_seconds))
        self._frame_samples = max(1, ceil(sample_rate * 0.020))
        self._quiet_reset_samples = max(1, ceil(sample_rate * 0.100))
        self._progress_samples = 0
        self._quiet_samples = 0
        self._pending = np.asarray([], dtype=np.float32)
        self._emitted = False

    def push(self, samples: np.ndarray) -> bool:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size:
            self._pending = np.concatenate((self._pending, incoming))

        while self._pending.size >= self._frame_samples:
            frame = self._pending[: self._frame_samples]
            self._pending = self._pending[self._frame_samples :]
            frame_peak = float(np.max(np.abs(frame))) if frame.size else 0.0

            if frame_peak >= self._threshold:
                self._progress_samples += frame.size
                self._quiet_samples = 0
            elif self._progress_samples:
                self._progress_samples += frame.size
                self._quiet_samples += frame.size
                if self._quiet_samples >= self._quiet_reset_samples:
                    self._progress_samples = 0
                    self._quiet_samples = 0
                    self._emitted = False

            if self._progress_samples >= self._required and not self._emitted:
                self._emitted = True
                return True

        return False

    def reset(self) -> None:
        self._progress_samples = 0
        self._quiet_samples = 0
        self._pending = np.asarray([], dtype=np.float32)
        self._emitted = False
