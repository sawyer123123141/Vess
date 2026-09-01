"""Deterministic sustained-speech interruption detection."""

from __future__ import annotations

from math import ceil

import numpy as np


class InterruptionDetector:
    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        pause_after_speech_seconds: float,
    ) -> None:
        self._threshold = float(threshold)
        self._required = max(1, ceil(sample_rate * pause_after_speech_seconds))
        self._audible_samples = 0
        self._emitted = False

    def push(self, samples: np.ndarray) -> bool:
        for sample in np.asarray(samples).reshape(-1):
            if abs(float(sample)) < self._threshold:
                self._audible_samples = 0
                self._emitted = False
                continue
            self._audible_samples += 1
            if self._audible_samples >= self._required and not self._emitted:
                self._emitted = True
                return True
        return False

    def reset(self) -> None:
        self._audible_samples = 0
        self._emitted = False
