"""Capture preprocessing contracts for barge-in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CapturedAudioBlock:
    samples: np.ndarray
    adc_time: float | None
    received_at: float


@dataclass(frozen=True)
class RenderedAudioBlock:
    samples: np.ndarray
    sample_rate: int
    dac_time: float | None


class CapturePreprocessor(Protocol):
    def push_render_reference(self, block: RenderedAudioBlock) -> None:
        raise NotImplementedError

    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray:
        raise NotImplementedError


class PassthroughCapturePreprocessor:
    def push_render_reference(self, block: RenderedAudioBlock) -> None:
        return None

    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray:
        return np.asarray(block.samples, dtype=np.float32).reshape(-1).copy()
