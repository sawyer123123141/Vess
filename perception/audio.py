"""Audio segmentation and wake-phrase matching."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

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
    ) -> None:
        self._threshold = threshold
        self._min_samples = ceil(sample_rate * min_utterance_seconds)
        self._silence_samples = ceil(sample_rate * silence_seconds)
        self._max_samples = ceil(sample_rate * max_utterance_seconds)
        self._samples: list[float] = []
        self._trailing_quiet = 0

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        """Return one finished utterance, if this block completes one."""
        for sample in np.asarray(samples).reshape(-1):
            amplitude = abs(float(sample))
            if not self._samples:
                if amplitude < self._threshold:
                    continue
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

    def _finish(self) -> np.ndarray | None:
        final_index = len(self._samples) - self._trailing_quiet
        utterance = np.asarray(self._samples[:final_index])
        self._samples = []
        self._trailing_quiet = 0
        if len(utterance) < self._min_samples:
            return None
        return utterance


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
