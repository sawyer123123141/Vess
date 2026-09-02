"""Regression tests for conservative expressive Chatterbox delivery."""

from __future__ import annotations

import unittest

import numpy as np

from output.tts.chatterbox_turbo import ChatterboxTurboEngine
from performance import PerformanceCue


class RecordingModel:
    sr = 24_000

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.ones(16, dtype=np.float32)


class ExpressiveTtsTests(unittest.TestCase):
    def make_engine(self) -> tuple[ChatterboxTurboEngine, RecordingModel]:
        engine = ChatterboxTurboEngine({})
        model = RecordingModel()
        engine._model = model
        return engine, model

    def test_neutral_weak_and_unknown_cues_leave_spoken_text_unchanged(self) -> None:
        engine, model = self.make_engine()

        engine.synthesize("Neutral line.", PerformanceCue())
        engine.synthesize("Barely playful.", PerformanceCue("playful", 0.59))
        engine.synthesize("Unknown delivery.", PerformanceCue("mysterious", 1.0))

        self.assertEqual(
            model.calls,
            ["Neutral line.", "Barely playful.", "Unknown delivery."],
        )

    def test_default_playful_cue_adds_one_trailing_chuckle(self) -> None:
        engine, model = self.make_engine()

        engine.synthesize("That's clever.", PerformanceCue("playful", 0.65))

        self.assertEqual(model.calls, ["That's clever. [chuckle]"])
        self.assertEqual(model.calls[0].count("[chuckle]"), 1)

    def test_cancellable_and_normal_paths_prepare_identical_expressive_text(self) -> None:
        engine, model = self.make_engine()
        cue = PerformanceCue("playful", 0.65)

        engine.synthesize("Same delivery.", cue)
        engine.synthesize_cancellable("Same delivery.", cue, lambda: False)

        self.assertEqual(
            model.calls,
            ["Same delivery. [chuckle]", "Same delivery. [chuckle]"],
        )


if __name__ == "__main__":
    unittest.main()
