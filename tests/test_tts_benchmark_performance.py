"""Regression test for benchmarking explicit performance cues."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from output.tts.base import SynthesisResult
from performance import PerformanceCue
from tools.benchmark_tts import run_benchmark


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, PerformanceCue]] = []

    def synthesize(self, text: str, performance: PerformanceCue) -> SynthesisResult:
        self.calls.append((text, performance))
        return SynthesisResult(np.ones(240, dtype=np.float32), 24_000)


class TtsBenchmarkPerformanceTests(unittest.TestCase):
    def test_run_benchmark_applies_one_explicit_cue_to_every_standard_text(self) -> None:
        engine = RecordingEngine()
        cue = PerformanceCue("playful", 0.65)

        with tempfile.TemporaryDirectory() as directory:
            with patch("output.tts.factory.create_tts_engine", return_value=engine):
                with patch("tools.benchmark_tts.write_wave"):
                    result = run_benchmark(
                        "chatterbox_turbo",
                        1,
                        Path(directory),
                        performance=cue,
                    )

        self.assertEqual(result, 0)
        self.assertGreater(len(engine.calls), 0)
        self.assertTrue(all(call_cue == cue for _, call_cue in engine.calls))


if __name__ == "__main__":
    unittest.main()
