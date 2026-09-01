"""Tests for model-independent TTS benchmark measurement and output."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from output.tts.base import SynthesisResult
from tests.tts_fakes import FakeTTSEngine
from tools.benchmark_tts import measure_one, write_results, write_wave


class TtsBenchmarkTests(unittest.TestCase):
    def test_measure_one_records_latency_duration_and_rtf(self) -> None:
        times = iter([10.0, 10.25])
        engine = FakeTTSEngine(
            lambda text, performance: SynthesisResult(
                np.ones(24_000, dtype=np.float32),
                24_000,
            )
        )

        row, result = measure_one(
            "fake",
            engine,
            "one",
            "hello",
            0,
            lambda: next(times),
        )

        self.assertTrue(row["success"])
        self.assertEqual(row["synthesis_ms"], 250.0)
        self.assertEqual(row["audio_duration_ms"], 1000.0)
        self.assertEqual(row["realtime_factor"], 0.25)
        self.assertEqual(row["sample_rate"], 24_000)
        self.assertEqual(row["sample_count"], 24_000)
        self.assertFalse(row["warm"])
        self.assertIs(result.audio.dtype.type, np.float32)

    def test_zero_length_audio_has_no_realtime_factor(self) -> None:
        times = iter([4.0, 4.1])
        engine = FakeTTSEngine(
            lambda text, performance: SynthesisResult(
                np.array([], dtype=np.float32),
                24_000,
            )
        )

        row, result = measure_one(
            "fake",
            engine,
            "empty",
            "",
            1,
            lambda: next(times),
        )

        self.assertTrue(row["success"])
        self.assertTrue(row["warm"])
        self.assertIsNone(row["realtime_factor"])
        self.assertEqual(row["audio_duration_ms"], 0.0)
        self.assertEqual(result.audio.size, 0)

    def test_failed_synthesis_records_error_without_result(self) -> None:
        def fail(text, performance):
            raise RuntimeError("model exploded")

        times = iter([1.0, 1.2])
        row, result = measure_one(
            "fake",
            FakeTTSEngine(fail),
            "broken",
            "hello",
            0,
            lambda: next(times),
        )

        self.assertFalse(row["success"])
        self.assertIn("model exploded", row["error"])
        self.assertIsNone(result)

    def test_write_results_emits_json_rows(self) -> None:
        rows = [{"engine": "fake", "success": True}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            write_results(path, rows)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), rows)

    def test_write_wave_uses_engine_sample_rate(self) -> None:
        calls: list[tuple[str, np.ndarray, int]] = []
        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.write = lambda path, audio, sample_rate: calls.append(
            (str(path), audio.copy(), sample_rate)
        )
        result = SynthesisResult(np.array([0.1, -0.1], dtype=np.float32), 16_000)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            old = sys.modules.get("soundfile")
            sys.modules["soundfile"] = fake_soundfile
            try:
                write_wave(path, result)
            finally:
                if old is None:
                    sys.modules.pop("soundfile", None)
                else:
                    sys.modules["soundfile"] = old

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], str(path))
        np.testing.assert_array_equal(calls[0][1], result.audio)
        self.assertEqual(calls[0][2], 16_000)


if __name__ == "__main__":
    unittest.main()
