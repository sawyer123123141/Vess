"""Tests for the lightweight TTS engine contract."""

import unittest

import numpy as np

from output.tts.base import SynthesisResult


class SynthesisResultTests(unittest.TestCase):
    def test_accepts_one_dimensional_float32_audio(self) -> None:
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)

        result = SynthesisResult(audio, 24_000)

        self.assertIs(result.audio, audio)
        self.assertEqual(result.sample_rate, 24_000)

    def test_rejects_non_float32_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "float32"):
            SynthesisResult(np.array([0.0], dtype=np.float64), 24_000)

    def test_rejects_non_1d_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            SynthesisResult(np.zeros((1, 3), dtype=np.float32), 24_000)

    def test_rejects_non_positive_sample_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            SynthesisResult(np.zeros(1, dtype=np.float32), 0)


if __name__ == "__main__":
    unittest.main()
