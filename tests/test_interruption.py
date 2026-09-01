"""Deterministic sustained-speech interruption detection."""

import unittest

import numpy as np

from perception.interruption import InterruptionDetector


class InterruptionDetectorTests(unittest.TestCase):
    def test_requires_sustained_speech(self) -> None:
        detector = InterruptionDetector(10, 0.1, 0.3)

        self.assertFalse(detector.push(np.array([0.2, 0.2])))
        self.assertTrue(detector.push(np.array([0.2])))
        self.assertFalse(detector.push(np.array([0.2])))

    def test_realistic_zero_crossing_waveform_counts_as_sustained_speech(self) -> None:
        sample_rate = 16_000
        detector = InterruptionDetector(sample_rate, 0.015, 0.25)
        time_axis = np.arange(int(sample_rate * 0.30), dtype=np.float32) / sample_rate
        speech_like = (0.04 * np.sin(2.0 * np.pi * 180.0 * time_axis)).astype(np.float32)

        emitted = False
        block_frames = 320
        for start in range(0, speech_like.size, block_frames):
            emitted = detector.push(speech_like[start : start + block_frames]) or emitted

        self.assertTrue(emitted)

    def test_quiet_resets_progress(self) -> None:
        detector = InterruptionDetector(10, 0.1, 0.3)

        self.assertFalse(detector.push(np.array([0.2, 0.2])))
        self.assertFalse(detector.push(np.array([0.0])))
        self.assertFalse(detector.push(np.array([0.2, 0.2])))

    def test_reset_allows_future_candidate(self) -> None:
        detector = InterruptionDetector(10, 0.1, 0.2)

        self.assertTrue(detector.push(np.array([0.2, 0.2])))
        detector.reset()
        self.assertTrue(detector.push(np.array([0.2, 0.2])))


if __name__ == "__main__":
    unittest.main()
