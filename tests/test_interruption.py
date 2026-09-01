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
