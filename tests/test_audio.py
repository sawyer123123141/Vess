"""Deterministic audio gate behavior."""

import unittest

import numpy as np

from perception.audio import WakeMatch, UtteranceAssembler, match_wake_phrase


class WakeMatchTests(unittest.TestCase):
    def test_matcher_accepts_whisper_mishear(self) -> None:
        self.assertEqual(
            match_wake_phrase("hey best tell me a joke", ["hey vess"], 2),
            WakeMatch("hey vess", 2, 2),
        )

    def test_matcher_rejects_unrelated_speech(self) -> None:
        self.assertIsNone(
            match_wake_phrase("turn on the lights", ["hey vess"], 2)
        )


class UtteranceAssemblerTests(unittest.TestCase):
    def test_assembler_emits_speech_after_trailing_silence(self) -> None:
        assembler = UtteranceAssembler(10, 0.1, 0.2, 0.3, 2.0)

        self.assertIsNone(assembler.push(np.array([0.0, 0.2, 0.2, 0.0])))
        self.assertTrue(
            np.array_equal(assembler.push(np.zeros(3)), np.array([0.2, 0.2]))
        )

if __name__ == "__main__":
    unittest.main()
