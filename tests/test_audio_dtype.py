"""Regression coverage for Whisper/Silero audio dtypes."""

import unittest

import numpy as np

from perception.audio import UtteranceAssembler


class AudioDtypeTests(unittest.TestCase):
    def test_assembler_emits_float32_with_pre_roll(self) -> None:
        assembler = UtteranceAssembler(10, 0.1, 0.2, 0.3, 2.0)

        self.assertIsNone(
            assembler.push(
                np.array([0.05, 0.05, 0.2, 0.2, 0.0], dtype=np.float32)
            )
        )
        utterance = assembler.push(np.zeros(2, dtype=np.float32))

        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
