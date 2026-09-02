"""Voice Lab endpoint replay tests."""

import unittest

import numpy as np

from voice_lab.endpointing import replay_endpoint


class VoiceLabEndpointingTests(unittest.TestCase):
    def test_internal_pause_splits_at_300ms_but_not_400ms(self) -> None:
        sample_rate = 16_000
        speech = np.full(int(sample_rate * 0.30), 0.08, dtype=np.float32)
        pause = np.zeros(int(sample_rate * 0.34), dtype=np.float32)
        samples = np.concatenate([speech, pause, speech])
        settings = {
            "sample_rate": sample_rate,
            "vad_threshold": 0.015,
            "min_utterance_seconds": 0.25,
            "max_utterance_seconds": 15.0,
            "pre_roll_seconds": 0.25,
        }

        fast = replay_endpoint(samples, settings, 0.30, expected_utterances=1)
        safe = replay_endpoint(samples, settings, 0.40, expected_utterances=1)

        self.assertEqual(fast["emitted_utterances"], 2)
        self.assertTrue(fast["premature_split"])
        self.assertFalse(fast["missed_split"])
        self.assertEqual(fast["configured_endpoint_ms"], 300.0)
        self.assertEqual(safe["emitted_utterances"], 1)
        self.assertFalse(safe["premature_split"])
        self.assertFalse(safe["missed_split"])

    def test_reports_missed_split(self) -> None:
        samples = np.full(8_000, 0.08, dtype=np.float32)
        settings = {
            "sample_rate": 16_000,
            "vad_threshold": 0.015,
            "min_utterance_seconds": 0.25,
            "max_utterance_seconds": 15.0,
            "pre_roll_seconds": 0.25,
        }
        result = replay_endpoint(samples, settings, 0.45, expected_utterances=2)
        self.assertEqual(result["emitted_utterances"], 1)
        self.assertTrue(result["missed_split"])


if __name__ == "__main__":
    unittest.main()
