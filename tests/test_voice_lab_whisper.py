"""Voice Lab Whisper metric tests."""

import unittest

import numpy as np

from voice_lab.whisper import measure_transcription, word_error_rate


class VoiceLabWhisperTests(unittest.TestCase):
    def test_word_error_rate_handles_exact_substitution_insertion_and_deletion(self) -> None:
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)
        self.assertEqual(word_error_rate("hello world", "hello there"), 0.5)
        self.assertEqual(word_error_rate("hello world", "hello brave world"), 0.5)
        self.assertEqual(word_error_rate("hello brave world", "hello world"), 1 / 3)

    def test_word_error_rate_empty_reference_is_zero_only_for_empty_hypothesis(self) -> None:
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "hello"), 1.0)

    def test_measure_transcription_records_latency_rtf_and_raw_hypothesis(self) -> None:
        samples = np.ones(32_000, dtype=np.float32)
        times = iter([10.0, 10.5])
        row = measure_transcription(
            samples,
            "Hello world",
            lambda audio: "hello there",
            now=lambda: next(times),
        )
        self.assertEqual(row["transcript"], "hello there")
        self.assertEqual(row["transcription_ms"], 500.0)
        self.assertEqual(row["utterance_seconds"], 2.0)
        self.assertEqual(row["realtime_factor"], 0.25)
        self.assertEqual(row["word_error_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
