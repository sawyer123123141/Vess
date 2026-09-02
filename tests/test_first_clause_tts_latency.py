"""Regression coverage for latency-sensitive first-clause streaming."""

import unittest

from brain.llm import SpeechClause, split_clauses
from performance import PerformanceCue


class FirstClauseTtsLatencyTests(unittest.TestCase):
    def test_balanced_first_comma_can_start_tts_before_sentence_finishes(self) -> None:
        chunks = [
            "Imagine floating together in a starry expanse,",
            " where our thoughts are the only light.",
        ]

        self.assertEqual(
            list(split_clauses(chunks)),
            [
                SpeechClause(
                    "Imagine floating together in a starry expanse,",
                    PerformanceCue(),
                ),
                SpeechClause(
                    "where our thoughts are the only light.",
                    PerformanceCue(),
                ),
            ],
        )

    def test_early_first_comma_does_not_split_short_intro_phrase(self) -> None:
        self.assertEqual(
            list(split_clauses(["First, then second."])),
            [SpeechClause("First, then second.", PerformanceCue())],
        )


if __name__ == "__main__":
    unittest.main()
