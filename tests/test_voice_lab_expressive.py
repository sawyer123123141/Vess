"""Voice Lab expressive-TTS forwarding tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from performance import PerformanceCue
from tools.voice_lab import _run_tts, build_parser


class VoiceLabExpressiveTests(unittest.TestCase):
    def test_tts_parser_accepts_expression_and_intensity(self) -> None:
        args = build_parser().parse_args(
            [
                "tts",
                "--engine",
                "chatterbox_turbo",
                "--runs",
                "2",
                "--expression",
                "playful",
                "--intensity",
                "0.65",
            ]
        )

        self.assertEqual(args.expression, "playful")
        self.assertEqual(args.intensity, 0.65)

    def test_tts_command_forwards_cue_and_keeps_expressive_artifacts_separate(self) -> None:
        args = build_parser().parse_args(
            [
                "tts",
                "--engine",
                "chatterbox_turbo",
                "--runs",
                "2",
                "--expression",
                "playful",
                "--intensity",
                "0.65",
            ]
        )

        with patch("tools.voice_lab.run_benchmark", return_value=0) as benchmark:
            result = _run_tts(args)

        self.assertEqual(result, 0)
        call = benchmark.call_args
        self.assertEqual(call.args[0], "chatterbox_turbo")
        self.assertEqual(call.args[1], 2)
        self.assertEqual(
            call.args[2].parts[-2:],
            ("chatterbox_turbo", "playful-0.65"),
        )
        self.assertEqual(call.kwargs["performance"], PerformanceCue("playful", 0.65))


if __name__ == "__main__":
    unittest.main()
