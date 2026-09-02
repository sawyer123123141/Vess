"""Voice Lab CLI contract tests."""

import subprocess
import sys
import unittest
from pathlib import Path

from tools.voice_lab import artifact_path, build_parser, with_whisper_beam


ROOT = Path(__file__).resolve().parents[1]


class VoiceLabCliTests(unittest.TestCase):
    def test_parser_accepts_endpoint_silence_and_whisper_beam_sweeps(self) -> None:
        parser = build_parser()
        endpoint = parser.parse_args(["endpoint", "--manifest", "corpus.json", "--silence", "0.30", "0.40"])
        whisper = parser.parse_args(["whisper", "--manifest", "corpus.json", "--beam-size", "1", "5"])
        self.assertEqual(endpoint.silence, [0.30, 0.40])
        self.assertEqual(whisper.beam_size, [1, 5])

    def test_whisper_variant_copies_config_without_mutating_original(self) -> None:
        config = {"whisper": {"beam_size": 5}, "audio": {"silence_seconds": 0.45}}
        variant = with_whisper_beam(config, 1)
        self.assertEqual(variant["whisper"]["beam_size"], 1)
        self.assertEqual(config["whisper"]["beam_size"], 5)
        self.assertIsNot(variant["whisper"], config["whisper"])

    def test_artifact_path_is_scoped_under_voice_lab(self) -> None:
        path = artifact_path(Path("artifacts"), "endpoint", "results.json")
        self.assertEqual(path, Path("artifacts") / "voice-lab" / "endpoint" / "results.json")

    def test_script_path_can_start_from_repository_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/voice_lab.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("endpoint", completed.stdout)
        self.assertIn("cancel", completed.stdout)


if __name__ == "__main__":
    unittest.main()
