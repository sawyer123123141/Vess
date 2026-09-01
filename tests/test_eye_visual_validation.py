"""Comprehensive visual-validation coverage for Vess eye behavior."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from performance import PerformanceCue
from tools.behavior_scenarios import get_scenario


MOODS = ("neutral", "happy", "sad", "annoyed", "curious")
PERFORMANCES = {
    "neutral": PerformanceCue(),
    "thoughtful": PerformanceCue("thoughtful", 0.55),
    "playful": PerformanceCue("playful", 0.65),
    "emphatic": PerformanceCue("emphatic", 0.70),
    "uncertain": PerformanceCue("uncertain", 0.50),
}
STATE_SUFFIXES = (
    "listening",
    "thinking",
    "speaking_neutral",
    "speaking_playful",
    "speaking_emphatic",
    "speaking_uncertain",
)


class EyeVisualValidationTests(unittest.TestCase):
    def test_validation_scenario_covers_every_mood_and_state(self) -> None:
        scenario = get_scenario(
            "mood_eye_validation",
            moods=MOODS,
            performances=PERFORMANCES,
        )
        names = {phase.name for phase in scenario.phases}

        self.assertEqual(len(scenario.phases), len(MOODS) * len(STATE_SUFFIXES))
        for mood in MOODS:
            for suffix in STATE_SUFFIXES:
                self.assertIn(f"{mood}__{suffix}", names)

    def test_validation_scenario_uses_the_requested_mood_in_every_phase(self) -> None:
        scenario = get_scenario(
            "mood_eye_validation",
            moods=MOODS,
            performances=PERFORMANCES,
        )

        for phase in scenario.phases:
            mood, _ = phase.name.split("__", 1)
            self.assertEqual(phase.state["mood"], mood)

    def test_eye_validation_writes_mobile_artifacts_and_trace(self) -> None:
        module = importlib.import_module("tools.render_eye_validation")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            code, summary = module.run_eye_validation(
                output_dir=output,
                fps=30,
                seed=1,
            )
            trace = json.loads(
                (output / "eye_validation_trace.json").read_text(encoding="utf-8")
            )

            self.assertEqual(code, 0)
            self.assertTrue((output / "eye_validation.gif").is_file())
            self.assertTrue((output / "eye_validation_contact_sheet.png").is_file())
            self.assertTrue((output / "eye_validation_summary.txt").is_file())
            self.assertEqual(trace["scenario"], "mood_eye_validation")
            self.assertEqual(trace["fps"], 30)
            self.assertEqual(trace["seed"], 1)
            self.assertTrue(trace["frames"])

        for mood in MOODS:
            self.assertIn(mood, summary)
        for eye_type in ("normal", "arc", "droop", "narrow", "wide"):
            self.assertIn(eye_type, summary)


if __name__ == "__main__":
    unittest.main()
