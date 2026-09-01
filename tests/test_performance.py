"""Validated transient performance vocabulary."""

import unittest

from performance import PerformanceCue, cue_for_label, load_performance_definitions


class PerformanceTests(unittest.TestCase):
    def test_neutral_is_required_and_unknown_labels_fall_back(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {"intensity": 0.0},
            "playful": {"intensity": 0.65},
        })
        self.assertEqual(
            cue_for_label("playful", definitions),
            PerformanceCue("playful", 0.65),
        )
        self.assertEqual(cue_for_label("nonsense", definitions), PerformanceCue())

    def test_numeric_values_are_clamped_and_missing_blocks_default_empty(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {"intensity": 0.0},
            "emphatic": {
                "intensity": 4.0,
                "shape": {"l_h": 99.0, "r_h": -99.0},
                "movement": {"hold_scale": 99.0, "gaze_y_bias": -9.0},
            },
        })
        emphatic = definitions["emphatic"]
        self.assertEqual(emphatic["intensity"], 1.0)
        self.assertEqual(emphatic["shape"]["l_h"], 3.0)
        self.assertEqual(emphatic["shape"]["r_h"], -3.0)
        self.assertEqual(emphatic["movement"]["hold_scale"], 1.6)
        self.assertEqual(emphatic["movement"]["gaze_y_bias"], -0.35)

    def test_malformed_numeric_values_fall_back_without_breaking_neutral(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {
                "intensity": "not-a-number",
                "shape": {"l_h": None},
                "movement": {"hold_scale": "oops"},
            },
            "playful": {"intensity": 0.65},
        })

        neutral = definitions["neutral"]
        self.assertEqual(neutral["intensity"], 0.0)
        self.assertEqual(neutral["shape"]["l_h"], 0.0)
        self.assertEqual(neutral["movement"]["hold_scale"], 1.0)
        self.assertEqual(cue_for_label("neutral", definitions), PerformanceCue())

    def test_missing_neutral_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "neutral"):
            load_performance_definitions({"playful": {"intensity": 0.6}})

    def test_eye_motion_defaults_to_neutral_values(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {"intensity": 0.0},
            "playful": {"intensity": 0.5},
        })
        self.assertEqual(
            definitions["playful"]["eye_motion"],
            {"l_x": 0.0, "l_y": 0.0, "r_x": 0.0, "r_y": 0.0, "reaction": 0.0},
        )

    def test_eye_motion_is_clamped_and_nonfinite_values_fall_back(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {"intensity": 0.0},
            "playful": {
                "intensity": 0.65,
                "eye_motion": {
                    "l_x": -9,
                    "l_y": float("nan"),
                    "r_x": 4,
                    "r_y": -0.7,
                    "reaction": float("inf"),
                },
            },
        })
        eye = definitions["playful"]["eye_motion"]
        self.assertEqual(eye["l_x"], -1.5)
        self.assertEqual(eye["l_y"], 0.0)
        self.assertEqual(eye["r_x"], 1.5)
        self.assertEqual(eye["r_y"], -0.7)
        self.assertEqual(eye["reaction"], 0.0)

    def test_neutral_eye_motion_is_forced_to_zero(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {
                "intensity": 0.0,
                "eye_motion": {"l_x": 1.0, "r_y": -1.0, "reaction": 1.0},
            }
        })
        self.assertEqual(
            definitions["neutral"]["eye_motion"],
            {"l_x": 0.0, "l_y": 0.0, "r_x": 0.0, "r_y": 0.0, "reaction": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
