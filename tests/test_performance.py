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

    def test_missing_neutral_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "neutral"):
            load_performance_definitions({"playful": {"intensity": 0.6}})


if __name__ == "__main__":
    unittest.main()
