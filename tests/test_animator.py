"""Deterministic conversational eye behavior and transient performance overlays."""

import unittest

from output.animator import FaceAnimator
from performance import PerformanceCue, load_performance_definitions
from state import State


MOODS = {
    "neutral": {"color": [100, 180, 255], "eye": "normal", "blink_rate": 1.0},
    "curious": {
        "color": [160, 140, 255],
        "eye": "wide",
        "blink_rate": 1.1,
        "movement": {"hold": 0.55, "spread": 1.35, "ease": 0.9, "bob": 1.1, "track_bias": 1.3},
    },
}

PERFORMANCES = load_performance_definitions({
    "neutral": {"intensity": 0.0},
    "playful": {
        "intensity": 0.65,
        "shape": {"l_h": -0.5, "r_h": 0.3, "l_slant": 0.7, "r_slant": -0.35},
        "eye_motion": {"l_x": -0.18, "l_y": 0.22, "r_x": 0.30, "r_y": -0.62, "reaction": 0.80},
        "movement": {"hold_scale": 0.75, "ease_scale": 0.85, "speaking_break_scale": 1.4},
    },
    "emphatic": {
        "intensity": 0.7,
        "shape": {"l_h": 1.2, "r_h": 1.0},
        "eye_motion": {"l_x": 0.0, "l_y": -0.32, "r_x": 0.0, "r_y": -0.27, "reaction": 0.55},
        "movement": {"track_bias_scale": 1.15, "speaking_break_scale": 0.4},
    },
    "thoughtful": {
        "intensity": 0.55,
        "shape": {"l_h": -0.5, "r_h": -0.4},
        "eye_motion": {"l_x": 0.0, "l_y": -0.16, "r_x": 0.0, "r_y": -0.24, "reaction": 0.20},
        "movement": {"hold_scale": 1.4, "ease_scale": 1.2, "gaze_y_bias": -0.22},
    },
})


class AnimatorTests(unittest.TestCase):
    def test_listening_prioritizes_person_and_suppresses_idle_fixation(self) -> None:
        state = State(listening=True, person_pos=(0.9, 0.5), person_present=True)
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        before = animator._fixation

        animator.tick(state, 0.05)

        self.assertEqual(animator._interaction_mode, "listening")
        self.assertEqual(animator._fixation, before)
        self.assertGreater(animator._gaze[0], 0.5)

    def test_thinking_breaks_eye_contact_up_and_away(self) -> None:
        state = State(thinking=True, person_pos=(0.9, 0.8), person_present=True)
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)

        animator.tick(state, 0.05)

        self.assertEqual(animator._interaction_mode, "thinking")
        self.assertLess(animator._gaze[1], 0.0)

    def test_speaking_outranks_plain_person_tracking(self) -> None:
        state = State(speaking=True, person_pos=(0.8, 0.5), person_present=True)
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)

        animator.tick(state, 0.05)

        self.assertEqual(animator._interaction_mode, "speaking")

    def test_runtime_mode_priority_is_deterministic(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        cases = [
            (State(listening=True, thinking=True, speaking=True, person_pos=(0.8, 0.5)), "listening"),
            (State(thinking=True, speaking=True, person_pos=(0.8, 0.5)), "thinking"),
            (State(speaking=True, person_pos=(0.8, 0.5)), "speaking"),
            (State(person_pos=(0.8, 0.5)), "tracking"),
            (State(), "idle"),
        ]
        for state, expected in cases:
            animator.tick(state, 0.01)
            self.assertEqual(animator._interaction_mode, expected)

    def test_neutral_performance_leaves_mood_shape_target_unchanged(self) -> None:
        state = State(mood="curious", performance=PerformanceCue())
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)

        animator.tick(state, 0.1)

        self.assertEqual(animator._performance_target["l_h"], 0.0)
        self.assertEqual(animator._performance_target["r_h"], 0.0)

    def test_performance_composes_with_mood_instead_of_replacing_it(self) -> None:
        state = State(mood="curious", performance=PerformanceCue("playful", 0.65))
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        base = animator._target_for("curious")

        animator.tick(state, 0.2)

        self.assertNotEqual(animator._last_shape["l_h"], base["l_h"])
        self.assertEqual(
            animator._last_color,
            tuple(animator.current[f"color_{channel}"] for channel in "rgb"),
        )

    def test_performance_overlay_eases_instead_of_snapping(self) -> None:
        state = State(performance=PerformanceCue("emphatic", 0.7))
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)

        animator.tick(state, 0.01)

        first = animator.performance_current["l_h"]
        target = animator._performance_target["l_h"]
        self.assertGreater(first, 0.0)
        self.assertLess(first, target)

    def test_unknown_performance_degrades_to_neutral_overlay(self) -> None:
        state = State(performance=PerformanceCue("made_up", 1.0))
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)

        animator.tick(state, 0.1)

        self.assertEqual(animator._performance_target["l_h"], 0.0)
        self.assertEqual(animator._performance_target["speaking_break_scale"], 1.0)

    def test_performance_does_not_change_mood_color(self) -> None:
        neutral = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        playful = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        neutral_state = State(mood="curious", performance=PerformanceCue())
        playful_state = State(
            mood="curious",
            performance=PerformanceCue("playful", 0.65),
        )

        for _ in range(30):
            neutral.tick(neutral_state, 1 / 30)
            playful.tick(playful_state, 1 / 30)

        self.assertEqual(
            neutral.debug_snapshot()["color"],
            playful.debug_snapshot()["color"],
        )

    def test_debug_snapshot_reports_render_values(self) -> None:
        state = State(
            speaking=True,
            person_present=True,
            person_pos=(0.8, 0.48),
            performance=PerformanceCue("playful", 0.65),
        )
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        animator.tick(state, 1 / 30)
        snap = animator.debug_snapshot()

        self.assertEqual(snap["interaction_mode"], "speaking")
        self.assertEqual(snap["render_gaze"], animator._last_render_gaze)
        self.assertEqual(snap["render_offset"], animator._last_render_offset)
        self.assertEqual(snap["shape"], animator._last_shape)
        self.assertEqual(snap["color"], animator._last_color)

    def test_debug_snapshot_copies_mutable_data(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        animator.tick(State(), 1 / 30)
        snap = animator.debug_snapshot()
        snap["shape"]["l_h"] = 999.0
        snap["performance_current"]["hold_scale"] = 999.0
        fresh = animator.debug_snapshot()

        self.assertNotEqual(fresh["shape"]["l_h"], 999.0)
        self.assertNotEqual(fresh["performance_current"]["hold_scale"], 999.0)

    def test_thinking_mode_adds_asymmetric_eye_body_lift(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        state = State(thinking=True, performance=PerformanceCue())
        for _ in range(20):
            animator.tick(state, 1 / 30)

        snap = animator.debug_snapshot()
        self.assertAlmostEqual(snap["left_eye_settled_target"][1], -0.12, places=6)
        self.assertAlmostEqual(snap["right_eye_settled_target"][1], -0.22, places=6)

    def test_non_neutral_performance_reacts_then_settles(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        state = State(performance=PerformanceCue("playful", 0.65))

        animator.tick(state, 1 / 30)
        first = animator.debug_snapshot()
        self.assertEqual(first["reaction_phase"], "entry")

        for _ in range(12):
            animator.tick(state, 1 / 30)
        settled = animator.debug_snapshot()
        self.assertEqual(settled["reaction_phase"], "hold")
        self.assertAlmostEqual(
            settled["left_eye_offset"][0],
            settled["left_eye_settled_target"][0],
            delta=0.03,
        )

    def test_release_to_neutral_has_no_overshoot_and_returns_to_mode_target(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        state = State(performance=PerformanceCue("playful", 0.65))
        for _ in range(12):
            animator.tick(state, 1 / 30)

        state.performance = PerformanceCue()
        previous_distance = None
        for _ in range(8):
            animator.tick(state, 1 / 30)
            snap = animator.debug_snapshot()
            left = snap["left_eye_offset"]
            target = snap["left_eye_settled_target"]
            distance = abs(left[0] - target[0]) + abs(left[1] - target[1])
            if previous_distance is not None:
                self.assertLessEqual(distance, previous_distance + 1e-6)
            previous_distance = distance

        final = animator.debug_snapshot()
        self.assertEqual(final["reaction_phase"], "hold")
        self.assertAlmostEqual(final["left_eye_offset"][0], 0.0, delta=1e-6)
        self.assertAlmostEqual(final["left_eye_offset"][1], 0.0, delta=1e-6)

    def test_direct_cue_change_starts_from_current_without_neutral_frame(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        state = State(performance=PerformanceCue("playful", 0.65))
        for _ in range(5):
            animator.tick(state, 1 / 30)
        before = animator.debug_snapshot()["left_eye_offset"]

        state.performance = PerformanceCue("emphatic", 0.70)
        animator.tick(state, 1 / 30)
        after = animator.debug_snapshot()

        self.assertEqual(after["reaction_phase"], "entry")
        self.assertNotEqual(after["left_eye_settled_target"], (0.0, 0.0))
        self.assertLess(
            abs(after["left_eye_offset"][0] - before[0])
            + abs(after["left_eye_offset"][1] - before[1]),
            0.5,
        )

    def test_same_expression_intensity_change_does_not_restart_reaction(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        state = State(performance=PerformanceCue("playful", 0.65))
        for _ in range(12):
            animator.tick(state, 1 / 30)
        self.assertEqual(animator.debug_snapshot()["reaction_phase"], "hold")

        state.performance = PerformanceCue("playful", 0.30)
        animator.tick(state, 1 / 30)
        self.assertNotEqual(animator.debug_snapshot()["reaction_phase"], "entry")

    def test_blink_does_not_reset_eye_motion(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        state = State(performance=PerformanceCue("playful", 0.65))
        for _ in range(12):
            animator.tick(state, 1 / 30)
        before = animator.debug_snapshot()["left_eye_offset"]

        animator.blink_phase = 0.1
        animator.tick(state, 1 / 30)
        after = animator.debug_snapshot()["left_eye_offset"]

        self.assertLess(
            abs(after[0] - before[0]) + abs(after[1] - before[1]),
            0.15,
        )

    def test_unknown_performance_does_not_start_eye_reaction(self) -> None:
        animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
        state = State(performance=PerformanceCue("made_up", 1.0))

        animator.tick(state, 1 / 30)
        snap = animator.debug_snapshot()

        self.assertEqual(snap["reaction_phase"], "hold")
        self.assertEqual(snap["left_eye_settled_target"], (0.0, 0.0))
        self.assertEqual(snap["right_eye_settled_target"], (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
