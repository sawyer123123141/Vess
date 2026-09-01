"""Headless behavior verification harness regressions."""

import json
import unittest

import numpy as np

from performance import PerformanceCue
from state import State
from tools.behavior_scenarios import (
    ScenarioPhase,
    apply_phase,
    get_scenario,
    phase_frame_count,
)
from tools.render_behavior_preview import (
    VerificationFailure,
    build_summary,
    calculate_metrics,
    check_invariants,
    simulate_scenario,
    verify_determinism,
)


class BehaviorVerificationTests(unittest.TestCase):
    def test_phase_frame_count(self) -> None:
        self.assertEqual(
            phase_frame_count(ScenarioPhase("thinking", 1.5, {}), 30),
            45,
        )

    def test_apply_phase_changes_only_declared_fields(self) -> None:
        state = State(mood="curious", thinking=False, person_pos=(0.2, 0.3))

        apply_phase(state, ScenarioPhase("thinking", 1.0, {"thinking": True}))

        self.assertTrue(state.thinking)
        self.assertEqual(state.mood, "curious")
        self.assertEqual(state.person_pos, (0.2, 0.3))

    def test_conversational_cycle_is_360_frames(self) -> None:
        scenario = get_scenario(
            "conversational_cycle",
            moods=["neutral"],
            performances={
                "neutral": PerformanceCue(),
                "thoughtful": PerformanceCue("thoughtful", 0.55),
                "playful": PerformanceCue("playful", 0.65),
                "emphatic": PerformanceCue("emphatic", 0.70),
            },
        )

        self.assertEqual(
            sum(phase_frame_count(phase, 30) for phase in scenario.phases),
            360,
        )

    def test_simulation_returns_native_frames_and_trace(self) -> None:
        result = simulate_scenario("conversational_cycle", fps=30, seed=1)

        self.assertEqual(len(result.frames), 360)
        self.assertEqual(len(result.trace), 360)
        self.assertTrue(all(frame.shape == (64, 64, 3) for frame in result.frames))
        self.assertTrue(all(frame.dtype == np.uint8 for frame in result.frames))
        self.assertEqual(
            [record["frame"] for record in result.trace],
            list(range(360)),
        )
        json.dumps(result.trace)

    def test_same_seed_is_deterministic(self) -> None:
        self.assertEqual(
            verify_determinism("conversational_cycle", fps=30, seed=1),
            [],
        )

    def test_bad_gaze_reports_exact_frame(self) -> None:
        result = simulate_scenario("conversational_cycle", fps=30, seed=1)
        result.trace[10]["gaze_y"] = -1.25

        failure = next(
            item
            for item in check_invariants(result)
            if item.invariant == "gaze bounds"
        )

        self.assertEqual(failure.frame, 10)
        self.assertEqual(failure.observed["gaze_y"], -1.25)

    def test_priority_conflicts_verify_full_order(self) -> None:
        result = simulate_scenario("priority_conflicts", fps=30, seed=1)

        self.assertEqual(check_invariants(result), [])

    def test_metrics_count_speaking_frames_from_trace(self) -> None:
        result = simulate_scenario("conversational_cycle", fps=30, seed=1)

        metrics = calculate_metrics(result)
        expected = sum(1 for row in result.trace if row["speaking"])

        self.assertEqual(metrics["speaking_frames"], expected)

    def test_summary_names_failure_and_observed_value(self) -> None:
        result = simulate_scenario("conversational_cycle", fps=30, seed=1)
        result.failures.append(
            VerificationFailure(
                scenario=result.scenario,
                phase="thinking",
                frame=143,
                time_seconds=4.766667,
                invariant="gaze bounds",
                observed={"gaze_y": -1.083},
            )
        )

        summary = build_summary([result], deterministic={result.scenario: True})

        self.assertIn("conversational_cycle", summary)
        self.assertIn("frame 143", summary)
        self.assertIn("gaze bounds", summary)
        self.assertIn("-1.083", summary)


if __name__ == "__main__":
    unittest.main()
