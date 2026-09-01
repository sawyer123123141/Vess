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
from tools.render_behavior_preview import simulate_scenario


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


if __name__ == "__main__":
    unittest.main()
