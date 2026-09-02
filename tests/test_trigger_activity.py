"""User activity must be distinct from Vess speaking proactively."""

import unittest
from unittest.mock import patch

from brain.proactive import ProactiveConversationWorker
from state import State


CONFIG = {
    "personas": {"friendly": "Warm."},
    "memory": {"short_term_minutes": 10, "short_term_turns": 8},
}


class TriggerActivityTests(unittest.TestCase):
    def test_user_submission_updates_interaction_clock_before_duplicate_collapse(self) -> None:
        state = State()
        worker = ProactiveConversationWorker(
            CONFIG,
            {"neutral": {}},
            state,
            RecordingLog(),
            object(),
            RecordingVoice(),
        )

        with patch("brain.proactive.time.time", side_effect=[100.0, 200.0]):
            worker.submit("hello")
            self.assertEqual(state.last_interaction, 100.0)
            worker.submit("hello")
            self.assertEqual(state.last_interaction, 200.0)

    def test_timed_user_submission_uses_same_interaction_clock(self) -> None:
        state = State()
        worker = ProactiveConversationWorker(
            CONFIG,
            {"neutral": {}},
            state,
            RecordingLog(),
            object(),
            RecordingVoice(),
        )

        with patch("brain.proactive.time.time", return_value=321.0):
            worker.submit_with_timing("hello", {"speech_ended_at": 300.0})

        self.assertEqual(state.last_interaction, 321.0)

    def test_proactive_submission_never_updates_user_interaction_clock(self) -> None:
        state = State(last_interaction=123.0)
        worker = ProactiveConversationWorker(
            CONFIG,
            {"neutral": {}},
            state,
            RecordingLog(),
            object(),
            RecordingVoice(),
        )

        with patch("brain.proactive.time.time", return_value=999.0):
            accepted = worker.submit_proactive(
                "quiet_interaction",
                "The room has been quiet between Vess and the person for about 30 minutes.",
            )

        self.assertTrue(accepted)
        self.assertEqual(state.last_interaction, 123.0)


class RecordingVoice:
    def begin_generation(self, generation_id: int) -> None:
        pass


class RecordingLog:
    def append(self, event_type, payload) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
