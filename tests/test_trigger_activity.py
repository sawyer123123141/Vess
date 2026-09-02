"""User activity must be distinct from Vess speaking proactively."""

import unittest
from unittest.mock import patch

from brain.llm import ConversationWorker
from state import State


CONFIG = {
    "personas": {"friendly": "Warm."},
    "memory": {"short_term_minutes": 10, "short_term_turns": 8},
}


class TriggerActivityTests(unittest.TestCase):
    def test_user_submission_updates_interaction_clock_before_duplicate_collapse(self) -> None:
        state = State()
        worker = ConversationWorker(
            CONFIG,
            {"neutral": {}},
            state,
            RecordingLog(),
            object(),
            RecordingVoice(),
        )

        with patch("brain.llm.time.time", side_effect=[100.0, 200.0]):
            worker.submit("hello")
            self.assertEqual(state.last_interaction, 100.0)
            worker.submit("hello")
            self.assertEqual(state.last_interaction, 200.0)

    def test_timed_user_submission_uses_same_interaction_clock(self) -> None:
        state = State()
        worker = ConversationWorker(
            CONFIG,
            {"neutral": {}},
            state,
            RecordingLog(),
            object(),
            RecordingVoice(),
        )

        with patch("brain.llm.time.time", return_value=321.0):
            worker.submit_with_timing("hello", {"speech_ended_at": 300.0})

        self.assertEqual(state.last_interaction, 321.0)


class RecordingVoice:
    def __init__(self) -> None:
        self.generations: list[int] = []

    def begin_generation(self, generation_id: int) -> None:
        self.generations.append(generation_id)


class RecordingLog:
    def append(self, event_type, payload) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
