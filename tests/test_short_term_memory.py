"""Short-term conversation memory behavior."""

import unittest

from brain.memory import append_conversation_turn, recent_conversation_turns
from state import State


class ShortTermMemoryTests(unittest.TestCase):
    def test_append_records_completed_turn(self) -> None:
        state = State()

        turn = append_conversation_turn(
            state,
            "How was your day?",
            "Pretty quiet so far.",
            timestamp=100.0,
            max_age_seconds=600.0,
            max_turns=8,
        )

        self.assertEqual(turn.timestamp, 100.0)
        self.assertEqual(turn.user, "How was your day?")
        self.assertEqual(turn.assistant, "Pretty quiet so far.")
        self.assertEqual(turn.status, "completed")
        self.assertIsNone(turn.interrupted_clause)
        self.assertEqual(state.conversation_turns, [turn])

    def test_append_records_interrupted_turn_without_claiming_partial_clause_was_heard(self) -> None:
        state = State()

        turn = append_conversation_turn(
            state,
            "Explain rainbows",
            "Light enters the droplet.",
            timestamp=100.0,
            max_age_seconds=600.0,
            max_turns=8,
            status="interrupted",
            interrupted_clause="Then it bends and separates into colors.",
        )

        self.assertEqual(turn.user, "Explain rainbows")
        self.assertEqual(turn.assistant, "Light enters the droplet.")
        self.assertEqual(turn.status, "interrupted")
        self.assertEqual(
            turn.interrupted_clause,
            "Then it bends and separates into colors.",
        )
        self.assertEqual(state.conversation_turns, [turn])

    def test_recent_turns_prune_by_age_and_count(self) -> None:
        state = State()
        append_conversation_turn(
            state,
            "old",
            "old reply",
            timestamp=0.0,
            max_age_seconds=600.0,
            max_turns=8,
        )
        append_conversation_turn(
            state,
            "one",
            "reply one",
            timestamp=995.0,
            max_age_seconds=600.0,
            max_turns=8,
        )
        append_conversation_turn(
            state,
            "two",
            "reply two",
            timestamp=996.0,
            max_age_seconds=600.0,
            max_turns=8,
        )
        append_conversation_turn(
            state,
            "three",
            "reply three",
            timestamp=997.0,
            max_age_seconds=600.0,
            max_turns=8,
        )

        turns = recent_conversation_turns(
            state,
            now=1000.0,
            max_age_seconds=600.0,
            max_turns=2,
        )

        self.assertEqual(
            [(turn.user, turn.assistant) for turn in turns],
            [("two", "reply two"), ("three", "reply three")],
        )
        self.assertEqual(state.conversation_turns, turns)

    def test_zero_turn_limit_returns_no_history(self) -> None:
        state = State()
        append_conversation_turn(
            state,
            "hello",
            "hi",
            timestamp=100.0,
            max_age_seconds=600.0,
            max_turns=8,
        )

        turns = recent_conversation_turns(
            state,
            now=101.0,
            max_age_seconds=600.0,
            max_turns=0,
        )

        self.assertEqual(turns, [])
        self.assertEqual(state.conversation_turns, [])


if __name__ == "__main__":
    unittest.main()
