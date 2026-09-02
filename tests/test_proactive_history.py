"""Assistant-only proactive memory must not render a fake blank user turn."""

import time
import unittest

from brain.llm import build_prompt
from brain.memory import append_conversation_turn
from state import State


class ProactiveHistoryTests(unittest.TestCase):
    def test_assistant_only_turn_omits_blank_user_history_line(self) -> None:
        state = State()
        append_conversation_turn(
            state,
            "",
            "Back in the room, I see.",
            timestamp=time.time(),
            max_age_seconds=600.0,
            max_turns=8,
        )

        prompt = build_prompt(
            {
                "personas": {"friendly": "Warm."},
                "memory": {"short_term_minutes": 10, "short_term_turns": 8},
            },
            {"neutral": {"prompt": ""}},
            state,
            "I know.",
        )

        self.assertIn("Vess: Back in the room, I see.", prompt)
        self.assertNotIn("User: \nVess: Back in the room, I see.", prompt)
        self.assertIn("Current request:\nI know.", prompt)


if __name__ == "__main__":
    unittest.main()
