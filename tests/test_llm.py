"""Prompt and stream parsing behavior for the local LLM client."""

import json
import unittest

from brain.llm import OllamaClient, build_prompt, split_clauses
from state import State


class LlmTests(unittest.TestCase):
    def test_split_clauses_emits_completed_punctuation(self) -> None:
        self.assertEqual(
            list(split_clauses(["First, then", " second.", " Last"])),
            ["First,", "then second.", "Last"],
        )

    def test_prompt_puts_stable_identity_before_dynamic_state(self) -> None:
        config = {
            "personas": {"friendly": "Warm and casual."},
        }
        state = State(persona="friendly", mood="annoyed", person_present=True)

        prompt = build_prompt(config, state, "What time is it?")

        self.assertLess(prompt.index("You are Vess."), prompt.index("Current state:"))
        self.assertIn("Mood: annoyed", prompt)
        self.assertIn("Request: What time is it?", prompt)

    def test_stream_uses_local_generate_json_lines(self) -> None:
        requests = []

        def open_request(request, *, timeout):
            requests.append((request, timeout))
            return FakeResponse([b'{"response":"Hello"}\n', b'{"response":"."}\n'])

        client = OllamaClient(opener=open_request)
        chunks = list(client.stream("Prompt", {"llm": {"num_predict": 80}}))

        self.assertEqual(chunks, ["Hello", "."])
        self.assertTrue(requests[0][0].full_url.endswith("/api/generate"))
        self.assertEqual(json.loads(requests[0][0].data)["options"], {
            "num_ctx": 4096,
            "num_predict": 80,
        })


class FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
