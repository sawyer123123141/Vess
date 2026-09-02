"""Ollama durable-fact extraction contract."""

from __future__ import annotations

import json
import unittest

from brain.llm import OllamaClient
from brain.memory import FactCandidate


class FactExtractionTests(unittest.TestCase):
    def test_valid_json_becomes_fact_candidates(self) -> None:
        client = OllamaClient(
            opener=SingleResponseOpener(
                [{"key": "dog_name", "value": "Rex"}]
            )
        )

        facts = client.extract_facts(
            "My dog's name is Rex.",
            ("favorite_color",),
            {"llm": {"model": "qwen2.5:7b", "num_predict": 80}},
        )

        self.assertEqual(facts, [FactCandidate("dog_name", "Rex")])

    def test_malformed_or_wrong_shape_returns_no_candidates(self) -> None:
        for response_text in ("not json", '{"key":"dog_name","value":"Rex"}', "null"):
            with self.subTest(response_text=response_text):
                client = OllamaClient(opener=RawResponseOpener(response_text))
                self.assertEqual(
                    client.extract_facts("My dog's name is Rex.", (), {}),
                    [],
                )

    def test_invalid_items_are_skipped_and_output_is_bounded(self) -> None:
        client = OllamaClient(
            opener=SingleResponseOpener(
                [
                    "wrong",
                    {"key": 12, "value": "bad"},
                    {"key": "first_fact", "value": "one"},
                    {"key": "second_fact", "value": "two"},
                    {"key": "third_fact", "value": "three"},
                    {"key": "fourth_fact", "value": "four"},
                ]
            )
        )

        facts = client.extract_facts("Several facts", (), {})

        self.assertEqual(
            facts,
            [
                FactCandidate("first_fact", "one"),
                FactCandidate("second_fact", "two"),
                FactCandidate("third_fact", "three"),
            ],
        )

    def test_prompt_is_explicit_bounded_and_reuses_known_keys(self) -> None:
        opener = SingleResponseOpener([])
        client = OllamaClient(opener=opener)

        client.extract_facts(
            "Actually my favorite color is green.",
            ("favorite_color", "dog_name"),
            {"llm": {"model": "qwen2.5:7b", "num_predict": 80}},
        )

        payload = opener.payloads[0]
        prompt = payload["prompt"].lower()
        self.assertIn("explicitly stated", prompt)
        self.assertIn("json only", prompt)
        self.assertIn("at most three", prompt)
        self.assertIn("favorite_color", prompt)
        self.assertIn("dog_name", prompt)
        self.assertIn("existing key", prompt)
        self.assertIn("password", prompt)
        self.assertIn("token", prompt)
        self.assertIn("credential", prompt)
        self.assertIn("temporary", prompt)
        self.assertIn("sensitive", prompt)
        self.assertIn("actually my favorite color is green", prompt)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 80)


class ReadResponse:
    def __init__(self, response_text: str) -> None:
        self._body = json.dumps({"response": response_text}).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


class SingleResponseOpener:
    def __init__(self, facts: list[object]) -> None:
        self._response_text = json.dumps(facts)
        self.payloads: list[dict[str, object]] = []

    def __call__(self, request, *, timeout: int):
        self.payloads.append(json.loads(request.data))
        return ReadResponse(self._response_text)


class RawResponseOpener:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def __call__(self, request, *, timeout: int):
        return ReadResponse(self._response_text)


if __name__ == "__main__":
    unittest.main()
