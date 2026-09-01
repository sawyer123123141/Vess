"""Prompt and stream parsing behavior for the local LLM client."""

import json
import unittest

from brain.llm import ConversationWorker, OllamaClient, build_prompt, split_clauses
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

    def test_conversation_streams_clauses_and_logs_valid_mood_change(self) -> None:
        state = State()
        voice = RecordingVoice()
        log = RecordingLog()
        worker = ConversationWorker(
            {"personas": {"friendly": "Warm."}},
            {"neutral": {}, "annoyed": {"decay": 400}},
            state,
            log,
            FakeClient(),
            voice,
        )

        worker.start()
        worker.submit("Tell me something")
        worker.close()

        self.assertEqual(voice.clauses, ["First,", "then second."])
        self.assertEqual(state.mood, "annoyed")
        self.assertGreater(state.mood_until, 0.0)
        self.assertFalse(state.thinking)
        self.assertEqual(
            log.events,
            [("mood_changed", {"from": "neutral", "to": "annoyed"})],
        )
        events = state.debug_snapshot()["events"]
        self.assertEqual(
            [event["event"] for event in events],
            ["llm_started", "llm_first_clause", "llm_complete", "mood_changed"],
        )
        first_clause = next(event for event in events if event["event"] == "llm_first_clause")
        self.assertIn("latency_ms", first_clause)
        self.assertGreaterEqual(first_clause["latency_ms"], 0.0)


class FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        pass


class FakeClient:
    def stream(self, prompt: str, config: dict) -> list[str]:
        return ["First, then second."]

    def classify_mood(self, transcript: str, mood_names: set[str], config: dict) -> str:
        return "annoyed"


class RecordingVoice:
    def __init__(self) -> None:
        self.clauses: list[str] = []

    def begin_generation(self, generation_id: int) -> None:
        pass

    def enqueue(self, text: str, generation_id: int | None = None) -> None:
        self.clauses.append(text)

    def enqueue_acknowledgement(self, generation_id: int | None = None) -> None:
        self.clauses.append("Yeah?")


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


if __name__ == "__main__":
    unittest.main()
