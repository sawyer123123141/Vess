"""Prompt and stream parsing behavior for the local LLM client."""

import json
import time
import unittest

from brain.llm import ConversationWorker, OllamaClient, build_prompt, split_clauses
from brain.memory import append_conversation_turn
from state import State


class LlmTests(unittest.TestCase):
    def test_split_clauses_emits_completed_punctuation(self) -> None:
        self.assertEqual(
            list(split_clauses(["First, then", " second.", " Last"])),
            ["First,", "then second.", "Last"],
        )

    def test_prompt_puts_grounded_identity_and_history_before_current_request(self) -> None:
        config = {
            "personas": {"friendly": "Warm and casual."},
            "memory": {"short_term_minutes": 10, "short_term_turns": 8},
        }
        moods = {
            "neutral": {"prompt": ""},
            "annoyed": {"prompt": "You're mildly irritated."},
        }
        state = State(persona="friendly", mood="annoyed", person_present=True)
        append_conversation_turn(
            state,
            "How was your day?",
            "Pretty quiet so far.",
            timestamp=time.time(),
            max_age_seconds=600.0,
            max_turns=8,
        )

        prompt = build_prompt(config, moods, state, "Why?")

        self.assertLess(prompt.index("You are Vess"), prompt.index("Current state:"))
        self.assertIn("human body", prompt.lower())
        self.assertIn("do not invent", prompt.lower())
        self.assertIn("Mood: annoyed. You're mildly irritated.", prompt)
        self.assertIn("Recent conversation:", prompt)
        self.assertIn("User: How was your day?", prompt)
        self.assertIn("Vess: Pretty quiet so far.", prompt)
        self.assertLess(
            prompt.index("User: How was your day?"),
            prompt.index("Current request:\nWhy?"),
        )

    def test_prompt_omits_expired_history(self) -> None:
        config = {
            "personas": {"friendly": "Warm and casual."},
            "memory": {"short_term_minutes": 1, "short_term_turns": 8},
        }
        state = State()
        append_conversation_turn(
            state,
            "ancient question",
            "ancient answer",
            timestamp=1.0,
            max_age_seconds=10_000_000_000.0,
            max_turns=8,
        )

        prompt = build_prompt(config, {"neutral": {"prompt": ""}}, state, "Hello")

        self.assertNotIn("Recent conversation:", prompt)
        self.assertNotIn("ancient question", prompt)
        self.assertIn("Current request:\nHello", prompt)

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
