"""Generation-safe command routing through the existing conversation lifecycle."""

import threading
import unittest

from brain.commands import CommandRegistry
from brain.llm import ConversationWorker
from state import State


CONFIG = {
    "personas": {"friendly": "Warm."},
    "memory": {"short_term_minutes": 10, "short_term_turns": 8},
    "commands": {"colors": {"blue": [10, 20, 30], "red": [200, 40, 30]}},
}


class CommandFlowTests(unittest.TestCase):
    def test_ordinary_conversation_never_calls_command_selector(self) -> None:
        state = State()
        client = RecordingClient(selection={"name": "set_color", "arguments": {"name": "blue"}})
        voice = DeliveringVoice()
        worker = _worker(state, client, voice)
        voice.delivery_callback = worker.handle_delivery

        worker.start()
        worker.submit("why is the sky blue?")
        worker.close()

        self.assertEqual(client.select_requests, [])
        self.assertEqual(client.stream_requests, 1)
        self.assertEqual([text for _, text in voice.clauses], ["Normal answer."])
        with state.locked():
            self.assertIsNone(state.color)

    def test_valid_command_executes_once_and_uses_delivered_turn_lifecycle(self) -> None:
        state = State()
        log = RecordingLog()
        client = RecordingClient(selection={"name": "set_color", "arguments": {"name": "blue"}})
        voice = DeliveringVoice()
        worker = _worker(state, client, voice, log=log)
        voice.delivery_callback = worker.handle_delivery

        worker.start()
        worker.submit("turn blue")
        worker.close()

        with state.locked():
            self.assertEqual(state.color, (10, 20, 30))
        self.assertEqual(client.select_requests, ["turn blue"])
        self.assertEqual(client.stream_requests, 0)
        self.assertEqual(client.mood_requests, [])
        self.assertEqual([text for _, text in voice.clauses], ["Blue."])
        self.assertEqual(
            [(turn.user, turn.assistant) for turn in state.conversation_turns],
            [("turn blue", "Blue.")],
        )
        self.assertIn(
            (
                "command_executed",
                {"name": "set_color", "arguments": {"name": "blue"}},
            ),
            log.events,
        )

    def test_invalid_selector_output_falls_through_to_normal_conversation(self) -> None:
        state = State()
        client = RecordingClient(
            selection={"name": "set_color", "arguments": {"name": "green"}}
        )
        voice = DeliveringVoice()
        worker = _worker(state, client, voice)
        voice.delivery_callback = worker.handle_delivery

        worker.start()
        worker.submit("turn blue")
        worker.close()

        self.assertEqual(client.select_requests, ["turn blue"])
        self.assertEqual(client.stream_requests, 1)
        self.assertEqual([text for _, text in voice.clauses], ["Normal answer."])
        with state.locked():
            self.assertIsNone(state.color)
        events = state.debug_snapshot()["events"]
        self.assertIn("command_selection_rejected", [event["event"] for event in events])

    def test_selector_error_falls_through_and_records_debug_failure(self) -> None:
        state = State()
        client = RecordingClient(selection_error=RuntimeError("selector unavailable"))
        voice = DeliveringVoice()
        worker = _worker(state, client, voice)
        voice.delivery_callback = worker.handle_delivery

        worker.start()
        worker.submit("turn blue")
        worker.close()

        self.assertEqual(client.stream_requests, 1)
        self.assertEqual([text for _, text in voice.clauses], ["Normal answer."])
        events = state.debug_snapshot()["events"]
        error = next(event for event in events if event["event"] == "command_selection_error")
        self.assertEqual(error["error"], "selector unavailable")

    def test_superseded_selection_never_executes_stale_command(self) -> None:
        state = State()
        client = BlockingSelectorClient()
        voice = DeliveringVoice()
        worker = _worker(state, client, voice)
        voice.delivery_callback = worker.handle_delivery
        worker.start()

        worker.submit("turn blue")
        self.assertTrue(client.selection_started.wait(timeout=0.5))
        worker.submit("what time is it?")
        client.release_selection.set()
        worker.close()

        with state.locked():
            self.assertIsNone(state.color)
        self.assertEqual([text for _, text in voice.clauses], ["Fresh answer."])
        self.assertEqual(client.stream_requests, 1)
        self.assertIn(
            "stale_response_cancelled",
            [event["event"] for event in state.debug_snapshot()["events"]],
        )


def _worker(state, client, voice, *, log=None):
    return ConversationWorker(
        CONFIG,
        {"neutral": {}},
        state,
        log or RecordingLog(),
        client,
        voice,
        command_registry=CommandRegistry(CONFIG, state),
    )


class RecordingClient:
    def __init__(self, *, selection=None, selection_error=None) -> None:
        self.selection = selection
        self.selection_error = selection_error
        self.select_requests: list[str] = []
        self.stream_requests = 0
        self.mood_requests: list[str] = []

    def select_command(self, transcript, catalog, config):
        self.select_requests.append(transcript)
        if self.selection_error is not None:
            raise self.selection_error
        return self.selection

    def stream(self, prompt, config):
        self.stream_requests += 1
        yield "Normal answer."

    def classify_mood(self, transcript, mood_names, config):
        self.mood_requests.append(transcript)
        return None


class BlockingSelectorClient:
    def __init__(self) -> None:
        self.selection_started = threading.Event()
        self.release_selection = threading.Event()
        self.stream_requests = 0

    def select_command(self, transcript, catalog, config):
        self.selection_started.set()
        self.release_selection.wait(timeout=1.0)
        return {"name": "set_color", "arguments": {"name": "blue"}}

    def stream(self, prompt, config):
        self.stream_requests += 1
        yield "Fresh answer."

    def classify_mood(self, transcript, mood_names, config):
        return None


class DeliveringVoice:
    def __init__(self) -> None:
        self.clauses: list[tuple[int, str]] = []
        self.delivery_callback = None

    def begin_generation(self, generation_id: int) -> None:
        pass

    def enqueue(self, text: str, generation_id: int | None = None, **kwargs) -> None:
        assert generation_id is not None
        self.clauses.append((generation_id, text))
        if self.delivery_callback is not None:
            self.delivery_callback(
                "clause_started",
                {"generation_id": generation_id, "text": text},
            )
            self.delivery_callback(
                "clause_completed",
                {"generation_id": generation_id, "text": text},
            )

    def finish_generation(self, generation_id: int) -> None:
        if self.delivery_callback is not None:
            self.delivery_callback(
                "generation_playback_drained",
                {"generation_id": generation_id},
            )

    def enqueue_acknowledgement(self, generation_id: int | None = None) -> None:
        pass


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


if __name__ == "__main__":
    unittest.main()
