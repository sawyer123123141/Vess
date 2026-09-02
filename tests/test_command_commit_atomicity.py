"""Command state commits must be atomic with generation freshness."""

import threading
import unittest

from brain.commands import CommandCall, CommandResult
from brain.llm import ConversationWorker
from state import State


class CommandCommitAtomicityTests(unittest.TestCase):
    def test_new_request_cannot_supersede_generation_during_command_commit(self) -> None:
        state = State()
        registry = BlockingCommitRegistry(state)
        client = Client()
        worker = ConversationWorker(
            {"memory": {"short_term_minutes": 10, "short_term_turns": 8}},
            {"neutral": {}},
            state,
            RecordingLog(),
            client,
            SilentVoice(),
            command_registry=registry,
        )
        worker.start()
        worker.submit("turn blue")
        self.assertTrue(registry.execute_started.wait(timeout=0.5))

        submit_started = threading.Event()
        submit_done = threading.Event()

        def submit_fresh_request() -> None:
            submit_started.set()
            worker.submit("what time is it?")
            submit_done.set()

        submit_thread = threading.Thread(target=submit_fresh_request, daemon=True)
        submit_thread.start()
        self.assertTrue(submit_started.wait(timeout=0.5))

        superseded_while_committing = submit_done.wait(timeout=0.25)
        registry.release_execute.set()
        submit_thread.join(timeout=1.0)
        worker.close()

        self.assertFalse(
            superseded_while_committing,
            "a newer generation became current while an older command was committing state",
        )
        self.assertTrue(submit_done.is_set())
        self.assertEqual(client.stream_calls, 1)


class BlockingCommitRegistry:
    def __init__(self, state: State) -> None:
        self._state = state
        self.execute_started = threading.Event()
        self.release_execute = threading.Event()

    def is_candidate(self, text: str) -> bool:
        return text == "turn blue"

    def catalog(self) -> dict[str, object]:
        return {"set_color": {"arguments": {"name": {"values": ["blue"]}}}}

    def validate(self, payload: object) -> CommandCall | None:
        return CommandCall("set_color", {"name": "blue"})

    def execute(self, call: CommandCall) -> CommandResult:
        self.execute_started.set()
        self.release_execute.wait(timeout=1.0)
        with self._state.locked():
            self._state.color = (10, 20, 30)
        return CommandResult(
            spoken_response="Blue.",
            event_payload={"name": "set_color", "arguments": {"name": "blue"}},
        )


class Client:
    def __init__(self) -> None:
        self.stream_calls = 0

    def select_command(self, transcript, catalog, config):
        return {"name": "set_color", "arguments": {"name": "blue"}}

    def stream(self, prompt, config):
        self.stream_calls += 1
        yield "Fresh answer."

    def classify_mood(self, transcript, mood_names, config):
        return None


class SilentVoice:
    def begin_generation(self, generation_id: int) -> None:
        pass

    def enqueue(self, text: str, generation_id: int | None = None, **kwargs) -> None:
        pass

    def finish_generation(self, generation_id: int) -> None:
        pass

    def enqueue_acknowledgement(self, generation_id: int | None = None) -> None:
        pass


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


if __name__ == "__main__":
    unittest.main()
