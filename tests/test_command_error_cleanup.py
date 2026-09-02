"""Command execution failures must close the active generation cleanly."""

import unittest

from brain.commands import CommandCall
from brain.llm import ConversationWorker
from state import State


class CommandErrorCleanupTests(unittest.TestCase):
    def test_execution_error_finishes_generation_without_falling_back_to_chat(self) -> None:
        state = State()
        log = RecordingLog()
        voice = DrainingVoice()
        worker = ConversationWorker(
            {"memory": {"short_term_minutes": 10, "short_term_turns": 8}},
            {"neutral": {}},
            state,
            log,
            Client(),
            voice,
            command_registry=FailingRegistry(),
        )
        voice.delivery_callback = worker.handle_delivery

        worker.start()
        worker.submit("turn blue")
        worker.close()

        self.assertEqual(voice.finished, [1])
        self.assertEqual(worker._delivery._generations, {})
        self.assertEqual(Client.stream_calls, 0)
        self.assertIn(
            ("command_error", {"name": "set_color", "error": "handler failed"}),
            log.events,
        )
        with state.locked():
            self.assertEqual(state.conversation_turns, [])


class FailingRegistry:
    def is_candidate(self, text: str) -> bool:
        return True

    def catalog(self) -> dict[str, object]:
        return {"set_color": {}}

    def validate(self, payload: object) -> CommandCall | None:
        return CommandCall("set_color", {"name": "blue"})

    def execute(self, call: CommandCall):
        raise RuntimeError("handler failed")


class Client:
    stream_calls = 0

    def select_command(self, transcript, catalog, config):
        return {"name": "set_color", "arguments": {"name": "blue"}}

    def stream(self, prompt, config):
        type(self).stream_calls += 1
        yield "This fallback should not run."

    def classify_mood(self, transcript, mood_names, config):
        return None


class DrainingVoice:
    def __init__(self) -> None:
        self.finished: list[int] = []
        self.delivery_callback = None

    def begin_generation(self, generation_id: int) -> None:
        pass

    def enqueue(self, text: str, generation_id: int | None = None, **kwargs) -> None:
        raise AssertionError("failed command must not enqueue successful acknowledgement")

    def finish_generation(self, generation_id: int) -> None:
        self.finished.append(generation_id)
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
