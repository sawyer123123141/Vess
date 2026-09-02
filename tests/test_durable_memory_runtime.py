"""Runtime ownership for durable fact memory."""

from __future__ import annotations

import unittest
from pathlib import Path

import main
from state import State


class DurableMemoryRuntimeTests(unittest.TestCase):
    def test_fact_memory_builder_binds_existing_client_and_config(self) -> None:
        client = RecordingClient()
        factory = RecordingMemoryFactory()
        config = {"llm": {"model": "qwen2.5:7b"}}
        path = Path("example-memory.sqlite3")

        memory = main._build_fact_memory(
            config,
            client,
            path=path,
            factory=factory,
        )
        result = factory.extractor("My dog is Rex", ("favorite_color",))

        self.assertIs(memory, factory.memory)
        self.assertEqual(factory.path, path)
        self.assertEqual(
            client.calls,
            [("My dog is Rex", ("favorite_color",), config)],
        )
        self.assertEqual(result, ["candidate"])

    def test_voice_runtime_passes_memory_to_conversation_and_drains_after_conversation(self) -> None:
        closed: list[str] = []
        memory = RecordingMemory(closed)
        FakeAudio.close_events = closed
        FakeCoordinator.close_events = closed
        FakeConversation.close_events = closed
        FakeVoice.close_events = closed
        try:
            runtime = main._build_voice_runtime(
                _runtime_config(),
                {"neutral": {}},
                {},
                State(),
                RecordingLog(),
                client=object(),
                durable_memory=memory,
                preprocessor=FakePreprocessor(),
                interruption_detector=object(),
                player_factory=FakePlayer,
                voice_factory=FakeVoice,
                conversation_factory=FakeConversation,
                coordinator_factory=FakeCoordinator,
                audio_factory=FakeAudio,
            )

            self.assertIs(runtime.durable_memory, memory)
            self.assertIs(runtime.conversation.durable_memory, memory)
            runtime.close()
        finally:
            FakeAudio.close_events = None
            FakeCoordinator.close_events = None
            FakeConversation.close_events = None
            FakeVoice.close_events = None

        self.assertEqual(
            closed,
            ["audio", "coordinator", "conversation", "memory", "voice"],
        )


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], dict]] = []

    def extract_facts(self, text: str, known: tuple[str, ...], config: dict):
        self.calls.append((text, known, config))
        return ["candidate"]


class RecordingMemoryFactory:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.extractor = None
        self.memory = object()

    def __call__(self, path: Path, extractor):
        self.path = path
        self.extractor = extractor
        return self.memory


class RecordingMemory:
    def __init__(self, close_events: list[str]) -> None:
        self.close_events = close_events

    def close(self) -> None:
        self.close_events.append("memory")


class RecordingLog:
    def append(self, event_type: str, payload: dict[str, object]) -> None:
        pass


class FakePreprocessor:
    def push_render_reference(self, block) -> None:
        pass

    def process_capture(self, block):
        return block.samples


class FakePlayer:
    def __init__(self, *, render_callback=None) -> None:
        self.render_callback = render_callback


class FakeVoice:
    close_events: list[str] | None = None

    def __init__(
        self,
        config,
        state,
        event_log,
        *,
        player=None,
        on_delivery=None,
        on_synthesis_timing=None,
    ) -> None:
        self.on_delivery = on_delivery
        self.on_synthesis_timing = on_synthesis_timing

    def close(self) -> None:
        if self.close_events is not None:
            self.close_events.append("voice")


class FakeConversation:
    close_events: list[str] | None = None

    def __init__(
        self,
        config,
        moods,
        state,
        event_log,
        client,
        voice,
        *,
        performances=None,
        durable_memory=None,
    ) -> None:
        self.durable_memory = durable_memory

    def handle_delivery(self, event_type: str, payload: dict[str, object]) -> None:
        pass

    def handle_synthesis_timing(self, payload: dict[str, object]) -> None:
        pass

    def submit(self, text: str) -> None:
        pass

    def submit_with_timing(self, text: str, timing: dict[str, object]) -> None:
        pass

    def cancel_generation(self, generation_id: int, reason: str) -> bool:
        return True

    def close(self) -> None:
        if self.close_events is not None:
            self.close_events.append("conversation")


class FakeCoordinator:
    close_events: list[str] | None = None

    def __init__(
        self,
        state,
        event_log,
        voice,
        conversation,
        transcript_submit,
        *,
        false_timeout_seconds,
        decision_watchdog_seconds,
        timed_transcript_submit=None,
    ) -> None:
        pass

    def close(self) -> None:
        if self.close_events is not None:
            self.close_events.append("coordinator")


class FakeAudio:
    close_events: list[str] | None = None

    def __init__(
        self,
        config,
        state,
        event_log,
        on_request,
        *,
        on_timed_request=None,
        preprocessor=None,
        interruption_detector=None,
        turn_coordinator=None,
    ) -> None:
        pass

    def close(self) -> None:
        if self.close_events is not None:
            self.close_events.append("audio")


def _runtime_config() -> dict[str, object]:
    return {
        "audio": {
            "sample_rate": 16_000,
            "vad_threshold": 0.015,
        },
        "barge_in": {
            "enabled": False,
            "pause_after_speech_seconds": 0.25,
            "false_interruption_timeout_seconds": 2.0,
            "max_interruption_decision_seconds": 5.0,
            "preprocessor": "passthrough",
            "disable_on_preprocessor_error": True,
        },
    }


if __name__ == "__main__":
    unittest.main()
