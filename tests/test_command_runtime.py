"""Shared closed-command registry wiring across voice and local web control."""

import unittest

from fastapi.testclient import TestClient

import main
from brain.commands import CommandRegistry
from control.web import WebPreview, create_app
from state import State


CONFIG = {
    "audio": {"sample_rate": 16_000, "vad_threshold": 0.015},
    "barge_in": {
        "enabled": False,
        "pause_after_speech_seconds": 0.25,
        "false_interruption_timeout_seconds": 2.0,
        "max_interruption_decision_seconds": 5.0,
        "preprocessor": "passthrough",
        "disable_on_preprocessor_error": True,
    },
    "commands": {
        "colors": {
            "blue": [10, 20, 30],
            "red": [200, 40, 30],
        }
    },
}


class CommandRuntimeTests(unittest.TestCase):
    def test_voice_runtime_forwards_exact_registry_object_to_conversation(self) -> None:
        state = State()
        registry = CommandRegistry(CONFIG, state)
        runtime = main._build_voice_runtime(
            CONFIG,
            {"neutral": {}},
            {},
            state,
            RecordingLog(),
            client=object(),
            command_registry=registry,
            preprocessor=FakePreprocessor(),
            interruption_detector=object(),
            player_factory=FakePlayer,
            voice_factory=FakeVoice,
            conversation_factory=CommandAwareConversation,
            coordinator_factory=FakeCoordinator,
            audio_factory=FakeAudio,
        )

        self.assertIs(runtime.conversation.command_registry, registry)

    def test_display_forwards_exact_registry_object_to_web_server(self) -> None:
        state = State()
        registry = CommandRegistry(CONFIG, state)
        display, server = main._build_display(
            {
                "display": {"cv2_enabled": False},
                "web": {"enabled": True, "port": 8080},
            },
            state,
            RecordingLog(),
            command_registry=registry,
        )

        self.assertIsNotNone(display)
        self.assertIsNotNone(server)
        self.assertIs(server.command_registry, registry)

    def test_web_catalog_and_execution_use_same_registry(self) -> None:
        state = State()
        log = RecordingLog()
        registry = CommandRegistry(CONFIG, state)
        app = create_app(
            state,
            WebPreview(),
            event_log=log,
            command_registry=registry,
        )

        with TestClient(app) as client:
            catalog = client.get("/commands")
            executed = client.post(
                "/commands",
                json={"name": "set_color", "arguments": {"name": "blue"}},
            )

        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json(), registry.catalog())
        self.assertEqual(executed.status_code, 200)
        self.assertEqual(
            executed.json(),
            {
                "spoken_response": "Blue.",
                "command": {"name": "set_color", "arguments": {"name": "blue"}},
            },
        )
        with state.locked():
            self.assertEqual(state.color, (10, 20, 30))
        self.assertEqual(
            log.events,
            [("command_executed", {"name": "set_color", "arguments": {"name": "blue"}})],
        )

    def test_web_rejects_unknown_values_and_extra_fields_without_mutation(self) -> None:
        state = State()
        registry = CommandRegistry(CONFIG, state)
        app = create_app(state, WebPreview(), command_registry=registry)

        invalid = (
            {"name": "set_color", "arguments": {"name": "green"}},
            {
                "name": "set_color",
                "arguments": {"name": "blue"},
                "extra": "must not disappear during request parsing",
            },
            {"name": "open_app", "arguments": {"name": "browser"}},
        )
        with TestClient(app) as client:
            responses = [client.post("/commands", json=payload) for payload in invalid]

        self.assertEqual([response.status_code for response in responses], [422, 422, 422])
        with state.locked():
            self.assertIsNone(state.color)

    def test_web_command_routes_are_unavailable_without_registry(self) -> None:
        app = create_app(State(), WebPreview())

        with TestClient(app) as client:
            catalog = client.get("/commands")
            execute = client.post(
                "/commands",
                json={"name": "set_color", "arguments": {"name": "blue"}},
            )

        self.assertEqual(catalog.status_code, 503)
        self.assertEqual(execute.status_code, 503)


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class FakePreprocessor:
    def push_render_reference(self, block) -> None:
        pass


class FakePlayer:
    def __init__(self, *, render_callback=None) -> None:
        self.render_callback = render_callback


class FakeVoice:
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
        pass


class CommandAwareConversation:
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
        command_registry=None,
    ) -> None:
        self.command_registry = command_registry

    def submit(self, text: str) -> None:
        pass

    def submit_with_timing(self, text: str, timing: dict[str, object]) -> None:
        pass

    def handle_delivery(self, event_type: str, payload: dict[str, object]) -> None:
        pass

    def handle_synthesis_timing(self, payload: dict[str, object]) -> None:
        pass

    def close(self) -> None:
        pass


class FakeCoordinator:
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
        pass


class FakeAudio:
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
        pass


if __name__ == "__main__":
    unittest.main()
