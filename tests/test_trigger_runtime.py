"""Runtime must enable proactive speech only for trusted live-camera presence."""

import inspect
import unittest

import main
from brain.triggers import TriggerWorker
from state import State


SETTINGS = {
    "min_absence_hours": 4,
    "idle_interaction_minutes": 30,
    "cooldown_minutes": 60,
    "quiet_after_hour": 22,
    "quiet_before_hour": 8,
}


class TriggerRuntimeTests(unittest.TestCase):
    def test_static_image_and_video_sources_never_build_trigger_worker(self) -> None:
        for source in ("image", "video"):
            with self.subTest(source=source):
                factory = RecordingTriggerFactory()
                worker = main._build_trigger_worker(
                    {"camera": {"source": source}, "triggers": SETTINGS},
                    State(),
                    RecordingLog(),
                    FakeConversation(),
                    trigger_factory=factory,
                )

                self.assertIsNone(worker)
                self.assertEqual(factory.calls, [])

    def test_live_camera_source_builds_worker_with_proactive_submit_callback(self) -> None:
        factory = RecordingTriggerFactory()
        state = State()
        log = RecordingLog()
        conversation = FakeConversation()

        worker = main._build_trigger_worker(
            {"camera": {"source": "camera"}, "triggers": SETTINGS},
            state,
            log,
            conversation,
            trigger_factory=factory,
        )

        self.assertIs(worker, factory.worker)
        self.assertEqual(len(factory.calls), 1)
        actual_state, actual_settings, callback, actual_log = factory.calls[0]
        self.assertIs(actual_state, state)
        self.assertEqual(actual_settings, SETTINGS)
        self.assertIs(callback.__self__, conversation)
        self.assertIs(callback.__func__, conversation.submit_proactive.__func__)
        self.assertIs(actual_log, log)

    def test_trigger_worker_default_poll_interval_is_about_half_a_second(self) -> None:
        default = inspect.signature(TriggerWorker).parameters["poll_seconds"].default

        self.assertEqual(default, 0.5)

    def test_voice_runtime_defaults_to_proactive_conversation_worker(self) -> None:
        default = inspect.signature(main._build_voice_runtime).parameters[
            "conversation_factory"
        ].default

        self.assertIs(default, main.ProactiveConversationWorker)

    def test_voice_runtime_shutdown_stops_triggers_before_voice_pipeline(self) -> None:
        closed = []
        runtime = main.VoiceRuntime(
            preprocessor=object(),
            player=object(),
            voice=Closer("voice", closed),
            conversation=Closer("conversation", closed),
            coordinator=Closer("coordinator", closed),
            audio=Closer("audio", closed),
            triggers=Closer("triggers", closed),
        )

        runtime.close()

        self.assertEqual(
            closed,
            ["triggers", "audio", "coordinator", "conversation", "voice"],
        )


class FakeConversation:
    def submit_proactive(self, trigger_name: str, context: str) -> bool:
        return True


class RecordingTriggerFactory:
    def __init__(self) -> None:
        self.worker = object()
        self.calls = []

    def __call__(self, state, settings, callback, event_log):
        self.calls.append((state, settings, callback, event_log))
        return self.worker


class RecordingLog:
    def append(self, event_type, payload) -> None:
        pass


class Closer:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed

    def close(self) -> None:
        self.closed.append(self.name)


if __name__ == "__main__":
    unittest.main()
