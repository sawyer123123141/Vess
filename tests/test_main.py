"""Runtime wiring for Vess targets."""

import unittest

import numpy as np

import main
from perception.audio_preprocess import RenderedAudioBlock
from state import State


class MainTests(unittest.TestCase):
    def test_default_display_fans_frames_to_the_browser_target(self) -> None:
        build_display = getattr(main, "_build_display", None)
        self.assertIsNotNone(build_display, "display target builder is missing")
        config = {
            "display": {"cv2_enabled": False, "preview_scale": 8},
            "web": {"enabled": True, "port": 8080},
        }
        display, server = build_display(config, State())
        self.assertIsNotNone(server)

        display.show(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertIsNotNone(server.preview.png())

    def test_expired_mood_is_logged(self) -> None:
        state = State(mood="annoyed", mood_until=10.0)
        log = RecordingLog()

        main._expire_mood(state, log, 10.1)

        self.assertEqual(
            log.events,
            [("mood_changed", {"from": "annoyed", "to": "neutral"})],
        )

    def test_repository_performance_config_contains_neutral(self) -> None:
        performances = main._load_performances()

        self.assertIn("neutral", performances)

    def test_repository_barge_in_config_is_dormant_and_complete(self) -> None:
        config = main._load("config.json")

        self.assertEqual(
            config["barge_in"],
            {
                "enabled": False,
                "pause_after_speech_seconds": 0.25,
                "false_interruption_timeout_seconds": 2.0,
                "max_interruption_decision_seconds": 5.0,
                "preprocessor": "passthrough",
                "disable_on_preprocessor_error": True,
            },
        )

    def test_voice_runtime_wires_one_preprocessor_and_shared_voice_conversation_objects(self) -> None:
        preprocessor = FakePreprocessor()
        detector = object()
        runtime = main._build_voice_runtime(
            _runtime_config(),
            {"neutral": {}},
            {},
            State(),
            RecordingLog(),
            client=object(),
            preprocessor=preprocessor,
            interruption_detector=detector,
            player_factory=FakePlayer,
            voice_factory=FakeVoice,
            conversation_factory=FakeConversation,
            coordinator_factory=FakeCoordinator,
            audio_factory=FakeAudio,
        )

        self.assertIs(runtime.preprocessor, preprocessor)
        self.assertIs(runtime.audio.preprocessor, preprocessor)
        self.assertIs(runtime.audio.interruption_detector, detector)
        self.assertIs(runtime.audio.turn_coordinator, runtime.coordinator)
        self.assertIs(runtime.coordinator.voice, runtime.voice)
        self.assertIs(runtime.coordinator.conversation, runtime.conversation)
        self.assertIs(runtime.conversation.voice, runtime.voice)
        self.assertIs(
            runtime.audio.on_timed_request.__self__,
            runtime.conversation,
        )
        self.assertIs(
            runtime.coordinator.timed_transcript_submit.__self__,
            runtime.conversation,
        )

        rendered = RenderedAudioBlock(
            samples=np.array([0.1], dtype=np.float32),
            sample_rate=24_000,
            dac_time=None,
        )
        runtime.player.render_callback(rendered)
        self.assertEqual(preprocessor.rendered, [rendered])

        runtime.voice.on_delivery(
            "clause_completed",
            {"generation_id": 3, "text": "done"},
        )
        self.assertEqual(
            runtime.conversation.deliveries,
            [("clause_completed", {"generation_id": 3, "text": "done"})],
        )

        synthesis_timing = {
            "generation_id": 3,
            "worker_wait_ms": 12.3,
            "synthesis_ms": 45.6,
        }
        runtime.voice.on_synthesis_timing(synthesis_timing)
        self.assertEqual(
            runtime.conversation.synthesis_timings,
            [synthesis_timing],
        )

    def test_voice_runtime_shutdown_stops_audio_then_coordinator_before_conversation_and_voice(self) -> None:
        closed: list[str] = []
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
                preprocessor=FakePreprocessor(),
                interruption_detector=object(),
                player_factory=FakePlayer,
                voice_factory=FakeVoice,
                conversation_factory=FakeConversation,
                coordinator_factory=FakeCoordinator,
                audio_factory=FakeAudio,
            )

            runtime.close()
        finally:
            FakeAudio.close_events = None
            FakeCoordinator.close_events = None
            FakeConversation.close_events = None
            FakeVoice.close_events = None

        self.assertEqual(closed, ["audio", "coordinator", "conversation", "voice"])


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class FakePreprocessor:
    def __init__(self) -> None:
        self.rendered: list[RenderedAudioBlock] = []

    def push_render_reference(self, block: RenderedAudioBlock) -> None:
        self.rendered.append(block)

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
        self.player = player
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
    ) -> None:
        self.voice = voice
        self.deliveries: list[tuple[str, dict[str, object]]] = []
        self.synthesis_timings: list[dict[str, object]] = []
        self.submitted: list[str] = []

    def handle_delivery(self, event_type: str, payload: dict[str, object]) -> None:
        self.deliveries.append((event_type, payload))

    def handle_synthesis_timing(self, payload: dict[str, object]) -> None:
        self.synthesis_timings.append(payload)

    def submit(self, text: str) -> None:
        self.submitted.append(text)

    def submit_with_timing(self, text: str, timing: dict[str, object]) -> None:
        self.submitted.append(text)

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
        self.voice = voice
        self.conversation = conversation
        self.transcript_submit = transcript_submit
        self.false_timeout_seconds = false_timeout_seconds
        self.decision_watchdog_seconds = decision_watchdog_seconds
        self.timed_transcript_submit = timed_transcript_submit

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
        self.on_request = on_request
        self.on_timed_request = on_timed_request
        self.preprocessor = preprocessor
        self.interruption_detector = interruption_detector
        self.turn_coordinator = turn_coordinator

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
