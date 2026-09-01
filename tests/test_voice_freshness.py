"""Freshness-first voice pipeline regressions."""

import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from brain.llm import ConversationWorker
from output.voice import VoiceOutput
from perception.audio import AudioLoop, _make_transcriber
from state import State


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class AudioFreshnessTests(unittest.TestCase):
    def test_only_latest_pending_utterance_survives_slow_transcription(self) -> None:
        transcribe_started = threading.Event()
        release_transcribe = threading.Event()
        seen: list[float] = []

        def slow_transcribe(samples: np.ndarray) -> str:
            seen.append(round(float(samples[0]), 1))
            if len(seen) == 1:
                transcribe_started.set()
                release_transcribe.wait(timeout=1.0)
            return "hey vess hello"

        config = {
            "audio": {
                "sample_rate": 10,
                "vad_threshold": 0.1,
                "min_utterance_seconds": 0.1,
                "silence_seconds": 0.1,
                "max_utterance_seconds": 2.0,
                "wake_variants": ["hey vess"],
                "wake_max_distance": 2,
            }
        }
        state = State()
        loop = AudioLoop(
            config,
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=slow_transcribe,
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        loop._blocks.put(np.array([0.2, 0.0]))
        self.assertTrue(transcribe_started.wait(timeout=0.5))
        loop._blocks.put(np.array([0.3, 0.0]))
        loop._blocks.put(np.array([0.4, 0.0]))
        time.sleep(0.05)

        release_transcribe.set()
        deadline = time.time() + 1.0
        while len(seen) < 2 and time.time() < deadline:
            time.sleep(0.01)

        loop._stop.set()
        loop._blocks.put(None)
        worker.join(timeout=1.0)
        if loop._transcribe_thread is not None:
            loop._transcribe_thread.join(timeout=1.0)

        self.assertEqual(seen, [0.2, 0.4])
        self.assertIn(
            "pending_utterance_replaced",
            [event["event"] for event in state.debug_snapshot()["events"]],
        )

    def test_transcriber_enables_vad_filter_and_keeps_short_speech(self) -> None:
        captured: dict[str, object] = {}

        class FakeWhisperModel:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def transcribe(self, samples: np.ndarray, **kwargs: object):
                captured.update(kwargs)
                return [
                    SimpleNamespace(
                        text="yes",
                        no_speech_prob=0.05,
                        avg_logprob=-0.1,
                    )
                ], None

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            transcribe = _make_transcriber({"whisper": {}})
            transcript = transcribe(np.ones(160, dtype=np.float32))

        self.assertEqual(transcript, "yes")
        self.assertTrue(captured["vad_filter"])


class RecordingGenerationVoice:
    def __init__(self) -> None:
        self.clauses: list[tuple[int | None, str]] = []
        self.first_clause = threading.Event()

    def begin_generation(self, generation_id: int) -> None:
        pass

    def enqueue(self, text: str, generation_id: int | None = None) -> None:
        self.clauses.append((generation_id, text))
        self.first_clause.set()

    def enqueue_acknowledgement(self, generation_id: int | None = None) -> None:
        self.clauses.append((generation_id, "Yeah?"))


class QueueClient:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts: list[str] = []
        self.mood_inputs: list[str] = []

    def stream(self, prompt: str, config: dict):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            self.started.set()
            self.release.wait(timeout=1.0)
        yield "Done."

    def classify_mood(self, transcript: str, mood_names: set[str], config: dict):
        self.mood_inputs.append(transcript)
        return None


class ImmediateClient:
    def stream(self, prompt: str, config: dict):
        yield "First, then second."

    def classify_mood(self, transcript: str, mood_names: set[str], config: dict):
        return None


class ConversationFreshnessTests(unittest.TestCase):
    def test_duplicate_active_requests_collapse_and_latest_pending_wins(self) -> None:
        client = QueueClient()
        state = State()
        worker = ConversationWorker(
            {"personas": {"friendly": "Warm."}},
            {"neutral": {}},
            state,
            RecordingLog(),
            client,
            RecordingGenerationVoice(),
        )
        worker.start()

        worker.submit("hello there")
        self.assertTrue(client.started.wait(timeout=0.5))
        for _ in range(8):
            worker.submit("hello there")
        worker.submit("newest question")
        client.release.set()
        worker.close()

        self.assertEqual(len(client.prompts), 2)
        self.assertIn("Current request:\nnewest question", client.prompts[-1])
        events = [event["event"] for event in state.debug_snapshot()["events"]]
        self.assertIn("duplicate_request", events)

    def test_completed_latest_response_is_remembered_once(self) -> None:
        voice = RecordingGenerationVoice()
        state = State()
        log = RecordingLog()
        worker = ConversationWorker(
            {
                "personas": {"friendly": "Warm."},
                "memory": {"short_term_minutes": 10, "short_term_turns": 8},
            },
            {"neutral": {}},
            state,
            log,
            ImmediateClient(),
            voice,
        )
        worker.start()
        worker.submit("Tell me something")
        worker.close()

        self.assertEqual(len(state.conversation_turns), 1)
        turn = state.conversation_turns[0]
        self.assertEqual(turn.user, "Tell me something")
        self.assertEqual(turn.assistant, "First, then second.")
        self.assertEqual(
            [event for event in log.events if event[0] == "conversation_turn"],
            [
                (
                    "conversation_turn",
                    {
                        "user": "Tell me something",
                        "assistant": "First, then second.",
                    },
                )
            ],
        )

    def test_new_request_cancels_old_unspoken_clauses_and_old_memory(self) -> None:
        release_second_clause = threading.Event()

        class SwitchingClient:
            def __init__(self) -> None:
                self.calls = 0

            def stream(self, prompt: str, config: dict):
                self.calls += 1
                if self.calls == 1:
                    yield "Old first."
                    release_second_clause.wait(timeout=1.0)
                    yield "Old second."
                else:
                    yield "New answer."

            def classify_mood(self, transcript: str, mood_names: set[str], config: dict):
                return None

        voice = RecordingGenerationVoice()
        state = State()
        log = RecordingLog()
        worker = ConversationWorker(
            {"personas": {"friendly": "Warm."}},
            {"neutral": {}},
            state,
            log,
            SwitchingClient(),
            voice,
        )
        worker.start()
        worker.submit("old question")
        self.assertTrue(voice.first_clause.wait(timeout=0.5))
        worker.submit("new question")
        release_second_clause.set()
        worker.close()

        spoken = [text for _, text in voice.clauses]
        self.assertIn("Old first.", spoken)
        self.assertNotIn("Old second.", spoken)
        self.assertIn("New answer.", spoken)
        self.assertEqual(
            [(turn.user, turn.assistant) for turn in state.conversation_turns],
            [("new question", "New answer.")],
        )
        remembered_events = [event for event in log.events if event[0] == "conversation_turn"]
        self.assertEqual(
            remembered_events,
            [
                (
                    "conversation_turn",
                    {"user": "new question", "assistant": "New answer."},
                )
            ],
        )
        self.assertIn(
            "stale_response_cancelled",
            [event["event"] for event in state.debug_snapshot()["events"]],
        )

    def test_wake_acknowledgement_is_not_conversation_memory(self) -> None:
        state = State()
        log = RecordingLog()
        voice = RecordingGenerationVoice()
        worker = ConversationWorker(
            {"personas": {"friendly": "Warm."}},
            {"neutral": {}},
            state,
            log,
            ImmediateClient(),
            voice,
        )
        worker.start()
        worker.submit("")
        worker.close()

        self.assertEqual(state.conversation_turns, [])
        self.assertEqual(
            [event for event in log.events if event[0] == "conversation_turn"],
            [],
        )
        self.assertEqual([text for _, text in voice.clauses], ["Yeah?"])

    def test_pending_request_skips_old_mood_classification(self) -> None:
        client = QueueClient()
        worker = ConversationWorker(
            {"personas": {"friendly": "Warm."}},
            {"neutral": {}},
            State(),
            RecordingLog(),
            client,
            RecordingGenerationVoice(),
        )
        worker.start()
        worker.submit("first")
        self.assertTrue(client.started.wait(timeout=0.5))
        worker.submit("second")
        client.release.set()
        worker.close()

        self.assertNotIn("first", client.mood_inputs)


class VoiceFreshnessTests(unittest.TestCase):
    def test_stale_queued_tts_is_skipped_when_generation_changes(self) -> None:
        first_play_started = threading.Event()
        release_first_play = threading.Event()
        played: list[int] = []

        codes = {"old first": 1.0, "old second": 2.0, "new answer": 3.0}

        def synthesize(text: str) -> np.ndarray:
            return np.array([codes[text]], dtype=np.float32)

        def play(audio: np.ndarray, sample_rate: int) -> None:
            played.append(int(audio[0]))
            if len(played) == 1:
                first_play_started.set()
                release_first_play.wait(timeout=1.0)

        state = State()
        voice = VoiceOutput(
            {"voice": {"sample_rate": 24_000}},
            state,
            RecordingLog(),
            synthesize=synthesize,
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("old first", generation_id=1)
        self.assertTrue(first_play_started.wait(timeout=0.5))
        voice.enqueue("old second", generation_id=1)
        voice.begin_generation(2)
        voice.enqueue("new answer", generation_id=2)
        release_first_play.set()
        voice.close()

        self.assertEqual(played, [1, 3])
        self.assertIn(
            "stale_tts_skipped",
            [event["event"] for event in state.debug_snapshot()["events"]],
        )


if __name__ == "__main__":
    unittest.main()
