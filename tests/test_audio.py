"""Deterministic audio gate behavior."""

import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from perception.audio import (
    AudioLoop,
    UtteranceAssembler,
    WakeMatch,
    _make_transcriber,
    match_wake_phrase,
)
from state import State


CONFIG = {
    "audio": {
        "wake_variants": ["hey vess"],
        "wake_max_distance": 2,
        "conversation_timeout_seconds": 30.0,
    }
}


class WakeMatchTests(unittest.TestCase):
    def test_matcher_accepts_whisper_mishear(self) -> None:
        self.assertEqual(
            match_wake_phrase("hey best tell me a joke", ["hey vess"], 2),
            WakeMatch("hey vess", 2, 2),
        )

    def test_matcher_rejects_unrelated_speech(self) -> None:
        self.assertIsNone(
            match_wake_phrase("turn on the lights", ["hey vess"], 2)
        )


class UtteranceAssemblerTests(unittest.TestCase):
    def test_assembler_emits_speech_after_trailing_silence(self) -> None:
        assembler = UtteranceAssembler(10, 0.1, 0.2, 0.3, 2.0)

        self.assertIsNone(assembler.push(np.array([0.0, 0.2, 0.2, 0.0])))
        utterance = assembler.push(np.zeros(3))
        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.dtype, np.float32)
        np.testing.assert_allclose(utterance, np.array([0.2, 0.2], dtype=np.float32))

    def test_assembler_preserves_audio_immediately_before_trigger(self) -> None:
        assembler = UtteranceAssembler(10, 0.1, 0.2, 0.3, 2.0)

        self.assertIsNone(
            assembler.push(np.array([0.05, 0.05, 0.2, 0.2, 0.0]))
        )

        utterance = assembler.push(np.zeros(2))
        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.dtype, np.float32)
        np.testing.assert_allclose(
            utterance,
            np.array([0.05, 0.05, 0.2, 0.2], dtype=np.float32),
        )

    def test_assembler_reports_live_vad_status(self) -> None:
        assembler = UtteranceAssembler(10, 0.1, 0.2, 0.3, 2.0)

        assembler.push(np.array([0.2, 0.2]))

        self.assertEqual(assembler.status(), {"vad_active": True, "vad_seconds": 0.2})


class AudioLoopTests(unittest.TestCase):
    def test_rejection_logs_without_dispatch(self) -> None:
        dispatched: list[str] = []
        log = RecordingLog()
        loop = AudioLoop(
            CONFIG,
            State(),
            log,
            dispatched.append,
            transcribe=lambda _: "turn on the lights",
        )

        loop.handle_utterance(np.ones(16_000, dtype=np.float32))

        self.assertEqual(dispatched, [])
        self.assertEqual(log.events[0][0], "wake_rejected")

    def test_acceptance_removes_matched_wake_prefix_before_dispatch(self) -> None:
        dispatched: list[str] = []
        log = RecordingLog()
        state = State()
        loop = AudioLoop(
            CONFIG,
            state,
            log,
            dispatched.append,
            transcribe=lambda _: "Hey best tell me a joke",
        )

        loop.handle_utterance(np.ones(16_000, dtype=np.float32))

        self.assertEqual(dispatched, ["tell me a joke"])
        self.assertEqual(log.events[0][0], "wake_accepted")
        self.assertEqual(
            [event["event"] for event in state.debug_snapshot()["events"]],
            ["transcript", "wake_accepted"],
        )

    def test_wake_opens_conversation_for_followup_without_wake_phrase(self) -> None:
        dispatched: list[str] = []
        transcripts = iter(["hey vess how are you", "pretty good"])
        loop = AudioLoop(
            CONFIG,
            State(),
            RecordingLog(),
            dispatched.append,
            transcribe=lambda _: next(transcripts),
        )

        loop.handle_utterance(np.ones(16_000, dtype=np.float32))
        loop.handle_utterance(np.ones(16_000, dtype=np.float32))

        self.assertEqual(dispatched, ["how are you", "pretty good"])

    def test_recent_vess_reply_keeps_conversation_open(self) -> None:
        dispatched: list[str] = []
        transcripts = iter(["hey vess how are you", "not bad"])
        state = State()
        loop = AudioLoop(
            CONFIG,
            state,
            RecordingLog(),
            dispatched.append,
            transcribe=lambda _: next(transcripts),
        )

        loop.handle_utterance(np.ones(16_000, dtype=np.float32))
        loop._conversation_until = 0.0
        state.last_spoke = time.time()
        loop.handle_utterance(np.ones(16_000, dtype=np.float32))

        self.assertEqual(dispatched, ["how are you", "not bad"])

    def test_transcription_records_latency_from_speech_end(self) -> None:
        state = State()
        loop = AudioLoop(
            CONFIG,
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "hey vess hello",
        )

        loop.handle_utterance(
            np.ones(16_000, dtype=np.float32),
            speech_ended_at=time.perf_counter() - 0.05,
        )

        values = state.debug_snapshot()["values"]
        self.assertIn("transcription_ms", values)
        self.assertIn("speech_to_transcript_ms", values)
        self.assertGreaterEqual(values["speech_to_transcript_ms"], values["transcription_ms"])

    def test_capture_keeps_draining_blocks_while_transcription_runs(self) -> None:
        transcribe_started = threading.Event()
        release_transcribe = threading.Event()

        def slow_transcribe(_: np.ndarray) -> str:
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
        loop = AudioLoop(
            config,
            State(),
            RecordingLog(),
            lambda _: None,
            transcribe=slow_transcribe,
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()
        loop._blocks.put(np.array([0.2, 0.0]))
        self.assertTrue(transcribe_started.wait(timeout=0.5))

        for _ in range(20):
            loop._on_audio(np.zeros((1, 1), dtype=np.float32))
        time.sleep(0.05)

        self.assertLess(loop._blocks.qsize(), 16)

        loop._stop.set()
        release_transcribe.set()
        loop._blocks.put(None)
        worker.join(timeout=1.0)
        if loop._transcribe_thread is not None:
            loop._transcribe_thread.join(timeout=1.0)

    def test_listening_tracks_live_vad_not_transcription(self) -> None:
        state = State()
        config = {
            "audio": {
                "sample_rate": 10,
                "vad_threshold": 0.1,
                "min_utterance_seconds": 0.1,
                "silence_seconds": 0.3,
                "max_utterance_seconds": 2.0,
                "wake_variants": ["hey vess"],
                "wake_max_distance": 2,
            }
        }
        loop = AudioLoop(
            config,
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "turn on the lights",
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        loop._blocks.put(np.array([0.2, 0.2]))
        time.sleep(0.05)
        self.assertTrue(state.listening)

        loop._blocks.put(np.zeros(3))
        time.sleep(0.05)
        self.assertFalse(state.listening)

        loop._stop.set()
        loop._blocks.put(None)
        worker.join(timeout=1.0)
        if loop._transcribe_thread is not None:
            loop._transcribe_thread.join(timeout=1.0)


class TranscriberTests(unittest.TestCase):
    def test_transcriber_uses_low_latency_english_decoding(self) -> None:
        captured: dict[str, object] = {}

        class FakeWhisperModel:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def transcribe(self, samples: np.ndarray, **kwargs: object):
                captured.update(kwargs)
                return [SimpleNamespace(text="hello")], None

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        config = {
            "whisper": {
                "model": "small",
                "device": "cpu",
                "compute_type": "int8",
                "language": "en",
                "beam_size": 1,
                "condition_on_previous_text": False,
                "vad_filter": True,
            }
        }
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            transcribe = _make_transcriber(config)
            transcript = transcribe(np.ones(160, dtype=np.float32))

        self.assertEqual(transcript, "hello")
        self.assertEqual(
            captured,
            {
                "language": "en",
                "beam_size": 1,
                "condition_on_previous_text": False,
                "vad_filter": True,
            },
        )


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))

if __name__ == "__main__":
    unittest.main()
