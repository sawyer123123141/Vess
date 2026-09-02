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
from perception.audio_preprocess import CapturedAudioBlock
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
    def test_timed_completion_locates_last_voiced_audio_within_input_block(self) -> None:
        assembler = UtteranceAssembler(10, 0.1, 0.2, 0.3, 2.0)

        completed = assembler.push_with_timing(
            np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.assertIsNotNone(completed)
        assert completed is not None
        np.testing.assert_allclose(
            completed.samples,
            np.array([0.2, 0.2], dtype=np.float32),
        )
        self.assertEqual(completed.speech_end_offset_samples, 5)

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
        timed: list[dict[str, object]] = []
        loop = AudioLoop(
            CONFIG,
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "hey vess hello",
            on_timed_request=lambda _, timing: timed.append(timing),
        )

        loop.handle_utterance(
            np.ones(16_000, dtype=np.float32),
            speech_ended_at=time.perf_counter() - 0.05,
        )

        self.assertEqual(len(timed), 1)
        self.assertIn("transcription_ms", timed[0])
        self.assertIn("speech_to_transcript_ms", timed[0])
        self.assertGreaterEqual(
            timed[0]["speech_to_transcript_ms"],
            timed[0]["transcription_ms"],
        )

    def test_latency_separates_endpoint_wait_and_forwards_speech_end(self) -> None:
        state = State()
        dispatched: list[str] = []
        timed: list[tuple[str, dict[str, object]]] = []

        loop = AudioLoop(
            CONFIG,
            state,
            RecordingLog(),
            dispatched.append,
            transcribe=lambda _: "hey vess hello",
            on_timed_request=lambda text, timing: timed.append((text, timing)),
        )

        with patch("perception.audio.time.perf_counter", side_effect=[10.0, 10.5]):
            loop.handle_utterance(
                np.ones(16_000, dtype=np.float32),
                speech_ended_at=9.4,
                utterance_finalized_at=9.85,
            )

        self.assertEqual(dispatched, [])
        self.assertEqual(len(timed), 1)
        request, timing = timed[0]
        self.assertEqual(request, "hello")
        self.assertEqual(timing["speech_ended_at"], 9.4)
        self.assertEqual(timing["endpoint_wait_ms"], 450.0)
        self.assertEqual(timing["transcription_queue_ms"], 150.0)
        self.assertEqual(timing["speech_to_transcript_ms"], 1100.0)
        self.assertNotIn("endpoint_wait_ms", state.debug_snapshot()["values"])

    def test_rejected_transcript_does_not_mix_with_accepted_latency_bundle(self) -> None:
        state = State()
        state.update_debug(
            latency_generation_id=3,
            transcription_ms=111.0,
            llm_first_clause_ms=222.0,
        )
        loop = AudioLoop(
            CONFIG,
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "unrelated background speech",
        )

        with patch("perception.audio.time.perf_counter", side_effect=[20.0, 21.0]):
            loop.handle_utterance(
                np.ones(16_000, dtype=np.float32),
                speech_ended_at=19.0,
                utterance_finalized_at=19.45,
            )

        values = state.debug_snapshot()["values"]
        self.assertEqual(values["latency_generation_id"], 3)
        self.assertEqual(values["transcription_ms"], 111.0)
        self.assertEqual(values["llm_first_clause_ms"], 222.0)

    def test_capture_sequence_invalidates_only_utterance_spanning_gap(self) -> None:
        state = State()
        timed: list[dict[str, object]] = []
        loop = AudioLoop(
            _small_audio_config(barge_in_enabled=False),
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "hey vess hello",
            on_timed_request=lambda text, timing: timed.append(timing),
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        now = time.perf_counter()
        for sequence, sample in ((1, 0.3), (2, 0.0), (3, 0.0)):
            loop._blocks.put(
                CapturedAudioBlock(
                    np.array([sample], dtype=np.float32),
                    None,
                    now + sequence / 10.0,
                    capture_sequence=sequence,
                )
            )
        self.assertTrue(_wait_until(lambda: len(timed) == 1))
        self.assertTrue(timed[0]["latency_timing_valid"])

        loop._audio_blocks_dropped = 1
        for sequence, sample in ((4, 0.3), (6, 0.0), (7, 0.0)):
            loop._blocks.put(
                CapturedAudioBlock(
                    np.array([sample], dtype=np.float32),
                    None,
                    now + sequence / 10.0,
                    capture_sequence=sequence,
                )
            )
        self.assertTrue(_wait_until(lambda: len(timed) == 2))

        self.assertFalse(timed[1]["latency_timing_valid"])
        self.assertIsNone(timed[1]["speech_ended_at"])
        self.assertEqual(state.debug_snapshot()["values"]["audio_blocks_dropped"], 1)
        _stop_loop(loop, worker)

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

    def test_audio_callback_queues_timestamped_capture_block(self) -> None:
        loop = AudioLoop(
            CONFIG,
            State(),
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "",
        )

        with patch("perception.audio.time.perf_counter", return_value=99.25):
            loop._on_audio(
                np.array([[0.1], [0.2]], dtype=np.float32),
                2,
                SimpleNamespace(inputBufferAdcTime=12.5),
                None,
            )

        block = loop._blocks.get_nowait()
        self.assertIsInstance(block, CapturedAudioBlock)
        assert isinstance(block, CapturedAudioBlock)
        np.testing.assert_allclose(block.samples, np.array([0.1, 0.2], dtype=np.float32))
        self.assertEqual(block.adc_time, 12.5)
        self.assertEqual(block.received_at, 99.25)
        self.assertEqual(block.capture_sequence, 1)

    def test_disabled_barge_in_preserves_speaking_time_discard(self) -> None:
        state = State(speaking=True)
        preprocessor = RecordingPreprocessor()
        detector = SequenceDetector([True])
        coordinator = RecordingCoordinator(state)
        config = _small_audio_config(barge_in_enabled=False)
        loop = AudioLoop(
            config,
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "wait",
            preprocessor=preprocessor,
            interruption_detector=detector,
            turn_coordinator=coordinator,
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        loop._blocks.put(np.array([0.3, 0.3], dtype=np.float32))
        self.assertTrue(_wait_until(lambda: state.debug_snapshot()["values"].get("audio_ignored") is True))

        self.assertEqual(preprocessor.calls, 0)
        self.assertEqual(detector.pushes, 0)
        self.assertEqual(coordinator.events, [])
        self.assertFalse(state.listening)
        _stop_loop(loop, worker)

    def test_speaking_energy_does_not_set_listening_until_candidate_is_accepted(self) -> None:
        state = State(speaking=True)
        preprocessor = RecordingPreprocessor()
        detector = SequenceDetector([False, True])
        coordinator = RecordingCoordinator(state)
        loop = AudioLoop(
            _small_audio_config(barge_in_enabled=True),
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "wait",
            preprocessor=preprocessor,
            interruption_detector=detector,
            turn_coordinator=coordinator,
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        loop._blocks.put(np.array([0.3], dtype=np.float32))
        self.assertTrue(_wait_until(lambda: detector.pushes == 1))
        self.assertFalse(state.listening)

        loop._blocks.put(np.array([0.3], dtype=np.float32))
        self.assertTrue(_wait_until(lambda: coordinator.events == [("candidate", None)]))
        self.assertTrue(state.listening)
        _stop_loop(loop, worker)

    def test_overlap_capture_continues_after_candidate_pause_and_transcript_bypasses_wake_gate(self) -> None:
        state = State(speaking=True)
        coordinator = RecordingCoordinator(state, pause_speaking=True)
        dispatched: list[str] = []
        loop = AudioLoop(
            _small_audio_config(barge_in_enabled=True),
            state,
            RecordingLog(),
            dispatched.append,
            transcribe=lambda _: "wait",
            preprocessor=RecordingPreprocessor(),
            interruption_detector=SequenceDetector([True, False]),
            turn_coordinator=coordinator,
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        loop._blocks.put(np.array([0.3, 0.3], dtype=np.float32))
        self.assertTrue(_wait_until(lambda: coordinator.events[:1] == [("candidate", None)]))
        self.assertFalse(state.speaking)
        loop._blocks.put(np.array([0.0, 0.0], dtype=np.float32))

        self.assertTrue(_wait_until(lambda: ("transcript", "wait") in coordinator.events))
        self.assertEqual(
            coordinator.events,
            [("candidate", None), ("queued", None), ("transcript", "wait")],
        )
        self.assertEqual(dispatched, [])
        _stop_loop(loop, worker)

    def test_empty_overlap_transcript_is_sent_to_coordinator_for_rollback(self) -> None:
        state = State(speaking=True)
        coordinator = RecordingCoordinator(state, pause_speaking=True)
        loop = AudioLoop(
            _small_audio_config(barge_in_enabled=True),
            state,
            RecordingLog(),
            lambda _: self.fail("barge-in transcript must bypass normal request dispatch"),
            transcribe=lambda _: "",
            preprocessor=RecordingPreprocessor(),
            interruption_detector=SequenceDetector([True, False]),
            turn_coordinator=coordinator,
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        loop._blocks.put(np.array([0.3, 0.3], dtype=np.float32))
        self.assertTrue(_wait_until(lambda: coordinator.events[:1] == [("candidate", None)]))
        loop._blocks.put(np.array([0.0, 0.0], dtype=np.float32))

        self.assertTrue(_wait_until(lambda: ("transcript", "") in coordinator.events))
        self.assertEqual(coordinator.events[-2:], [("queued", None), ("transcript", "")])
        _stop_loop(loop, worker)

    def test_fail_closed_preprocessor_error_disables_only_speaking_barge_in(self) -> None:
        state = State(speaking=True)
        preprocessor = RecordingPreprocessor(error=RuntimeError("aec failed"))
        coordinator = RecordingCoordinator(state)
        dispatched: list[str] = []
        loop = AudioLoop(
            _small_audio_config(barge_in_enabled=True),
            state,
            RecordingLog(),
            dispatched.append,
            transcribe=lambda _: "hey vess hello",
            preprocessor=preprocessor,
            interruption_detector=SequenceDetector([True]),
            turn_coordinator=coordinator,
        )
        worker = threading.Thread(target=loop._run, daemon=True)
        worker.start()

        loop._blocks.put(np.array([0.3, 0.3], dtype=np.float32))
        self.assertTrue(_wait_until(lambda: preprocessor.calls == 1))
        state.speaking = True
        loop._blocks.put(np.array([0.3, 0.3], dtype=np.float32))
        time.sleep(0.05)
        self.assertEqual(preprocessor.calls, 1)
        self.assertEqual(coordinator.events, [])

        state.speaking = False
        loop._blocks.put(np.array([0.3, 0.3, 0.0, 0.0], dtype=np.float32))
        self.assertTrue(_wait_until(lambda: dispatched == ["hello"]))
        self.assertEqual(preprocessor.calls, 1)
        _stop_loop(loop, worker)


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


class RecordingPreprocessor:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return np.asarray(block.samples, dtype=np.float32).copy()

    def push_render_reference(self, block: object) -> None:
        return None


class SequenceDetector:
    def __init__(self, answers: list[bool]) -> None:
        self.answers = list(answers)
        self.pushes = 0
        self.resets = 0

    def push(self, samples: np.ndarray) -> bool:
        self.pushes += 1
        if self.answers:
            return self.answers.pop(0)
        return False

    def reset(self) -> None:
        self.resets += 1


class RecordingCoordinator:
    def __init__(self, state: State, *, pause_speaking: bool = False) -> None:
        self.state = state
        self.pause_speaking = pause_speaking
        self.events: list[tuple[str, object | None]] = []

    def on_candidate(self) -> bool:
        self.events.append(("candidate", None))
        with self.state.locked():
            self.state.listening = True
            if self.pause_speaking:
                self.state.speaking = False
        return True

    def on_utterance_queued_for_transcription(self) -> None:
        self.events.append(("queued", None))

    def on_transcript(
        self,
        text: str,
        *,
        timing: dict[str, object] | None = None,
    ) -> None:
        self.events.append(("transcript", text))
        with self.state.locked():
            self.state.listening = False

    def on_transcription_error(self, error: Exception) -> None:
        self.events.append(("error", str(error)))
        with self.state.locked():
            self.state.listening = False


def _small_audio_config(*, barge_in_enabled: bool) -> dict[str, object]:
    return {
        "audio": {
            "sample_rate": 10,
            "vad_threshold": 0.1,
            "min_utterance_seconds": 0.1,
            "silence_seconds": 0.2,
            "max_utterance_seconds": 2.0,
            "pre_roll_seconds": 0.0,
            "wake_variants": ["hey vess"],
            "wake_max_distance": 2,
        },
        "barge_in": {
            "enabled": barge_in_enabled,
            "disable_on_preprocessor_error": True,
        },
    }


def _wait_until(predicate, timeout: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _stop_loop(loop: AudioLoop, worker: threading.Thread) -> None:
    loop._stop.set()
    loop._blocks.put(None)
    worker.join(timeout=1.0)
    if loop._transcribe_thread is not None:
        loop._transcribe_thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
