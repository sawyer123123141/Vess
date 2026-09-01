"""Deterministic audio gate behavior."""

import unittest

import numpy as np

from perception.audio import (
    AudioLoop,
    UtteranceAssembler,
    WakeMatch,
    match_wake_phrase,
)
from state import State


CONFIG = {
    "audio": {
        "wake_variants": ["hey vess"],
        "wake_max_distance": 2,
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
        self.assertTrue(
            np.array_equal(assembler.push(np.zeros(3)), np.array([0.2, 0.2]))
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


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))

if __name__ == "__main__":
    unittest.main()
