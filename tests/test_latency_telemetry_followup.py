"""Regression tests for the measured conversational latency follow-up."""

from __future__ import annotations

import json
from pathlib import Path
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


class RecordingVoice:
    def __init__(self) -> None:
        self.generations: list[int] = []

    def begin_generation(self, generation_id: int) -> None:
        self.generations.append(generation_id)


class LatencyTelemetryFollowupTests(unittest.TestCase):
    def test_repository_and_runtime_whisper_beam_default_are_five(self) -> None:
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["whisper"]["beam_size"], 5)

        captured: dict[str, object] = {}

        class FakeWhisperModel:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                captured["model"] = model_name
                captured["init"] = kwargs

            def transcribe(self, samples: np.ndarray, **kwargs: object):
                captured["transcribe"] = kwargs
                return [SimpleNamespace(text="hello")], None

        with patch.dict(
            sys.modules,
            {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
        ):
            transcribe = _make_transcriber({"whisper": {}})
            self.assertEqual(transcribe(np.ones(160, dtype=np.float32)), "hello")

        transcribe_kwargs = captured["transcribe"]
        assert isinstance(transcribe_kwargs, dict)
        self.assertEqual(transcribe_kwargs["beam_size"], 5)

    def test_audio_blocks_dropped_starts_at_zero_in_debug_state(self) -> None:
        state = State()
        AudioLoop(
            {"audio": {"sample_rate": 16_000}},
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "",
        )

        self.assertEqual(
            state.debug_snapshot()["values"].get("audio_blocks_dropped"),
            0,
        )

    def test_first_tts_timings_are_generation_scoped_and_frozen(self) -> None:
        state = State()
        voice = RecordingVoice()
        worker = ConversationWorker(
            {},
            {"neutral": {}},
            state,
            RecordingLog(),
            object(),
            voice,
        )
        worker.submit_with_timing("first", {"latency_timing_valid": True})
        first_generation = voice.generations[-1]

        initial = state.debug_snapshot()["values"]
        self.assertIsNone(initial.get("tts_worker_wait_ms"))
        self.assertIsNone(initial.get("tts_first_synthesis_ms"))

        worker.handle_delivery(
            "clause_synthesized",
            {
                "generation_id": first_generation,
                "worker_wait_ms": 123.4,
                "synthesis_ms": 456.7,
            },
        )
        first = state.debug_snapshot()["values"]
        self.assertEqual(first["tts_worker_wait_ms"], 123.4)
        self.assertEqual(first["tts_first_synthesis_ms"], 456.7)

        worker.handle_delivery(
            "clause_synthesized",
            {
                "generation_id": first_generation,
                "worker_wait_ms": 8.0,
                "synthesis_ms": 9.0,
            },
        )
        still_first = state.debug_snapshot()["values"]
        self.assertEqual(still_first["tts_worker_wait_ms"], 123.4)
        self.assertEqual(still_first["tts_first_synthesis_ms"], 456.7)

        worker.submit_with_timing("second", {"latency_timing_valid": True})
        second_generation = voice.generations[-1]
        reset = state.debug_snapshot()["values"]
        self.assertIsNone(reset["tts_worker_wait_ms"])
        self.assertIsNone(reset["tts_first_synthesis_ms"])

        worker.handle_delivery(
            "clause_synthesized",
            {
                "generation_id": first_generation,
                "worker_wait_ms": 999.0,
                "synthesis_ms": 999.0,
            },
        )
        after_stale = state.debug_snapshot()["values"]
        self.assertEqual(after_stale["latency_generation_id"], second_generation)
        self.assertIsNone(after_stale["tts_worker_wait_ms"])
        self.assertIsNone(after_stale["tts_first_synthesis_ms"])

    def test_worker_wait_includes_time_blocked_behind_stale_synthesis(self) -> None:
        old_started = threading.Event()
        release_old = threading.Event()
        new_started = threading.Event()
        deliveries: list[tuple[str, dict[str, object]]] = []

        def synthesize(text: str) -> np.ndarray:
            if text == "old":
                old_started.set()
                release_old.wait(timeout=1.0)
            elif text == "new":
                new_started.set()
            return np.ones(10, dtype=np.float32)

        voice = VoiceOutput(
            {"voice": {"sample_rate": 24_000}},
            State(),
            RecordingLog(),
            synthesize=synthesize,
            play=lambda audio, sample_rate: None,
            on_delivery=lambda event, payload: deliveries.append((event, payload)),
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("old", generation_id=1)
        self.assertTrue(old_started.wait(timeout=0.5))

        voice.begin_generation(2)
        voice.enqueue("new", generation_id=2)
        time.sleep(0.05)
        release_old.set()
        self.assertTrue(new_started.wait(timeout=0.5))
        voice.close()

        new_synthesis = next(
            payload
            for event, payload in deliveries
            if event == "clause_synthesized" and payload.get("generation_id") == 2
        )
        self.assertGreaterEqual(new_synthesis["worker_wait_ms"], 40.0)
        self.assertLess(new_synthesis["synthesis_ms"], new_synthesis["worker_wait_ms"])


if __name__ == "__main__":
    unittest.main()
