"""Whisper runtime tuning and latency telemetry."""

import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from perception.audio import AudioLoop, _make_transcriber
from state import State


class WhisperRuntimeTests(unittest.TestCase):
    def test_transcriber_forwards_runtime_tuning_to_model(self) -> None:
        captured: dict[str, object] = {}

        class FakeWhisperModel:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                captured["model"] = model_name
                captured["init"] = kwargs

            def transcribe(self, samples: np.ndarray, **kwargs: object):
                return [SimpleNamespace(text="hello")], None

        config = {
            "whisper": {
                "model": "small",
                "device": "cuda",
                "compute_type": "int8_float16",
                "device_index": 0,
                "cpu_threads": 4,
                "num_workers": 1,
            }
        }
        with patch.dict(
            sys.modules,
            {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
        ):
            transcribe = _make_transcriber(config)
            self.assertEqual(transcribe(np.ones(160, dtype=np.float32)), "hello")

        self.assertEqual(captured["model"], "small")
        self.assertEqual(
            captured["init"],
            {
                "device": "cuda",
                "compute_type": "int8_float16",
                "device_index": 0,
                "cpu_threads": 4,
                "num_workers": 1,
            },
        )

    def test_audio_loop_reports_utterance_duration_and_realtime_factor(self) -> None:
        state = State()
        loop = AudioLoop(
            {"audio": {"sample_rate": 16_000, "conversation_timeout_seconds": 0}},
            state,
            RecordingLog(),
            lambda _: None,
            transcribe=lambda _: "hello",
        )

        with patch("perception.audio.time.perf_counter", side_effect=[10.0, 10.5]):
            loop.handle_utterance(
                np.ones(32_000, dtype=np.float32),
                speech_ended_at=9.9,
            )

        values = state.debug_snapshot()["values"]
        self.assertEqual(values["utterance_seconds"], 2.0)
        self.assertEqual(values["transcription_ms"], 500.0)
        self.assertEqual(values["speech_to_transcript_ms"], 600.0)
        self.assertEqual(values["transcription_rtf"], 0.25)

    def test_transcription_worker_reports_loaded_backend(self) -> None:
        state = State()
        config = {
            "audio": {"sample_rate": 16_000},
            "whisper": {
                "model": "small",
                "device": "cuda",
                "compute_type": "int8_float16",
                "device_index": 0,
            },
        }
        loop = AudioLoop(config, state, RecordingLog(), lambda _: None)

        with patch("perception.audio._make_transcriber", return_value=lambda _: ""):
            with patch("perception.audio.time.perf_counter", side_effect=[20.0, 20.125]):
                worker = threading.Thread(target=loop._run_transcription, daemon=True)
                worker.start()
                self.assertTrue(
                    _wait_until(
                        lambda: "whisper_load_ms" in state.debug_snapshot()["values"]
                    )
                )
                loop._utterances.put(None)
                worker.join(timeout=1.0)

        values = state.debug_snapshot()["values"]
        self.assertEqual(values["whisper_model"], "small")
        self.assertEqual(values["whisper_device"], "cuda")
        self.assertEqual(values["whisper_compute_type"], "int8_float16")
        self.assertEqual(values["whisper_device_index"], 0)
        self.assertEqual(values["whisper_load_ms"], 125.0)


class RecordingLog:
    def append(self, event_type: str, payload: dict[str, object]) -> None:
        return None


def _wait_until(predicate, timeout: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


if __name__ == "__main__":
    unittest.main()
