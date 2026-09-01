"""One-clause-ahead TTS pipeline behavior."""

import threading
import time
import unittest

import numpy as np

from output.voice import VoiceOutput
from state import State


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class TtsPipelineTests(unittest.TestCase):
    def test_next_clause_synthesizes_while_current_clause_plays(self) -> None:
        first_play_started = threading.Event()
        release_first_play = threading.Event()
        second_synthesized = threading.Event()
        play_count = 0

        def synthesize(text: str) -> np.ndarray:
            if text == "second":
                second_synthesized.set()
            return np.array([1.0], dtype=np.float32)

        def play(audio: np.ndarray, sample_rate: int) -> None:
            nonlocal play_count
            play_count += 1
            if play_count == 1:
                first_play_started.set()
                release_first_play.wait(timeout=1.0)

        voice = VoiceOutput(
            {"voice": {"sample_rate": 24_000}},
            State(),
            RecordingLog(),
            synthesize=synthesize,
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("first", generation_id=1)
        voice.enqueue("second", generation_id=1)

        self.assertTrue(first_play_started.wait(timeout=0.5))
        synthesized_during_playback = second_synthesized.wait(timeout=0.25)
        release_first_play.set()
        voice.close()

        self.assertTrue(synthesized_during_playback)

    def test_only_one_waveform_is_prepared_ahead(self) -> None:
        first_play_started = threading.Event()
        release_first_play = threading.Event()
        second_synthesized = threading.Event()
        third_synthesis_started = threading.Event()
        play_count = 0
        state = State()

        def synthesize(text: str) -> np.ndarray:
            if text == "second":
                second_synthesized.set()
            elif text == "third":
                third_synthesis_started.set()
            return np.array([1.0], dtype=np.float32)

        def play(audio: np.ndarray, sample_rate: int) -> None:
            nonlocal play_count
            play_count += 1
            if play_count == 1:
                first_play_started.set()
                release_first_play.wait(timeout=1.0)

        voice = VoiceOutput(
            {"voice": {"sample_rate": 24_000}},
            state,
            RecordingLog(),
            synthesize=synthesize,
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("first", generation_id=1)
        voice.enqueue("second", generation_id=1)
        voice.enqueue("third", generation_id=1)

        self.assertTrue(first_play_started.wait(timeout=0.5))
        prepared_second = second_synthesized.wait(timeout=0.5)
        if prepared_second:
            time.sleep(0.05)
            prepared_third_too_early = third_synthesis_started.is_set()
            ready_depth = state.debug_snapshot()["values"].get("tts_ready_queue")
        else:
            prepared_third_too_early = False
            ready_depth = None

        release_first_play.set()
        voice.close()

        self.assertTrue(prepared_second)
        self.assertFalse(prepared_third_too_early)
        self.assertEqual(ready_depth, 1)

    def test_prepared_stale_generation_is_skipped_before_playback(self) -> None:
        first_play_started = threading.Event()
        release_first_play = threading.Event()
        old_second_synthesized = threading.Event()
        played: list[int] = []

        codes = {"old first": 1.0, "old second": 2.0, "new answer": 3.0}

        def synthesize(text: str) -> np.ndarray:
            if text == "old second":
                old_second_synthesized.set()
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
        voice.enqueue("old second", generation_id=1)

        self.assertTrue(first_play_started.wait(timeout=0.5))
        prepared_old_second = old_second_synthesized.wait(timeout=0.5)
        voice.begin_generation(2)
        voice.enqueue("new answer", generation_id=2)
        release_first_play.set()
        voice.close()

        self.assertTrue(prepared_old_second)
        self.assertEqual(played, [1, 3])
        self.assertIn(
            "stale_tts_skipped",
            [event["event"] for event in state.debug_snapshot()["events"]],
        )

    def test_records_actual_gap_between_played_clauses(self) -> None:
        state = State()

        def synthesize(text: str) -> np.ndarray:
            return np.array([1.0], dtype=np.float32)

        def play(audio: np.ndarray, sample_rate: int) -> None:
            time.sleep(0.01)

        voice = VoiceOutput(
            {"voice": {"sample_rate": 24_000}},
            state,
            RecordingLog(),
            synthesize=synthesize,
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("first", generation_id=1)
        voice.enqueue("second", generation_id=1)
        voice.close()

        values = state.debug_snapshot()["values"]
        self.assertIn("tts_gap_ms", values)
        self.assertGreaterEqual(values["tts_gap_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
