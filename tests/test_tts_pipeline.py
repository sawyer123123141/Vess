"""One-clause-ahead TTS pipeline behavior."""

import threading
import time
import unittest

import numpy as np

from output.voice import VoiceOutput
from performance import PerformanceCue
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

    def test_playback_reports_waveform_edge_silence(self) -> None:
        state = State()

        def synthesize(text: str) -> np.ndarray:
            return np.concatenate(
                [
                    np.zeros(20, dtype=np.float32),
                    np.ones(50, dtype=np.float32),
                    np.zeros(30, dtype=np.float32),
                ]
            )

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            synthesize=synthesize,
            play=lambda audio, sample_rate: None,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("measured", generation_id=1)
        voice.close()

        playback_events = [
            event
            for event in state.debug_snapshot()["events"]
            if event["event"] == "tts_playback_started"
        ]
        self.assertEqual(len(playback_events), 1)
        event = playback_events[0]
        self.assertEqual(event["text"], "measured")
        self.assertEqual(event["leading_silence_ms"], 20.0)
        self.assertEqual(event["trailing_silence_ms"], 30.0)

    def test_excess_edge_silence_is_trimmed_to_safety_margins(self) -> None:
        state = State()
        played: list[np.ndarray] = []

        def synthesize(text: str) -> np.ndarray:
            return np.concatenate(
                [
                    np.zeros(330, dtype=np.float32),
                    np.ones(50, dtype=np.float32),
                    np.zeros(480, dtype=np.float32),
                ]
            )

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            synthesize=synthesize,
            play=lambda audio, sample_rate: played.append(audio.copy()),
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("trimmed", generation_id=1)
        voice.close()

        self.assertEqual(len(played), 1)
        self.assertEqual(played[0].size, 310)
        playback_event = next(
            event
            for event in state.debug_snapshot()["events"]
            if event["event"] == "tts_playback_started"
        )
        self.assertEqual(playback_event["raw_leading_silence_ms"], 330.0)
        self.assertEqual(playback_event["raw_trailing_silence_ms"], 480.0)
        self.assertEqual(playback_event["leading_silence_ms"], 100.0)
        self.assertEqual(playback_event["trailing_silence_ms"], 160.0)
        values = state.debug_snapshot()["values"]
        self.assertEqual(values["tts_raw_leading_silence_ms"], 330.0)
        self.assertEqual(values["tts_raw_trailing_silence_ms"], 480.0)
        self.assertEqual(values["tts_leading_silence_ms"], 100.0)
        self.assertEqual(values["tts_trailing_silence_ms"], 160.0)

    def test_trimming_preserves_quiet_onset_inside_leading_safety_margin(self) -> None:
        played: list[np.ndarray] = []
        quiet_onset = np.full(80, 0.0005, dtype=np.float32)

        def synthesize(text: str) -> np.ndarray:
            return np.concatenate(
                [
                    np.zeros(300, dtype=np.float32),
                    quiet_onset,
                    np.ones(50, dtype=np.float32),
                    np.zeros(300, dtype=np.float32),
                ]
            )

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            State(),
            RecordingLog(),
            synthesize=synthesize,
            play=lambda audio, sample_rate: played.append(audio.copy()),
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("quiet onset", generation_id=1)
        voice.close()

        self.assertEqual(len(played), 1)
        np.testing.assert_allclose(played[0][20:100], quiet_onset)
        self.assertEqual(float(played[0][100]), 1.0)

    def test_cached_acknowledgement_audio_is_not_trimmed(self) -> None:
        played: list[np.ndarray] = []
        acknowledgement = np.concatenate(
            [
                np.zeros(300, dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(400, dtype=np.float32),
            ]
        )

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            State(),
            RecordingLog(),
            synthesize=lambda text: acknowledgement.copy(),
            play=lambda audio, sample_rate: played.append(audio.copy()),
        )
        voice.start()
        voice.prepare_acknowledgement()
        voice.enqueue_acknowledgement()
        voice.close()

        self.assertEqual(len(played), 1)
        np.testing.assert_array_equal(played[0], acknowledgement)

    def test_performance_activates_only_during_physical_playback(self) -> None:
        state = State()
        play_started = threading.Event()
        release_play = threading.Event()

        def play(audio: np.ndarray, sample_rate: int) -> None:
            self.assertEqual(state.performance.expression, "playful")
            play_started.set()
            release_play.wait(timeout=1.0)

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            synthesize=lambda text: np.ones(10, dtype=np.float32),
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue(
            "hello",
            generation_id=1,
            performance=PerformanceCue("playful", 0.65),
        )
        self.assertTrue(play_started.wait(timeout=0.5))
        release_play.set()
        voice.close()

        self.assertEqual(state.performance, PerformanceCue())

    def test_prepared_cue_does_not_activate_before_playback(self) -> None:
        first_play_started = threading.Event()
        release_first_play = threading.Event()
        second_synthesized = threading.Event()
        state = State()

        def synthesize(text: str) -> np.ndarray:
            if text == "second":
                second_synthesized.set()
            return np.ones(10, dtype=np.float32)

        def play(audio: np.ndarray, sample_rate: int) -> None:
            if not first_play_started.is_set():
                first_play_started.set()
                release_first_play.wait(timeout=1.0)

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            synthesize=synthesize,
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("first", generation_id=1)
        voice.enqueue(
            "second",
            generation_id=1,
            performance=PerformanceCue("thoughtful", 0.55),
        )

        self.assertTrue(first_play_started.wait(timeout=0.5))
        self.assertTrue(second_synthesized.wait(timeout=0.5))
        self.assertEqual(state.performance, PerformanceCue())
        release_first_play.set()
        voice.close()

    def test_stale_prepared_audio_never_activates_its_performance(self) -> None:
        first_play_started = threading.Event()
        release_first_play = threading.Event()
        second_synthesized = threading.Event()
        seen: list[str] = []
        state = State()

        def synthesize(text: str) -> np.ndarray:
            if text == "old second":
                second_synthesized.set()
            return np.ones(10, dtype=np.float32)

        def play(audio: np.ndarray, sample_rate: int) -> None:
            seen.append(state.performance.expression)
            if len(seen) == 1:
                first_play_started.set()
                release_first_play.wait(timeout=1.0)

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            synthesize=synthesize,
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue("old first", generation_id=1)
        voice.enqueue(
            "old second",
            generation_id=1,
            performance=PerformanceCue("playful", 0.65),
        )
        self.assertTrue(first_play_started.wait(timeout=0.5))
        self.assertTrue(second_synthesized.wait(timeout=0.5))
        voice.begin_generation(2)
        voice.enqueue(
            "new answer",
            generation_id=2,
            performance=PerformanceCue("thoughtful", 0.55),
        )
        release_first_play.set()
        voice.close()

        self.assertNotIn("playful", seen)
        self.assertIn("thoughtful", seen)

    def test_playback_error_clears_performance_and_records_lifecycle(self) -> None:
        state = State()

        def fail_play(audio: np.ndarray, sample_rate: int) -> None:
            self.assertEqual(state.performance.expression, "emphatic")
            raise RuntimeError("speaker failed")

        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            synthesize=lambda text: np.ones(10, dtype=np.float32),
            play=fail_play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue(
            "important",
            generation_id=1,
            performance=PerformanceCue("emphatic", 0.7),
        )
        voice.close()

        self.assertEqual(state.performance, PerformanceCue())
        events = state.debug_snapshot()["events"]
        started = next(event for event in events if event["event"] == "performance_started")
        ended = next(event for event in events if event["event"] == "performance_ended")
        self.assertEqual(started["text"], "important")
        self.assertEqual(started["expression"], "emphatic")
        self.assertEqual(started["generation_id"], 1)
        self.assertEqual(ended["expression"], "emphatic")
        self.assertEqual(ended["generation_id"], 1)


if __name__ == "__main__":
    unittest.main()
