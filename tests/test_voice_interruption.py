"""VoiceOutput pause, resume, commit, and delivery ordering."""

from __future__ import annotations

import threading
import unittest

import numpy as np

from output.audio_player import PlaybackReceipt
from output.voice import VoiceOutput
from performance import PerformanceCue
from state import State


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class ControlledPlayer:
    def __init__(self, *, pauseable: bool = False, on_resume=None) -> None:
        self.pauseable = pauseable
        self.on_resume = on_resume
        self.started = threading.Event()
        self.resume_started = threading.Event()
        self._release = threading.Event()
        self._current_generation: int | None = None
        self._sample_rate = 0
        self._total_frames = 0
        self._paused_receipt: PlaybackReceipt | None = None
        self.discarded = False

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generation_id: int | None,
    ) -> PlaybackReceipt:
        self._current_generation = generation_id
        self._sample_rate = sample_rate
        self._total_frames = int(audio.size)
        self.started.set()
        if self.pauseable:
            self._release.wait(timeout=1.0)
            if self._paused_receipt is not None:
                return self._paused_receipt
        return PlaybackReceipt(
            "completed",
            generation_id,
            0,
            self._total_frames,
            self._total_frames,
            sample_rate,
        )

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        if not self.started.is_set() or not self.pauseable:
            return None
        completed = min(1, max(self._total_frames - 1, 0))
        self._paused_receipt = PlaybackReceipt(
            "paused",
            self._current_generation,
            0,
            completed,
            self._total_frames,
            self._sample_rate,
        )
        self._release.set()
        return self._paused_receipt

    def resume(self) -> PlaybackReceipt:
        if self._paused_receipt is None or self.discarded:
            raise RuntimeError("no paused waveform")
        if self.on_resume is not None:
            self.on_resume()
        self.resume_started.set()
        paused = self._paused_receipt
        self._paused_receipt = None
        return PlaybackReceipt(
            "completed",
            paused.generation_id,
            paused.frames_completed,
            paused.total_frames,
            paused.total_frames,
            paused.sample_rate,
        )

    def discard_paused(self) -> None:
        self.discarded = True
        self._paused_receipt = None


class VoiceInterruptionTests(unittest.TestCase):
    def _voice(self, state: State, player: ControlledPlayer, events: list) -> VoiceOutput:
        return VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            synthesize=lambda text: np.ones(8, dtype=np.float32),
            player=player,
            on_delivery=lambda event_type, payload: events.append((event_type, payload)),
        )

    def test_pause_clears_physical_speaking_and_commit_abandons_exact_clause(self) -> None:
        state = State()
        events: list[tuple[str, dict[str, object]]] = []
        player = ControlledPlayer(pauseable=True)
        voice = self._voice(state, player, events)
        cue = PerformanceCue("playful", 0.7)
        voice.start()
        voice.begin_generation(11)
        voice.enqueue("hello", generation_id=11, performance=cue)
        self.assertTrue(player.started.wait(timeout=0.5))
        self.assertTrue(state.speaking)
        self.assertEqual(state.performance, cue)

        paused = voice.pause_for_interruption()

        self.assertIsNotNone(paused)
        assert paused is not None
        self.assertEqual(paused.generation_id, 11)
        self.assertFalse(state.speaking)
        self.assertEqual(state.performance, PerformanceCue())
        self.assertTrue(voice.commit_interruption(11))
        self.assertFalse(voice.commit_interruption(11))
        voice.close()

        self.assertTrue(player.discarded)
        names = [event_type for event_type, _ in events]
        self.assertIn("clause_paused", names)
        self.assertIn("clause_abandoned", names)
        self.assertNotIn("clause_completed", names)

    def test_false_resume_restores_original_performance_at_physical_resume(self) -> None:
        state = State()
        events: list[tuple[str, dict[str, object]]] = []
        seen_on_resume: list[PerformanceCue] = []
        player = ControlledPlayer(
            pauseable=True,
            on_resume=lambda: seen_on_resume.append(state.performance),
        )
        voice = self._voice(state, player, events)
        cue = PerformanceCue("thoughtful", 0.55)
        voice.start()
        voice.begin_generation(12)
        voice.enqueue("consider this", generation_id=12, performance=cue)
        self.assertTrue(player.started.wait(timeout=0.5))
        self.assertIsNotNone(voice.pause_for_interruption())

        self.assertTrue(voice.resume_after_false_interruption(12))
        self.assertTrue(player.resume_started.wait(timeout=0.5))
        voice.close()

        self.assertEqual(seen_on_resume, [cue])
        self.assertEqual(state.performance, PerformanceCue())
        names = [event_type for event_type, _ in events]
        self.assertIn("clause_resumed", names)
        self.assertIn("clause_completed", names)
        self.assertNotIn("clause_abandoned", names)

    def test_finish_generation_drains_after_all_preceding_clauses(self) -> None:
        state = State()
        events: list[tuple[str, dict[str, object]]] = []
        player = ControlledPlayer()
        voice = self._voice(state, player, events)
        voice.start()
        voice.begin_generation(13)
        voice.enqueue("one", generation_id=13)
        voice.enqueue("two", generation_id=13)
        voice.finish_generation(13)
        voice.close()

        lifecycle = [
            (event_type, payload.get("text"))
            for event_type, payload in events
        ]
        self.assertEqual(
            lifecycle,
            [
                ("clause_started", "one"),
                ("clause_completed", "one"),
                ("clause_started", "two"),
                ("clause_completed", "two"),
                ("generation_playback_drained", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
