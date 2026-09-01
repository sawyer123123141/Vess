"""Cross-component barge-in verification across conversation, voice, and turn control."""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from brain.llm import ConversationWorker
from brain.turn_coordinator import TurnCoordinator
from output.audio_player import PlaybackReceipt
from output.voice import VoiceOutput
from performance import PerformanceCue
from state import State


CONFIG = {
    "personas": {"friendly": "Warm."},
    "memory": {"short_term_minutes": 10, "short_term_turns": 8},
    "voice": {"sample_rate": 24_000},
}


class FakeTimer:
    def __init__(self, seconds: float, callback) -> None:
        self.seconds = seconds
        self.callback = callback
        self.started = False
        self.cancelled = False
        self.fired = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if self.cancelled or self.fired:
            return
        self.fired = True
        self.callback()


class FakeTimerFactory:
    def __init__(self) -> None:
        self.timers: list[FakeTimer] = []

    def __call__(self, seconds: float, callback) -> FakeTimer:
        timer = FakeTimer(seconds, callback)
        self.timers.append(timer)
        return timer


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class ScriptedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def stream(self, prompt: str, config: dict):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("unexpected LLM request")
        yield self.responses.pop(0)

    def classify_mood(self, transcript: str, mood_names: set[str], config: dict):
        return None


class InterruptiblePlayer:
    """Complete ordinary calls and block selected playback calls until pause is requested."""

    def __init__(self, pause_on_calls: set[int]) -> None:
        self.pause_on_calls = set(pause_on_calls)
        self.pause_started = threading.Event()
        self.resume_started = threading.Event()
        self._release = threading.Event()
        self._lock = threading.Lock()
        self._play_count = 0
        self._current_call = 0
        self._generation_id: int | None = None
        self._sample_rate = 0
        self._total_frames = 0
        self._paused_receipt: PlaybackReceipt | None = None
        self.discarded_generations: list[int | None] = []
        self.resumed_generations: list[int | None] = []

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generation_id: int | None,
    ) -> PlaybackReceipt:
        with self._lock:
            self._play_count += 1
            call = self._play_count
            self._current_call = call
            self._generation_id = generation_id
            self._sample_rate = sample_rate
            self._total_frames = int(audio.size)
            self._paused_receipt = None
            self._release.clear()

        if call in self.pause_on_calls:
            self.pause_started.set()
            self._release.wait(timeout=2.0)
            with self._lock:
                paused = self._paused_receipt
            if paused is not None:
                return paused

        return PlaybackReceipt(
            "completed",
            generation_id,
            0,
            int(audio.size),
            int(audio.size),
            sample_rate,
        )

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        with self._lock:
            if self._current_call not in self.pause_on_calls:
                return None
            completed = min(2, max(self._total_frames - 1, 0))
            receipt = PlaybackReceipt(
                "paused",
                self._generation_id,
                0,
                completed,
                self._total_frames,
                self._sample_rate,
            )
            self._paused_receipt = receipt
        self._release.set()
        return receipt

    def resume(self) -> PlaybackReceipt:
        with self._lock:
            paused = self._paused_receipt
            if paused is None:
                raise RuntimeError("no paused waveform")
            self._paused_receipt = None
        self.resumed_generations.append(paused.generation_id)
        self.resume_started.set()
        return PlaybackReceipt(
            "completed",
            paused.generation_id,
            paused.frames_completed,
            paused.total_frames,
            paused.total_frames,
            paused.sample_rate,
        )

    def discard_paused(self) -> None:
        with self._lock:
            generation_id = (
                self._paused_receipt.generation_id
                if self._paused_receipt is not None
                else self._generation_id
            )
            self._paused_receipt = None
        self.discarded_generations.append(generation_id)


class BargeInFlowTests(unittest.TestCase):
    def _stack(self, responses: list[str], *, pause_on_calls: set[int]):
        state = State()
        log = RecordingLog()
        player = InterruptiblePlayer(pause_on_calls)
        timers = FakeTimerFactory()
        holder: dict[str, ConversationWorker] = {}
        deliveries: list[tuple[str, dict[str, object]]] = []

        def on_delivery(event_type: str, payload: dict[str, object]) -> None:
            deliveries.append((event_type, payload))
            conversation = holder.get("conversation")
            if conversation is not None:
                conversation.handle_delivery(event_type, payload)

        voice = VoiceOutput(
            CONFIG,
            state,
            log,
            synthesize=lambda _: np.ones(16, dtype=np.float32),
            player=player,
            on_delivery=on_delivery,
        )
        conversation = ConversationWorker(
            CONFIG,
            {"neutral": {}},
            state,
            log,
            ScriptedClient(responses),
            voice,
        )
        holder["conversation"] = conversation
        coordinator = TurnCoordinator(
            state,
            log,
            voice,
            conversation,
            conversation.submit,
            false_timeout_seconds=2.0,
            decision_watchdog_seconds=5.0,
            timer_factory=timers,
        )
        voice.start()
        conversation.start()
        return state, log, player, timers, deliveries, voice, conversation, coordinator

    def test_real_interruption_discards_old_remainder_and_remembers_only_delivered_old_speech(self) -> None:
        (
            state,
            _,
            player,
            _,
            deliveries,
            voice,
            conversation,
            coordinator,
        ) = self._stack(
            ["Old complete. Old interrupted. Old never.", "New answer."],
            pause_on_calls={2},
        )

        conversation.submit("old question")
        self.assertTrue(player.pause_started.wait(timeout=1.0))
        self.assertTrue(coordinator.on_candidate())
        coordinator.on_utterance_queued_for_transcription()
        coordinator.on_transcript("new question")
        conversation.close()
        voice.close()
        coordinator.close()

        started = [
            payload["text"]
            for event_type, payload in deliveries
            if event_type == "clause_started"
        ]
        self.assertIn("Old complete.", started)
        self.assertIn("Old interrupted.", started)
        self.assertNotIn("Old never.", started)
        self.assertIn("New answer.", started)
        self.assertIn(1, player.discarded_generations)

        self.assertGreaterEqual(len(state.conversation_turns), 2)
        old_turn = state.conversation_turns[0]
        new_turn = state.conversation_turns[-1]
        self.assertEqual(old_turn.user, "old question")
        self.assertEqual(old_turn.assistant, "Old complete.")
        self.assertEqual(old_turn.status, "interrupted")
        self.assertEqual(old_turn.interrupted_clause, "Old interrupted.")
        self.assertEqual((new_turn.user, new_turn.assistant), ("new question", "New answer."))
        self.assertEqual(new_turn.status, "completed")

    def test_false_interruption_resumes_same_waveform_and_finalizes_completed_turn(self) -> None:
        state, _, player, timers, _, voice, conversation, coordinator = self._stack(
            ["Keep going."],
            pause_on_calls={1},
        )

        conversation.submit("continue")
        self.assertTrue(player.pause_started.wait(timeout=1.0))
        self.assertTrue(coordinator.on_candidate())
        timers.timers[0].fire()
        self.assertTrue(player.resume_started.wait(timeout=1.0))
        conversation.close()
        voice.close()
        coordinator.close()

        self.assertEqual(player.resumed_generations, [1])
        self.assertEqual(len(state.conversation_turns), 1)
        turn = state.conversation_turns[0]
        self.assertEqual((turn.user, turn.assistant), ("continue", "Keep going."))
        self.assertEqual(turn.status, "completed")

    def test_slow_transcription_does_not_false_resume_before_watchdog(self) -> None:
        state, _, player, timers, _, voice, conversation, coordinator = self._stack(
            ["Interrupted response.", "Replacement."],
            pause_on_calls={1},
        )

        conversation.submit("first")
        self.assertTrue(player.pause_started.wait(timeout=1.0))
        self.assertTrue(coordinator.on_candidate())
        false_timer = timers.timers[0]
        coordinator.on_utterance_queued_for_transcription()
        watchdog = timers.timers[1]

        false_timer.fire()
        self.assertEqual(player.resumed_generations, [])
        self.assertTrue(state.listening)
        self.assertFalse(watchdog.cancelled)

        coordinator.on_transcript("replacement")
        self.assertTrue(watchdog.cancelled)
        conversation.close()
        voice.close()
        coordinator.close()
        self.assertEqual(player.resumed_generations, [])

    def test_delayed_old_decision_never_compare_cancels_newer_generation(self) -> None:
        _, log, player, _, _, voice, conversation, coordinator = self._stack(
            ["Old response.", "Independent response.", "Barge response."],
            pause_on_calls={1},
        )

        conversation.submit("old")
        self.assertTrue(player.pause_started.wait(timeout=1.0))
        self.assertTrue(coordinator.on_candidate())
        coordinator.on_utterance_queued_for_transcription()

        conversation.submit("independent newer request")
        coordinator.on_transcript("delayed barge text")
        conversation.close()
        voice.close()
        coordinator.close()

        cancellations = [
            payload
            for event_type, payload in log.events
            if event_type == "generation_cancelled"
        ]
        self.assertFalse(
            any(payload.get("expected_generation") == 2 for payload in cancellations)
        )
        self.assertIn(1, player.discarded_generations)

    def test_shutdown_while_paused_cannot_resume_or_leave_transient_face_state(self) -> None:
        state, _, player, timers, _, voice, conversation, coordinator = self._stack(
            ["Paused response."],
            pause_on_calls={1},
        )

        conversation.submit("pause me")
        self.assertTrue(player.pause_started.wait(timeout=1.0))
        self.assertTrue(coordinator.on_candidate())
        false_timer = timers.timers[0]

        coordinator.close()
        conversation.close()
        voice.close()
        false_timer.fire()

        self.assertEqual(player.resumed_generations, [])
        self.assertFalse(state.speaking)
        self.assertFalse(state.listening)
        self.assertEqual(state.performance, PerformanceCue())


if __name__ == "__main__":
    unittest.main()
