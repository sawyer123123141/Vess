"""Two-phase reversible barge-in decision behavior."""

from __future__ import annotations

import unittest

from brain.turn_coordinator import TurnCoordinator
from output.audio_player import PlaybackReceipt
from state import State


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


class FakeVoice:
    def __init__(self, effects: list[tuple], generation_id: int = 7) -> None:
        self.effects = effects
        self.generation_id = generation_id
        self.pause_calls = 0
        self.resume_result = True

    def pause_for_interruption(self) -> PlaybackReceipt:
        self.pause_calls += 1
        self.effects.append(("pause", self.generation_id))
        return PlaybackReceipt(
            status="paused",
            generation_id=self.generation_id,
            frames_started=100,
            frames_completed=40,
            total_frames=100,
            sample_rate=24_000,
        )

    def commit_interruption(self, generation_id: int) -> bool:
        self.effects.append(("commit", generation_id))
        return True

    def resume_after_false_interruption(self, generation_id: int) -> bool:
        self.effects.append(("resume", generation_id))
        return self.resume_result


class FakeConversation:
    def __init__(self, effects: list[tuple], latest_generation: int = 7) -> None:
        self.effects = effects
        self.latest_generation = latest_generation

    def cancel_generation(self, generation_id: int, reason: str) -> bool:
        self.effects.append(("cancel", generation_id, reason))
        if generation_id != self.latest_generation:
            return False
        self.latest_generation += 1
        return True


class TurnCoordinatorTests(unittest.TestCase):
    def _coordinator(self):
        effects: list[tuple] = []
        timers = FakeTimerFactory()
        state = State(speaking=True)
        log = RecordingLog()
        voice = FakeVoice(effects)
        conversation = FakeConversation(effects)

        def submit(text: str) -> None:
            effects.append(("submit", text))

        coordinator = TurnCoordinator(
            state,
            log,
            voice,
            conversation,
            submit,
            false_timeout_seconds=2.0,
            decision_watchdog_seconds=5.0,
            timer_factory=timers,
        )
        return coordinator, state, log, voice, conversation, timers, effects

    def test_nonempty_transcript_commits_in_cancel_commit_submit_order(self) -> None:
        coordinator, state, _, _, _, timers, effects = self._coordinator()

        self.assertTrue(coordinator.on_candidate())
        self.assertTrue(state.listening)
        coordinator.on_utterance_queued_for_transcription()
        coordinator.on_transcript("wait")

        self.assertEqual(
            effects[-3:],
            [("cancel", 7, "barge_in"), ("commit", 7), ("submit", "wait")],
        )
        self.assertFalse(state.listening)
        self.assertTrue(timers.timers[0].cancelled)
        self.assertTrue(timers.timers[1].cancelled)

    def test_false_timeout_resumes_same_generation_once(self) -> None:
        coordinator, state, _, voice, _, timers, effects = self._coordinator()

        self.assertTrue(coordinator.on_candidate())
        self.assertEqual(voice.pause_calls, 1)
        false_timer = timers.timers[0]
        false_timer.fire()
        false_timer.fire()
        coordinator.on_transcript("")

        self.assertEqual([effect for effect in effects if effect[0] == "resume"], [("resume", 7)])
        self.assertFalse(state.listening)

    def test_transcription_phase_suspends_false_timeout_and_uses_watchdog(self) -> None:
        coordinator, _, _, _, _, timers, effects = self._coordinator()

        self.assertTrue(coordinator.on_candidate())
        false_timer = timers.timers[0]
        coordinator.on_utterance_queued_for_transcription()
        watchdog = timers.timers[1]

        self.assertTrue(false_timer.cancelled)
        self.assertEqual(watchdog.seconds, 5.0)
        false_timer.fire()
        self.assertNotIn(("resume", 7), effects)

        coordinator.on_transcript("real words")
        self.assertTrue(watchdog.cancelled)

    def test_watchdog_discards_old_pause_when_same_generation_cannot_resume(self) -> None:
        coordinator, state, log, voice, _, timers, effects = self._coordinator()
        voice.resume_result = False

        self.assertTrue(coordinator.on_candidate())
        coordinator.on_utterance_queued_for_transcription()
        timers.timers[1].fire()

        self.assertIn(("resume", 7), effects)
        self.assertIn(("commit", 7), effects)
        self.assertFalse(state.listening)
        self.assertIn("barge_in_watchdog", [event_type for event_type, _ in log.events])

    def test_delayed_old_interruption_never_cancels_newer_generation(self) -> None:
        coordinator, _, _, _, conversation, _, effects = self._coordinator()

        self.assertTrue(coordinator.on_candidate())
        coordinator.on_utterance_queued_for_transcription()
        conversation.latest_generation = 8
        coordinator.on_transcript("new interruption text")

        cancel_effects = [effect for effect in effects if effect[0] == "cancel"]
        self.assertEqual(cancel_effects, [("cancel", 7, "barge_in")])
        self.assertEqual(conversation.latest_generation, 8)
        self.assertIn(("commit", 7), effects)
        self.assertIn(("submit", "new interruption text"), effects)

    def test_duplicate_candidates_and_decisions_are_idempotent(self) -> None:
        coordinator, _, _, voice, _, _, effects = self._coordinator()

        self.assertTrue(coordinator.on_candidate())
        self.assertTrue(coordinator.on_candidate())
        self.assertEqual(voice.pause_calls, 1)

        coordinator.on_transcript("stop")
        coordinator.on_transcript("stop again")
        coordinator.on_transcription_error(RuntimeError("late"))

        self.assertEqual(len([effect for effect in effects if effect[0] == "cancel"]), 1)
        self.assertEqual(len([effect for effect in effects if effect[0] == "commit"]), 1)
        self.assertEqual(len([effect for effect in effects if effect[0] == "submit"]), 1)

    def test_close_cancels_timers_discards_pause_and_cannot_later_resume(self) -> None:
        coordinator, state, _, _, _, timers, effects = self._coordinator()

        self.assertTrue(coordinator.on_candidate())
        timer = timers.timers[0]
        coordinator.close()
        timer.fire()
        coordinator.on_transcript("")

        self.assertTrue(timer.cancelled)
        self.assertIn(("commit", 7), effects)
        self.assertNotIn(("resume", 7), effects)
        self.assertFalse(state.listening)
        self.assertFalse(coordinator.on_candidate())


if __name__ == "__main__":
    unittest.main()
