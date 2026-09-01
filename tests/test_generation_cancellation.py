"""Conversation generation cancellation must never invalidate newer intent."""

import unittest

from brain.llm import ConversationWorker
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


class GenerationCancellationTests(unittest.TestCase):
    def _worker(self) -> tuple[ConversationWorker, RecordingVoice, RecordingLog]:
        voice = RecordingVoice()
        log = RecordingLog()
        worker = ConversationWorker(
            {},
            {"neutral": {}},
            State(),
            log,
            object(),
            voice,
        )
        return worker, voice, log

    def test_exact_current_generation_is_invalidated_without_submitting_text(self) -> None:
        worker, voice, log = self._worker()
        worker.submit("first")
        original = worker._latest_generation

        cancelled = worker.cancel_generation(original, "barge_in")

        self.assertTrue(cancelled)
        self.assertGreater(worker._latest_generation, original)
        self.assertEqual(voice.generations, [original, worker._latest_generation])
        self.assertTrue(
            any(
                event_type == "generation_cancelled"
                and payload["expected_generation"] == original
                and payload["replacement_generation"] == worker._latest_generation
                and payload["reason"] == "barge_in"
                for event_type, payload in log.events
            )
        )

    def test_duplicate_cancel_is_harmless(self) -> None:
        worker, voice, _ = self._worker()
        worker.submit("first")
        original = worker._latest_generation
        self.assertTrue(worker.cancel_generation(original, "barge_in"))
        replacement = worker._latest_generation

        self.assertFalse(worker.cancel_generation(original, "barge_in"))
        self.assertEqual(worker._latest_generation, replacement)
        self.assertEqual(voice.generations, [original, replacement])

    def test_delayed_cancel_of_old_generation_cannot_cancel_newer_request(self) -> None:
        worker, voice, _ = self._worker()
        worker.submit("first")
        old_generation = worker._latest_generation
        worker.submit("newer")
        newer_generation = worker._latest_generation

        cancelled = worker.cancel_generation(old_generation, "barge_in")

        self.assertFalse(cancelled)
        self.assertEqual(worker._latest_generation, newer_generation)
        self.assertEqual(voice.generations, [old_generation, newer_generation])


if __name__ == "__main__":
    unittest.main()
