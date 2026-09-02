"""The trigger worker polls State but consumes cooldown only after accepted speech."""

import unittest

from brain.triggers import TriggerWorker
from state import State


SETTINGS = {
    "min_absence_hours": 4,
    "idle_interaction_minutes": 30,
    "cooldown_minutes": 60,
    "quiet_after_hour": 22,
    "quiet_before_hour": 8,
}


class TriggerWorkerTests(unittest.TestCase):
    def test_successful_poll_submits_once_and_records_grounded_event(self) -> None:
        state = State(person_present=True, present_since=0.0, last_interaction=0.0)
        submitted = []
        log = RecordingLog()
        worker = TriggerWorker(
            state,
            SETTINGS,
            lambda name, context: submitted.append((name, context)) or True,
            log,
        )

        event = worker.poll_once(now=1800.0, local_hour=12)
        repeated = worker.poll_once(now=7200.0, local_hour=12)

        self.assertIsNotNone(event)
        self.assertIsNone(repeated)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0][0], "quiet_interaction")
        self.assertEqual(log.events[0][0], "trigger_fired")
        self.assertEqual(log.events[0][1]["trigger"], "quiet_interaction")
        self.assertEqual(log.events[0][1]["duration_seconds"], 1800.0)

    def test_rejected_submission_does_not_consume_trigger(self) -> None:
        state = State(person_present=True, present_since=0.0, last_interaction=0.0)
        attempts = []
        worker = TriggerWorker(
            state,
            SETTINGS,
            lambda name, context: attempts.append(name) or False,
            RecordingLog(),
        )

        first = worker.poll_once(now=1800.0, local_hour=12)
        second = worker.poll_once(now=1801.0, local_hour=12)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(attempts, ["quiet_interaction", "quiet_interaction"])

    def test_poll_snapshots_all_busy_and_activity_fields_under_state_lock(self) -> None:
        state = State(
            person_present=True,
            present_since=100.0,
            last_interaction=200.0,
            muted_until=300.0,
            listening=True,
            thinking=True,
            speaking=True,
        )
        worker = TriggerWorker(
            state,
            SETTINGS,
            lambda name, context: True,
            RecordingLog(),
        )

        snapshot = worker.snapshot()

        self.assertTrue(snapshot.person_present)
        self.assertEqual(snapshot.present_since, 100.0)
        self.assertEqual(snapshot.last_interaction, 200.0)
        self.assertEqual(snapshot.muted_until, 300.0)
        self.assertTrue(snapshot.listening)
        self.assertTrue(snapshot.thinking)
        self.assertTrue(snapshot.speaking)

    def test_start_and_close_are_idempotent(self) -> None:
        state = State()
        worker = TriggerWorker(
            state,
            SETTINGS,
            lambda name, context: False,
            RecordingLog(),
            poll_seconds=0.01,
        )

        worker.start()
        worker.start()
        worker.close()
        worker.close()

        self.assertFalse(worker.is_running)


class RecordingLog:
    def __init__(self) -> None:
        self.events = []

    def append(self, event_type, payload) -> None:
        self.events.append((event_type, payload))


if __name__ == "__main__":
    unittest.main()
