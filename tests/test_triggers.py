"""Pure Step 7 trigger decisions without threads, hardware, or an LLM."""

import unittest

from brain.triggers import TriggerDecider, TriggerEvent, TriggerSnapshot


SETTINGS = {
    "min_absence_hours": 4,
    "idle_interaction_minutes": 30,
    "cooldown_minutes": 60,
    "quiet_after_hour": 22,
    "quiet_before_hour": 8,
}


def snapshot(
    *,
    present: bool,
    present_since: float | None = None,
    last_interaction: float = 0.0,
    muted_until: float = 0.0,
    listening: bool = False,
    thinking: bool = False,
    speaking: bool = False,
) -> TriggerSnapshot:
    return TriggerSnapshot(
        person_present=present,
        present_since=present_since,
        last_interaction=last_interaction,
        muted_until=muted_until,
        listening=listening,
        thinking=thinking,
        speaking=speaking,
    )


class TriggerDeciderTests(unittest.TestCase):
    def test_starting_absent_does_not_manufacture_a_long_absence(self) -> None:
        decider = TriggerDecider(SETTINGS)

        self.assertIsNone(decider.evaluate(snapshot(present=False), now=0.0, local_hour=12))
        event = decider.evaluate(
            snapshot(present=True, present_since=20_000.0),
            now=20_000.0,
            local_hour=12,
        )

        self.assertIsNone(event)

    def test_return_fires_only_after_observed_absence_threshold(self) -> None:
        decider = TriggerDecider(SETTINGS)
        decider.evaluate(snapshot(present=True, present_since=0.0), now=0.0, local_hour=12)
        decider.evaluate(snapshot(present=False), now=100.0, local_hour=12)

        too_soon = decider.evaluate(
            snapshot(present=True, present_since=100.0 + 3 * 3600.0),
            now=100.0 + 3 * 3600.0,
            local_hour=12,
        )
        self.assertIsNone(too_soon)

        decider.evaluate(snapshot(present=False), now=20_000.0, local_hour=12)
        returned_at = 20_000.0 + 4.5 * 3600.0
        event = decider.evaluate(
            snapshot(present=True, present_since=returned_at),
            now=returned_at,
            local_hour=12,
        )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.name, "returned_after_absence")
        self.assertAlmostEqual(event.duration_seconds, 4.5 * 3600.0)
        self.assertIn("about 4", event.context)

    def test_quiet_interaction_fires_once_until_real_user_activity_resets_it(self) -> None:
        decider = TriggerDecider(SETTINGS)
        start = 1_000.0
        present = snapshot(present=True, present_since=start, last_interaction=start)
        decider.evaluate(present, now=start, local_hour=12)

        event = decider.evaluate(present, now=start + 30 * 60.0, local_hour=12)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.name, "quiet_interaction")
        decider.accept(event, now=start + 30 * 60.0)

        self.assertIsNone(
            decider.evaluate(present, now=start + 3 * 3600.0, local_hour=12)
        )

        interacted = snapshot(
            present=True,
            present_since=start,
            last_interaction=start + 3 * 3600.0,
        )
        self.assertIsNone(
            decider.evaluate(interacted, now=start + 3 * 3600.0, local_hour=12)
        )
        later = decider.evaluate(
            interacted,
            now=start + 3 * 3600.0 + 30 * 60.0,
            local_hour=12,
        )
        self.assertIsNotNone(later)
        assert later is not None
        self.assertEqual(later.name, "quiet_interaction")

    def test_accepted_return_prevents_later_quiet_line_without_user_response(self) -> None:
        decider = TriggerDecider(SETTINGS)
        decider.evaluate(snapshot(present=True, present_since=0.0), now=0.0, local_hour=12)
        decider.evaluate(snapshot(present=False), now=100.0, local_hour=12)
        returned_at = 100.0 + 5 * 3600.0
        return_event = decider.evaluate(
            snapshot(present=True, present_since=returned_at),
            now=returned_at,
            local_hour=12,
        )
        self.assertIsNotNone(return_event)
        assert return_event is not None
        decider.accept(return_event, returned_at)

        self.assertIsNone(
            decider.evaluate(
                snapshot(present=True, present_since=returned_at),
                now=returned_at + 2 * 3600.0,
                local_hour=12,
            )
        )

    def test_failed_attempt_does_not_consume_latch_or_cooldown(self) -> None:
        decider = TriggerDecider(SETTINGS)
        start = 5_000.0
        idle = snapshot(present=True, present_since=start, last_interaction=start)
        decider.evaluate(idle, now=start, local_hour=12)

        first = decider.evaluate(idle, now=start + 30 * 60.0, local_hour=12)
        second = decider.evaluate(idle, now=start + 30 * 60.0 + 1.0, local_hour=12)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.name, second.name)

    def test_common_gates_suppress_proactive_speech(self) -> None:
        base = 10_000.0
        cases = (
            (snapshot(present=False), 12),
            (snapshot(present=True, present_since=base, muted_until=base + 9999), 12),
            (snapshot(present=True, present_since=base, listening=True), 12),
            (snapshot(present=True, present_since=base, thinking=True), 12),
            (snapshot(present=True, present_since=base, speaking=True), 12),
            (snapshot(present=True, present_since=base), 23),
            (snapshot(present=True, present_since=base), 7),
        )
        for blocked, hour in cases:
            with self.subTest(snapshot=blocked, hour=hour):
                decider = TriggerDecider(SETTINGS)
                decider.evaluate(blocked, now=base, local_hour=hour)
                self.assertIsNone(
                    decider.evaluate(blocked, now=base + 3 * 3600.0, local_hour=hour)
                )

    def test_global_cooldown_applies_across_trigger_types(self) -> None:
        decider = TriggerDecider(SETTINGS)
        start = 2_000.0
        idle = snapshot(present=True, present_since=start, last_interaction=start)
        decider.evaluate(idle, now=start, local_hour=12)
        event = decider.evaluate(idle, now=start + 30 * 60.0, local_hour=12)
        assert event is not None
        decider.accept(event, now=start + 30 * 60.0)

        # A real interaction resets the one-shot latch, but not the global cooldown.
        interacted_at = start + 31 * 60.0
        interacted = snapshot(
            present=True,
            present_since=start,
            last_interaction=interacted_at,
        )
        decider.evaluate(interacted, now=interacted_at, local_hour=12)
        blocked = decider.evaluate(
            interacted,
            now=interacted_at + 30 * 60.0,
            local_hour=12,
        )
        self.assertIsNone(blocked)

        allowed = decider.evaluate(
            interacted,
            now=start + 90 * 60.0,
            local_hour=12,
        )
        self.assertIsNotNone(allowed)


if __name__ == "__main__":
    unittest.main()
