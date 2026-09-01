"""Generation-scoped delivered-speech accounting."""

import unittest

from brain.delivery import DeliveryLedger


class DeliveryLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finalized: list[tuple[int, str, str, str, str | None]] = []
        self.ledger = DeliveryLedger(
            lambda generation_id, user, assistant, status, interrupted_clause: self.finalized.append(
                (generation_id, user, assistant, status, interrupted_clause)
            )
        )

    def test_normal_finalization_requires_llm_finished_and_playback_drained(self) -> None:
        self.ledger.begin(7, "Why is the sky blue?")
        self.ledger.generated(7, "Blue light scatters more.")
        self.ledger.handle(
            "clause_started",
            {"generation_id": 7, "text": "Blue light scatters more."},
        )
        self.ledger.handle(
            "clause_completed",
            {"generation_id": 7, "text": "Blue light scatters more."},
        )

        self.ledger.llm_finished(7)
        self.assertEqual(self.finalized, [])

        self.ledger.handle("generation_playback_drained", {"generation_id": 7})

        self.assertEqual(
            self.finalized,
            [(7, "Why is the sky blue?", "Blue light scatters more.", "completed", None)],
        )

    def test_interruption_keeps_completed_clauses_and_marks_active_clause_partial(self) -> None:
        self.ledger.begin(8, "Explain rainbows")
        self.ledger.generated(8, "Light enters the droplet.")
        self.ledger.generated(8, "Then it bends and separates into colors.")
        self.ledger.handle(
            "clause_started",
            {"generation_id": 8, "text": "Light enters the droplet."},
        )
        self.ledger.handle(
            "clause_completed",
            {"generation_id": 8, "text": "Light enters the droplet."},
        )
        self.ledger.handle(
            "clause_started",
            {
                "generation_id": 8,
                "text": "Then it bends and separates into colors.",
            },
        )
        self.ledger.handle(
            "clause_paused",
            {
                "generation_id": 8,
                "text": "Then it bends and separates into colors.",
            },
        )

        self.assertTrue(self.ledger.interrupt(8))

        self.assertEqual(
            self.finalized,
            [
                (
                    8,
                    "Explain rainbows",
                    "Light enters the droplet.",
                    "interrupted",
                    "Then it bends and separates into colors.",
                )
            ],
        )

    def test_late_receipts_after_finalization_do_nothing(self) -> None:
        self.ledger.begin(9, "Keep it short")
        self.ledger.generated(9, "Sure.")
        self.ledger.handle(
            "clause_started",
            {"generation_id": 9, "text": "Sure."},
        )
        self.assertTrue(self.ledger.interrupt(9))
        snapshot = list(self.finalized)

        self.ledger.handle(
            "clause_completed",
            {"generation_id": 9, "text": "Sure."},
        )
        self.ledger.handle("generation_playback_drained", {"generation_id": 9})
        self.ledger.llm_finished(9)
        self.assertFalse(self.ledger.interrupt(9))

        self.assertEqual(self.finalized, snapshot)


if __name__ == "__main__":
    unittest.main()
