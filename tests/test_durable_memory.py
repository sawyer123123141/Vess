"""Durable fact persistence, validation, and deterministic retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brain.memory import FactCandidate, FactMemory


class DurableMemoryTests(unittest.TestCase):
    def test_fact_persists_and_same_key_updates_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            values = iter(("blue", "green"))

            def extractor(_text: str, _known: tuple[str, ...]) -> list[FactCandidate]:
                return [FactCandidate("favorite_color", next(values))]

            first = FactMemory(path, extractor)
            first.remember("My favorite color is blue")
            first.close()

            second = FactMemory(path, extractor)
            facts = second.relevant_facts("What is my favorite color?")
            self.assertEqual([(fact.key, fact.value) for fact in facts], [("favorite_color", "blue")])

            second.remember("Actually my favorite color is green")
            second.close()

            third = FactMemory(path, lambda _text, _known: [])
            facts = third.relevant_facts("favorite color")
            third.close()

        self.assertEqual([(fact.key, fact.value) for fact in facts], [("favorite_color", "green")])

    def test_invalid_candidates_are_rejected_before_storage(self) -> None:
        candidates = [
            FactCandidate("Favorite Color", "blue"),
            FactCandidate("", "empty key"),
            FactCandidate("x" * 65, "long key"),
            FactCandidate("valid_key", ""),
            FactCandidate("other_valid_key", "v" * 241),
            FactCandidate("api_token", "definitely-not-memory"),
            FactCandidate("favorite_snack", "pretzels"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            memory = FactMemory(path, lambda _text, _known: candidates)
            memory.remember("Several candidate facts")
            memory.close()

            reopened = FactMemory(path, lambda _text, _known: [])
            facts = reopened.relevant_facts("What is my favorite snack?")
            keys = reopened.known_keys()
            reopened.close()

        self.assertEqual([(fact.key, fact.value) for fact in facts], [("favorite_snack", "pretzels")])
        self.assertEqual(keys, ("favorite_snack",))

    def test_retrieval_is_relevant_bounded_and_quiet_for_unrelated_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"

            def extractor(text: str, _known: tuple[str, ...]) -> list[FactCandidate]:
                if text == "seed":
                    return [
                        FactCandidate("dog_name", "Rex"),
                        FactCandidate("favorite_color", "navy blue"),
                        FactCandidate("current_project", "Vess voice assistant"),
                    ]
                return []

            memory = FactMemory(path, extractor)
            memory.remember("seed")
            memory.close()

            reopened = FactMemory(path, lambda _text, _known: [])
            dog = reopened.relevant_facts("What is my dog's name?", limit=1)
            unrelated = reopened.relevant_facts("How does photosynthesis work?")
            broad = reopened.relevant_facts("What do you remember about me?", limit=2)
            reopened.close()

        self.assertEqual([(fact.key, fact.value) for fact in dog], [("dog_name", "Rex")])
        self.assertEqual(unrelated, [])
        self.assertEqual(len(broad), 2)

    def test_extractor_receives_existing_keys(self) -> None:
        seen: list[tuple[str, ...]] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"

            def extractor(text: str, known: tuple[str, ...]) -> list[FactCandidate]:
                seen.append(known)
                if text == "first":
                    return [FactCandidate("favorite_color", "blue")]
                return []

            memory = FactMemory(path, extractor)
            memory.remember("first")
            memory.remember("second")
            memory.close()

        self.assertEqual(seen[0], ())
        self.assertEqual(seen[1], ("favorite_color",))


if __name__ == "__main__":
    unittest.main()
