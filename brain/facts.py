"""Durable user facts persisted in the local Vess database."""

from __future__ import annotations

import queue
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


_FACT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_FACT_VALUE_CHARS = 240
_SENSITIVE_KEY_PARTS = (
    "password",
    "passcode",
    "api_key",
    "token",
    "secret",
    "credential",
    "social_security",
    "ssn",
    "credit_card",
    "debit_card",
    "bank_account",
    "routing_number",
    "medical",
    "diagnosis",
    "medication",
    "religion",
    "political",
    "sexual",
    "race",
    "ethnicity",
)
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "do",
    "does",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "s",
    "the",
    "to",
    "what",
    "you",
}
_BROAD_MEMORY_PHRASES = (
    "remember about me",
    "remember about myself",
    "what do you remember",
    "what do you know about me",
    "what do you know about myself",
)


@dataclass(frozen=True)
class FactCandidate:
    key: str
    value: str


@dataclass(frozen=True)
class DurableFact:
    key: str
    value: str
    source_text: str
    created_at: float
    updated_at: float


class FactMemory:
    """Extract and persist explicit user facts without blocking conversation delivery."""

    def __init__(
        self,
        path: Path,
        extractor: Callable[[str, tuple[str, ...]], list[FactCandidate]],
    ) -> None:
        self._path = path
        self._extractor = extractor
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._closed = False
        self._ensure_schema()
        self._thread = threading.Thread(
            target=self._run,
            name="fact-memory",
            daemon=True,
        )
        self._thread.start()

    def remember(self, text: str) -> None:
        """Queue one user utterance for background durable-fact extraction."""
        clean = text.strip()
        if not clean or self._closed:
            return
        self._queue.put(clean)

    def known_keys(self) -> tuple[str, ...]:
        """Return stable fact keys for extraction updates and diagnostics."""
        try:
            with sqlite3.connect(self._path, timeout=1.0) as connection:
                rows = connection.execute(
                    "SELECT key FROM facts ORDER BY key"
                ).fetchall()
        except sqlite3.Error:
            return ()
        return tuple(str(row[0]) for row in rows)

    def relevant_facts(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[DurableFact]:
        """Return only query-relevant facts, with recent fallback for memory questions."""
        if limit <= 0:
            return []
        try:
            with sqlite3.connect(self._path, timeout=1.0) as connection:
                rows = connection.execute(
                    "SELECT key, value, source_text, created_at, updated_at FROM facts"
                ).fetchall()
        except sqlite3.Error:
            return []

        facts = [
            DurableFact(
                key=str(row[0]),
                value=str(row[1]),
                source_text=str(row[2]),
                created_at=float(row[3]),
                updated_at=float(row[4]),
            )
            for row in rows
        ]
        normalized_query = _normalize_text(query)
        if any(phrase in normalized_query for phrase in _BROAD_MEMORY_PHRASES):
            return sorted(facts, key=lambda fact: fact.updated_at, reverse=True)[:limit]

        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        ranked: list[tuple[int, float, DurableFact]] = []
        for fact in facts:
            fact_tokens = _tokens(f"{fact.key.replace('_', ' ')} {fact.value}")
            score = len(query_tokens & fact_tokens)
            if score > 0:
                ranked.append((score, fact.updated_at, fact))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [fact for _score, _updated, fact in ranked[:limit]]

    def close(self) -> None:
        """Drain extraction work before shutdown."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._path, timeout=1.0) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.commit()

    def _run(self) -> None:
        connection = sqlite3.connect(self._path, timeout=1.0)
        try:
            while True:
                text = self._queue.get()
                if text is None:
                    return
                try:
                    known = tuple(
                        str(row[0])
                        for row in connection.execute(
                            "SELECT key FROM facts ORDER BY key"
                        ).fetchall()
                    )
                    candidates = self._extractor(text, known)
                    now = time.time()
                    stored = 0
                    for candidate in candidates:
                        clean = _validated_candidate(candidate)
                        if clean is None:
                            continue
                        connection.execute(
                            """
                            INSERT INTO facts (
                                key, value, source_text, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(key) DO UPDATE SET
                                value = excluded.value,
                                source_text = excluded.source_text,
                                updated_at = excluded.updated_at
                            """,
                            (clean.key, clean.value, text, now, now),
                        )
                        stored += 1
                        if stored >= 3:
                            break
                    connection.commit()
                except Exception:
                    connection.rollback()
        finally:
            connection.close()


def _validated_candidate(candidate: FactCandidate) -> FactCandidate | None:
    key = candidate.key.strip()
    value = candidate.value.strip()
    if not _FACT_KEY.fullmatch(key):
        return None
    if not value or len(value) > _MAX_FACT_VALUE_CHARS:
        return None
    if any(part in key for part in _SENSITIVE_KEY_PARTS):
        return None
    return FactCandidate(key, value)


def _normalize_text(value: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in value).split()
    )


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize_text(value).split()
        if len(token) > 1 and token not in _STOP_WORDS
    }
