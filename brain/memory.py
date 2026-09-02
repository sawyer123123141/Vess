"""Short-term conversation memory and durable local history."""

from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state import ConversationTurn


EventRecord = tuple[float, str, str]
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
                    for candidate in candidates[:3]:
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


def append_conversation_turn(
    state: Any,
    user: str,
    assistant: str,
    *,
    timestamp: float | None = None,
    max_age_seconds: float,
    max_turns: int,
    status: str = "completed",
    interrupted_clause: str | None = None,
) -> ConversationTurn:
    """Store one delivered exchange and keep short-term history bounded."""
    now = time.time() if timestamp is None else float(timestamp)
    turn = ConversationTurn(
        now,
        user,
        assistant,
        status=status,
        interrupted_clause=interrupted_clause,
    )
    with state.locked():
        state.conversation_turns.append(turn)
        _prune_turns_locked(
            state,
            now=now,
            max_age_seconds=max_age_seconds,
            max_turns=max_turns,
        )
    return turn


def recent_conversation_turns(
    state: Any,
    *,
    now: float | None = None,
    max_age_seconds: float,
    max_turns: int,
) -> list[ConversationTurn]:
    """Return the bounded recent transcript without exposing the live list."""
    current = time.time() if now is None else float(now)
    with state.locked():
        _prune_turns_locked(
            state,
            now=current,
            max_age_seconds=max_age_seconds,
            max_turns=max_turns,
        )
        return list(state.conversation_turns)


def _prune_turns_locked(
    state: Any,
    *,
    now: float,
    max_age_seconds: float,
    max_turns: int,
) -> None:
    if max_age_seconds <= 0.0 or max_turns <= 0:
        state.conversation_turns.clear()
        return

    cutoff = now - max_age_seconds
    state.conversation_turns[:] = [
        turn for turn in state.conversation_turns if turn.timestamp >= cutoff
    ][-max_turns:]


class EventLog:
    """Write timestamped JSON events without blocking the caller."""

    def __init__(self, path: Path) -> None:
        self._queue: queue.SimpleQueue[EventRecord | None] = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._write_events,
            args=(path,),
            name="event-log",
            daemon=True,
        )
        self._thread.start()

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        timestamp: float | None = None,
    ) -> None:
        """Queue an event for durable, append-only storage."""
        self._queue.put(
            (
                time.time() if timestamp is None else timestamp,
                event_type,
                json.dumps(payload, separators=(",", ":")),
            )
        )

    def close(self) -> None:
        """Flush queued events before application shutdown."""
        self._queue.put(None)
        self._thread.join()

    def _write_events(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.commit()
            while True:
                event = self._queue.get()
                if event is None:
                    break
                connection.execute(
                    "INSERT INTO events (timestamp, event_type, payload_json) "
                    "VALUES (?, ?, ?)",
                    event,
                )
                connection.commit()
        finally:
            connection.close()
