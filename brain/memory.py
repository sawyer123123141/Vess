"""Short-term conversation memory and append-only event history."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from state import ConversationTurn


EventRecord = tuple[float, str, str]


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
