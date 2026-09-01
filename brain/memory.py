"""Append-only event history for later memory work."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


EventRecord = tuple[float, str, str]


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
