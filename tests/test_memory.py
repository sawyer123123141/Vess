"""Append-only event history behavior."""

import importlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


class EventLogTests(unittest.TestCase):
    def test_append_persists_timestamp_type_and_json_payload(self) -> None:
        event_log = _event_log_class()
        self.assertIsNotNone(event_log, "EventLog is missing")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            log = event_log(path)
            log.append("wake_rejected", {"transcript": "hey guess"}, timestamp=12.5)
            log.close()

            connection = sqlite3.connect(path)
            try:
                row = connection.execute(
                    "SELECT timestamp, event_type, payload_json FROM events").fetchone()
            finally:
                connection.close()

        self.assertEqual(row[0:2], (12.5, "wake_rejected"))
        self.assertEqual(json.loads(row[2]), {"transcript": "hey guess"})


def _event_log_class():
    try:
        return importlib.import_module("brain.memory").EventLog
    except ModuleNotFoundError:
        return None


if __name__ == "__main__":
    unittest.main()
