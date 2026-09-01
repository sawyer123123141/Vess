"""Runtime wiring for Vess targets."""

import unittest

import numpy as np

import main
from state import State


class MainTests(unittest.TestCase):
    def test_default_display_fans_frames_to_the_browser_target(self) -> None:
        build_display = getattr(main, "_build_display", None)
        self.assertIsNotNone(build_display, "display target builder is missing")
        config = {
            "display": {"cv2_enabled": False, "preview_scale": 8},
            "web": {"enabled": True, "port": 8080},
        }
        display, server = build_display(config, State())
        self.assertIsNotNone(server)

        display.show(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertIsNotNone(server.preview.png())

    def test_expired_mood_is_logged(self) -> None:
        state = State(mood="annoyed", mood_until=10.0)
        log = RecordingLog()

        main._expire_mood(state, log, 10.1)

        self.assertEqual(
            log.events,
            [("mood_changed", {"from": "annoyed", "to": "neutral"})],
        )


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


if __name__ == "__main__":
    unittest.main()
