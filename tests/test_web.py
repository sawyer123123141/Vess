"""Browser-preview behavior."""

import importlib
import socket
import time
import unittest
from urllib.request import urlopen

import cv2
import numpy as np
from fastapi.testclient import TestClient

from state import State


class WebPreviewTests(unittest.TestCase):
    def test_frame_endpoint_returns_the_latest_lossless_rgb_frame(self) -> None:
        web = _web_module()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[10, 20] = (12, 34, 56)
        expected = frame.copy()
        preview = web.WebPreview()
        preview.show(frame)
        frame[10, 20] = (99, 99, 99)

        app = web.create_app(State(), preview)
        with TestClient(app) as client:
            response = client.get("/frame.png")

        decoded = cv2.imdecode(
            np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertTrue(np.array_equal(
            cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
            expected,
        ))

    def test_color_endpoint_sets_the_explicit_state_override(self) -> None:
        web = _web_module()
        state = State()
        app = web.create_app(state, web.WebPreview())

        with TestClient(app) as client:
            response = client.put("/color", json={"color": [12, 34, 56]})

        self.assertEqual(response.status_code, 200)
        with state.locked():
            self.assertEqual(state.color, (12, 34, 56))

    def test_color_reset_clears_the_explicit_state_override(self) -> None:
        web = _web_module()
        state = State(color=(12, 34, 56))
        app = web.create_app(state, web.WebPreview())

        with TestClient(app) as client:
            response = client.delete("/color")

        self.assertEqual(response.status_code, 200)
        with state.locked():
            self.assertIsNone(state.color)

    def test_color_endpoints_record_explicit_override_history(self) -> None:
        web = _web_module()
        log = RecordingLog()
        app = web.create_app(State(), web.WebPreview(), event_log=log)

        with TestClient(app) as client:
            client.put("/color", json={"color": [12, 34, 56]})
            client.delete("/color")

        self.assertEqual(
            log.events,
            [
                ("color_override_set", {"color": [12, 34, 56]}),
                ("color_override_cleared", {}),
            ],
        )

    def test_debug_endpoint_reports_runtime_state_and_recent_events(self) -> None:
        web = _web_module()
        state = State(listening=True, thinking=True)
        app = web.create_app(state, web.WebPreview())

        with TestClient(app) as client:
            response = client.get("/debug")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["runtime"],
            {"listening": True, "thinking": True, "speaking": False},
        )
        self.assertEqual(response.json()["events"], [])

    def test_debug_endpoint_retains_recent_worker_event(self) -> None:
        web = _web_module()
        state = State()
        state.record_debug("wake_rejected", transcript="hey guest")
        app = web.create_app(state, web.WebPreview())

        with TestClient(app) as client:
            response = client.get("/debug")

        self.assertEqual(response.json()["events"][0]["event"], "wake_rejected")
        self.assertEqual(response.json()["events"][0]["transcript"], "hey guest")

    def test_homepage_polls_lossless_preview_at_30fps_with_color_controls(self) -> None:
        web = _web_module()
        app = web.create_app(State(), web.WebPreview())

        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("frame.png", response.text)
        self.assertIn("1000 / 30", response.text)
        self.assertIn('id="color"', response.text)
        self.assertIn('id="reset"', response.text)
        self.assertIn('id="debug"', response.text)
        self.assertIn("/debug", response.text)

    def test_server_serves_preview_from_its_own_thread(self) -> None:
        web = _web_module()
        server_class = getattr(web, "WebServer", None)
        self.assertIsNotNone(server_class, "web server lifecycle is missing")
        port = _free_port()
        server = server_class(State(), port)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        server.preview.show(frame)

        server.start()
        try:
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/frame.png", timeout=0.2) as response:
                        status = response.status
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        finally:
            server.close()

        self.assertEqual(status, 200)


def _web_module():
    try:
        return importlib.import_module("control.web")
    except ModuleNotFoundError:
        raise AssertionError("browser preview target is missing") from None


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
