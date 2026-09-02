"""Voice Lab cancellable TTS timing tests."""

import threading
import unittest

from performance import PerformanceCue
from voice_lab.tts import measure_cancellation


class _CancellableEngine:
    def __init__(self) -> None:
        self.started = threading.Event()

    def synthesize_cancellable(self, text, performance, should_cancel):
        self.started.set()
        while not should_cancel():
            pass
        raise RuntimeError("cancelled")


class _CompletesImmediately:
    def synthesize_cancellable(self, text, performance, should_cancel):
        return object()


class VoiceLabTtsTests(unittest.TestCase):
    def test_measure_cancellation_reports_release_after_request(self) -> None:
        times = iter([10.0, 10.025])
        result = measure_cancellation(
            _CancellableEngine(),
            "long sentence",
            PerformanceCue(),
            cancel_after_ms=50.0,
            now=lambda: next(times),
            sleep=lambda seconds: None,
        )
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["release_ms_after_cancel"], 25.0)
        self.assertEqual(result["cancel_after_ms"], 50.0)

    def test_measure_cancellation_marks_unsupported_engine(self) -> None:
        result = measure_cancellation(
            object(),
            "text",
            PerformanceCue(),
            cancel_after_ms=100.0,
        )
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["release_ms_after_cancel"])

    def test_measure_cancellation_detects_completion_before_cancel(self) -> None:
        result = measure_cancellation(
            _CompletesImmediately(),
            "text",
            PerformanceCue(),
            cancel_after_ms=100.0,
            sleep=lambda seconds: None,
        )
        self.assertEqual(result["status"], "completed_before_cancel")


if __name__ == "__main__":
    unittest.main()
