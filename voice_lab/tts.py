"""Real cancellable-TTS timing helpers for Voice Lab."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time

from performance import PerformanceCue


def measure_cancellation(
    engine: object,
    text: str,
    performance: PerformanceCue,
    cancel_after_ms: float,
    now: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Measure how long a cancellable synthesis keeps the worker after invalidation."""
    cancellable = getattr(engine, "synthesize_cancellable", None)
    if not callable(cancellable):
        return {
            "cancel_after_ms": float(cancel_after_ms),
            "release_ms_after_cancel": None,
            "status": "unsupported",
            "error": None,
        }
    if cancel_after_ms < 0.0:
        raise ValueError("cancel_after_ms must be non-negative")

    cancel = threading.Event()
    done = threading.Event()
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = cancellable(text, performance, cancel.is_set)
        except Exception as error:
            outcome["error"] = str(error)
        finally:
            done.set()

    worker = threading.Thread(target=run, name="voice-lab-cancel", daemon=True)
    worker.start()
    sleep(cancel_after_ms / 1000.0)
    # Give a synthesis that already returned a scheduling instant to publish `done`.
    done.wait(0.001)
    if done.is_set():
        worker.join()
        return {
            "cancel_after_ms": float(cancel_after_ms),
            "release_ms_after_cancel": None,
            "status": "completed_before_cancel" if "error" not in outcome else "failed_before_cancel",
            "error": outcome.get("error"),
        }

    cancel_requested = now()
    cancel.set()
    worker.join(timeout=30.0)
    released = now()
    if worker.is_alive():
        return {
            "cancel_after_ms": float(cancel_after_ms),
            "release_ms_after_cancel": round((released - cancel_requested) * 1000.0, 3),
            "status": "timeout",
            "error": "cancellable synthesis did not exit within 30 seconds",
        }

    return {
        "cancel_after_ms": float(cancel_after_ms),
        "release_ms_after_cancel": round(max(released - cancel_requested, 0.0) * 1000.0, 3),
        "status": "cancelled",
        "error": outcome.get("error"),
    }
