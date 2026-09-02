"""Regression tests for cancelling obsolete Chatterbox synthesis work."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from output.tts.chatterbox_turbo import ChatterboxTurboEngine
from output.tts.base import SynthesisResult
from output.voice import VoiceOutput
from performance import PerformanceCue
from state import State


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class CooperativeFakeEngine:
    """Expose both the old blocking path and the desired cancellable path."""

    def __init__(self) -> None:
        self.old_started = threading.Event()
        self.new_started = threading.Event()
        self.release_old = threading.Event()

    def synthesize(
        self,
        text: str,
        performance: PerformanceCue,
    ) -> SynthesisResult:
        if text == "old":
            self.old_started.set()
            self.release_old.wait(timeout=0.5)
        elif text == "new":
            self.new_started.set()
        return SynthesisResult(np.ones(10, dtype=np.float32), 24_000)

    def synthesize_cancellable(
        self,
        text: str,
        performance: PerformanceCue,
        should_cancel,
    ) -> SynthesisResult:
        if text == "old":
            self.old_started.set()
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                if should_cancel():
                    from output.tts import base as tts_base

                    cancelled = getattr(tts_base, "SynthesisCancelled", RuntimeError)
                    raise cancelled("synthesis cancelled")
                time.sleep(0.002)
        elif text == "new":
            self.new_started.set()
        return SynthesisResult(np.ones(10, dtype=np.float32), 24_000)


class FakeHookHandle:
    def __init__(self, transformer) -> None:
        self._transformer = transformer

    def remove(self) -> None:
        self._transformer.hook = None


class FakeTransformer:
    def __init__(self) -> None:
        self.hook = None
        self.steps = 0

    def register_forward_pre_hook(self, hook):
        self.hook = hook
        return FakeHookHandle(self)

    def step(self) -> None:
        self.steps += 1
        if self.hook is not None:
            self.hook(self, ())


class FakeChatterboxModel:
    def __init__(self) -> None:
        self.sr = 24_000
        self.t3 = SimpleNamespace(tfmr=FakeTransformer())

    def generate(self, text: str):
        for _ in range(10):
            self.t3.tfmr.step()
            time.sleep(0.001)
        return np.ones(10, dtype=np.float32)


class StaleTTSPreemptionTests(unittest.TestCase):
    def test_new_generation_does_not_wait_for_obsolete_cancellable_synthesis(self) -> None:
        state = State()
        log = RecordingLog()
        engine = CooperativeFakeEngine()
        voice = VoiceOutput(
            {"voice": {"sample_rate": 24_000}},
            state,
            log,
            engine=engine,
            play=lambda audio, sample_rate: None,
        )
        voice.start()
        try:
            voice.begin_generation(1)
            voice.enqueue("old", generation_id=1)
            self.assertTrue(engine.old_started.wait(timeout=0.2))

            voice.begin_generation(2)
            voice.enqueue("new", generation_id=2)

            self.assertTrue(
                engine.new_started.wait(timeout=0.12),
                "new synthesis stayed blocked behind obsolete work",
            )
        finally:
            engine.release_old.set()
            voice.close()

        events = state.debug_snapshot()["events"]
        self.assertTrue(
            any(
                event["event"] == "stale_tts_skipped"
                and event.get("generation_id") == 1
                and event.get("stage") == "during_synthesis"
                for event in events
            )
        )
        self.assertFalse(
            any(event["event"] == "tts_error" for event in events),
            "cooperative cancellation must not be reported as a voice error",
        )

    def test_chatterbox_cancellable_path_aborts_inside_token_generation(self) -> None:
        engine = ChatterboxTurboEngine({})
        model = FakeChatterboxModel()
        engine._model = model
        checks = 0

        def should_cancel() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 3

        self.assertTrue(
            hasattr(engine, "synthesize_cancellable"),
            "Chatterbox needs a cooperative synthesis entry point",
        )

        with self.assertRaisesRegex(RuntimeError, "cancel"):
            engine.synthesize_cancellable(
                "obsolete text",
                PerformanceCue(),
                should_cancel,
            )

        self.assertLess(model.t3.tfmr.steps, 10)
        self.assertIsNone(model.t3.tfmr.hook)


if __name__ == "__main__":
    unittest.main()
