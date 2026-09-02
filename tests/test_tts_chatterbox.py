"""Structural tests for the optional lazy Chatterbox Turbo adapter."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from performance import PerformanceCue


class FakeTensor:
    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return np.array([[0.1, -0.2]], dtype=np.float64)


class FakeModel:
    sr = 24_000

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.conditioning_calls: list[str] = []

    def prepare_conditionals(self, audio_prompt_path: str) -> None:
        self.conditioning_calls.append(audio_prompt_path)

    def generate(self, text: str, **kwargs: object) -> FakeTensor:
        self.calls.append((text, kwargs))
        return FakeTensor()


class FakeTurboTTS:
    loads: list[str] = []
    model = FakeModel()

    @classmethod
    def reset(cls) -> None:
        cls.loads.clear()
        cls.model = FakeModel()

    @classmethod
    def from_pretrained(cls, *, device: str) -> FakeModel:
        cls.loads.append(device)
        return cls.model


def fake_chatterbox_modules() -> dict[str, types.ModuleType]:
    package = types.ModuleType("chatterbox")
    package.__path__ = []
    turbo = types.ModuleType("chatterbox.tts_turbo")
    turbo.ChatterboxTurboTTS = FakeTurboTTS
    return {"chatterbox": package, "chatterbox.tts_turbo": turbo}


class ChatterboxTurboEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeTurboTTS.reset()

    def test_constructor_does_not_load_model(self) -> None:
        from output.tts.chatterbox_turbo import ChatterboxTurboEngine

        engine = ChatterboxTurboEngine({"voice": {"chatterbox": {"device": "cuda"}}})

        self.assertIsNone(engine._model)
        self.assertEqual(FakeTurboTTS.loads, [])

    def test_first_synthesis_loads_once_and_reuses_model(self) -> None:
        with patch.dict(sys.modules, fake_chatterbox_modules()):
            from output.tts.chatterbox_turbo import ChatterboxTurboEngine

            engine = ChatterboxTurboEngine(
                {"voice": {"chatterbox": {"device": "cuda"}}}
            )
            first = engine.synthesize("first", PerformanceCue())
            second = engine.synthesize("second", PerformanceCue("playful", 0.65))

        self.assertEqual(FakeTurboTTS.loads, ["cuda"])
        self.assertEqual(
            FakeTurboTTS.model.calls,
            [("first", {}), ("second [chuckle]", {})],
        )
        self.assertEqual(first.sample_rate, 24_000)
        self.assertEqual(second.sample_rate, 24_000)

    def test_reference_audio_is_conditioned_once_and_reused(self) -> None:
        with patch.dict(sys.modules, fake_chatterbox_modules()):
            from output.tts.chatterbox_turbo import ChatterboxTurboEngine

            engine = ChatterboxTurboEngine(
                {
                    "voice": {
                        "chatterbox": {
                            "device": "cuda",
                            "reference_audio": "voice.wav",
                        }
                    }
                }
            )
            engine.synthesize("hello", PerformanceCue())
            engine.synthesize("again", PerformanceCue("curious", 0.4))

        self.assertEqual(FakeTurboTTS.model.conditioning_calls, ["voice.wav"])
        self.assertEqual(
            FakeTurboTTS.model.calls,
            [("hello", {}), ("again", {})],
        )

    def test_output_is_flat_float32_and_uses_model_sample_rate(self) -> None:
        with patch.dict(sys.modules, fake_chatterbox_modules()):
            from output.tts.chatterbox_turbo import ChatterboxTurboEngine

            result = ChatterboxTurboEngine({"voice": {}}).synthesize(
                "hello",
                PerformanceCue(),
            )

        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.audio.dtype, np.float32)
        np.testing.assert_allclose(
            result.audio,
            np.array([0.1, -0.2], dtype=np.float32),
        )
        self.assertEqual(result.sample_rate, 24_000)

    def test_nonapproved_performance_does_not_rewrite_text(self) -> None:
        with patch.dict(sys.modules, fake_chatterbox_modules()):
            from output.tts.chatterbox_turbo import ChatterboxTurboEngine

            engine = ChatterboxTurboEngine({"voice": {}})
            engine.synthesize("Do not rewrite me.", PerformanceCue("emphatic", 0.7))

        self.assertEqual(FakeTurboTTS.model.calls, [("Do not rewrite me.", {})])

    def test_missing_dependency_raises_clear_engine_error(self) -> None:
        from output.tts.chatterbox_turbo import ChatterboxTurboEngine

        engine = ChatterboxTurboEngine({"voice": {"chatterbox": {"device": "cuda"}}})
        with patch.dict(
            sys.modules,
            {"chatterbox": None, "chatterbox.tts_turbo": None},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "chatterbox_turbo.*chatterbox-tts",
            ):
                engine.synthesize("hello", PerformanceCue())


if __name__ == "__main__":
    unittest.main()
