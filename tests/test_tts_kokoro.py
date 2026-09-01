"""Regression tests for the lazy Kokoro TTS adapter."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from performance import PerformanceCue


class Result:
    def __init__(self, audio: object) -> None:
        self.audio = audio


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return np.array([self._values], dtype=np.float64)


class FakePipeline:
    builds = 0
    build_args: list[tuple[str, str]] = []
    calls: list[tuple[str, str]] = []
    outputs: list[Result] = []

    def __init__(self, lang_code: str, device: str) -> None:
        type(self).builds += 1
        type(self).build_args.append((lang_code, device))

    def __call__(self, text: str, *, voice: str) -> list[Result]:
        type(self).calls.append((text, voice))
        return list(type(self).outputs)


class KokoroEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        FakePipeline.builds = 0
        FakePipeline.build_args.clear()
        FakePipeline.calls.clear()
        FakePipeline.outputs = [
            Result(np.array([0.1, 0.2], dtype=np.float32)),
            Result(np.array([0.3], dtype=np.float32)),
        ]

    def test_constructor_does_not_build_pipeline(self) -> None:
        from output.tts.kokoro import KokoroEngine

        engine = KokoroEngine({"voice": {"name": "af_heart"}})

        self.assertIsNone(engine._pipeline)
        self.assertEqual(FakePipeline.builds, 0)

    def test_first_synthesis_builds_cpu_pipeline_once_and_reuses_it(self) -> None:
        fake_module = types.ModuleType("kokoro")
        fake_module.KPipeline = FakePipeline
        with patch.dict(sys.modules, {"kokoro": fake_module}):
            from output.tts.kokoro import KokoroEngine

            engine = KokoroEngine({"voice": {"name": "af_heart"}})
            first = engine.synthesize("first", PerformanceCue())
            second = engine.synthesize("second", PerformanceCue("playful", 0.65))

        self.assertEqual(FakePipeline.builds, 1)
        self.assertEqual(FakePipeline.build_args, [("a", "cpu")])
        self.assertEqual(
            FakePipeline.calls,
            [("first", "af_heart"), ("second", "af_heart")],
        )
        np.testing.assert_allclose(
            first.audio,
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
        self.assertEqual(first.audio.dtype, np.float32)
        self.assertEqual(first.sample_rate, 24_000)
        self.assertEqual(second.sample_rate, 24_000)

    def test_tensor_like_audio_is_flattened_and_converted_to_float32(self) -> None:
        FakePipeline.outputs = [Result(FakeTensor([0.25, -0.5]))]
        fake_module = types.ModuleType("kokoro")
        fake_module.KPipeline = FakePipeline
        with patch.dict(sys.modules, {"kokoro": fake_module}):
            from output.tts.kokoro import KokoroEngine

            result = KokoroEngine({"voice": {}}).synthesize(
                "tensor",
                PerformanceCue(),
            )

        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.audio.dtype, np.float32)
        np.testing.assert_allclose(
            result.audio,
            np.array([0.25, -0.5], dtype=np.float32),
        )

    def test_empty_model_output_returns_empty_float32_audio(self) -> None:
        FakePipeline.outputs = []
        fake_module = types.ModuleType("kokoro")
        fake_module.KPipeline = FakePipeline
        with patch.dict(sys.modules, {"kokoro": fake_module}):
            from output.tts.kokoro import KokoroEngine

            result = KokoroEngine({"voice": {}}).synthesize(
                "empty",
                PerformanceCue(),
            )

        self.assertEqual(result.audio.shape, (0,))
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertEqual(result.sample_rate, 24_000)


if __name__ == "__main__":
    unittest.main()
