"""Capture preprocessing contracts for barge-in."""

import unittest

import numpy as np

from perception.audio_preprocess import (
    CapturedAudioBlock,
    PassthroughCapturePreprocessor,
    RenderedAudioBlock,
)


class AudioPreprocessTests(unittest.TestCase):
    def test_passthrough_returns_float32_copy(self) -> None:
        source = np.array([0.1, -0.2], dtype=np.float64)

        result = PassthroughCapturePreprocessor().process_capture(
            CapturedAudioBlock(source, 1.0, 2.0)
        )

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [0.1, -0.2])
        self.assertIsNot(result, source)

    def test_render_reference_does_not_modify_capture(self) -> None:
        preprocessor = PassthroughCapturePreprocessor()
        preprocessor.push_render_reference(
            RenderedAudioBlock(
                np.array([0.8], dtype=np.float32),
                24_000,
                None,
            )
        )

        result = preprocessor.process_capture(
            CapturedAudioBlock(
                np.array([0.2], dtype=np.float32),
                None,
                1.0,
            )
        )

        np.testing.assert_allclose(result, [0.2])


if __name__ == "__main__":
    unittest.main()
