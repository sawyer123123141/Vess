"""Tests for explicit TTS engine selection."""

import unittest

from output.tts.factory import create_tts_engine


class TtsFactoryTests(unittest.TestCase):
    def test_default_engine_is_kokoro(self) -> None:
        engine = create_tts_engine({"voice": {}})
        self.assertEqual(type(engine).__name__, "KokoroEngine")
        self.assertIsNone(engine._pipeline)

    def test_explicit_kokoro_engine_is_kokoro(self) -> None:
        engine = create_tts_engine({"voice": {"engine": "kokoro"}})
        self.assertEqual(type(engine).__name__, "KokoroEngine")
        self.assertIsNone(engine._pipeline)

    def test_chatterbox_selection_builds_lazy_adapter(self) -> None:
        engine = create_tts_engine({"voice": {"engine": "chatterbox_turbo"}})
        self.assertEqual(type(engine).__name__, "ChatterboxTurboEngine")
        self.assertIsNone(engine._model)

    def test_unknown_engine_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown TTS engine"):
            create_tts_engine({"voice": {"engine": "made_up"}})


if __name__ == "__main__":
    unittest.main()
