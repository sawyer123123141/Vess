"""Voice Lab corpus tests."""

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from voice_lab.corpus import load_manifest, read_wav


class VoiceLabCorpusTests(unittest.TestCase):
    def test_manifest_resolves_audio_and_defaults_performance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "pause.wav"
            self._write_wav(audio, np.array([0, 16384, -16384], dtype=np.int16))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"items": [{"id": "pause", "audio": "pause.wav", "transcript": "hello", "source": "owner", "tags": ["hesitation"]}]}), encoding="utf-8")

            [item] = load_manifest(manifest)

            self.assertEqual(item.audio_path, audio)
            self.assertEqual(item.expected_utterances, 1)
            self.assertEqual(item.expression, "neutral")
            self.assertEqual(item.intensity, 0.0)
            self.assertEqual(item.tags, ("hesitation",))

    def test_manifest_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"items": [
                {"id": "same", "audio": "a.wav", "transcript": "a"},
                {"id": "same", "audio": "b.wav", "transcript": "b"},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate corpus id"):
                load_manifest(manifest)

    def test_read_wav_decodes_mono_16khz_pcm16_float32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            raw = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
            self._write_wav(path, raw)
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"items": [{"id": "x", "audio": "sample.wav", "transcript": "x"}]}), encoding="utf-8")
            [item] = load_manifest(manifest)

            audio = read_wav(item)

            self.assertEqual(audio.dtype, np.float32)
            np.testing.assert_allclose(audio, raw.astype(np.float32) / 32768.0)

    def test_read_wav_rejects_wrong_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            self._write_wav(path, np.array([0, 1], dtype=np.int16), sample_rate=8000)
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"items": [{"id": "x", "audio": "sample.wav", "transcript": "x"}]}), encoding="utf-8")
            [item] = load_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "16000 Hz"):
                read_wav(item)

    @staticmethod
    def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(samples.astype("<i2").tobytes())


if __name__ == "__main__":
    unittest.main()
