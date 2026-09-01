"""Serialized speech-output behavior."""

import time
import unittest

import numpy as np

from output.voice import VoiceOutput
from state import State


CONFIG = {
    "voice": {
        "name": "af_heart",
        "sample_rate": 24_000,
    }
}


class VoiceOutputTests(unittest.TestCase):
    def test_voice_plays_in_order_and_clears_speaking(self) -> None:
        played: list[int] = []
        state = State()
        voice = VoiceOutput(
            CONFIG,
            state,
            RecordingLog(),
            synthesize=lambda text: np.array([len(text)], dtype=np.float32),
            play=lambda audio, _: played.append(int(audio[0])),
        )

        started = time.time()
        voice.start()
        voice.enqueue("one")
        voice.enqueue("four")
        voice.close()

        self.assertEqual(played, [3, 4])
        self.assertFalse(state.speaking)
        self.assertGreaterEqual(state.last_spoke, started)
        self.assertEqual(
            [event["event"] for event in state.debug_snapshot()["events"]],
            [
                "tts_started",
                "tts_playback_started",
                "tts_complete",
                "tts_started",
                "tts_playback_started",
                "tts_complete",
            ],
        )


class RecordingLog:
    def append(self, event_type: str, payload: dict[str, object]) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
