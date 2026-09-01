"""Serialized speech-output behavior."""

import time
import unittest

import numpy as np

from output.tts.base import SynthesisResult
from output.voice import VoiceOutput
from performance import PerformanceCue
from state import State
from tests.tts_fakes import FakeTTSEngine


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
            [
                event["event"]
                for event in state.debug_snapshot()["events"]
                if event["event"].startswith("tts_")
            ],
            [
                "tts_started",
                "tts_playback_started",
                "tts_complete",
                "tts_started",
                "tts_playback_started",
                "tts_complete",
            ],
        )

    def test_engine_sample_rate_and_performance_reach_voice_pipeline(self) -> None:
        played_rates: list[int] = []
        cue = PerformanceCue("playful", 0.65)
        engine = FakeTTSEngine(sample_rate=16_000)
        voice = VoiceOutput(
            CONFIG,
            State(),
            RecordingLog(),
            engine=engine,
            play=lambda audio, sample_rate: played_rates.append(sample_rate),
        )

        voice.start()
        voice.enqueue("hello", performance=cue)
        voice.close()

        self.assertEqual(played_rates, [16_000])
        self.assertEqual(engine.calls, [("hello", cue)])

    def test_cached_acknowledgement_preserves_engine_sample_rate(self) -> None:
        played_rates: list[int] = []
        engine = FakeTTSEngine(
            lambda text, performance: SynthesisResult(
                np.ones(20, dtype=np.float32),
                22_050,
            )
        )
        voice = VoiceOutput(
            CONFIG,
            State(),
            RecordingLog(),
            engine=engine,
            play=lambda audio, sample_rate: played_rates.append(sample_rate),
        )

        voice.start()
        voice.prepare_acknowledgement("Ready?")
        voice.enqueue_acknowledgement()
        voice.close()

        self.assertEqual(played_rates, [22_050])
        self.assertEqual(engine.calls, [("Ready?", PerformanceCue())])


class RecordingLog:
    def append(self, event_type: str, payload: dict[str, object]) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
