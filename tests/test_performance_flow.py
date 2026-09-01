"""Integration-style performance flow from LLM markup into physical playback state."""

import unittest

import numpy as np

from brain.llm import split_clauses
from output.tts.base import SynthesisResult
from output.voice import VoiceOutput
from performance import PerformanceCue, load_performance_definitions
from state import State
from tests.tts_fakes import FakeTTSEngine


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((event_type, payload))


class PerformanceFlowTests(unittest.TestCase):
    def test_clean_tagged_clause_activates_only_while_audio_plays(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {"intensity": 0.0},
            "playful": {"intensity": 0.65},
        })
        clause = next(
            split_clauses(["[[vess:playful]] That's actually pretty neat!"], definitions)
        )
        self.assertNotIn("[[vess:", clause.text)
        self.assertEqual(clause.performance.expression, "playful")

        state = State()

        def play(audio: np.ndarray, sample_rate: int) -> None:
            self.assertEqual(state.performance.expression, "playful")

        engine = FakeTTSEngine(
            lambda text, performance: SynthesisResult(
                np.ones(20, dtype=np.float32),
                1_000,
            )
        )
        voice = VoiceOutput(
            {"voice": {"sample_rate": 1_000}},
            state,
            RecordingLog(),
            engine=engine,
            play=play,
        )
        voice.start()
        voice.begin_generation(1)
        voice.enqueue(
            clause.text,
            generation_id=1,
            performance=clause.performance,
        )
        voice.close()

        self.assertEqual(engine.calls, [(clause.text, clause.performance)])
        self.assertEqual(state.performance, PerformanceCue())
        snapshot = state.debug_snapshot()
        self.assertEqual(snapshot["values"]["performance_expression"], "neutral")
        self.assertEqual(snapshot["values"]["performance_intensity"], 0.0)
        started = next(
            event for event in snapshot["events"] if event["event"] == "performance_started"
        )
        ended = next(
            event for event in snapshot["events"] if event["event"] == "performance_ended"
        )
        self.assertEqual(started["text"], "That's actually pretty neat!")
        self.assertEqual(started["expression"], "playful")
        self.assertEqual(started["generation_id"], 1)
        self.assertEqual(ended["expression"], "playful")
        self.assertEqual(ended["generation_id"], 1)


if __name__ == "__main__":
    unittest.main()
