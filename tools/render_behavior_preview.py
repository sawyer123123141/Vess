"""Run deterministic headless Vess behavior simulations and capture native traces."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output.animator import FaceAnimator
from performance import PerformanceCue, load_performance_definitions
from state import State
from tools.behavior_scenarios import (
    ScenarioPhase,
    apply_phase,
    get_scenario,
    phase_frame_count,
)


@dataclass(frozen=True)
class VerificationFailure:
    scenario: str
    phase: str
    frame: int
    time_seconds: float
    invariant: str
    observed: dict[str, object]


@dataclass
class SimulationResult:
    scenario: str
    fps: int
    seed: int
    frames: list[np.ndarray] = field(default_factory=list)
    frame_hashes: list[str] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)
    failures: list[VerificationFailure] = field(default_factory=list)


def simulate_scenario(
    name: str,
    *,
    fps: int = 30,
    seed: int = 1,
) -> SimulationResult:
    """Run one scripted scenario using the production animator without sleeping."""
    if fps <= 0:
        raise ValueError("fps must be positive")

    moods = _load_json("moods.json")
    performance_definitions = load_performance_definitions(_load_json("performance.json"))
    cues = {
        expression: PerformanceCue(expression, float(entry["intensity"]))
        for expression, entry in performance_definitions.items()
    }
    scenario = get_scenario(
        name,
        moods=moods.keys(),
        performances=cues,
    )

    state = State(mood_until=0.0)
    animator = FaceAnimator(moods, performance_definitions, seed=seed)
    result = SimulationResult(scenario=name, fps=fps, seed=seed)

    frame_index = 0
    dt = 1.0 / fps
    for phase in scenario.phases:
        apply_phase(state, phase)
        for _ in range(phase_frame_count(phase, fps)):
            frame = animator.tick(state, dt)
            snapshot = animator.debug_snapshot()
            native = frame.copy()
            result.frames.append(native)
            result.frame_hashes.append(hashlib.sha256(native.tobytes()).hexdigest())
            result.trace.append(
                _trace_record(
                    frame_index,
                    frame_index / fps,
                    phase,
                    state,
                    snapshot,
                )
            )
            frame_index += 1

    return result


def _load_json(name: str) -> dict[str, object]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _trace_record(
    frame_index: int,
    time_seconds: float,
    phase: ScenarioPhase,
    state: State,
    snapshot: dict[str, object],
) -> dict[str, object]:
    with state.locked():
        mood = state.mood
        performance = state.performance
        listening = state.listening
        thinking = state.thinking
        speaking = state.speaking
        person_present = state.person_present
        person_pos = state.person_pos

    shape = dict(snapshot["shape"])
    color = tuple(snapshot["color"])
    gaze = tuple(snapshot["render_gaze"])
    offset = tuple(snapshot["render_offset"])
    fixation = tuple(snapshot["fixation"])

    person_x = float(person_pos[0]) if person_pos is not None else None
    person_y = float(person_pos[1]) if person_pos is not None else None

    return {
        "frame": frame_index,
        "time_seconds": round(time_seconds, 6),
        "phase": phase.name,
        "interaction_mode": str(snapshot["interaction_mode"]),
        "mood": mood,
        "performance_expression": performance.expression,
        "performance_intensity": float(performance.intensity),
        "listening": listening,
        "thinking": thinking,
        "speaking": speaking,
        "person_present": person_present,
        "person_x": person_x,
        "person_y": person_y,
        "gaze_x": float(gaze[0]),
        "gaze_y": float(gaze[1]),
        "face_offset_x": float(offset[0]),
        "face_offset_y": float(offset[1]),
        "blink_openness": float(snapshot["blink_openness"]),
        "left_eye_x": float(shape["l_cx"]),
        "left_eye_y": float(shape["l_cy"]),
        "right_eye_x": float(shape["r_cx"]),
        "right_eye_y": float(shape["r_cy"]),
        "left_eye_offset_x": 0.0,
        "left_eye_offset_y": 0.0,
        "right_eye_offset_x": 0.0,
        "right_eye_offset_y": 0.0,
        "left_eye_width": float(shape["l_w"]),
        "left_eye_height": float(shape["l_h"]),
        "right_eye_width": float(shape["r_w"]),
        "right_eye_height": float(shape["r_h"]),
        "left_eye_slant": float(shape["l_slant"]),
        "right_eye_slant": float(shape["r_slant"]),
        "color_r": float(color[0]),
        "color_g": float(color[1]),
        "color_b": float(color[2]),
        "fixation_x": float(fixation[0]),
        "fixation_y": float(fixation[1]),
        "speaking_break_active": bool(snapshot["speaking_break_active"]),
        "speaking_break_remaining": float(snapshot["speaking_break_remaining"]),
        "performance_current": {
            key: float(value)
            for key, value in dict(snapshot["performance_current"]).items()
        },
        "performance_target": {
            key: float(value)
            for key, value in dict(snapshot["performance_target"]).items()
        },
    }


if __name__ == "__main__":
    simulate_scenario("conversational_cycle")
