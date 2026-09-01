"""Run deterministic headless Vess behavior simulations and capture native traces."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output.animator import FaceAnimator, _SPEAK_BREAK_LENGTH
from performance import PerformanceCue, load_performance_definitions
from state import State
from tools.behavior_scenarios import (
    ScenarioPhase,
    apply_phase,
    get_scenario,
    phase_frame_count,
)

_PANEL_MIN = 1.0
_PANEL_MAX = 63.0
_TRANSITION_SECONDS = 0.25
_SHAPE_DELTA_KEYS = (
    "l_h",
    "r_h",
    "l_slant",
    "r_slant",
    "l_cy",
    "r_cy",
)
_UNIT_SCALE_KEYS = (
    "hold_scale",
    "ease_scale",
    "track_bias_scale",
    "speaking_break_scale",
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


def check_invariants(result: SimulationResult) -> list[VerificationFailure]:
    """Return every hard behavioral/safety violation found in one simulation."""
    failures: list[VerificationFailure] = []
    stable_rows = _stable_rows_by_phase(result)

    for index, row in enumerate(result.trace):
        frame = result.frames[index] if index < len(result.frames) else None
        failures.extend(_global_failures(result, row, frame))

    for rows in stable_rows.values():
        if not rows:
            continue
        for row in rows:
            expected = _expected_mode(row)
            if row["interaction_mode"] != expected:
                failures.append(
                    _failure(
                        result,
                        row,
                        "interaction priority",
                        {"expected": expected, "actual": row["interaction_mode"]},
                    )
                )

        mode = str(rows[0]["interaction_mode"])
        if mode == "listening":
            failures.extend(_listening_failures(result, rows))
        elif mode == "thinking":
            failures.extend(_thinking_failures(result, rows))

    failures.extend(_speaking_failures(result, stable_rows))
    return failures


def verify_determinism(
    name: str,
    *,
    fps: int = 30,
    seed: int = 1,
) -> list[VerificationFailure]:
    """Run one scenario twice and require exact trace/hash equality."""
    first = simulate_scenario(name, fps=fps, seed=seed)
    second = simulate_scenario(name, fps=fps, seed=seed)
    if first.trace == second.trace and first.frame_hashes == second.frame_hashes:
        return []

    length = min(len(first.trace), len(second.trace))
    mismatch = next(
        (
            index
            for index in range(length)
            if first.trace[index] != second.trace[index]
            or first.frame_hashes[index] != second.frame_hashes[index]
        ),
        length,
    )
    row = first.trace[mismatch] if mismatch < len(first.trace) else {
        "phase": "<length>",
        "frame": mismatch,
        "time_seconds": mismatch / fps,
    }
    return [
        VerificationFailure(
            scenario=name,
            phase=str(row["phase"]),
            frame=int(row["frame"]),
            time_seconds=float(row["time_seconds"]),
            invariant="determinism",
            observed={
                "first_trace_length": len(first.trace),
                "second_trace_length": len(second.trace),
                "first_hash_length": len(first.frame_hashes),
                "second_hash_length": len(second.frame_hashes),
            },
        )
    ]


def calculate_metrics(result: SimulationResult) -> dict[str, object]:
    """Calculate review-only measurements directly from the trace."""
    speaking_rows = [row for row in result.trace if row["speaking"]]
    directed_rows = [
        row
        for row in speaking_rows
        if row["person_present"]
        and row["person_x"] is not None
        and not row["speaking_break_active"]
    ]
    directed_count = sum(1 for row in directed_rows if _person_dot(row) > 0.0)
    directed_percent = (
        100.0 * directed_count / len(directed_rows)
        if directed_rows
        else None
    )

    break_lengths = _speaking_break_lengths(result.trace, result.fps)
    peak_face_offset = max(
        (
            math.hypot(float(row["face_offset_x"]), float(row["face_offset_y"]))
            for row in result.trace
        ),
        default=0.0,
    )
    max_gaze_delta = _max_pair_delta(result.trace, "gaze_x", "gaze_y")
    max_face_delta = _max_pair_delta(
        result.trace,
        "face_offset_x",
        "face_offset_y",
    )

    return {
        "total_frames": len(result.trace),
        "speaking_frames": len(speaking_rows),
        "person_directed_percent": directed_percent,
        "speaking_break_count": len(break_lengths),
        "average_break_seconds": (
            sum(break_lengths) / len(break_lengths) if break_lengths else None
        ),
        "max_break_seconds": max(break_lengths) if break_lengths else None,
        "peak_face_offset": peak_face_offset,
        "max_frame_gaze_delta": max_gaze_delta,
        "max_frame_face_delta": max_face_delta,
        "performance_eye_deltas": _performance_eye_deltas(result),
    }


def build_summary(
    results: Iterable[SimulationResult],
    *,
    deterministic: dict[str, bool],
) -> str:
    """Build a compact, measured GitHub/mobile-friendly verification summary."""
    items = list(results)
    all_failures = [failure for result in items for failure in result.failures]
    passed = sum(
        1
        for result in items
        if not result.failures and deterministic.get(result.scenario, False)
    )
    invalid_invariants = {
        "numeric finiteness",
        "frame format",
        "gaze bounds",
        "performance intensity",
        "eye dimensions",
        "geometry bounds",
    }
    invalid_frames = {
        (failure.scenario, failure.frame)
        for failure in all_failures
        if failure.invariant in invalid_invariants
    }
    geometry_ok = not any(
        failure.invariant == "geometry bounds" for failure in all_failures
    )
    determinism_ok = all(deterministic.get(result.scenario, False) for result in items)

    lines = [
        "Vess behavior verification",
        "",
        f"Scenarios: {passed}/{len(items)} PASS",
        f"Invalid frames: {len(invalid_frames)}",
        f"Geometry: {'PASS' if geometry_ok else 'FAIL'}",
        f"Determinism: {'PASS' if determinism_ok else 'FAIL'}",
    ]

    conversational = next(
        (result for result in items if result.scenario == "conversational_cycle"),
        None,
    )
    if conversational is not None:
        metrics = calculate_metrics(conversational)
        lines.extend(
            [
                "",
                "Conversational cycle",
                f"  speaking frames: {metrics['speaking_frames']}",
                "  speaking person-directed frames: "
                + _format_percent(metrics["person_directed_percent"]),
                f"  speaking gaze breaks: {metrics['speaking_break_count']}",
                "  average break duration: "
                + _format_seconds(metrics["average_break_seconds"]),
                "  maximum break duration: "
                + _format_seconds(metrics["max_break_seconds"]),
                f"  peak face offset: {float(metrics['peak_face_offset']):.3f} px",
                f"  max frame gaze delta: {float(metrics['max_frame_gaze_delta']):.3f}",
                f"  max frame face delta: {float(metrics['max_frame_face_delta']):.3f} px",
            ]
        )
        deltas = dict(metrics["performance_eye_deltas"])
        if deltas:
            lines.extend(["", "Performance"])
            for expression, values in deltas.items():
                lines.append(
                    f"  {expression} max L/R eye delta: "
                    f"{float(values['left']):.3f} / {float(values['right']):.3f} px"
                )

    for failure in all_failures:
        lines.extend(
            [
                "",
                f"FAIL {failure.scenario} frame {failure.frame} ({failure.time_seconds:.3f}s)",
                f"Invariant: {failure.invariant}",
                "Observed: " + json.dumps(failure.observed, sort_keys=True),
            ]
        )

    return "\n".join(lines) + "\n"


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


def _global_failures(
    result: SimulationResult,
    row: dict[str, object],
    frame: np.ndarray | None,
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []

    nonfinite = [
        path
        for path, value in _numeric_values(row)
        if not math.isfinite(value)
    ]
    if nonfinite:
        failures.append(
            _failure(result, row, "numeric finiteness", {"fields": nonfinite})
        )

    if frame is None or frame.shape != (64, 64, 3) or frame.dtype != np.uint8:
        failures.append(
            _failure(
                result,
                row,
                "frame format",
                {
                    "shape": None if frame is None else list(frame.shape),
                    "dtype": None if frame is None else str(frame.dtype),
                },
            )
        )

    gaze_x = float(row["gaze_x"])
    gaze_y = float(row["gaze_y"])
    if not (-1.0 <= gaze_x <= 1.0 and -1.0 <= gaze_y <= 1.0):
        failures.append(
            _failure(
                result,
                row,
                "gaze bounds",
                {"gaze_x": gaze_x, "gaze_y": gaze_y},
            )
        )

    intensity = float(row["performance_intensity"])
    if not 0.0 <= intensity <= 1.0:
        failures.append(
            _failure(
                result,
                row,
                "performance intensity",
                {"performance_intensity": intensity},
            )
        )

    for side in ("left", "right"):
        width = float(row[f"{side}_eye_width"])
        height = float(row[f"{side}_eye_height"])
        if width <= 0.0 or height < 2.0:
            failures.append(
                _failure(
                    result,
                    row,
                    "eye dimensions",
                    {"side": side, "width": width, "height": height},
                )
            )
        bounds = _eye_bounds(row, side)
        if (
            bounds[0] < _PANEL_MIN
            or bounds[1] > _PANEL_MAX
            or bounds[2] < _PANEL_MIN
            or bounds[3] > _PANEL_MAX
        ):
            failures.append(
                _failure(
                    result,
                    row,
                    "geometry bounds",
                    {
                        "side": side,
                        "left": bounds[0],
                        "right": bounds[1],
                        "top": bounds[2],
                        "bottom": bounds[3],
                    },
                )
            )

    if row["performance_expression"] == "neutral":
        target = dict(row["performance_target"])
        wrong = {
            key: target.get(key)
            for key in _SHAPE_DELTA_KEYS
            if float(target.get(key, 0.0)) != 0.0
        }
        wrong.update(
            {
                key: target.get(key)
                for key in _UNIT_SCALE_KEYS
                if float(target.get(key, 1.0)) != 1.0
            }
        )
        if float(target.get("gaze_y_bias", 0.0)) != 0.0:
            wrong["gaze_y_bias"] = target.get("gaze_y_bias")
        if wrong:
            failures.append(
                _failure(result, row, "neutral performance target", wrong)
            )

    return failures


def _eye_bounds(
    row: dict[str, object],
    side: str,
) -> tuple[float, float, float, float]:
    cx = (
        float(row[f"{side}_eye_x"])
        + float(row["face_offset_x"])
        + float(row[f"{side}_eye_offset_x"])
    )
    cy = (
        float(row[f"{side}_eye_y"])
        + float(row["face_offset_y"])
        + float(row[f"{side}_eye_offset_y"])
    )
    half_width = float(row[f"{side}_eye_width"]) / 2.0
    half_height = (
        float(row[f"{side}_eye_height"]) / 2.0
        + abs(float(row[f"{side}_eye_slant"]))
    )
    return (
        cx - half_width,
        cx + half_width,
        cy - half_height,
        cy + half_height,
    )


def _numeric_values(
    value: object,
    path: str = "",
) -> Iterable[tuple[str, float]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _numeric_values(nested, child)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        yield path, float(value)


def _stable_rows_by_phase(
    result: SimulationResult,
) -> dict[str, list[dict[str, object]]]:
    transition_frames = int(math.ceil(_TRANSITION_SECONDS * result.fps))
    groups: dict[str, list[dict[str, object]]] = {}
    for row in result.trace:
        groups.setdefault(str(row["phase"]), []).append(row)
    return {
        phase: rows[transition_frames:]
        for phase, rows in groups.items()
    }


def _expected_mode(row: dict[str, object]) -> str:
    if row["listening"]:
        return "listening"
    if row["thinking"]:
        return "thinking"
    if row["speaking"]:
        return "speaking"
    if row["person_present"] and row["person_x"] is not None:
        return "tracking"
    return "idle"


def _listening_failures(
    result: SimulationResult,
    rows: list[dict[str, object]],
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []
    first_fixation = (rows[0]["fixation_x"], rows[0]["fixation_y"])
    for row in rows:
        fixation = (row["fixation_x"], row["fixation_y"])
        if fixation != first_fixation:
            failures.append(
                _failure(
                    result,
                    row,
                    "listening fixation",
                    {"expected": first_fixation, "actual": fixation},
                )
            )
            break
        if row["person_present"] and _person_dot(row) <= 0.0:
            failures.append(
                _failure(
                    result,
                    row,
                    "listening person direction",
                    {
                        "gaze_x": row["gaze_x"],
                        "gaze_y": row["gaze_y"],
                        "person_x": row["person_x"],
                        "person_y": row["person_y"],
                    },
                )
            )
            break
        person_x = row["person_x"]
        if person_x is not None and float(person_x) != 0.5:
            direction = math.copysign(1.0, float(person_x) - 0.5)
            if float(row["face_offset_x"]) * direction <= 0.0:
                failures.append(
                    _failure(
                        result,
                        row,
                        "listening face lean",
                        {
                            "face_offset_x": row["face_offset_x"],
                            "person_x": person_x,
                        },
                    )
                )
                break
    return failures


def _thinking_failures(
    result: SimulationResult,
    rows: list[dict[str, object]],
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []
    first_fixation = (rows[0]["fixation_x"], rows[0]["fixation_y"])
    for row in rows:
        if float(row["gaze_y"]) >= 0.0:
            failures.append(
                _failure(
                    result,
                    row,
                    "thinking upward gaze",
                    {"gaze_y": row["gaze_y"]},
                )
            )
            break
        fixation = (row["fixation_x"], row["fixation_y"])
        if fixation != first_fixation:
            failures.append(
                _failure(
                    result,
                    row,
                    "thinking fixation",
                    {"expected": first_fixation, "actual": fixation},
                )
            )
            break
    return failures


def _speaking_failures(
    result: SimulationResult,
    stable_rows: dict[str, list[dict[str, object]]],
) -> list[VerificationFailure]:
    failures: list[VerificationFailure] = []
    rows = [
        row
        for phase_rows in stable_rows.values()
        for row in phase_rows
        if row["speaking"]
        and row["person_present"]
        and row["person_x"] is not None
        and not row["speaking_break_active"]
    ]
    if rows:
        directed = sum(1 for row in rows if _person_dot(row) > 0.0)
        percent = 100.0 * directed / len(rows)
        if percent <= 50.0:
            failures.append(
                _failure(
                    result,
                    rows[0],
                    "speaking person direction",
                    {"person_directed_percent": percent},
                )
            )

    low = _SPEAK_BREAK_LENGTH[0] - 1.0 / result.fps
    high = _SPEAK_BREAK_LENGTH[1] + 1.0 / result.fps
    for start_row, duration in _speaking_break_runs(result.trace, result.fps):
        if duration < low or duration > high:
            failures.append(
                _failure(
                    result,
                    start_row,
                    "speaking break duration",
                    {
                        "duration_seconds": duration,
                        "allowed_min": low,
                        "allowed_max": high,
                    },
                )
            )
    return failures


def _person_dot(row: dict[str, object]) -> float:
    person_x = row["person_x"]
    person_y = row["person_y"]
    if person_x is None or person_y is None:
        return 0.0
    return (
        float(row["gaze_x"]) * (float(person_x) - 0.5)
        + float(row["gaze_y"]) * (float(person_y) - 0.5)
    )


def _speaking_break_runs(
    trace: list[dict[str, object]],
    fps: int,
) -> list[tuple[dict[str, object], float]]:
    runs: list[tuple[dict[str, object], float]] = []
    start: dict[str, object] | None = None
    count = 0
    previous_frame: int | None = None
    for row in trace:
        active = bool(row["speaking"]) and bool(row["speaking_break_active"])
        frame = int(row["frame"])
        contiguous = previous_frame is None or frame == previous_frame + 1
        if active and start is None:
            start = row
            count = 1
        elif active and start is not None and contiguous:
            count += 1
        elif active:
            runs.append((start, count / fps))
            start = row
            count = 1
        elif start is not None:
            runs.append((start, count / fps))
            start = None
            count = 0
        previous_frame = frame
    if start is not None:
        runs.append((start, count / fps))
    return runs


def _speaking_break_lengths(
    trace: list[dict[str, object]],
    fps: int,
) -> list[float]:
    return [duration for _, duration in _speaking_break_runs(trace, fps)]


def _max_pair_delta(
    trace: list[dict[str, object]],
    x_key: str,
    y_key: str,
) -> float:
    best = 0.0
    for previous, current in zip(trace, trace[1:]):
        delta = math.hypot(
            float(current[x_key]) - float(previous[x_key]),
            float(current[y_key]) - float(previous[y_key]),
        )
        best = max(best, delta)
    return best


def _performance_eye_deltas(
    result: SimulationResult,
) -> dict[str, dict[str, float]]:
    stable = _stable_rows_by_phase(result)
    baseline = stable.get("speaking_neutral", [])
    if not baseline:
        return {}

    baseline_values = {
        "left": {
            key: sum(float(row[key]) for row in baseline) / len(baseline)
            for key in ("left_eye_y", "left_eye_height", "left_eye_slant")
        },
        "right": {
            key: sum(float(row[key]) for row in baseline) / len(baseline)
            for key in ("right_eye_y", "right_eye_height", "right_eye_slant")
        },
    }

    deltas: dict[str, dict[str, float]] = {}
    for phase, rows in stable.items():
        if not rows or not phase.startswith("speaking_") or phase == "speaking_neutral":
            continue
        expression = str(rows[0]["performance_expression"])
        side_deltas: dict[str, float] = {}
        for side in ("left", "right"):
            keys = (
                f"{side}_eye_y",
                f"{side}_eye_height",
                f"{side}_eye_slant",
            )
            side_deltas[side] = max(
                (
                    abs(float(row[key]) - baseline_values[side][key])
                    for row in rows
                    for key in keys
                ),
                default=0.0,
            )
        deltas[expression] = side_deltas
    return deltas


def _failure(
    result: SimulationResult,
    row: dict[str, object],
    invariant: str,
    observed: dict[str, object],
) -> VerificationFailure:
    return VerificationFailure(
        scenario=result.scenario,
        phase=str(row["phase"]),
        frame=int(row["frame"]),
        time_seconds=float(row["time_seconds"]),
        invariant=invariant,
        observed=observed,
    )


def _format_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def _format_seconds(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f} s"


if __name__ == "__main__":
    simulate_scenario("conversational_cycle")
