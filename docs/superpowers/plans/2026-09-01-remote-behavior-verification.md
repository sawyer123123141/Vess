# Remote Behavior Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, headless Vess behavior verification that produces machine-checkable invariants, a frame-by-frame numeric trace, and a mobile-viewable GIF from the same real `State` + `FaceAnimator` + `face.py` simulation.

**Architecture:** Expose one read-only `FaceAnimator.debug_snapshot()` containing the exact render parameters used on the most recent frame. A lightweight runner drives scripted `State` phases at fixed 30 FPS, captures the exact native frame and animator snapshot from every tick, checks invariants/metrics, and emits `trace.json`, `summary.txt`, and `preview.gif`. GitHub Actions runs the ordinary unit suite first and only then runs the lightweight behavior-preview job.

**Tech Stack:** Python 3.11 in CI, standard-library `dataclasses`/`argparse`/`json`/`hashlib`/`unittest`, NumPy, Pillow for GIF encoding, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-01-remote-behavior-verification-design.md`

## Global Constraints

- Exercise production `State`, `FaceAnimator.tick`, and `face.render`; never create a second animator or redraw Vess independently.
- Simulate at fixed `30 FPS`, `dt = 1.0 / 30.0`; no `sleep()` or wall-clock pacing.
- Default RNG seed is `1`.
- Scripted scenarios use `mood_until = 0.0` unless explicitly testing expiry elsewhere.
- Trace, invariants, frame hashes, and GIF must derive from the same simulated ticks.
- Trace the exact gaze and whole-face offset passed to `face.render`, including drift/bob.
- Exit codes: `0` pass, `1` hard invariant failure, `2` harness/configuration error.
- Informational metrics never fail CI in v1.
- Generated artifacts are not committed.
- CI must never start/import runtime paths that require Ollama, Whisper, Kokoro, YOLO, microphone, camera, physical audio playback, or physical display hardware.
- CI uses synthetic state only and never uploads conversations, camera/audio captures, databases, credentials, or local-machine identifiers.
- Keep `requirements-ci.txt` lightweight; do not install the runtime `requirements.txt` merely for preview verification.
- No GitHub Pages, permanent dashboard, automatic aesthetic scoring, or automatic merge logic.
- Independent whole-eye motion remains out of scope, but schema-v1 left/right eye offset fields exist and are `0.0` until production supports them.

---

## File Structure

- Modify `output/animator.py`: retain exact render-time gaze/offset and expose read-only diagnostics.
- Create `tools/behavior_scenarios.py`: deterministic scenario data only.
- Create `tools/render_behavior_preview.py`: simulation, trace, invariants, metrics, summary, GIF, CLI.
- Create `tests/test_behavior_verification.py`: harness tests and regression checks.
- Create `requirements-ci.txt`: lightweight dependencies for repository tests.
- Create `.github/workflows/verify.yml`: `unit-tests` then dependent `behavior-preview`.
- Modify `.gitignore`: ignore `artifacts/behavior-verification/`.

---

### Task 1: Exact Render-Time Animator Diagnostics

**Files:**
- Modify: `output/animator.py`
- Modify: `tests/test_animator.py`

**Interfaces:**
- Produces `FaceAnimator.debug_snapshot() -> dict[str, object]`.
- Snapshot keys: `interaction_mode`, `render_gaze`, `render_offset`, `blink_openness`, `shape`, `color`, `fixation`, `speaking_break_active`, `speaking_break_remaining`, `performance_current`, `performance_target`.

- [ ] **Step 1: Write failing diagnostics tests**

Add to `tests/test_animator.py`:

```python
def test_debug_snapshot_reports_exact_values_used_for_render(self) -> None:
    state = State(
        speaking=True,
        person_present=True,
        person_pos=(0.8, 0.48),
        performance=PerformanceCue("playful", 0.65),
    )
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(state, 1.0 / 30.0)

    snapshot = animator.debug_snapshot()

    self.assertEqual(snapshot["interaction_mode"], "speaking")
    self.assertEqual(snapshot["render_gaze"], animator._last_render_gaze)
    self.assertEqual(snapshot["render_offset"], animator._last_render_offset)
    self.assertEqual(snapshot["shape"], animator._last_shape)
    self.assertEqual(snapshot["color"], animator._last_color)
    self.assertEqual(snapshot["blink_openness"], animator._openness())
    self.assertEqual(snapshot["fixation"], animator._fixation)
    self.assertEqual(snapshot["speaking_break_active"], animator._speak_break_left > 0.0)


def test_debug_snapshot_returns_copies_of_mutable_data(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(State(), 1.0 / 30.0)
    snapshot = animator.debug_snapshot()

    snapshot["shape"]["l_h"] = 999.0
    snapshot["performance_current"]["hold_scale"] = 999.0

    fresh = animator.debug_snapshot()
    self.assertNotEqual(fresh["shape"]["l_h"], 999.0)
    self.assertNotEqual(fresh["performance_current"]["hold_scale"], 999.0)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_animator -v
```

Expected: diagnostics fields/method missing.

- [ ] **Step 3: Store exact render values**

In `FaceAnimator.__init__`:

```python
self._last_render_gaze: tuple[float, float] = (0.0, 0.0)
self._last_render_offset: tuple[float, float] = (0.0, 0.0)
```

In `tick`, immediately after `_advance_gaze` and `_advance_face` return:

```python
self._last_render_gaze = gaze
self._last_render_offset = offset
```

Do not substitute `self._gaze` because the returned gaze includes drift. Do not substitute `face_offset` because the returned offset includes bob.

- [ ] **Step 4: Add `debug_snapshot`**

```python
def debug_snapshot(self) -> dict[str, object]:
    return {
        "interaction_mode": self._interaction_mode,
        "render_gaze": tuple(self._last_render_gaze),
        "render_offset": tuple(self._last_render_offset),
        "blink_openness": self._openness(),
        "shape": dict(self._last_shape),
        "color": tuple(self._last_color),
        "fixation": tuple(self._fixation),
        "speaking_break_active": self._speak_break_left > 0.0,
        "speaking_break_remaining": max(self._speak_break_left, 0.0),
        "performance_current": dict(self.performance_current),
        "performance_target": dict(self._performance_target),
    }
```

The method must not mutate timers, RNG, animation physics, or shared `State`.

- [ ] **Step 5: Verify GREEN**

```powershell
python -m unittest tests.test_animator -v
```

- [ ] **Step 6: Commit**

```powershell
git add output/animator.py tests/test_animator.py
git commit -m "test: expose animator render diagnostics"
```

---

### Task 2: Deterministic Scenarios and Native Trace Simulation

**Files:**
- Create: `tools/behavior_scenarios.py`
- Create: `tools/render_behavior_preview.py`
- Create: `tests/test_behavior_verification.py`

**Interfaces:**
- `ScenarioPhase(name: str, duration_seconds: float, state: dict[str, object])`.
- `BehaviorScenario(name: str, phases: tuple[ScenarioPhase, ...], seed: int = 1)`.
- `phase_frame_count(phase, fps) -> int`.
- `get_scenario(name, *, moods, performances) -> BehaviorScenario`.
- `apply_phase(state, phase) -> None`.
- `SimulationResult` contains `frames`, `frame_hashes`, `trace`, `failures`.
- `simulate_scenario(name, *, fps=30, seed=1) -> SimulationResult`.

- [ ] **Step 1: Write failing scenario/simulation tests**

Create `tests/test_behavior_verification.py`:

```python
import json
import unittest

import numpy as np

from performance import PerformanceCue
from state import State
from tools.behavior_scenarios import ScenarioPhase, get_scenario, phase_frame_count
from tools.render_behavior_preview import apply_phase, simulate_scenario


class BehaviorVerificationTests(unittest.TestCase):
    def test_phase_duration_converts_to_frame_count(self) -> None:
        self.assertEqual(phase_frame_count(ScenarioPhase("thinking", 1.5, {}), 30), 45)

    def test_apply_phase_changes_only_declared_fields(self) -> None:
        state = State(
            mood="curious",
            listening=False,
            thinking=False,
            person_present=True,
            person_pos=(0.2, 0.3),
        )
        apply_phase(state, ScenarioPhase("thinking", 1.0, {"thinking": True}))
        self.assertTrue(state.thinking)
        self.assertEqual(state.mood, "curious")
        self.assertFalse(state.listening)
        self.assertEqual(state.person_pos, (0.2, 0.3))

    def test_conversational_cycle_is_twelve_seconds(self) -> None:
        scenario = get_scenario(
            "conversational_cycle",
            moods=["neutral"],
            performances={
                "neutral": PerformanceCue(),
                "thoughtful": PerformanceCue("thoughtful", 0.55),
                "playful": PerformanceCue("playful", 0.65),
                "emphatic": PerformanceCue("emphatic", 0.70),
            },
        )
        self.assertEqual(sum(phase_frame_count(p, 30) for p in scenario.phases), 360)

    def test_simulation_produces_native_frames_and_monotonic_trace(self) -> None:
        result = simulate_scenario("conversational_cycle", fps=30, seed=1)
        self.assertEqual(len(result.frames), 360)
        self.assertEqual(len(result.trace), 360)
        self.assertTrue(all(frame.shape == (64, 64, 3) for frame in result.frames))
        self.assertTrue(all(frame.dtype == np.uint8 for frame in result.frames))
        self.assertEqual([row["frame"] for row in result.trace], list(range(360)))
        self.assertEqual(result.trace[0]["time_seconds"], 0.0)
        self.assertAlmostEqual(result.trace[-1]["time_seconds"], 359 / 30.0)
        json.dumps(result.trace)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

- [ ] **Step 3: Implement scenario types and dispatch**

Create `tools/behavior_scenarios.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from performance import PerformanceCue


@dataclass(frozen=True)
class ScenarioPhase:
    name: str
    duration_seconds: float
    state: dict[str, object]


@dataclass(frozen=True)
class BehaviorScenario:
    name: str
    phases: tuple[ScenarioPhase, ...]
    seed: int = 1


def phase_frame_count(phase: ScenarioPhase, fps: int) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = int(round(phase.duration_seconds * fps))
    if frames <= 0:
        raise ValueError(f"phase {phase.name!r} must contain at least one frame")
    return frames


def _require_cue(performances: dict[str, PerformanceCue], name: str) -> PerformanceCue:
    try:
        return performances[name]
    except KeyError as error:
        raise ValueError(f"scenario requires performance {name!r}") from error
```

Primary scenario:

```python
def _conversational_cycle(performances: dict[str, PerformanceCue]) -> BehaviorScenario:
    neutral = _require_cue(performances, "neutral")
    thoughtful = _require_cue(performances, "thoughtful")
    playful = _require_cue(performances, "playful")
    emphatic = _require_cue(performances, "emphatic")
    person = (0.80, 0.48)
    return BehaviorScenario("conversational_cycle", (
        ScenarioPhase("idle", 1.0, {
            "listening": False, "thinking": False, "speaking": False,
            "person_present": False, "person_pos": None,
            "mood": "neutral", "mood_until": 0.0, "performance": neutral,
        }),
        ScenarioPhase("tracking", 1.0, {
            "listening": False, "thinking": False, "speaking": False,
            "person_present": True, "person_pos": person, "performance": neutral,
        }),
        ScenarioPhase("listening", 1.5, {
            "listening": True, "thinking": False, "speaking": False,
            "person_present": True, "person_pos": person, "performance": neutral,
        }),
        ScenarioPhase("thinking", 1.5, {
            "listening": False, "thinking": True, "speaking": False,
            "person_present": True, "person_pos": person, "performance": thoughtful,
        }),
        ScenarioPhase("speaking_neutral", 2.0, {
            "listening": False, "thinking": False, "speaking": True,
            "person_present": True, "person_pos": person, "performance": neutral,
        }),
        ScenarioPhase("speaking_playful", 2.0, {
            "listening": False, "thinking": False, "speaking": True,
            "person_present": True, "person_pos": person, "performance": playful,
        }),
        ScenarioPhase("speaking_emphatic", 2.0, {
            "listening": False, "thinking": False, "speaking": True,
            "person_present": True, "person_pos": person, "performance": emphatic,
        }),
        ScenarioPhase("return_idle", 1.0, {
            "listening": False, "thinking": False, "speaking": False,
            "person_present": False, "person_pos": None, "performance": neutral,
        }),
    ))
```

Priority-conflict scenario:

```python
def _priority_conflicts(performances: dict[str, PerformanceCue]) -> BehaviorScenario:
    neutral = _require_cue(performances, "neutral")
    person = (0.80, 0.48)
    return BehaviorScenario("priority_conflicts", (
        ScenarioPhase("idle", 0.5, {
            "listening": False, "thinking": False, "speaking": False,
            "person_present": False, "person_pos": None, "performance": neutral,
        }),
        ScenarioPhase("tracking", 0.5, {
            "listening": False, "thinking": False, "speaking": False,
            "person_present": True, "person_pos": person,
        }),
        ScenarioPhase("speaking", 0.5, {
            "listening": False, "thinking": False, "speaking": True,
            "person_present": True, "person_pos": person,
        }),
        ScenarioPhase("thinking_over_speaking", 0.5, {
            "listening": False, "thinking": True, "speaking": True,
            "person_present": True, "person_pos": person,
        }),
        ScenarioPhase("listening_over_all", 0.5, {
            "listening": True, "thinking": True, "speaking": True,
            "person_present": True, "person_pos": person,
        }),
    ))
```

Geometry stress:

```python
def _geometry_stress(
    moods: list[str],
    performances: dict[str, PerformanceCue],
) -> BehaviorScenario:
    positions = ((0.10, 0.15), (0.90, 0.15), (0.10, 0.85), (0.90, 0.85))
    phases: list[ScenarioPhase] = []
    index = 0
    for mood in moods:
        for cue in performances.values():
            phases.append(ScenarioPhase(
                f"stress_{mood}_{cue.expression}",
                0.2,
                {
                    "mood": mood,
                    "mood_until": 0.0,
                    "performance": cue,
                    "listening": False,
                    "thinking": False,
                    "speaking": True,
                    "person_present": True,
                    "person_pos": positions[index % len(positions)],
                },
            ))
            index += 1
    return BehaviorScenario("geometry_stress", tuple(phases))
```

Dispatch:

```python
def get_scenario(
    name: str,
    *,
    moods: list[str],
    performances: dict[str, PerformanceCue],
) -> BehaviorScenario:
    if name == "conversational_cycle":
        return _conversational_cycle(performances)
    if name == "priority_conflicts":
        return _priority_conflicts(performances)
    if name == "geometry_stress":
        return _geometry_stress(moods, performances)
    raise KeyError(f"unknown behavior scenario: {name}")
```

- [ ] **Step 4: Implement runner bootstrap and data types**

The spec requires this exact CLI form:

```powershell
python tools/render_behavior_preview.py
```

A directly executed script gets `tools/` as `sys.path[0]`, so bootstrap the repository root **before** importing production modules:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Then import only standard library, NumPy, and production face-stack modules. Pillow is imported later inside GIF encoding.

Define:

```python
from dataclasses import dataclass, field
from hashlib import sha256
import json

import numpy as np

from output.animator import FaceAnimator
from performance import PerformanceCue, cue_for_label, load_performance_definitions
from state import State
from tools.behavior_scenarios import ScenarioPhase, get_scenario, phase_frame_count

DEFAULT_FPS = 30


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
```

- [ ] **Step 5: Implement config loading and phase application**

```python
def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runtime_definitions() -> tuple[dict[str, dict], dict[str, dict[str, object]]]:
    moods = _load_json(ROOT / "moods.json")
    performances = load_performance_definitions(_load_json(ROOT / "performance.json"))
    return moods, performances


def _performance_cues(definitions: dict[str, dict[str, object]]) -> dict[str, PerformanceCue]:
    return {name: cue_for_label(name, definitions) for name in definitions}


def apply_phase(state: State, phase: ScenarioPhase) -> None:
    with state.locked():
        for field_name, value in phase.state.items():
            if not hasattr(state, field_name):
                raise ValueError(f"unknown State field in phase {phase.name}: {field_name}")
            setattr(state, field_name, value)
```

- [ ] **Step 6: Build one trace record from the exact post-tick snapshot**

```python
def _trace_row(
    *, phase: str, frame_index: int, fps: int,
    state: State, snapshot: dict[str, object],
) -> dict[str, object]:
    shape = snapshot["shape"]
    gaze_x, gaze_y = snapshot["render_gaze"]
    face_x, face_y = snapshot["render_offset"]
    with state.locked():
        person = state.person_pos
        return {
            "frame": frame_index,
            "time_seconds": frame_index / fps,
            "phase": phase,
            "interaction_mode": snapshot["interaction_mode"],
            "mood": state.mood,
            "performance_expression": state.performance.expression,
            "performance_intensity": state.performance.intensity,
            "listening": state.listening,
            "thinking": state.thinking,
            "speaking": state.speaking,
            "person_present": state.person_present,
            "person_x": person[0] if person is not None else None,
            "person_y": person[1] if person is not None else None,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "face_offset_x": face_x,
            "face_offset_y": face_y,
            "blink_openness": snapshot["blink_openness"],
            "left_eye_x": shape["l_cx"],
            "left_eye_y": shape["l_cy"],
            "right_eye_x": shape["r_cx"],
            "right_eye_y": shape["r_cy"],
            "left_eye_offset_x": 0.0,
            "left_eye_offset_y": 0.0,
            "right_eye_offset_x": 0.0,
            "right_eye_offset_y": 0.0,
            "left_eye_width": shape["l_w"],
            "left_eye_height": shape["l_h"],
            "right_eye_width": shape["r_w"],
            "right_eye_height": shape["r_h"],
            "left_eye_slant": shape["l_slant"],
            "right_eye_slant": shape["r_slant"],
            "color_r": snapshot["color"][0],
            "color_g": snapshot["color"][1],
            "color_b": snapshot["color"][2],
            "fixation_x": snapshot["fixation"][0],
            "fixation_y": snapshot["fixation"][1],
            "speaking_break_active": snapshot["speaking_break_active"],
            "speaking_break_remaining": snapshot["speaking_break_remaining"],
            "performance_current": dict(snapshot["performance_current"]),
            "performance_target": dict(snapshot["performance_target"]),
        }
```

- [ ] **Step 7: Implement simulation**

```python
def simulate_scenario(name: str, *, fps: int = DEFAULT_FPS, seed: int = 1) -> SimulationResult:
    moods, definitions = _load_runtime_definitions()
    scenario = get_scenario(
        name,
        moods=list(moods),
        performances=_performance_cues(definitions),
    )
    state = State(mood_until=0.0)
    animator = FaceAnimator(moods, definitions, seed=seed)
    result = SimulationResult(name, fps, seed)
    frame_index = 0

    for phase in scenario.phases:
        apply_phase(state, phase)
        for _ in range(phase_frame_count(phase, fps)):
            frame = animator.tick(state, 1.0 / fps)
            snapshot = animator.debug_snapshot()
            result.frames.append(frame.copy())
            result.frame_hashes.append(sha256(frame.tobytes()).hexdigest())
            result.trace.append(_trace_row(
                phase=phase.name,
                frame_index=frame_index,
                fps=fps,
                state=state,
                snapshot=snapshot,
            ))
            frame_index += 1
    return result
```

- [ ] **Step 8: Verify GREEN and commit**

```powershell
python -m unittest tests.test_behavior_verification -v
git add tools/behavior_scenarios.py tools/render_behavior_preview.py tests/test_behavior_verification.py
git commit -m "feat: add deterministic face behavior simulation"
```

---

### Task 3: Hard Invariants, Determinism, Metrics, and Summary

**Files:**
- Modify: `tools/render_behavior_preview.py`
- Modify: `tests/test_behavior_verification.py`
- Modify: `tests/test_animator.py`

**Interfaces:**
- `check_invariants(result) -> list[VerificationFailure]`.
- `verify_determinism(name, *, fps, seed) -> list[VerificationFailure]`.
- `calculate_metrics(result) -> dict[str, object]`.
- `build_summary(results, *, deterministic) -> str`.

- [ ] **Step 1: Write failing invariant/metrics tests**

Add:

```python
def test_same_scenario_and_seed_are_deterministic(self) -> None:
    self.assertEqual(
        verify_determinism("conversational_cycle", fps=30, seed=1),
        [],
    )


def test_bad_gaze_reports_exact_frame(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    result.trace[10]["gaze_y"] = -1.25
    failure = next(f for f in check_invariants(result) if f.invariant == "gaze bounds")
    self.assertEqual(failure.frame, 10)
    self.assertEqual(failure.phase, result.trace[10]["phase"])
    self.assertEqual(failure.observed["gaze_y"], -1.25)


def test_priority_conflict_scenario_checks_full_mode_order(self) -> None:
    result = simulate_scenario("priority_conflicts", fps=30, seed=1)
    self.assertEqual(check_invariants(result), [])
    final_modes = {
        row["phase"]: row["interaction_mode"]
        for row in result.trace
        if row["frame"] % 15 == 14
    }
    self.assertEqual(final_modes["tracking"], "tracking")
    self.assertEqual(final_modes["speaking"], "speaking")
    self.assertEqual(final_modes["thinking_over_speaking"], "thinking")
    self.assertEqual(final_modes["listening_over_all"], "listening")


def test_metrics_are_calculated_from_trace(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    metrics = calculate_metrics(result)
    self.assertEqual(metrics["speaking_frames"], sum(bool(r["speaking"]) for r in result.trace))
    self.assertGreaterEqual(metrics["person_directed_percent"], 0.0)
    self.assertLessEqual(metrics["person_directed_percent"], 100.0)
```

Add summary failure coverage:

```python
def test_summary_names_failure_location_and_values(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    result.failures.append(VerificationFailure(
        result.scenario, "thinking", 143, 143 / 30.0,
        "gaze bounds", {"gaze_y": -1.083},
    ))
    summary = build_summary([result], deterministic=True)
    self.assertIn("frame 143", summary)
    self.assertIn("gaze bounds", summary)
    self.assertIn("-1.083", summary)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

- [ ] **Step 3: Implement global/geometry helpers**

```python
import math
from output.animator import _SPEAK_BREAK_LENGTH

_PANEL_MIN = 1.0
_PANEL_MAX = 63.0
_TRANSITION_SECONDS = 0.25


def _finite(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def _eye_bounds(row: dict[str, object], prefix: str) -> tuple[float, float, float, float]:
    cx = float(row[f"{prefix}_eye_x"]) + float(row["face_offset_x"]) + float(row[f"{prefix}_eye_offset_x"])
    cy = float(row[f"{prefix}_eye_y"]) + float(row["face_offset_y"]) + float(row[f"{prefix}_eye_offset_y"])
    width = float(row[f"{prefix}_eye_width"])
    height = float(row[f"{prefix}_eye_height"])
    slant = abs(float(row[f"{prefix}_eye_slant"]))
    return (
        cx - width / 2.0,
        cy - height / 2.0 - slant,
        cx + width / 2.0,
        cy + height / 2.0 + slant,
    )


def _failure(result, row, invariant: str, **observed: object) -> VerificationFailure:
    return VerificationFailure(
        result.scenario,
        str(row["phase"]),
        int(row["frame"]),
        float(row["time_seconds"]),
        invariant,
        observed,
    )
```

For each row/frame, fail on:

```python
not _finite(row)
frame.shape != (64, 64, 3)
frame.dtype != np.uint8
not (-1.0 <= gaze_x <= 1.0 and -1.0 <= gaze_y <= 1.0)
not (0.0 <= performance_intensity <= 1.0)
left/right eye width <= 0.0
left/right eye height < 2.0
any composed eye bound < _PANEL_MIN or > _PANEL_MAX
```

Geometry uses composed eye centers + whole-face offset + per-eye offsets + width/height/slant, never static config alone.

- [ ] **Step 4: Implement generic interaction-priority invariant**

For every stable row after the first `round(0.25 * fps)` frames of each phase:

```python
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
```

Hard-fail if `interaction_mode != _expected_mode(row)`. This single rule proves:

```text
listening > thinking > speaking > tracking > idle
```

including the overlap scenario.

- [ ] **Step 5: Implement listening/thinking/speaking invariants**

Group trace rows into contiguous phases and skip the transition allowance.

Listening stable rows:

```python
fixation remains unchanged through the stable phase
person_direction_dot = gaze_x * (person_x - 0.5) + gaze_y * (person_y - 0.5)
person_direction_dot > 0.0
face_offset_x * (person_x - 0.5) > 0.0
```

Thinking stable rows:

```python
gaze_y < 0.0
fixation remains unchanged through the stable phase
```

Speaking stable rows with a person, excluding `speaking_break_active` rows:

```python
person_direction_dot = gaze_x * (person_x - 0.5) + gaze_y * (person_y - 0.5)
```

The spec says **most** non-break speaking frames are person-directed, so the hard requirement is mathematically minimal and non-arbitrary:

```python
person_directed_percent > 50.0
```

Do not invent a 70/80/90% threshold before real regressions justify one.

Measure contiguous observed speaking breaks. For each completed run:

```python
low = _SPEAK_BREAK_LENGTH[0] - 1.0 / result.fps
high = _SPEAK_BREAK_LENGTH[1] + 1.0 / result.fps
```

Require duration inside `[low, high]`. Never require a break to occur.

- [ ] **Step 6: Implement performance invariants and exact color regression test**

For rows whose `performance_expression == "neutral"`, require `performance_target` to contain:

```python
shape deltas l_h/r_h/l_slant/r_slant/l_cy/r_cy == 0.0
hold_scale/ease_scale/track_bias_scale/speaking_break_scale == 1.0
gaze_y_bias == 0.0
```

Add to `tests/test_animator.py`:

```python
def test_performance_overlay_does_not_change_mood_color(self) -> None:
    neutral_state = State(mood="curious", performance=PerformanceCue())
    playful_state = State(
        mood="curious",
        performance=PerformanceCue("playful", 0.65),
    )
    neutral = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    playful = FaceAnimator(MOODS, PERFORMANCES, seed=1)

    for _ in range(30):
        neutral.tick(neutral_state, 1.0 / 30.0)
        playful.tick(playful_state, 1.0 / 30.0)

    self.assertEqual(
        neutral.debug_snapshot()["color"],
        playful.debug_snapshot()["color"],
    )
```

Add to verification tests:

```python
def test_unknown_performance_targets_neutral(self) -> None:
    moods, definitions = _load_runtime_definitions()
    animator = FaceAnimator(moods, definitions, seed=1)
    animator.tick(State(performance=PerformanceCue("not-real", 1.0)), 1.0 / 30.0)
    target = animator.debug_snapshot()["performance_target"]
    self.assertEqual(target["l_h"], 0.0)
    self.assertEqual(target["hold_scale"], 1.0)
```

- [ ] **Step 7: Implement determinism**

```python
def verify_determinism(name: str, *, fps: int, seed: int) -> list[VerificationFailure]:
    first = simulate_scenario(name, fps=fps, seed=seed)
    second = simulate_scenario(name, fps=fps, seed=seed)
    if first.trace == second.trace and first.frame_hashes == second.frame_hashes:
        return []
    row = first.trace[0] if first.trace else {"phase": "<none>", "frame": 0, "time_seconds": 0.0}
    return [_failure(
        first, row, "determinism",
        trace_equal=first.trace == second.trace,
        frame_hashes_equal=first.frame_hashes == second.frame_hashes,
    )]
```

Do not round trace values to manufacture equality.

- [ ] **Step 8: Implement metrics**

`calculate_metrics(result)` returns:

```python
{
    "total_frames": len(result.trace),
    "speaking_frames": <int>,
    "person_directed_percent": <float>,
    "speaking_break_count": <int>,
    "average_break_seconds": <float | None>,
    "max_break_seconds": <float | None>,
    "peak_face_offset": <float>,
    "max_frame_gaze_delta": <float>,
    "max_frame_face_delta": <float>,
    "performance_eye_deltas": <dict>,
}
```

Use `math.hypot` for vector magnitudes. `performance_eye_deltas` compares stable performance-speaking rows against the most recent stable `speaking_neutral` baseline; if no baseline exists, return `{}`.

- [ ] **Step 9: Implement generated summary**

`build_summary(results, deterministic=...)` derives all numbers from passed results. Required shape:

```text
Vess behavior verification

Scenarios: <passed>/<total> PASS|FAIL
Invalid frames: <count>
Geometry: PASS|FAIL
Determinism: PASS|FAIL

Conversational cycle
  listening mode: PASS|FAIL
  thinking mode: PASS|FAIL
  speaking mode: PASS|FAIL
  speaking person-directed frames: <percent>%
  speaking gaze breaks: <count>
  average break duration: <seconds|n/a> s
  peak face offset: <pixels> px

Performance
  <expression> max eye-height delta: L <value> / R <value> px
```

Every failure appends:

```text
FAIL <scenario> frame <frame> (<time> s)
Invariant: <invariant>
Observed: <JSON observed values>
```

Never substitute example metrics when data is absent; use `n/a`.

- [ ] **Step 10: Verify GREEN and commit**

```powershell
python -m unittest tests.test_behavior_verification tests.test_animator -v
git add tools/render_behavior_preview.py tests/test_behavior_verification.py tests/test_animator.py
git commit -m "test: add behavior invariants and metrics"
```

---

### Task 4: GIF, Trace Artifact, Summary Artifact, and CLI Contract

**Files:**
- Modify: `tools/render_behavior_preview.py`
- Modify: `tests/test_behavior_verification.py`
- Modify: `.gitignore`

**Interfaces:**
- `write_trace(result, output_dir) -> Path`.
- `write_preview(result, output_dir, *, sample_every=2, scale=6) -> Path`.
- `run_verification(...) -> tuple[int, str]`.
- CLI: `python tools/render_behavior_preview.py [--scenario ...] [--seed ...] [--output ...] [--no-gif]`.

- [ ] **Step 1: Write failing artifact tests**

Add:

```python
import tempfile
from pathlib import Path
from PIL import Image


def test_write_trace_uses_schema_v1(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    with tempfile.TemporaryDirectory() as directory:
        payload = json.loads(write_trace(result, Path(directory)).read_text(encoding="utf-8"))
    self.assertEqual(payload["schema_version"], 1)
    self.assertEqual(payload["fps"], 30)
    self.assertEqual(payload["seed"], 1)
    self.assertEqual(payload["scenario"], "conversational_cycle")
    self.assertEqual(payload["frames"], result.trace)


def test_preview_preserves_a_real_nonblack_native_pixel(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    native = result.frames[0]
    y, x = np.argwhere(np.any(native != 0, axis=2))[0]
    expected = tuple(int(v) for v in native[y, x])

    with tempfile.TemporaryDirectory() as directory:
        path = write_preview(result, Path(directory), sample_every=30, scale=2)
        image = Image.open(path)
        image.seek(0)
        encoded = np.asarray(image.convert("RGB"))

    self.assertEqual(tuple(int(v) for v in encoded[24 + y * 2, x * 2]), expected)


def test_run_verification_writes_primary_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
        code, summary = run_verification(output=Path(directory), include_gif=True)
        names = {p.name for p in Path(directory).iterdir()}
    self.assertEqual(code, 0)
    self.assertEqual({"preview.gif", "trace.json", "summary.txt"} - names, set())
    self.assertIn("Vess behavior verification", summary)
```

Add one test patching `check_invariants` to return a failure and expect code `1`; add one patching `_load_runtime_definitions` to raise `ValueError` and expect code `2` plus `summary.txt` containing `HARNESS ERROR`.

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

- [ ] **Step 3: Implement `trace.json`**

```python
def write_trace(result: SimulationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trace.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "fps": result.fps,
        "seed": result.seed,
        "scenario": result.scenario,
        "frames": result.trace,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return path
```

- [ ] **Step 4: Implement GIF encoding from native frames only**

Import Pillow only inside the function:

```python
def write_preview(
    result: SimulationResult,
    output_dir: Path,
    *, sample_every: int = 2,
    scale: int = 6,
) -> Path:
    from PIL import Image, ImageDraw

    if sample_every <= 0 or scale <= 0:
        raise ValueError("sample_every and scale must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    label_height = 24

    for index in range(0, len(result.frames), sample_every):
        native = Image.fromarray(result.frames[index], mode="RGB")
        scaled = native.resize((64 * scale, 64 * scale), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (64 * scale, 64 * scale + label_height), (12, 12, 15))
        canvas.paste(scaled, (0, label_height))
        row = result.trace[index]
        label = f"{row['phase']} | {row['interaction_mode']} | {row['performance_expression']}"
        ImageDraw.Draw(canvas).text((6, 5), label, fill=(235, 235, 240))
        images.append(canvas)

    if not images:
        raise ValueError("cannot write preview with no frames")
    path = output_dir / "preview.gif"
    duration_ms = round(1000 * sample_every / result.fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return path
```

Pillow only scales/labels/encodes real frames; it never draws Vess geometry.

- [ ] **Step 5: Implement orchestration**

```python
DEFAULT_SCENARIOS = (
    "conversational_cycle",
    "priority_conflicts",
    "geometry_stress",
)
```

`run_verification` simulates every selected scenario, appends `check_invariants`, then runs `verify_determinism` for every selected scenario. Attach any determinism failure to its corresponding result.

Artifact source:

```python
artifact_result = next(
    (r for r in results if r.scenario == "conversational_cycle"),
    results[0],
)
```

Thus default `all` produces the conversational-cycle artifact; `--scenario priority_conflicts` still produces a useful trace/GIF for the explicitly selected scenario.

Always write `summary.txt`. Write `trace.json` for `artifact_result`. Write GIF unless `--no-gif`.

Return code `1` if any result has failures, else `0`.

Catch top-level harness/config exceptions, create output directory, write:

```text
Vess behavior verification

HARNESS ERROR: <exception>
```

and return `(2, summary)`.

- [ ] **Step 6: Implement CLI**

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render deterministic Vess behavior verification")
    parser.add_argument(
        "--scenario",
        choices=("all", "conversational_cycle", "priority_conflicts", "geometry_stress"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "behavior-verification")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scenarios = DEFAULT_SCENARIOS if args.scenario == "all" else (args.scenario,)
    code, summary = run_verification(
        scenarios=scenarios,
        seed=args.seed,
        fps=30,
        output=args.output,
        include_gif=not args.no_gif,
    )
    print(summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
```

Do not expose animator tuning parameters.

- [ ] **Step 7: Ignore generated artifacts**

Append to `.gitignore`:

```text
artifacts/behavior-verification/
```

- [ ] **Step 8: Verify locally and commit**

```powershell
python -m unittest tests.test_behavior_verification -v
python tools/render_behavior_preview.py
git add tools/render_behavior_preview.py tests/test_behavior_verification.py .gitignore
git commit -m "feat: emit mobile behavior verification artifacts"
```

Expected harness exit `0`, with `preview.gif`, `trace.json`, `summary.txt`.

---

### Task 5: Lightweight CI and Artifact Publication

**Files:**
- Create: `requirements-ci.txt`
- Create: `.github/workflows/verify.yml`
- Modify: `tests/test_behavior_verification.py`

**Interfaces:**
- CI `unit-tests` job runs full `unittest` discovery.
- `behavior-preview` depends on `unit-tests`, publishes summary/artifacts even when behavioral invariants fail, then returns the original verification exit code.

- [ ] **Step 1: Add a no-heavy-runtime execution test**

The test must run an actual simulation, not merely import the module:

```python
def test_verification_simulation_does_not_import_heavy_runtime_packages(self) -> None:
    import subprocess
    import sys

    code = r'''
import builtins
blocked = {"kokoro", "faster_whisper", "ultralytics", "sounddevice", "ollama"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    root = name.split(".", 1)[0]
    if root in blocked:
        raise AssertionError(f"verification imported blocked runtime package: {root}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from tools.render_behavior_preview import simulate_scenario
result = simulate_scenario("conversational_cycle", fps=30, seed=1)
assert len(result.frames) == 360
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)
```

- [ ] **Step 2: Run that test before adding dependencies**

```powershell
python -m unittest tests.test_behavior_verification.BehaviorVerificationTests.test_verification_simulation_does_not_import_heavy_runtime_packages -v
```

If it fails, fix the verification import graph. Do not solve it by installing blocked packages.

- [ ] **Step 3: Create lightweight unit-test requirements**

Create `requirements-ci.txt` exactly:

```text
numpy
opencv-python-headless
fastapi
uvicorn
httpx
pillow
```

The current production modules import heavyweight model/audio libraries lazily. If the complete suite later proves one additional **lightweight** package is genuinely required, add that exact package; never replace this file with the runtime requirements wholesale.

- [ ] **Step 4: Create GitHub Actions workflow**

Create `.github/workflows/verify.yml`:

```yaml
name: Verify Vess

on:
  push:
  pull_request:

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: requirements-ci.txt

      - name: Install CI dependencies
        run: python -m pip install --upgrade pip && pip install -r requirements-ci.txt

      - name: Run unit tests
        run: python -m unittest discover -s tests -v

  behavior-preview:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install preview dependencies
        run: python -m pip install --upgrade pip && pip install numpy pillow

      - name: Run behavior verification
        id: behavior
        shell: bash
        run: |
          set +e
          python tools/render_behavior_preview.py
          code=$?
          echo "exit_code=$code" >> "$GITHUB_OUTPUT"
          exit 0

      - name: Publish verification summary
        if: always()
        shell: bash
        run: |
          if [ -f artifacts/behavior-verification/summary.txt ]; then
            cat artifacts/behavior-verification/summary.txt >> "$GITHUB_STEP_SUMMARY"
          else
            echo "Vess behavior verification did not produce summary.txt" >> "$GITHUB_STEP_SUMMARY"
          fi

      - name: Upload behavior artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: vess-behavior-verification
          path: artifacts/behavior-verification/
          if-no-files-found: error

      - name: Fail job if verification failed
        if: always()
        shell: bash
        run: exit "${{ steps.behavior.outputs.exit_code }}"
```

This deliberately preserves artifacts on invariant failure before failing the job.

- [ ] **Step 5: Run full local verification**

```powershell
python -m unittest discover -s tests -v
python tools/render_behavior_preview.py
```

Expected: 0 test failures/errors and harness exit 0.

- [ ] **Step 6: Commit**

```powershell
git add requirements-ci.txt .github/workflows/verify.yml tests/test_behavior_verification.py
git commit -m "ci: add remote Vess behavior verification"
```

---

## Final Verification Checklist

- [ ] Full unit suite reports 0 failures/errors.
- [ ] `python tools/render_behavior_preview.py` exits 0.
- [ ] `preview.gif`, `trace.json`, `summary.txt` exist in `artifacts/behavior-verification/`.
- [ ] GIF pixels originate from exact `FaceAnimator.tick` frames, not a duplicate renderer.
- [ ] Trace schema version is 1 and default primary trace contains all 360 native 30-FPS frames.
- [ ] Trace gaze/offset equal exact render-time gaze/offset including drift/bob.
- [ ] Left/right eye-offset fields exist and remain `0.0` until whole-eye movement arrives.
- [ ] Same scenario + seed has identical trace and frame hashes.
- [ ] Priority invariant proves `listening > thinking > speaking > tracking > idle`.
- [ ] Listening, thinking, speaking, performance, finite-value, frame-format, and composed-geometry invariants run automatically.
- [ ] Every failure identifies scenario, phase, frame/time, invariant, and measured values.
- [ ] Summary numbers are trace-derived; missing metrics show `n/a`, not invented constants.
- [ ] Verification simulation imports no Kokoro, Whisper, YOLO, sounddevice, or Ollama package.
- [ ] `requirements-ci.txt` remains lightweight.
- [ ] GitHub Actions has `unit-tests` and dependent `behavior-preview` jobs.
- [ ] Failed behavior verification still uploads summary/artifacts before returning failure.
- [ ] Artifact name is exactly `vess-behavior-verification`.
- [ ] No real camera/audio/conversation/database/private-machine data enters CI artifacts.
- [ ] Final target-PC acceptance remains required for hardware, latency, and subjective visual quality.
