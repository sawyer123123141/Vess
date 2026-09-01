# Remote Behavior Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, headless Vess behavior verification that produces machine-checkable invariants, a frame-by-frame numeric trace, and a mobile-viewable GIF from the same real `State` + `FaceAnimator` + `face.py` simulation.

**Architecture:** Expose one read-only `FaceAnimator.debug_snapshot()` containing the exact render parameters used on the most recent frame. A lightweight verification runner drives scripted `State` phases at fixed 30 FPS, captures native frames plus animator snapshots, checks invariants and metrics, then emits `trace.json`, `summary.txt`, and `preview.gif`. GitHub Actions runs ordinary unit tests first and only then runs the lightweight behavior preview job.

**Tech Stack:** Python 3.11 in CI, standard library `dataclasses`/`argparse`/`json`/`hashlib`/`unittest`, NumPy, Pillow for GIF encoding, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-01-remote-behavior-verification-design.md`

## Global Constraints

- The verification harness must exercise production `State`, `FaceAnimator.tick`, and `face.render`; never create a second animator or redraw Vess independently.
- Simulation runs at fixed `30 FPS` with `dt = 1.0 / 30.0`; no `sleep()` or wall-clock pacing.
- Default deterministic RNG seed is `1`.
- Scripted scenarios keep `State.mood_until = 0.0` unless specifically testing mood expiry elsewhere.
- `trace.json`, invariant checks, and GIF frames must come from the same simulated ticks.
- Exact gaze and whole-face offset in the trace must be the values actually passed to `face.render`, including drift/bob, not nearby pre-render internal values.
- Hard invariant failures return exit code `1`; harness/configuration errors return `2`; success returns `0`.
- Informational metrics do not fail CI in version 1.
- Generated verification artifacts are not committed.
- CI must never start Ollama, Whisper, Kokoro, YOLO, camera, microphone, physical audio playback, or a physical display path.
- CI uses synthetic state only and never uploads conversations, camera frames, audio, databases, credentials, or local machine identifiers.
- `requirements-ci.txt` stays lightweight; do not install the full runtime `requirements.txt` merely for the behavior preview.
- No GitHub Pages, permanent dashboard, automatic aesthetic scoring, or automatic merge logic in this slice.
- Independent whole-eye motion itself remains out of scope, but schema-v1 per-eye offset fields must exist and start at `0.0`.

---

## File Structure

- Modify `output/animator.py`: retain exact render-time gaze/offset and expose one read-only diagnostic snapshot.
- Create `tools/behavior_scenarios.py`: deterministic scenario data only.
- Create `tools/render_behavior_preview.py`: simulation, trace capture, invariant checking, metrics, summary, GIF output, CLI.
- Create `tests/test_behavior_verification.py`: harness, determinism, invariants, metrics, preview, and no-heavy-import tests.
- Create `requirements-ci.txt`: lightweight dependencies needed by unit tests and CI preview.
- Create `.github/workflows/verify.yml`: `unit-tests` then dependent `behavior-preview` job.
- Modify `.gitignore`: ignore `artifacts/behavior-verification/`.

---

### Task 1: Expose Exact Render-Time Animator Diagnostics

**Files:**
- Modify: `output/animator.py`
- Modify: `tests/test_animator.py`

**Interfaces:**
- Produces: `FaceAnimator.debug_snapshot() -> dict[str, object]`.
- Snapshot keys consumed later: `interaction_mode`, `render_gaze`, `render_offset`, `blink_openness`, `shape`, `color`, `fixation`, `speaking_break_active`, `speaking_break_remaining`, `performance_current`, `performance_target`.
- The method is read-only and animator-local; it does not touch shared `State`.

- [ ] **Step 1: Write failing tests for exact render diagnostics**

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
    self.assertEqual(
        snapshot["speaking_break_active"],
        animator._speak_break_left > 0.0,
    )
```

Add immutability coverage:

```python
def test_debug_snapshot_returns_copies_of_mutable_animator_data(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(State(), 1.0 / 30.0)

    snapshot = animator.debug_snapshot()
    snapshot["shape"]["l_h"] = 999.0
    snapshot["performance_current"]["hold_scale"] = 999.0

    fresh = animator.debug_snapshot()
    self.assertNotEqual(fresh["shape"]["l_h"], 999.0)
    self.assertNotEqual(fresh["performance_current"]["hold_scale"], 999.0)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_animator -v
```

Expected: failures because `debug_snapshot`, `_last_render_gaze`, and `_last_render_offset` do not exist.

- [ ] **Step 3: Store the exact values passed into `face.render`**

In `FaceAnimator.__init__`, initialize:

```python
self._last_render_gaze: tuple[float, float] = (0.0, 0.0)
self._last_render_offset: tuple[float, float] = (0.0, 0.0)
```

In `tick`, immediately after computing:

```python
gaze = self._advance_gaze(...)
offset = self._advance_face(...)
```

store:

```python
self._last_render_gaze = gaze
self._last_render_offset = offset
```

Do not use `self._gaze` for the trace because `_advance_gaze` adds visual drift after `_gaze` is updated. Do not use `face_offset` for the trace because `_advance_face` adds bob to the returned render offset.

- [ ] **Step 4: Add the read-only diagnostic method**

Add to `FaceAnimator`:

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

This method only reports state already computed by the animator. It must not mutate RNG state, timers, `State`, or animation physics.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_animator -v
```

Expected: all animator tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add output/animator.py tests/test_animator.py
git commit -m "test: expose animator render diagnostics"
```

---

### Task 2: Deterministic Scenario Model and Native Trace Simulation

**Files:**
- Create: `tools/behavior_scenarios.py`
- Create: `tools/render_behavior_preview.py`
- Create: `tests/test_behavior_verification.py`

**Interfaces:**
- Produces: `ScenarioPhase(name: str, duration_seconds: float, state: dict[str, object])`.
- Produces: `BehaviorScenario(name: str, phases: tuple[ScenarioPhase, ...], seed: int = 1)`.
- Produces: `get_scenario(name: str, *, moods: list[str], performances: dict[str, PerformanceCue]) -> BehaviorScenario`.
- Produces: `phase_frame_count(phase: ScenarioPhase, fps: int) -> int`.
- Produces: `apply_phase(state: State, phase: ScenarioPhase) -> None`.
- Produces: `SimulationResult` with native frames, trace records, frame hashes, and failures.
- Produces: `simulate_scenario(...) -> SimulationResult` using real `State`, `FaceAnimator`, and JSON configs.

- [ ] **Step 1: Write failing scenario and simulation tests**

Create `tests/test_behavior_verification.py` with:

```python
import json
import math
import unittest

import numpy as np

from performance import PerformanceCue
from state import State
from tools.behavior_scenarios import ScenarioPhase, get_scenario, phase_frame_count
from tools.render_behavior_preview import apply_phase, simulate_scenario


class BehaviorVerificationTests(unittest.TestCase):
    def test_phase_duration_converts_to_deterministic_frame_count(self) -> None:
        phase = ScenarioPhase("thinking", 1.5, {"thinking": True})
        self.assertEqual(phase_frame_count(phase, 30), 45)

    def test_apply_phase_changes_only_declared_state_fields(self) -> None:
        state = State(
            mood="curious",
            listening=False,
            thinking=False,
            speaking=False,
            person_present=True,
            person_pos=(0.2, 0.3),
        )
        phase = ScenarioPhase("thinking", 1.0, {"thinking": True})

        apply_phase(state, phase)

        self.assertTrue(state.thinking)
        self.assertEqual(state.mood, "curious")
        self.assertFalse(state.listening)
        self.assertFalse(state.speaking)
        self.assertEqual(state.person_pos, (0.2, 0.3))

    def test_conversational_cycle_has_expected_total_frames(self) -> None:
        scenario = get_scenario(
            "conversational_cycle",
            moods=["neutral"],
            performances={
                "neutral": PerformanceCue(),
                "playful": PerformanceCue("playful", 0.65),
                "emphatic": PerformanceCue("emphatic", 0.7),
            },
        )
        total = sum(phase_frame_count(phase, 30) for phase in scenario.phases)
        self.assertEqual(total, 360)

    def test_simulation_uses_native_frames_and_monotonic_trace(self) -> None:
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

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_behavior_verification -v
```

Expected: import failures because the scenario and runner modules do not exist.

- [ ] **Step 3: Implement scenario dataclasses and fixed scenarios**

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
```

Implement the primary scenario exactly:

```python
def _conversational_cycle(performances: dict[str, PerformanceCue]) -> BehaviorScenario:
    neutral = performances.get("neutral", PerformanceCue())
    playful = performances.get("playful", PerformanceCue("playful", 0.65))
    emphatic = performances.get("emphatic", PerformanceCue("emphatic", 0.7))
    person = (0.80, 0.48)
    return BehaviorScenario(
        "conversational_cycle",
        (
            ScenarioPhase("idle", 1.0, {
                "listening": False, "thinking": False, "speaking": False,
                "person_present": False, "person_pos": None,
                "mood": "neutral", "mood_until": 0.0, "performance": neutral,
            }),
            ScenarioPhase("tracking", 1.0, {
                "listening": False, "thinking": False, "speaking": False,
                "person_present": True, "person_pos": person,
                "performance": neutral,
            }),
            ScenarioPhase("listening", 1.5, {
                "listening": True, "thinking": False, "speaking": False,
                "person_present": True, "person_pos": person,
                "performance": neutral,
            }),
            ScenarioPhase("thinking", 1.5, {
                "listening": False, "thinking": True, "speaking": False,
                "person_present": True, "person_pos": person,
                "performance": performances.get(
                    "thoughtful", PerformanceCue("thoughtful", 0.55)
                ),
            }),
            ScenarioPhase("speaking_neutral", 2.0, {
                "listening": False, "thinking": False, "speaking": True,
                "person_present": True, "person_pos": person,
                "performance": neutral,
            }),
            ScenarioPhase("speaking_playful", 2.0, {
                "listening": False, "thinking": False, "speaking": True,
                "person_present": True, "person_pos": person,
                "performance": playful,
            }),
            ScenarioPhase("speaking_emphatic", 2.0, {
                "listening": False, "thinking": False, "speaking": True,
                "person_present": True, "person_pos": person,
                "performance": emphatic,
            }),
            ScenarioPhase("return_idle", 1.0, {
                "listening": False, "thinking": False, "speaking": False,
                "person_present": False, "person_pos": None,
                "performance": neutral,
            }),
        ),
    )
```

Implement `priority_conflicts` with five 0.5-second phases: idle; tracking; speaking; thinking+speaking; listening+thinking+speaking. Keep a person at `(0.80, 0.48)` in every non-idle phase.

Implement geometry stress as a generated deterministic scenario:

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
            position = positions[index % len(positions)]
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
                    "person_pos": position,
                },
            ))
            index += 1
    return BehaviorScenario("geometry_stress", tuple(phases))
```

Add `get_scenario` dispatch that raises `KeyError` for unknown names.

- [ ] **Step 4: Implement native simulation and trace assembly**

Create `tools/render_behavior_preview.py` with imports limited to standard library, NumPy, Pillow only in the GIF function, production `State`, `FaceAnimator`, `PerformanceCue`/loader, and scenario definitions.

Define:

```python
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from output.animator import FaceAnimator
from performance import PerformanceCue, cue_for_label, load_performance_definitions
from state import State
from tools.behavior_scenarios import BehaviorScenario, ScenarioPhase, get_scenario, phase_frame_count


ROOT = Path(__file__).resolve().parents[1]
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

Add config loading:

```python
def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runtime_definitions() -> tuple[dict[str, dict], dict[str, dict[str, object]]]:
    moods = _load_json(ROOT / "moods.json")
    performances = load_performance_definitions(_load_json(ROOT / "performance.json"))
    return moods, performances
```

Add cue map:

```python
def _performance_cues(definitions: dict[str, dict[str, object]]) -> dict[str, PerformanceCue]:
    return {name: cue_for_label(name, definitions) for name in definitions}
```

Add phase application that validates fields and changes only declared fields:

```python
def apply_phase(state: State, phase: ScenarioPhase) -> None:
    with state.locked():
        for field_name, value in phase.state.items():
            if not hasattr(state, field_name):
                raise ValueError(f"unknown State field in phase {phase.name}: {field_name}")
            setattr(state, field_name, value)
```

Build one trace row from the exact post-`tick` snapshot:

```python
def _trace_row(
    *,
    scenario: str,
    phase: str,
    frame_index: int,
    fps: int,
    state: State,
    animator_snapshot: dict[str, object],
) -> dict[str, object]:
    shape = animator_snapshot["shape"]
    gaze_x, gaze_y = animator_snapshot["render_gaze"]
    face_x, face_y = animator_snapshot["render_offset"]
    with state.locked():
        person = state.person_pos
        row = {
            "frame": frame_index,
            "time_seconds": frame_index / fps,
            "phase": phase,
            "interaction_mode": animator_snapshot["interaction_mode"],
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
            "blink_openness": animator_snapshot["blink_openness"],
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
            "color_r": animator_snapshot["color"][0],
            "color_g": animator_snapshot["color"][1],
            "color_b": animator_snapshot["color"][2],
            "fixation_x": animator_snapshot["fixation"][0],
            "fixation_y": animator_snapshot["fixation"][1],
            "speaking_break_active": animator_snapshot["speaking_break_active"],
            "speaking_break_remaining": animator_snapshot["speaking_break_remaining"],
            "performance_current": dict(animator_snapshot["performance_current"]),
            "performance_target": dict(animator_snapshot["performance_target"]),
        }
    return row
```

Implement `simulate_scenario`:

```python
def simulate_scenario(name: str, *, fps: int = DEFAULT_FPS, seed: int = 1) -> SimulationResult:
    moods, definitions = _load_runtime_definitions()
    cues = _performance_cues(definitions)
    scenario = get_scenario(name, moods=list(moods), performances=cues)
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
                scenario=name,
                phase=phase.name,
                frame_index=frame_index,
                fps=fps,
                state=state,
                animator_snapshot=snapshot,
            ))
            frame_index += 1
    return result
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_behavior_verification -v
```

Expected: scenario/frame/trace tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add tools/behavior_scenarios.py tools/render_behavior_preview.py tests/test_behavior_verification.py
git commit -m "feat: add deterministic face behavior simulation"
```

---

### Task 3: Hard Invariants, Determinism, Metrics, and Text Summary

**Files:**
- Modify: `tools/render_behavior_preview.py`
- Modify: `tests/test_behavior_verification.py`

**Interfaces:**
- Produces: `check_invariants(result: SimulationResult) -> list[VerificationFailure]`.
- Produces: `verify_determinism(name: str, *, fps: int, seed: int) -> list[VerificationFailure]`.
- Produces: `calculate_metrics(result: SimulationResult) -> dict[str, object]`.
- Produces: `build_summary(results: list[SimulationResult], deterministic: bool) -> str`.
- Later tasks consume `summary`, `failures`, and primary trace.

- [ ] **Step 1: Write failing invariant and determinism tests**

Add tests:

```python
from tools.render_behavior_preview import (
    SimulationResult,
    VerificationFailure,
    build_summary,
    calculate_metrics,
    check_invariants,
    verify_determinism,
)


def test_same_scenario_and_seed_are_deterministic(self) -> None:
    failures = verify_determinism("conversational_cycle", fps=30, seed=1)
    self.assertEqual(failures, [])


def test_global_invariant_reports_exact_bad_frame(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    result.trace[10]["gaze_y"] = -1.25

    failures = check_invariants(result)

    failure = next(item for item in failures if item.invariant == "gaze bounds")
    self.assertEqual(failure.frame, 10)
    self.assertEqual(failure.phase, result.trace[10]["phase"])
    self.assertEqual(failure.observed["gaze_y"], -1.25)


def test_metrics_are_calculated_from_trace_not_constants(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    metrics = calculate_metrics(result)
    speaking_rows = [row for row in result.trace if row["speaking"]]

    self.assertEqual(metrics["speaking_frames"], len(speaking_rows))
    self.assertGreaterEqual(metrics["person_directed_percent"], 0.0)
    self.assertLessEqual(metrics["person_directed_percent"], 100.0)
```

Add summary failure coverage:

```python
def test_summary_names_scenario_frame_time_and_values_on_failure(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    result.failures.append(VerificationFailure(
        scenario=result.scenario,
        phase="thinking",
        frame=143,
        time_seconds=143 / 30.0,
        invariant="gaze bounds",
        observed={"gaze_y": -1.083},
    ))

    summary = build_summary([result], deterministic=True)
    self.assertIn("conversational_cycle", summary)
    self.assertIn("frame 143", summary)
    self.assertIn("gaze bounds", summary)
    self.assertIn("-1.083", summary)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_behavior_verification -v
```

Expected: failures because invariant/metric/summary functions do not exist.

- [ ] **Step 3: Implement finite/global/geometry invariants**

In `tools/render_behavior_preview.py`, define explicit geometry safety margin:

```python
_PANEL_MIN = 1.0
_PANEL_MAX = 63.0
```

Add helpers:

```python
def _finite_numbers(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_numbers(item) for item in value)
    return True


def _eye_bounds(row: dict[str, object], side: str) -> tuple[float, float, float, float]:
    prefix = "left" if side == "l" else "right"
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
```

For every row, hard-fail when:

```python
not _finite_numbers(row)
not (-1.0 <= gaze_x <= 1.0 and -1.0 <= gaze_y <= 1.0)
not (0.0 <= performance_intensity <= 1.0)
left/right width <= 0.0
left/right height < 2.0
any eye bound < _PANEL_MIN or > _PANEL_MAX
```

Also verify each native frame is `(64, 64, 3)` and `np.uint8`.

Build failures through one helper:

```python
def _failure(result: SimulationResult, row: dict[str, object], invariant: str, **observed: object) -> VerificationFailure:
    return VerificationFailure(
        scenario=result.scenario,
        phase=str(row["phase"]),
        frame=int(row["frame"]),
        time_seconds=float(row["time_seconds"]),
        invariant=invariant,
        observed=observed,
    )
```

- [ ] **Step 4: Implement interaction and performance invariants**

Use the first `0.25 s` of each phase as a transition allowance:

```python
_transition_frames = max(1, int(round(0.25 * result.fps)))
```

Group rows by contiguous `phase` name.

Listening stable rows:

```python
interaction_mode == "listening"
fixation_x/fixation_y remain equal to the first stable listening row
if person_x > 0.5, gaze_x > -0.05
if person_x < 0.5, gaze_x < 0.05
face_offset_x has the same horizontal sign as person_x - 0.5
```

Thinking stable rows:

```python
interaction_mode == "thinking"
gaze_y < 0.0
fixation remains constant through the phase
```

Speaking stable rows:

```python
interaction_mode == "speaking"
```

For non-break speaking rows with a person present, count person-directed rows using a general direction dot product:

```python
person_dx = float(row["person_x"]) - 0.5
person_dy = float(row["person_y"]) - 0.5
person_directed = float(row["gaze_x"]) * person_dx + float(row["gaze_y"]) * person_dy > 0.0
```

Hard-fail if fewer than `70%` of stable, non-break speaking rows are person-directed. This threshold encodes only the spec word "most" and deliberately leaves room for drift/easing; metrics will report the exact percentage.

Measure contiguous `speaking_break_active` runs. Import the production range instead of duplicating it:

```python
from output.animator import _SPEAK_BREAK_LENGTH
```

For each observed completed break, require duration within:

```python
low = _SPEAK_BREAK_LENGTH[0] - 1.0 / result.fps
high = _SPEAK_BREAK_LENGTH[1] + 1.0 / result.fps
```

Do not require any break to occur.

Performance target invariants use `performance_target`, not eased `performance_current`:

```python
if performance_expression == "neutral":
    shape fields == 0.0
    hold_scale/ease_scale/track_bias_scale/speaking_break_scale == 1.0
    gaze_y_bias == 0.0
```

Unknown performance fallback is covered with a focused test using a fresh animator/state:

```python
def test_unknown_performance_expression_targets_neutral(self) -> None:
    moods, definitions = _load_runtime_definitions()
    state = State(performance=PerformanceCue("not-real", 1.0))
    animator = FaceAnimator(moods, definitions, seed=1)
    animator.tick(state, 1 / 30)
    target = animator.debug_snapshot()["performance_target"]
    self.assertEqual(target["l_h"], 0.0)
    self.assertEqual(target["hold_scale"], 1.0)
```

Add a focused production-composition test that two fresh animators with the same mood but neutral/playful performance keep identical rendered color after the same number of ticks.

- [ ] **Step 5: Implement determinism verification**

```python
def verify_determinism(name: str, *, fps: int, seed: int) -> list[VerificationFailure]:
    first = simulate_scenario(name, fps=fps, seed=seed)
    second = simulate_scenario(name, fps=fps, seed=seed)
    if first.frame_hashes == second.frame_hashes and first.trace == second.trace:
        return []
    row = first.trace[0] if first.trace else {"phase": "<none>", "frame": 0, "time_seconds": 0.0}
    return [_failure(
        first,
        row,
        "determinism",
        trace_equal=first.trace == second.trace,
        frame_hashes_equal=first.frame_hashes == second.frame_hashes,
    )]
```

Exact equality is appropriate because both runs execute identical floating-point operations in the same process with the same seed. Do not round trace values merely to force determinism.

- [ ] **Step 6: Implement metrics from trace data**

`calculate_metrics(result)` returns at least:

```python
{
    "total_frames": len(result.trace),
    "speaking_frames": ..., 
    "person_directed_percent": ..., 
    "speaking_break_count": ..., 
    "average_break_seconds": ..., 
    "max_break_seconds": ..., 
    "peak_face_offset": ..., 
    "max_frame_gaze_delta": ..., 
    "max_frame_face_delta": ..., 
    "performance_eye_deltas": {...},
}
```

Use `math.hypot` for vector magnitudes.

For `performance_eye_deltas`, compare each expression's rows against the most recent stable neutral-speaking baseline shape in the same conversational scenario. Report max absolute left/right height delta and max left/right slant delta. If a scenario has no neutral baseline, return an empty mapping instead of inventing values.

- [ ] **Step 7: Implement generated summary text**

`build_summary(results, deterministic)` must derive all counts/metrics from passed results. Structure:

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
  average break duration: <seconds> s
  peak face offset: <pixels> px

Performance
  <expression> max eye-height delta: L <value> / R <value> px
```

Append one failure block per failure:

```text
FAIL <scenario> frame <frame> (<time_seconds> s)
Invariant: <invariant>
Observed: <json-encoded observed dict>
Mode: <interaction mode when available>
Performance: <performance expression when available>
```

Do not print example constants when data is absent; use `n/a`.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_behavior_verification tests.test_animator -v
```

Expected: all verification and animator tests pass.

- [ ] **Step 9: Commit Task 3**

```powershell
git add tools/render_behavior_preview.py tests/test_behavior_verification.py tests/test_animator.py
git commit -m "test: add behavior invariants and metrics"
```

---

### Task 4: Artifact Output, GIF Preview, and CLI Exit Contract

**Files:**
- Modify: `tools/render_behavior_preview.py`
- Modify: `tests/test_behavior_verification.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `write_trace(result, output_dir) -> Path`.
- Produces: `write_preview(result, output_dir, *, sample_every: int = 2, scale: int = 6) -> Path`.
- Produces: `run_verification(...) -> tuple[int, str]`.
- CLI: `python tools/render_behavior_preview.py [--scenario NAME] [--seed N] [--output PATH] [--no-gif]`.

- [ ] **Step 1: Write failing artifact/CLI tests**

Add imports:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from tools.render_behavior_preview import run_verification, write_preview, write_trace
```

Add tests:

```python
def test_write_trace_uses_schema_v1_and_primary_native_trace(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    with tempfile.TemporaryDirectory() as directory:
        path = write_trace(result, Path(directory))
        payload = json.loads(path.read_text(encoding="utf-8"))

    self.assertEqual(payload["schema_version"], 1)
    self.assertEqual(payload["fps"], 30)
    self.assertEqual(payload["seed"], 1)
    self.assertEqual(payload["scenario"], "conversational_cycle")
    self.assertEqual(payload["frames"], result.trace)


def test_preview_is_encoded_from_real_simulation_frames(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    first_pixel = tuple(int(v) for v in result.frames[0][0, 0])
    with tempfile.TemporaryDirectory() as directory:
        path = write_preview(result, Path(directory), sample_every=30, scale=2)
        image = Image.open(path)
        image.seek(0)
        rendered = np.asarray(image.convert("RGB"))

    # The label strip is above the scaled native frame; native pixel (0,0)
    # starts at y=24 in the encoded preview and is nearest-neighbor doubled.
    self.assertEqual(tuple(int(v) for v in rendered[24, 0]), first_pixel)


def test_run_verification_writes_all_primary_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
        code, summary = run_verification(output=Path(directory), include_gif=True)
        names = {path.name for path in Path(directory).iterdir()}

    self.assertEqual(code, 0)
    self.assertIn("preview.gif", names)
    self.assertIn("trace.json", names)
    self.assertIn("summary.txt", names)
    self.assertIn("Vess behavior verification", summary)
```

Add failure exit-code test by patching `check_invariants` to return one `VerificationFailure`; expect `run_verification` code `1`. Patch `_load_runtime_definitions` to raise `ValueError`; expect code `2` and a concise configuration-error summary.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

Expected: artifact functions/CLI orchestration missing.

- [ ] **Step 3: Implement `trace.json` writing**

```python
def write_trace(result: SimulationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trace.json"
    payload = {
        "schema_version": 1,
        "fps": result.fps,
        "seed": result.seed,
        "scenario": result.scenario,
        "frames": result.trace,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
```

- [ ] **Step 4: Implement GIF encoding from real frames only**

Import Pillow only inside the function:

```python
def write_preview(
    result: SimulationResult,
    output_dir: Path,
    *,
    sample_every: int = 2,
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

Pillow only presents frames. It must never call Vess geometry functions or redraw eyes.

- [ ] **Step 5: Implement verification orchestration**

`run_verification` runs all three scenarios:

```python
DEFAULT_SCENARIOS = (
    "conversational_cycle",
    "priority_conflicts",
    "geometry_stress",
)
```

For each scenario:

```python
result = simulate_scenario(name, fps=fps, seed=seed)
result.failures.extend(check_invariants(result))
```

Run determinism on `conversational_cycle` with the same seed/fps and include its failure in the primary result.

Always write `summary.txt`, even on invariant failure, so CI artifacts explain the problem. Write `trace.json` for `conversational_cycle`. Write `preview.gif` for `conversational_cycle` unless `--no-gif`.

Return:

```python
code = 1 if any(result.failures for result in results) else 0
return code, summary
```

Catch configuration/harness exceptions at the top of `run_verification`, create output directory, write:

```text
Vess behavior verification

HARNESS ERROR: <exception text>
```

and return `(2, summary)`.

- [ ] **Step 6: Implement CLI**

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render deterministic Vess behavior verification")
    parser.add_argument("--scenario", choices=("all", "conversational_cycle", "priority_conflicts", "geometry_stress"), default="all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "behavior-verification")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected = DEFAULT_SCENARIOS if args.scenario == "all" else (args.scenario,)
    code, summary = run_verification(
        scenarios=selected,
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

Do not expose animator physics/tuning flags.

- [ ] **Step 7: Ignore generated artifacts**

Append to `.gitignore`:

```text
artifacts/behavior-verification/
```

- [ ] **Step 8: Run focused tests and a local harness execution**

Run:

```powershell
python -m unittest tests.test_behavior_verification -v
python tools/render_behavior_preview.py
```

Expected:

```text
exit code 0
artifacts/behavior-verification/preview.gif exists
artifacts/behavior-verification/trace.json exists
artifacts/behavior-verification/summary.txt exists
summary reports 3/3 scenarios PASS
```

- [ ] **Step 9: Commit Task 4**

```powershell
git add tools/render_behavior_preview.py tests/test_behavior_verification.py .gitignore
git commit -m "feat: emit mobile behavior verification artifacts"
```

---

### Task 5: Lightweight CI Dependencies and GitHub Actions Publication

**Files:**
- Create: `requirements-ci.txt`
- Create: `.github/workflows/verify.yml`
- Modify: `tests/test_behavior_verification.py`

**Interfaces:**
- CI job `unit-tests` runs the repository suite.
- CI job `behavior-preview` depends on `unit-tests` and uploads `vess-behavior-verification`.
- No production interface changes.

- [ ] **Step 1: Add a test proving the verification module imports without heavyweight runtime packages**

Add to `tests/test_behavior_verification.py`:

```python
def test_verification_import_does_not_require_heavy_runtime_packages(self) -> None:
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
import tools.render_behavior_preview
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)
```

Import `ROOT` from the runner or define the repo root in the test file with `Path(__file__).resolve().parents[1]`.

- [ ] **Step 2: Run the new import test and verify current behavior**

```powershell
python -m unittest tests.test_behavior_verification.BehaviorVerificationTests.test_verification_import_does_not_require_heavy_runtime_packages -v
```

Expected: PASS if Tasks 1-4 respected dependency boundaries. If it fails, fix imports rather than adding blocked packages to CI.

- [ ] **Step 3: Create lightweight CI requirements**

Create `requirements-ci.txt` exactly:

```text
numpy
opencv-python-headless
fastapi
uvicorn
httpx
pillow
```

Do not include `ultralytics`, `faster-whisper`, `sounddevice`, `kokoro`, or `ollama`.

- [ ] **Step 4: Create the two-job GitHub Actions workflow**

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
          cache: pip

      - name: Install preview dependencies
        run: python -m pip install --upgrade pip && pip install numpy pillow

      - name: Run behavior verification
        run: python tools/render_behavior_preview.py

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
```

Important ordering: the preview command may return nonzero, but the summary/artifact steps still need to run. GitHub normally stops later steps after a failed step, so change the preview step to capture and re-emit its exit code only after publication:

```yaml
      - name: Run behavior verification
        id: behavior
        shell: bash
        run: |
          set +e
          python tools/render_behavior_preview.py
          code=$?
          echo "exit_code=$code" >> "$GITHUB_OUTPUT"
          exit 0
```

Then after summary + upload add:

```yaml
      - name: Fail job if verification failed
        if: always()
        shell: bash
        run: exit "${{ steps.behavior.outputs.exit_code }}"
```

Use this final form in the committed workflow so failed verification still leaves inspectable mobile artifacts.

- [ ] **Step 5: Run the complete repository suite locally**

```powershell
python -m unittest discover -s tests -v
```

Expected: `0 failures`, `0 errors`.

- [ ] **Step 6: Run the behavior harness one final time**

```powershell
python tools/render_behavior_preview.py
```

Expected exit `0` and all three primary artifacts.

- [ ] **Step 7: Commit Task 5**

```powershell
git add requirements-ci.txt .github/workflows/verify.yml tests/test_behavior_verification.py
git commit -m "ci: add remote Vess behavior verification"
```

---

## Final Verification Checklist

Before marking this implementation ready for review:

- [ ] `python -m unittest discover -s tests -v` reports 0 failures/errors.
- [ ] `python tools/render_behavior_preview.py` exits 0.
- [ ] `preview.gif`, `trace.json`, and `summary.txt` are produced under `artifacts/behavior-verification/`.
- [ ] The GIF uses exact `FaceAnimator.tick` frames and does not redraw eyes independently.
- [ ] `trace.json` schema version is 1 and contains all 30 FPS primary-scenario frames.
- [ ] Traced gaze/offset values are the exact values passed to `face.render`, including drift/bob.
- [ ] Per-eye offset fields exist and are `0.0` until whole-eye translation is implemented.
- [ ] Same scenario + seed produces identical trace and frame hashes.
- [ ] Global, listening, thinking, speaking, performance, and geometry hard invariants execute.
- [ ] Any failure names scenario, phase/frame/time, invariant, and measured values.
- [ ] Summary metrics come from trace data and do not use example constants.
- [ ] The behavior harness imports no Kokoro, Whisper, YOLO, sounddevice, or Ollama package.
- [ ] `requirements-ci.txt` remains lightweight.
- [ ] GitHub Actions has `unit-tests` and dependent `behavior-preview` jobs.
- [ ] A failed behavior run still publishes `summary.txt` and artifacts before the job fails.
- [ ] Uploaded artifact name is exactly `vess-behavior-verification`.
- [ ] No real camera/audio/conversation/database/private machine data is uploaded.
- [ ] Final target-PC acceptance remains required for hardware, latency, and subjective visual quality.
