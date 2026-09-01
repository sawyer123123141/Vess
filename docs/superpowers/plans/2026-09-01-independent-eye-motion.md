# Independent Eye Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add restrained, deterministic independent left/right eye-body translation to Vess while keeping pupil gaze, eye shape, and whole-face motion separate and fully observable through the existing remote verification system.

**Architecture:** Extend the stateless renderer with optional per-eye offsets, extend validated performance config with bounded eye-motion targets, and let `FaceAnimator` own a deterministic entry/settle/release reaction state. The existing verification harness records the exact render-time offsets, checks hard invariants, and produces a real-renderer GIF plus numeric metrics.

**Tech Stack:** Python 3.11, NumPy, Pillow, unittest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-01-independent-eye-motion-design.md`

## Global Constraints

- Pupil gaze, independent eye motion, and whole-face motion remain separate channels.
- Final eye center is `authored center + independent eye offset + whole-face offset`.
- Eye-body tracking of the person is out of scope; person tracking stays pupil/whole-face driven.
- Hard per-eye translation limits are `[-1.5, +1.5]` px on X and Y.
- Neutral performance resolves to zero performance eye offsets and zero reaction strength.
- Interaction-mode contribution is fixed in V1: idle/tracking/listening/speaking are `(0,0)` for both eyes; thinking is left `(0,-0.12)`, right `(0,-0.22)`.
- Performance target is `configured eye offset * cue.intensity`, composed with mode contribution, then clamped.
- Performance reaction triggers only when the performance expression name changes to a non-neutral expression.
- Same-expression intensity changes retarget smoothly without starting a new overshoot.
- Reaction constants: entry leg `0.12 s`, settle leg `0.16 s`, release `0.22 s`, maximum overshoot factor `0.15`.
- Overshoot formula is `settled + (settled - start) * (0.15 * reaction * cue.intensity)`.
- Direct non-neutral cue changes start from the current rendered eye offset and never force an intermediate neutral frame.
- Blinks and gaze breaks do not reset or retrigger independent eye motion.
- Zero eye offsets must produce byte-for-byte renderer-equivalent frames.
- Trace/GIF/invariants must all derive from the same real `FaceAnimator.tick()` frames.
- No audio, LLM, memory, microphone, camera, or TTS production code changes.
- The verification path must remain lightweight and must not import heavyweight model/audio packages.

---

### Task 1: Renderer Support for Independent Eye Offsets

**Files:**
- Modify: `output/face.py`
- Create: `tests/test_face.py`

**Interfaces:**
- Produces: `face.render(..., eye_offsets=((0.0, 0.0), (0.0, 0.0))) -> np.ndarray`
- Semantics: tuple index `0` is left eye, index `1` is right eye.
- Existing callers remain valid because the new argument has a zero default.

- [ ] **Step 1: Write renderer regression tests first**

Create `tests/test_face.py`:

```python
"""Independent eye-offset renderer regressions."""

import unittest

import numpy as np

from output import face


class FaceRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = face.shape_params("normal")
        self.color = (100.0, 180.0, 255.0)

    def test_zero_eye_offsets_are_byte_identical_to_legacy_call(self) -> None:
        legacy = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.25, -0.10),
            (0.4, -0.3),
        )
        explicit = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.25, -0.10),
            (0.4, -0.3),
            eye_offsets=((0.0, 0.0), (0.0, 0.0)),
        )
        np.testing.assert_array_equal(legacy, explicit)

    def test_left_eye_offset_moves_left_eye_without_moving_right_eye(self) -> None:
        baseline = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.0, 0.0),
            eye_offsets=((0.0, 0.0), (0.0, 0.0)),
        )
        shifted = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.0, 0.0),
            eye_offsets=((1.0, 0.0), (0.0, 0.0)),
        )

        left_slice = np.s_[:, :34, :]
        right_slice = np.s_[:, 34:, :]
        self.assertFalse(np.array_equal(baseline[left_slice], shifted[left_slice]))
        np.testing.assert_array_equal(baseline[right_slice], shifted[right_slice])

    def test_gaze_stays_relative_to_translated_eye_body(self) -> None:
        centered = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.8, 0.0),
            eye_offsets=((0.0, 0.0), (0.0, 0.0)),
        )
        translated = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.8, 0.0),
            eye_offsets=((0.75, -0.25), (0.0, 0.0)),
        )
        self.assertFalse(np.array_equal(centered, translated))
```

- [ ] **Step 2: Verify RED through CI/local test**

Run:

```powershell
python -m unittest tests.test_face -v
```

Expected before implementation: calls using `eye_offsets=` fail because `render` does not accept that argument.

- [ ] **Step 3: Extend `_eye` with a local eye offset**

Change the private signature to:

```python
def _eye(
    side: str,
    shape: dict[str, float],
    openness: float,
    gaze: tuple[float, float],
    offset: tuple[float, float],
    eye_offset: tuple[float, float],
) -> np.ndarray:
```

Use the composed center:

```python
center_x = p("cx") + offset[0] + eye_offset[0]
center_y = p("cy") + offset[1] + eye_offset[1]
px = _XX - center_x
tilt = p("slant") * _INNER[side] * (px / half_w)
py = _YY - center_y - tilt
```

Do not modify gaze math. Pupil reach remains relative to the translated eye body through `px/py`.

- [ ] **Step 4: Extend `render` compatibly**

Use:

```python
def render(
    shape: dict[str, float],
    color: tuple[float, float, float],
    brightness: float,
    openness: float,
    gaze: tuple[float, float],
    offset: tuple[float, float] = (0.0, 0.0),
    eye_offsets: tuple[
        tuple[float, float],
        tuple[float, float],
    ] = ((0.0, 0.0), (0.0, 0.0)),
) -> np.ndarray:
```

Then:

```python
cover = np.maximum(
    _eye("l", shape, openness, gaze, offset, eye_offsets[0]),
    _eye("r", shape, openness, gaze, offset, eye_offsets[1]),
)
```

- [ ] **Step 5: Verify GREEN**

```powershell
python -m unittest tests.test_face -v
python -m unittest tests.test_animator -v
```

Expected: all tests pass and old animator callers remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add output/face.py tests/test_face.py
git commit -m "feat: support independent eye offsets in renderer"
```

---

### Task 2: Validate and Configure Performance Eye Motion

**Files:**
- Modify: `performance.py`
- Modify: `performance.json`
- Modify: `tests/test_performance.py`

**Interfaces:**
- Produces validated `entry["eye_motion"]` dictionaries containing exactly:
  `l_x`, `l_y`, `r_x`, `r_y`, `reaction`.
- Offset limits: `[-1.5, 1.5]`.
- Reaction limit: `[0.0, 1.0]`.

- [ ] **Step 1: Add failing validation tests**

Add to `tests/test_performance.py`:

```python
def test_eye_motion_defaults_to_neutral_values(self) -> None:
    definitions = load_performance_definitions({
        "neutral": {"intensity": 0.0},
        "playful": {"intensity": 0.5},
    })
    self.assertEqual(
        definitions["playful"]["eye_motion"],
        {"l_x": 0.0, "l_y": 0.0, "r_x": 0.0, "r_y": 0.0, "reaction": 0.0},
    )


def test_eye_motion_is_clamped_and_nonfinite_values_fall_back(self) -> None:
    definitions = load_performance_definitions({
        "neutral": {"intensity": 0.0},
        "playful": {
            "intensity": 0.65,
            "eye_motion": {
                "l_x": -9,
                "l_y": float("nan"),
                "r_x": 4,
                "r_y": -0.7,
                "reaction": float("inf"),
            },
        },
    })
    eye = definitions["playful"]["eye_motion"]
    self.assertEqual(eye["l_x"], -1.5)
    self.assertEqual(eye["l_y"], 0.0)
    self.assertEqual(eye["r_x"], 1.5)
    self.assertEqual(eye["r_y"], -0.7)
    self.assertEqual(eye["reaction"], 0.0)


def test_neutral_eye_motion_is_forced_to_zero(self) -> None:
    definitions = load_performance_definitions({
        "neutral": {
            "intensity": 0.0,
            "eye_motion": {"l_x": 1.0, "r_y": -1.0, "reaction": 1.0},
        }
    })
    self.assertEqual(
        definitions["neutral"]["eye_motion"],
        {"l_x": 0.0, "l_y": 0.0, "r_x": 0.0, "r_y": 0.0, "reaction": 0.0},
    )
```

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_performance -v
```

Expected: `eye_motion` is missing.

- [ ] **Step 3: Add eye-motion validation constants**

In `performance.py`:

```python
_EYE_MOTION_LIMITS = {
    "l_x": (-1.5, 1.5),
    "l_y": (-1.5, 1.5),
    "r_x": (-1.5, 1.5),
    "r_y": (-1.5, 1.5),
    "reaction": (0.0, 1.0),
}

_EYE_MOTION_DEFAULTS = {
    "l_x": 0.0,
    "l_y": 0.0,
    "r_x": 0.0,
    "r_y": 0.0,
    "reaction": 0.0,
}
```

Inside `load_performance_definitions`, validate `entry.get("eye_motion", {})` through the existing `_number` and `_clamp` helpers. After validation, if `name == "neutral"`, replace it with `dict(_EYE_MOTION_DEFAULTS)` regardless of authored content.

Include the cleaned mapping in every returned definition:

```python
cleaned[name] = {
    "intensity": intensity,
    "shape": shape,
    "eye_motion": eye_motion,
    "movement": movement,
}
```

- [ ] **Step 4: Add conservative initial authored values**

Update `performance.json` with these exact starting values:

```json
"neutral": {
  "eye_motion": {"l_x": 0.0, "l_y": 0.0, "r_x": 0.0, "r_y": 0.0, "reaction": 0.0}
},
"curious": {
  "eye_motion": {"l_x": 0.0, "l_y": -0.28, "r_x": 0.0, "r_y": -0.48, "reaction": 0.45}
},
"amused": {
  "eye_motion": {"l_x": 0.10, "l_y": 0.12, "r_x": -0.06, "r_y": 0.08, "reaction": 0.25}
},
"playful": {
  "eye_motion": {"l_x": -0.18, "l_y": 0.22, "r_x": 0.30, "r_y": -0.62, "reaction": 0.80}
},
"emphatic": {
  "eye_motion": {"l_x": 0.0, "l_y": -0.32, "r_x": 0.0, "r_y": -0.27, "reaction": 0.55}
},
"thoughtful": {
  "eye_motion": {"l_x": 0.0, "l_y": -0.16, "r_x": 0.0, "r_y": -0.24, "reaction": 0.20}
},
"sympathetic": {
  "eye_motion": {"l_x": 0.10, "l_y": 0.18, "r_x": -0.08, "r_y": 0.15, "reaction": 0.15}
},
"uncertain": {
  "eye_motion": {"l_x": 0.0, "l_y": 0.10, "r_x": 0.0, "r_y": -0.46, "reaction": 0.55}
}
```

Preserve each expression's existing `intensity`, `shape`, and `movement` values exactly.

- [ ] **Step 5: Verify GREEN**

```powershell
python -m unittest tests.test_performance tests.test_main -v
```

- [ ] **Step 6: Commit**

```bash
git add performance.py performance.json tests/test_performance.py
git commit -m "feat: validate performance eye motion"
```

---

### Task 3: Animator-Owned Reaction State and Exact Render Diagnostics

**Files:**
- Modify: `output/animator.py`
- Modify: `tests/test_animator.py`

**Interfaces:**
- Produces exact render-time `left_eye_offset` and `right_eye_offset`.
- Adds snapshot fields:
  `left_eye_offset`, `right_eye_offset`, `left_eye_settled_target`, `right_eye_settled_target`, `reaction_phase`, `reaction_elapsed`.
- Calls `face.render(..., eye_offsets=(left_offset, right_offset))`.

- [ ] **Step 1: Add failing animator lifecycle tests**

Add constants at test level if useful:

```python
DT = 1 / 30
```

Add these tests to `tests/test_animator.py`:

```python
def test_thinking_mode_adds_asymmetric_eye_body_lift(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    state = State(thinking=True, performance=PerformanceCue())
    for _ in range(20):
        animator.tick(state, 1 / 30)
    snap = animator.debug_snapshot()
    self.assertAlmostEqual(snap["left_eye_settled_target"][1], -0.12, places=6)
    self.assertAlmostEqual(snap["right_eye_settled_target"][1], -0.22, places=6)


def test_non_neutral_performance_reacts_then_settles(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    state = State(performance=PerformanceCue("playful", 0.65))

    animator.tick(state, 1 / 30)
    first = animator.debug_snapshot()
    self.assertEqual(first["reaction_phase"], "entry")

    for _ in range(12):
        animator.tick(state, 1 / 30)
    settled = animator.debug_snapshot()
    self.assertEqual(settled["reaction_phase"], "hold")
    self.assertAlmostEqual(
        settled["left_eye_offset"][0],
        settled["left_eye_settled_target"][0],
        delta=0.03,
    )


def test_release_to_neutral_has_no_overshoot_and_returns_to_mode_target(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    state = State(performance=PerformanceCue("playful", 0.65))
    for _ in range(12):
        animator.tick(state, 1 / 30)

    state.performance = PerformanceCue()
    previous_distance = None
    for _ in range(8):
        animator.tick(state, 1 / 30)
        snap = animator.debug_snapshot()
        distance = abs(snap["left_eye_offset"][0]) + abs(snap["left_eye_offset"][1])
        if previous_distance is not None:
            self.assertLessEqual(distance, previous_distance + 1e-6)
        previous_distance = distance
    self.assertEqual(animator.debug_snapshot()["reaction_phase"], "hold")


def test_direct_cue_change_starts_from_current_without_neutral_frame(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    state = State(performance=PerformanceCue("playful", 0.65))
    for _ in range(5):
        animator.tick(state, 1 / 30)
    before = animator.debug_snapshot()["left_eye_offset"]

    state.performance = PerformanceCue("emphatic", 0.70)
    animator.tick(state, 1 / 30)
    after = animator.debug_snapshot()

    self.assertEqual(after["reaction_phase"], "entry")
    self.assertNotEqual(after["left_eye_settled_target"], (0.0, 0.0))
    self.assertLess(
        abs(after["left_eye_offset"][0] - before[0])
        + abs(after["left_eye_offset"][1] - before[1]),
        0.5,
    )


def test_same_expression_intensity_change_does_not_restart_reaction(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    state = State(performance=PerformanceCue("playful", 0.65))
    for _ in range(12):
        animator.tick(state, 1 / 30)
    self.assertEqual(animator.debug_snapshot()["reaction_phase"], "hold")

    state.performance = PerformanceCue("playful", 0.30)
    animator.tick(state, 1 / 30)
    self.assertNotEqual(animator.debug_snapshot()["reaction_phase"], "entry")


def test_blink_does_not_reset_eye_motion(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    state = State(performance=PerformanceCue("playful", 0.65))
    for _ in range(12):
        animator.tick(state, 1 / 30)
    before = animator.debug_snapshot()["left_eye_offset"]

    animator.blink_phase = 0.1
    animator.tick(state, 1 / 30)
    after = animator.debug_snapshot()["left_eye_offset"]

    self.assertLess(
        abs(after[0] - before[0]) + abs(after[1] - before[1]),
        0.15,
    )
```

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_animator -v
```

Expected: snapshot eye-motion fields do not exist and the renderer is not receiving independent offsets.

- [ ] **Step 3: Add animator constants and neutral helpers**

In `output/animator.py`:

```python
_EYE_LIMIT = 1.5
_EYE_REACTION_LEG = 0.12
_EYE_SETTLE_LEG = 0.16
_EYE_RELEASE = 0.22
_EYE_MAX_OVERSHOOT = 0.15

_MODE_EYE_OFFSETS = {
    "idle": ((0.0, 0.0), (0.0, 0.0)),
    "tracking": ((0.0, 0.0), (0.0, 0.0)),
    "listening": ((0.0, 0.0), (0.0, 0.0)),
    "thinking": ((0.0, -0.12), (0.0, -0.22)),
    "speaking": ((0.0, 0.0), (0.0, 0.0)),
}
```

Add helpers:

```python
def _clamp_eye(offset: tuple[float, float]) -> tuple[float, float]:
    return (
        _clamp_range(offset[0], -_EYE_LIMIT, _EYE_LIMIT),
        _clamp_range(offset[1], -_EYE_LIMIT, _EYE_LIMIT),
    )


def _lerp2(
    start: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    eased = _smoothstep(t)
    return (
        start[0] + (end[0] - start[0]) * eased,
        start[1] + (end[1] - start[1]) * eased,
    )
```

- [ ] **Step 4: Add animator eye-motion state**

In `__init__` initialize:

```python
self._left_eye_offset = (0.0, 0.0)
self._right_eye_offset = (0.0, 0.0)
self._left_eye_settled = (0.0, 0.0)
self._right_eye_settled = (0.0, 0.0)
self._eye_start_left = (0.0, 0.0)
self._eye_start_right = (0.0, 0.0)
self._eye_overshoot_left = (0.0, 0.0)
self._eye_overshoot_right = (0.0, 0.0)
self._eye_reaction_phase = "hold"
self._eye_reaction_elapsed = 0.0
self._eye_expression = "neutral"
```

- [ ] **Step 5: Calculate settled target from mode + performance**

Add a method with this contract:

```python
def _eye_settled_targets(
    self,
    mode: str,
    cue: PerformanceCue,
) -> tuple[tuple[float, float], tuple[float, float], float]:
```

Implementation semantics:

```python
entry = self._performances.get(cue.expression)
if entry is None:
    entry = self._performances["neutral"]
    cue = PerformanceCue()

eye = dict(entry.get("eye_motion", {}))
intensity = _clamp_range(float(cue.intensity), 0.0, 1.0)
mode_left, mode_right = _MODE_EYE_OFFSETS[mode]
left = _clamp_eye((
    mode_left[0] + float(eye.get("l_x", 0.0)) * intensity,
    mode_left[1] + float(eye.get("l_y", 0.0)) * intensity,
))
right = _clamp_eye((
    mode_right[0] + float(eye.get("r_x", 0.0)) * intensity,
    mode_right[1] + float(eye.get("r_y", 0.0)) * intensity,
))
reaction = _clamp_range(float(eye.get("reaction", 0.0)), 0.0, 1.0)
return left, right, reaction
```

- [ ] **Step 6: Implement deterministic reaction transitions**

Add `_advance_eye_motion(mode, cue, dt)`.

Expression-change rules:

```python
expression_changed = cue.expression != self._eye_expression
if expression_changed:
    self._eye_expression = cue.expression
    self._eye_start_left = self._left_eye_offset
    self._eye_start_right = self._right_eye_offset
    self._eye_reaction_elapsed = 0.0

    if cue.expression == "neutral":
        self._eye_reaction_phase = "release"
    else:
        self._eye_reaction_phase = "entry"
        factor = _EYE_MAX_OVERSHOOT * reaction * intensity
        self._eye_overshoot_left = _clamp_eye((
            left_settled[0] + (left_settled[0] - self._eye_start_left[0]) * factor,
            left_settled[1] + (left_settled[1] - self._eye_start_left[1]) * factor,
        ))
        self._eye_overshoot_right = _clamp_eye((
            right_settled[0] + (right_settled[0] - self._eye_start_right[0]) * factor,
            right_settled[1] + (right_settled[1] - self._eye_start_right[1]) * factor,
        ))
```

Advance phase behavior:

```python
if self._eye_reaction_phase == "entry":
    t = self._eye_reaction_elapsed / _EYE_REACTION_LEG
    if t < 1.0:
        current = lerp(start, overshoot, t)
    else:
        self._eye_reaction_phase = "settle"
        self._eye_reaction_elapsed = 0.0
elif self._eye_reaction_phase == "settle":
    t = self._eye_reaction_elapsed / _EYE_SETTLE_LEG
    current = lerp(overshoot, settled, t)
    if t >= 1.0:
        current = settled
        self._eye_reaction_phase = "hold"
elif self._eye_reaction_phase == "release":
    t = self._eye_reaction_elapsed / _EYE_RELEASE
    current = lerp(start, settled, t)
    if t >= 1.0:
        current = settled
        self._eye_reaction_phase = "hold"
else:
    # Hold or same-expression/mode/intensity retarget.
    # Use exponential easing toward the recomputed settled target without overshoot.
    alpha = 1.0 - math.exp(-dt / 0.10)
    current = current + (settled - current) * alpha
```

Implement left and right symmetrically with `_lerp2`; increment elapsed by `dt` exactly once per tick. Clamp the final current offsets before returning them.

When mode changes during an active cue, only the settled targets change. Do not reset `_eye_reaction_phase` or `_eye_reaction_elapsed`.

- [ ] **Step 7: Wire render and debug snapshot**

In `tick`, after interaction mode is known and before rendering:

```python
left_eye_offset, right_eye_offset = self._advance_eye_motion(
    self._interaction_mode,
    performance,
    dt,
)
```

Pass them to renderer:

```python
return face.render(
    shape,
    color,
    brightness,
    self._openness(),
    gaze,
    offset,
    eye_offsets=(left_eye_offset, right_eye_offset),
)
```

Extend `debug_snapshot()` with copies/tuples:

```python
"left_eye_offset": tuple(self._left_eye_offset),
"right_eye_offset": tuple(self._right_eye_offset),
"left_eye_settled_target": tuple(self._left_eye_settled),
"right_eye_settled_target": tuple(self._right_eye_settled),
"reaction_phase": self._eye_reaction_phase,
"reaction_elapsed": self._eye_reaction_elapsed,
```

- [ ] **Step 8: Verify GREEN**

```powershell
python -m unittest tests.test_animator tests.test_face tests.test_performance -v
```

- [ ] **Step 9: Commit**

```bash
git add output/animator.py tests/test_animator.py
git commit -m "feat: animate independent eye reactions"
```

---

### Task 4: Trace, Invariants, Eye-Reaction Scenario, and Metrics

**Files:**
- Modify: `tools/behavior_scenarios.py`
- Modify: `tools/render_behavior_preview.py`
- Modify: `tests/test_behavior_verification.py`

**Interfaces:**
- Existing trace fields `left_eye_offset_x/y` and `right_eye_offset_x/y` become real render-time values.
- Adds scenario `eye_reaction_cycle`.
- Adds hard eye-offset and lifecycle invariants.
- Extends metrics/summary with measured eye-body motion.

- [ ] **Step 1: Add failing trace tests**

Add to `tests/test_behavior_verification.py`:

```python
def test_trace_records_real_independent_eye_offsets(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    playful = [row for row in result.trace if row["phase"] == "speaking_playful"]
    self.assertTrue(playful)
    self.assertTrue(
        any(
            abs(float(row["left_eye_offset_x"])) > 1e-4
            or abs(float(row["left_eye_offset_y"])) > 1e-4
            or abs(float(row["right_eye_offset_x"])) > 1e-4
            or abs(float(row["right_eye_offset_y"])) > 1e-4
            for row in playful
        )
    )


def test_eye_reaction_cycle_exists_and_is_deterministic(self) -> None:
    result = simulate_scenario("eye_reaction_cycle", fps=30, seed=1)
    self.assertGreater(len(result.trace), 0)
    self.assertEqual(verify_determinism("eye_reaction_cycle", fps=30, seed=1), [])


def test_eye_offset_bounds_failure_reports_exact_frame(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    result.trace[20]["left_eye_offset_y"] = -1.75
    failure = next(
        item for item in check_invariants(result)
        if item.invariant == "eye offset bounds"
    )
    self.assertEqual(failure.frame, 20)
    self.assertEqual(failure.observed["left_eye_offset_y"], -1.75)
```

Add a lifecycle test that finds a `playful -> neutral` portion of `eye_reaction_cycle` and asserts distance to the neutral/mode target decreases after release begins.

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

- [ ] **Step 3: Replace zero trace placeholders with snapshot values**

In `_trace_record`:

```python
left_eye_offset = tuple(snapshot["left_eye_offset"])
right_eye_offset = tuple(snapshot["right_eye_offset"])
```

Then record:

```python
"left_eye_offset_x": float(left_eye_offset[0]),
"left_eye_offset_y": float(left_eye_offset[1]),
"right_eye_offset_x": float(right_eye_offset[0]),
"right_eye_offset_y": float(right_eye_offset[1]),
"left_eye_settled_target_x": float(snapshot["left_eye_settled_target"][0]),
"left_eye_settled_target_y": float(snapshot["left_eye_settled_target"][1]),
"right_eye_settled_target_x": float(snapshot["right_eye_settled_target"][0]),
"right_eye_settled_target_y": float(snapshot["right_eye_settled_target"][1]),
"eye_reaction_phase": str(snapshot["reaction_phase"]),
"eye_reaction_elapsed": float(snapshot["reaction_elapsed"]),
```

- [ ] **Step 4: Add `eye_reaction_cycle` scenario**

In `tools/behavior_scenarios.py`, define exact phases at 30 FPS:

```text
neutral_1       0.6 s
curious         0.8 s
neutral_2       0.6 s
playful         0.8 s
neutral_3       0.6 s
emphatic        0.8 s
thoughtful      0.8 s
sympathetic     0.8 s
uncertain       0.8 s
neutral_4       0.8 s
```

All phases use `speaking=True`, `person_present=True`, `person_pos=(0.80, 0.48)`, and explicit performance cues matching the phase name. Neutral phases use `PerformanceCue()`.

Total duration is `7.6 s`, or `228` frames at 30 FPS.

- [ ] **Step 5: Add hard independent-eye invariants**

For every frame, fail `eye offset bounds` if any of:

```python
left_eye_offset_x
left_eye_offset_y
right_eye_offset_x
right_eye_offset_y
```

falls outside `[-1.5, 1.5]` or is non-finite.

Keep the existing composed geometry check but ensure it already uses per-eye offsets. The helper `_eye_bounds(...)` should now be exercised with real nonzero values.

For rows where `performance_expression == "neutral"`, require the **performance** eye-motion target to be zero while allowing the fixed thinking mode target when mode is thinking. Do not incorrectly require the final settled target to be zero in thinking mode.

- [ ] **Step 6: Add eye-motion metrics**

Extend `calculate_metrics` with:

```text
eye_motion_by_performance
max_frame_left_eye_delta
max_frame_right_eye_delta
```

For each non-neutral performance, report:

```python
{
    "left_peak_x": max(abs(x)),
    "left_peak_y": max(abs(y)),
    "right_peak_x": max(abs(x)),
    "right_peak_y": max(abs(y)),
    "max_asymmetry": max(hypot(left_x-right_x, left_y-right_y)),
}
```

Use actual trace values only.

Extend `build_summary` with a compact block:

```text
Eye motion
  playful L peak x/y: ... / ... px
  playful R peak x/y: ... / ... px
  playful asymmetry: ... px
```

No invented thresholds.

- [ ] **Step 7: Update GIF label with eye offsets**

The label for each sampled frame becomes two lines:

```text
phase | mode | performance
L +0.00,-0.12  R +0.00,-0.22
```

Use the existing real simulation trace row corresponding to that frame. Do not recalculate offsets in the preview writer.

- [ ] **Step 8: Include new scenario in default verification**

Default scenario tuple becomes:

```python
(
    "conversational_cycle",
    "priority_conflicts",
    "geometry_stress",
    "eye_reaction_cycle",
)
```

- [ ] **Step 9: Verify GREEN**

```powershell
python -m unittest tests.test_behavior_verification -v
python tools/render_behavior_preview.py
```

Expected: verifier exit `0`, summary written, and preview/trace still use real animator output.

- [ ] **Step 10: Commit**

```bash
git add tools/behavior_scenarios.py tools/render_behavior_preview.py tests/test_behavior_verification.py
git commit -m "test: verify independent eye motion remotely"
```

---

### Task 5: Full CI Verification and Evidence-Based Visual Tuning

**Files:**
- Modify only if evidence requires tuning: `performance.json`
- Modify tests only when an actual correctness issue is discovered.
- Do not change `.github/workflows/verify.yml` unless the existing workflow fails to include the new scenario/artifacts for a concrete reason.

**Interfaces:**
- Existing `Verify Vess` GitHub Actions workflow is the execution environment.
- Final evidence: full unittest log, behavior summary, `preview.gif`, `trace.json`.

- [ ] **Step 1: Run the complete suite on the branch**

```powershell
python -m unittest discover -s tests -v
```

Expected baseline target: at least the previous 97 tests plus the newly added face/performance/animator/behavior tests, with zero failures/errors.

- [ ] **Step 2: Run behavior verification**

```powershell
python tools/render_behavior_preview.py
```

Expected:

```text
Scenarios: 4/4 PASS
Invalid frames: 0
Geometry: PASS
Determinism: PASS
```

Do not hard-code measured eye-motion numbers in tests.

- [ ] **Step 3: Review the real GIF and numeric summary together**

Use these review rules:

```text
neutral: no independent performance eye motion
thinking: visible but restrained asymmetric lift
curious: right eye lifts more than left
playful: clearest asymmetry of normal cues
emphatic: coordinated, smaller asymmetry than playful
thoughtful: subtle upward bias
sympathetic: subtle down/inward settling
uncertain: visibly asymmetric vertical response
```

Reject/tune values if any cue:

```text
looks like eye-body tracking rather than expression
visibly jumps instead of reacting/easing
looks constantly restless
makes the eyes collide or approach panel edges awkwardly
is visually indistinguishable from neutral when the intended gesture should be readable
```

- [ ] **Step 4: If tuning is needed, change config only first**

Adjust only `performance.json` eye-motion values while preserving the hard limits and directional intent. Re-run the full behavior verifier after every tuning commit.

Do not alter reaction code merely to make one expression stronger if the config can express the desired change.

- [ ] **Step 5: Fresh final verification**

After the final tuning commit, obtain fresh evidence from GitHub Actions or local equivalent:

```text
full unittest discovery: 0 failures/errors
behavior verifier: exit 0
4/4 scenarios PASS
0 invalid frames
geometry PASS
determinism PASS
artifact contains preview.gif, trace.json, summary.txt
```

- [ ] **Step 6: Final diff review**

Compare `design/independent-eye-motion` against `design/remote-behavior-verification` and confirm production changes are limited to:

```text
output/face.py
output/animator.py
performance.py
performance.json
```

plus focused tests/verification/docs. No audio/LLM/TTS/memory/camera/microphone production files should appear.

- [ ] **Step 7: Commit any final config tuning**

If tuning changed `performance.json` after Task 4:

```bash
git add performance.json
git commit -m "tune: refine independent eye motion"
```

If no tuning was required, do not create a no-op commit.

---

## Final Verification Checklist

- [ ] Zero eye offsets are byte-for-byte renderer-compatible.
- [ ] Left/right eye bodies can translate independently.
- [ ] Pupil gaze remains relative to translated eye bodies.
- [ ] Eye offsets never exceed ±1.5 px per axis.
- [ ] Neutral performance eye-motion config is forced to zero.
- [ ] Thinking adds left `(0,-0.12)` and right `(0,-0.22)` mode targets.
- [ ] Reaction only triggers on performance-expression name changes to non-neutral.
- [ ] Same-expression intensity changes do not retrigger overshoot.
- [ ] Entry uses 0.12 s reaction leg + 0.16 s settle leg.
- [ ] Release uses 0.22 s and no overshoot.
- [ ] Direct cue changes do not force neutral.
- [ ] Blink/gaze-break events do not reset eye motion.
- [ ] Debug snapshot reports exact render-time offsets and targets.
- [ ] Existing trace placeholder fields contain real offsets.
- [ ] `eye_reaction_cycle` runs deterministically at 228 frames/30 FPS.
- [ ] Composed geometry checks include real eye offsets.
- [ ] Summary reports measured eye-motion peaks/asymmetry.
- [ ] GIF labels include measured L/R offsets from the same trace.
- [ ] Full unit suite passes.
- [ ] Behavior verifier passes all four scenarios.
- [ ] No heavyweight runtime dependencies enter the preview job.
- [ ] Final diff contains no unrelated production subsystem changes.
- [ ] Target-PC subjective acceptance remains separate from automated correctness.