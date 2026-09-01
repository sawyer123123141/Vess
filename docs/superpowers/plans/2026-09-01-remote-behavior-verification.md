# Remote Behavior Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic remote verification for Vess that produces hard behavioral checks, a numeric frame trace, and a mobile-viewable GIF from the same real animation run.

**Architecture:** Add a read-only `FaceAnimator.debug_snapshot()` exposing the exact gaze, whole-face offset, shape, color, and transient state used on the last rendered frame. A headless runner drives scripted `State` phases at 30 FPS, captures the real `FaceAnimator.tick()` frame plus the matching snapshot, evaluates invariants/metrics, and writes `trace.json`, `summary.txt`, and `preview.gif`. GitHub Actions runs the full unit suite first, then a lightweight behavior-preview job.

**Tech Stack:** Python 3.11 CI, unittest, NumPy, Pillow, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-01-remote-behavior-verification-design.md`

## Global Constraints

- Use production `State`, `FaceAnimator.tick`, and `face.render`; never duplicate animator/render logic.
- Simulate at exactly 30 FPS with `dt = 1 / 30`; never `sleep()`.
- Default seed is `1`.
- Use `mood_until = 0.0` in scripted scenarios.
- Trace/invariants/GIF all derive from the same ticks.
- Trace the exact gaze and exact whole-face offset passed to `face.render`, including drift/bob.
- Exit codes: `0` pass, `1` invariant failure, `2` harness/config error.
- Metrics are informational in v1.
- CI must not require/start Ollama, Whisper, Kokoro, YOLO, microphone, camera, speakers, or physical display paths.
- Keep CI dependencies lightweight.
- Artifacts contain only synthetic behavior data.
- Schema v1 includes left/right eye offset fields at `0.0` until future independent whole-eye movement exists.

---

### Task 1: Exact Animator Diagnostics

**Files:**
- Modify: `output/animator.py`
- Modify: `tests/test_animator.py`

**Interfaces:**
- Produce `FaceAnimator.debug_snapshot() -> dict[str, object]` with:
  `interaction_mode`, `render_gaze`, `render_offset`, `blink_openness`, `shape`, `color`, `fixation`, `speaking_break_active`, `speaking_break_remaining`, `performance_current`, `performance_target`.

- [ ] **Step 1: Write failing tests**

```python
def test_debug_snapshot_reports_render_values(self) -> None:
    state = State(
        speaking=True,
        person_present=True,
        person_pos=(0.8, 0.48),
        performance=PerformanceCue("playful", 0.65),
    )
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(state, 1 / 30)
    snap = animator.debug_snapshot()

    self.assertEqual(snap["interaction_mode"], "speaking")
    self.assertEqual(snap["render_gaze"], animator._last_render_gaze)
    self.assertEqual(snap["render_offset"], animator._last_render_offset)
    self.assertEqual(snap["shape"], animator._last_shape)
    self.assertEqual(snap["color"], animator._last_color)


def test_debug_snapshot_copies_mutable_data(self) -> None:
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(State(), 1 / 30)
    snap = animator.debug_snapshot()
    snap["shape"]["l_h"] = 999.0
    snap["performance_current"]["hold_scale"] = 999.0
    fresh = animator.debug_snapshot()
    self.assertNotEqual(fresh["shape"]["l_h"], 999.0)
    self.assertNotEqual(fresh["performance_current"]["hold_scale"], 999.0)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_animator -v
```

- [ ] **Step 3: Store exact render values**

In `__init__`:

```python
self._last_render_gaze = (0.0, 0.0)
self._last_render_offset = (0.0, 0.0)
```

In `tick`, immediately before `face.render(...)`:

```python
self._last_render_gaze = gaze
self._last_render_offset = offset
```

Do not use `_gaze` because it excludes the final drift. Do not use `face_offset` because it excludes the returned bob component.

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

- [ ] **Step 5: Verify GREEN and commit**

```powershell
python -m unittest tests.test_animator -v
git add output/animator.py tests/test_animator.py
git commit -m "test: expose animator render diagnostics"
```

---

### Task 2: Scenarios and Deterministic Native Trace

**Files:**
- Create: `tools/behavior_scenarios.py`
- Create: `tools/render_behavior_preview.py`
- Create: `tests/test_behavior_verification.py`

**Interfaces:**
- `ScenarioPhase(name, duration_seconds, state)`
- `BehaviorScenario(name, phases, seed=1)`
- `phase_frame_count(phase, fps)`
- `get_scenario(name, *, moods, performances)`
- `apply_phase(state, phase)`
- `SimulationResult`
- `simulate_scenario(name, *, fps=30, seed=1)`

- [ ] **Step 1: Write failing tests**

```python
def test_phase_frame_count(self) -> None:
    self.assertEqual(phase_frame_count(ScenarioPhase("thinking", 1.5, {}), 30), 45)


def test_apply_phase_changes_only_declared_fields(self) -> None:
    state = State(mood="curious", thinking=False, person_pos=(0.2, 0.3))
    apply_phase(state, ScenarioPhase("thinking", 1.0, {"thinking": True}))
    self.assertTrue(state.thinking)
    self.assertEqual(state.mood, "curious")
    self.assertEqual(state.person_pos, (0.2, 0.3))


def test_conversational_cycle_is_360_frames(self) -> None:
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


def test_simulation_returns_native_frames_and_trace(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    self.assertEqual(len(result.frames), 360)
    self.assertEqual(len(result.trace), 360)
    self.assertTrue(all(f.shape == (64, 64, 3) for f in result.frames))
    self.assertTrue(all(f.dtype == np.uint8 for f in result.frames))
    self.assertEqual([r["frame"] for r in result.trace], list(range(360)))
    json.dumps(result.trace)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

- [ ] **Step 3: Implement scenario types**

```python
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

Use `_require_cue()` so missing required labels are configuration errors, not silently invented fallback expressions.

- [ ] **Step 4: Define `conversational_cycle` exactly**

```text
idle                  1.0s
tracking              1.0s
listening             1.5s
thinking              1.5s
speaking_neutral      2.0s
speaking_playful      2.0s
speaking_emphatic     2.0s
return_idle           1.0s
```

Use `(0.80, 0.48)` for the person during conversational phases. Explicitly set listening/thinking/speaking flags in every phase. Use required cues `neutral`, `thoughtful`, `playful`, `emphatic`.

- [ ] **Step 5: Define `priority_conflicts` exactly**

Five 0.5s phases:

```text
idle: all flags false, no person
tracking: person present, all flags false
speaking: speaking true
thinking_over_speaking: thinking + speaking true
listening_over_all: listening + thinking + speaking true
```

Use the same `(0.80, 0.48)` person position.

- [ ] **Step 6: Define `geometry_stress`**

Generate one 0.2s phase for every `(mood, performance)` pair. Cycle positions:

```python
((0.10, 0.15), (0.90, 0.15), (0.10, 0.85), (0.90, 0.85))
```

Each phase sets `speaking=True`, `person_present=True`, `mood_until=0.0`.

- [ ] **Step 7: Add direct-script import bootstrap**

The required CLI is `python tools/render_behavior_preview.py`, so before production imports:

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 8: Add runner data types and phase application**

```python
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


def apply_phase(state: State, phase: ScenarioPhase) -> None:
    with state.locked():
        for name, value in phase.state.items():
            if not hasattr(state, name):
                raise ValueError(f"unknown State field in phase {phase.name}: {name}")
            setattr(state, name, value)
```

- [ ] **Step 9: Build trace from `debug_snapshot`**

Every row contains:

```text
frame, time_seconds, phase
interaction_mode, mood, performance_expression, performance_intensity
listening, thinking, speaking
person_present, person_x, person_y
gaze_x, gaze_y
face_offset_x, face_offset_y
blink_openness
left/right eye x, y, width, height, slant
left/right eye offset x/y = 0.0
color_r/g/b
fixation_x/y
speaking_break_active, speaking_break_remaining
performance_current, performance_target
```

Eye shape values come from `snapshot["shape"]`. Gaze/offset come from `render_gaze`/`render_offset`.

- [ ] **Step 10: Implement `simulate_scenario`**

Load `moods.json` and validated `performance.json`, construct fresh `State(mood_until=0.0)` and `FaceAnimator(..., seed=seed)`, run every frame without sleeping, append:

```python
frame.copy()
sha256(frame.tobytes()).hexdigest()
matching trace row
```

- [ ] **Step 11: Verify GREEN and commit**

```powershell
python -m unittest tests.test_behavior_verification -v
git add tools/behavior_scenarios.py tools/render_behavior_preview.py tests/test_behavior_verification.py
git commit -m "feat: add deterministic face behavior simulation"
```

---

### Task 3: Invariants, Determinism, Metrics, Summary

**Files:**
- Modify: `tools/render_behavior_preview.py`
- Modify: `tests/test_behavior_verification.py`
- Modify: `tests/test_animator.py`

**Interfaces:**
- `check_invariants(result) -> list[VerificationFailure]`
- `verify_determinism(name, *, fps, seed) -> list[VerificationFailure]`
- `calculate_metrics(result) -> dict[str, object]`
- `build_summary(results, *, deterministic) -> str`

- [ ] **Step 1: Write failing tests**

```python
def test_same_seed_is_deterministic(self) -> None:
    self.assertEqual(verify_determinism("conversational_cycle", fps=30, seed=1), [])


def test_bad_gaze_reports_exact_frame(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    result.trace[10]["gaze_y"] = -1.25
    failure = next(f for f in check_invariants(result) if f.invariant == "gaze bounds")
    self.assertEqual(failure.frame, 10)
    self.assertEqual(failure.observed["gaze_y"], -1.25)


def test_priority_conflicts_verify_full_order(self) -> None:
    result = simulate_scenario("priority_conflicts", fps=30, seed=1)
    self.assertEqual(check_invariants(result), [])
```

Add metrics test asserting `speaking_frames` equals the count derived from trace rows, and summary failure test asserting scenario/frame/invariant/measured value appear.

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

- [ ] **Step 3: Add global invariants**

Explicit limits:

```python
_PANEL_MIN = 1.0
_PANEL_MAX = 63.0
_TRANSITION_SECONDS = 0.25
```

Fail when any trace numeric is non-finite, frame is not `(64,64,3)` `uint8`, gaze leaves `[-1,1]`, performance intensity leaves `[0,1]`, eye width <= 0, eye height < 2, or composed geometry leaves the panel safety boundary.

Composed eye bounds must use:

```text
eye center + whole-face offset + per-eye offset
width/height
abs(slant) added to vertical extent
```

- [ ] **Step 4: Add interaction-priority invariant**

For stable frames after 0.25s transition:

```python
def _expected_mode(row):
    if row["listening"]: return "listening"
    if row["thinking"]: return "thinking"
    if row["speaking"]: return "speaking"
    if row["person_present"] and row["person_x"] is not None: return "tracking"
    return "idle"
```

Require actual mode == expected mode. This proves:

```text
listening > thinking > speaking > tracking > idle
```

- [ ] **Step 5: Add listening/thinking/speaking invariants**

Listening stable frames:

```text
fixation remains unchanged
person-direction dot product > 0
face_offset_x points toward person's horizontal side
```

Thinking stable frames:

```text
gaze_y < 0
fixation remains unchanged
```

Speaking stable non-break frames with person present:

```python
person_dot = gaze_x * (person_x - 0.5) + gaze_y * (person_y - 0.5)
```

The spec says **most**, so require `person_directed_percent > 50.0`; do not invent a stricter threshold yet.

For observed speaking-break runs, import production `_SPEAK_BREAK_LENGTH` and allow one-frame timing tolerance:

```python
low = _SPEAK_BREAK_LENGTH[0] - 1 / fps
high = _SPEAK_BREAK_LENGTH[1] + 1 / fps
```

Never require a break to occur.

- [ ] **Step 6: Add performance invariants**

Neutral `performance_target` must contain zero shape deltas, unit scale fields, and zero gaze-y bias.

Add exact animator test:

```python
def test_performance_does_not_change_mood_color(self) -> None:
    a = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    b = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    sa = State(mood="curious", performance=PerformanceCue())
    sb = State(mood="curious", performance=PerformanceCue("playful", 0.65))
    for _ in range(30):
        a.tick(sa, 1/30)
        b.tick(sb, 1/30)
    self.assertEqual(a.debug_snapshot()["color"], b.debug_snapshot()["color"])
```

Add verification test that unknown performance expression produces neutral target values.

- [ ] **Step 7: Add determinism**

Run the same scenario twice in the same process. Determinism passes only when both full trace dictionaries and full frame-hash lists are exactly equal. Do not round values to force equality.

- [ ] **Step 8: Add metrics**

Calculate from trace:

```text
total_frames
speaking_frames
person_directed_percent
speaking_break_count
average_break_seconds
max_break_seconds
peak_face_offset
max_frame_gaze_delta
max_frame_face_delta
performance_eye_deltas
```

`performance_eye_deltas` compares stable performance-speaking frames against stable `speaking_neutral` baseline. Missing baseline => `{}`.

- [ ] **Step 9: Add readable summary**

Generate all numbers from results. Include scenario pass count, invalid-frame count, geometry/determinism status, conversational metrics, performance deltas, and one block per failure:

```text
FAIL <scenario> frame <frame> (<time>s)
Invariant: <name>
Observed: <JSON values>
```

Absent metrics display `n/a`, never example constants.

- [ ] **Step 10: Verify GREEN and commit**

```powershell
python -m unittest tests.test_behavior_verification tests.test_animator -v
git add tools/render_behavior_preview.py tests/test_behavior_verification.py tests/test_animator.py
git commit -m "test: add behavior invariants and metrics"
```

---

### Task 4: GIF/JSON/Text Artifacts and CLI

**Files:**
- Modify: `tools/render_behavior_preview.py`
- Modify: `tests/test_behavior_verification.py`
- Modify: `.gitignore`

**Interfaces:**
- `write_trace(result, output_dir) -> Path`
- `write_preview(result, output_dir, *, sample_every=2, scale=6) -> Path`
- `run_verification(...) -> tuple[int, str]`
- CLI: `python tools/render_behavior_preview.py`

- [ ] **Step 1: Write failing artifact tests**

Trace test verifies schema v1/fps/seed/scenario/frames.

Preview test must verify the encoder is given a real simulation frame **before GIF palette quantization**:

```python
def test_preview_encoder_receives_real_animator_frame(self) -> None:
    result = simulate_scenario("conversational_cycle", fps=30, seed=1)
    with tempfile.TemporaryDirectory() as directory:
        with patch("PIL.Image.fromarray", wraps=Image.fromarray) as fromarray:
            write_preview(result, Path(directory), sample_every=30, scale=2)
    np.testing.assert_array_equal(fromarray.call_args_list[0].args[0], result.frames[0])
```

Do **not** assert exact RGB after GIF encoding; palette quantization may legitimately alter colors slightly.

Add `run_verification` success test requiring `preview.gif`, `trace.json`, `summary.txt`; failure test expecting exit `1`; config-error test expecting exit `2` plus `HARNESS ERROR` summary.

- [ ] **Step 2: Verify RED**

```powershell
python -m unittest tests.test_behavior_verification -v
```

- [ ] **Step 3: Write `trace.json`**

Top-level shape exactly:

```json
{"schema_version":1,"fps":30,"seed":1,"scenario":"conversational_cycle","frames":[]}
```

Pretty-print JSON; frame list is the real trace.

- [ ] **Step 4: Write GIF**

Use Pillow only to:

```text
Image.fromarray(real frame)
nearest-neighbor upscale
paste beneath a 24px label strip
label phase | mode | performance
save GIF
```

Use every second simulation frame by default. Underlying trace/invariants still use every 30-FPS frame.

- [ ] **Step 5: Implement `run_verification`**

Default scenarios:

```python
("conversational_cycle", "priority_conflicts", "geometry_stress")
```

For each: simulate, check invariants, verify determinism. Write summary always. Artifact source is conversational cycle when present, otherwise first selected scenario. Return `1` for any hard failure, `0` otherwise. Catch harness/config exceptions, write `summary.txt` with `HARNESS ERROR`, return `2`.

- [ ] **Step 6: Implement CLI**

Arguments:

```text
--scenario all|conversational_cycle|priority_conflicts|geometry_stress
--seed <int>, default 1
--output <path>, default artifacts/behavior-verification
--no-gif
```

No animator tuning arguments.

- [ ] **Step 7: Ignore artifacts**

Append:

```text
artifacts/behavior-verification/
```

to `.gitignore`.

- [ ] **Step 8: Verify and commit**

```powershell
python -m unittest tests.test_behavior_verification -v
python tools/render_behavior_preview.py
git add tools/render_behavior_preview.py tests/test_behavior_verification.py .gitignore
git commit -m "feat: emit mobile behavior verification artifacts"
```

Expected exit 0 and all three artifacts.

---

### Task 5: Lightweight GitHub Actions CI

**Files:**
- Create: `requirements-ci.txt`
- Create: `.github/workflows/verify.yml`
- Modify: `tests/test_behavior_verification.py`

- [ ] **Step 1: Add no-heavy-runtime simulation test**

Run a subprocess that replaces `builtins.__import__` with a guard rejecting roots:

```python
{"kokoro", "faster_whisper", "ultralytics", "sounddevice", "ollama"}
```

Then import `simulate_scenario`, execute `conversational_cycle`, and assert 360 frames. The test must exercise simulation, not merely module import.

- [ ] **Step 2: Verify that test**

If it fails, fix import boundaries. Do not install the blocked package.

- [ ] **Step 3: Create `requirements-ci.txt`**

```text
numpy
opencv-python-headless
fastapi
uvicorn
httpx
pillow
```

If full unittest discovery proves one additional lightweight dependency is genuinely required, add only that exact dependency.

- [ ] **Step 4: Create `.github/workflows/verify.yml`**

Workflow name: `Verify Vess`; triggers: `push`, `pull_request`.

`unit-tests` job:

```text
ubuntu-latest
checkout@v4
setup-python@v5 Python 3.11
pip cache keyed to requirements-ci.txt
pip install -r requirements-ci.txt
python -m unittest discover -s tests -v
```

`behavior-preview` job:

```text
needs: unit-tests
ubuntu-latest
checkout@v4
setup-python@v5 Python 3.11
pip install numpy pillow
run python tools/render_behavior_preview.py while capturing exit code
append summary.txt to $GITHUB_STEP_SUMMARY
upload artifact named vess-behavior-verification
finally exit captured verification code
```

Use `if: always()` on summary/upload/final-fail steps so invariant failures remain inspectable from mobile.

- [ ] **Step 5: Run complete local verification**

```powershell
python -m unittest discover -s tests -v
python tools/render_behavior_preview.py
```

Expected: 0 unit-test failures/errors and harness exit 0.

- [ ] **Step 6: Commit**

```powershell
git add requirements-ci.txt .github/workflows/verify.yml tests/test_behavior_verification.py
git commit -m "ci: add remote Vess behavior verification"
```

---

## Final Verification Checklist

- [ ] Full unittest discovery passes.
- [ ] Behavior CLI exits 0.
- [ ] `preview.gif`, `trace.json`, `summary.txt` exist.
- [ ] Preview is encoded from real animator frames, not a duplicate renderer.
- [ ] Default primary trace contains 360 native 30-FPS frames.
- [ ] Trace gaze/offset match exact render-time values including drift/bob.
- [ ] Left/right eye offset fields exist and are zero for now.
- [ ] Same scenario + seed has identical trace and frame hashes.
- [ ] Mode priority is automatically checked: listening > thinking > speaking > tracking > idle.
- [ ] Global, listening, thinking, speaking, performance, and composed-geometry invariants run.
- [ ] Failure output includes scenario, phase, frame/time, invariant, observed values.
- [ ] Metrics are trace-derived and informational.
- [ ] Verification simulation imports no heavyweight runtime/model/audio package.
- [ ] GitHub Actions runs unit tests before behavior preview.
- [ ] Failed behavior preview still uploads artifact `vess-behavior-verification` before job failure.
- [ ] No private/real sensor/conversation data is uploaded.
- [ ] Target-PC acceptance remains required for hardware/latency/subjective quality.
