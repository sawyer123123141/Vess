# Remote Behavior Verification Design

## Goal

Make ordinary Vess development verifiable without requiring the owner to be physically at the target PC for every iteration.

The verification system should automatically run deterministic behavior simulations using the real runtime face stack, produce machine-checkable numeric traces, generate a human-viewable animation, and summarize the results in GitHub Actions so changes can be reviewed from mobile.

This system does **not** replace final target-PC acceptance for microphone, camera, speakers, local model latency, or subjective visual quality. It reduces how often that manual acceptance is required.

## Why this is a separate subsystem

Vess currently has unit tests for behavior and can be previewed live through its web output, but there is no single repeatable path that answers all three questions:

1. Did the implementation violate any known behavioral rule?
2. What exactly did the face do numerically over time?
3. What did that same run actually look like?

A GIF alone answers only the third question. Unit tests alone answer only selected rules. A useful remote verification path must generate all three forms of evidence from the **same deterministic simulation**.

The verification subsystem therefore owns scripted scenarios, trace capture, invariant checking, preview rendering, human-readable summaries, and CI publication. It must not own or duplicate face behavior itself.

## Core principle

The verification harness must exercise the **real production state and animation code**.

The path is:

```text
BehaviorScenario
    -> real State
    -> real FaceAnimator.tick(...)
    -> real face.render(...)
    -> rendered frame
    -> trace record from the same tick
    -> invariant checks
    -> GIF + JSON trace + text summary
```

Do not create a second simplified animator for testing. Do not manually redraw Vess in the preview tool. If the production face changes, the remote preview should change automatically because it imports the production modules.

## Scope

This first verification slice covers:

- deterministic scripted face/runtime scenarios;
- full frame-by-frame numeric trace capture;
- automatic behavioral invariant checks;
- a GIF preview built from the exact frames returned by `FaceAnimator.tick`;
- a readable summary suitable for GitHub Actions output;
- GitHub Actions execution on pushes and pull requests;
- downloadable CI artifacts containing the preview and raw trace;
- ordinary Python unit-test execution in the same workflow.

It is intentionally small. It does not build a permanent analytics website, remote-control the target PC, exercise real hardware, or attempt computer-vision scoring of whether an expression is aesthetically good.

## Files and responsibilities

The implementation should prefer focused files:

```text
tools/
  behavior_scenarios.py
  render_behavior_preview.py

.github/workflows/
  verify.yml

tests/
  test_behavior_verification.py
```

Generated files are build artifacts, not committed source:

```text
artifacts/behavior-verification/
  preview.gif
  trace.json
  summary.txt
```

`behavior_scenarios.py` defines deterministic input timelines only.

`render_behavior_preview.py` owns simulation execution, trace collection, invariant evaluation, artifact generation, and command-line exit status.

The existing `State`, `FaceAnimator`, `face.py`, `moods.json`, and `performance.json` remain production-owned and are imported rather than copied.

## Deterministic simulation clock

The verification runner must not depend on real wall-clock pacing.

Use a fixed simulated frame rate of **30 FPS**, matching the production render target. Each step advances by:

```python
dt = 1.0 / 30.0
```

A scenario is a sequence of timed state directives. Example:

```python
ScenarioPhase(
    name="thinking",
    duration_seconds=1.5,
    state={
        "listening": False,
        "thinking": True,
        "speaking": False,
        "person_present": True,
        "person_pos": (0.78, 0.46),
        "performance": PerformanceCue("thoughtful", 0.55),
    },
)
```

The runner converts each duration into a deterministic integer frame count. No `sleep()` is used.

The animator is constructed with a fixed RNG seed for verification. The initial seed should be **1** unless a scenario explicitly declares another seed for a specific regression case.

### Wall-clock dependency

The production animator currently may inspect real time for mood expiry. The verification runner must avoid nondeterminism from that path by setting `mood_until = 0.0` for scripted scenarios unless the test specifically targets expiry elsewhere.

Do not monkeypatch production animation physics merely to make tests deterministic.

## Scenario model

Start with a small fixed scenario set that exercises the conversational behavior already present and leaves room for whole-eye movement later.

### Scenario 1: conversational cycle

Timeline:

```text
idle                 1.0 s
person tracking      1.0 s
listening             1.5 s
thinking              1.5 s
speaking neutral      2.0 s
speaking playful      2.0 s
speaking emphatic     2.0 s
idle                  1.0 s
```

Use one stable person position, initially `(0.80, 0.48)`, so state transitions are easy to compare.

This scenario produces the primary mobile preview.

### Scenario 2: priority conflicts

Deliberately enable overlapping flags to verify:

```text
listening > thinking > speaking > tracking > idle
```

This scenario may be trace/test only and does not need its own GIF initially.

### Scenario 3: geometry stress

Cycle all configured moods and performance expressions through deterministic frames while varying person position across safe normalized extremes.

Its purpose is to prove numeric validity and geometry bounds, not visual storytelling.

### Future scenario compatibility

The schema must support future fields for independent whole-eye motion without redesigning the harness. Trace fields for left/right eye translation should exist from the first version and remain `0.0` until production animation exposes nonzero values.

## Trace schema

`trace.json` is the authoritative machine-readable record of the simulated run.

Top level:

```json
{
  "schema_version": 1,
  "fps": 30,
  "seed": 1,
  "scenario": "conversational_cycle",
  "frames": []
}
```

Each frame contains at least:

```json
{
  "frame": 0,
  "time_seconds": 0.0,
  "phase": "listening",

  "interaction_mode": "listening",
  "mood": "neutral",
  "performance_expression": "neutral",
  "performance_intensity": 0.0,

  "listening": true,
  "thinking": false,
  "speaking": false,

  "person_present": true,
  "person_x": 0.8,
  "person_y": 0.48,

  "gaze_x": 0.0,
  "gaze_y": 0.0,
  "face_offset_x": 0.0,
  "face_offset_y": 0.0,
  "blink_openness": 1.0,

  "left_eye_x": 24.0,
  "left_eye_y": 37.0,
  "right_eye_x": 42.0,
  "right_eye_y": 31.0,

  "left_eye_offset_x": 0.0,
  "left_eye_offset_y": 0.0,
  "right_eye_offset_x": 0.0,
  "right_eye_offset_y": 0.0,

  "left_eye_height": 22.0,
  "right_eye_height": 16.0,
  "left_eye_slant": 0.0,
  "right_eye_slant": 0.0
}
```

The exact numeric eye positions/heights come from the animator's composed render parameters for that frame, not from re-reading static eye-shape definitions after the fact.

### Exposing animator diagnostics

The runner may read animator-owned diagnostic/test attributes such as `_interaction_mode`, `_gaze`, `face_offset`, `_last_shape`, `_last_color`, and performance overlay state when those already exist.

If a needed value is not currently exposed, prefer adding a single non-mutating snapshot method to `FaceAnimator` rather than having the verification tool reconstruct internal physics.

Recommended future-safe interface:

```python
def debug_snapshot(self) -> dict[str, object]:
    ...
```

This is animator-local diagnostic state only. It is not shared runtime `State`, persistence, or a production control interface.

## Behavioral invariants

The runner must distinguish **hard invariants** from **informational metrics**.

A hard invariant failure exits nonzero and fails CI. Informational metrics appear in the summary but do not fail the build.

### Global hard invariants

Every frame must satisfy:

- all traced numeric values are finite;
- no NaN or infinity;
- rendered frame shape is exactly `64 x 64 x 3`;
- rendered frame dtype is `uint8`;
- gaze remains inside `[-1.0, 1.0]` on each axis;
- eye heights remain above the renderer's usable minimum;
- eye geometry stays inside defined safe panel bounds;
- performance intensity remains in `[0.0, 1.0]`;
- repeated runs with the same scenario/seed produce identical trace values within a strict numeric tolerance and identical rendered frames.

The geometry bounds should be calculated from composed eye centers, widths/heights, slants, and whole-face offsets with a small explicit safety margin. Do not assert only static parameter ranges while ignoring where the eye actually lands on the panel.

### Listening hard invariants

During stable listening frames after a short transition allowance:

- `interaction_mode == "listening"`;
- idle fixation does not change;
- if a person exists, gaze is directionally closer to the person's gaze target than to unrelated idle fixations;
- the whole-face offset leans toward the person's horizontal direction rather than away from it.

Do not require an exact gaze coordinate. The animator intentionally eases and drifts.

### Thinking hard invariants

During stable thinking frames:

- `interaction_mode == "thinking"`;
- gaze has a negative vertical component, meaning upward in Vess gaze coordinates;
- thinking does not use direct person tracking even when a person remains present;
- idle fixation does not mutate.

### Speaking hard invariants

During stable speaking frames:

- `interaction_mode == "speaking"`;
- when a person exists, most non-break frames look generally toward that person;
- gaze breaks, when they occur, stay within configured duration bounds;
- no waveform or audio input is required to produce speaking eye behavior.

Do **not** require at least one gaze break in every short scenario because deterministic RNG may legitimately produce none in a small window. Instead record break count and validate every observed break.

### Performance hard invariants

For every configured performance expression:

- composed values remain finite and inside geometry bounds;
- neutral leaves zero transient shape deltas and unit movement scales;
- mood color is not modified by performance;
- unknown performance names degrade to neutral behavior.

### Informational metrics

Examples:

- time for gaze to move 90% of the distance toward a new interaction target;
- peak face-offset magnitude;
- percentage of speaking frames generally directed toward the person;
- speaking gaze-break count;
- average and maximum gaze-break duration;
- maximum left/right eye shape delta per performance expression;
- maximum asymmetry introduced by playful/uncertain expressions;
- maximum frame-to-frame gaze delta;
- maximum frame-to-frame whole-face offset delta.

These are useful for review and regression comparison but should not initially fail CI unless a later bug proves a particular threshold must become contractual.

## Human-readable summary

`summary.txt` should be concise enough to read directly in GitHub Actions.

Example shape:

```text
Vess behavior verification

Unit tests: PASS
Scenarios: 3/3 PASS
Invalid frames: 0
Geometry: PASS
Determinism: PASS

Conversational cycle
  listening mode: PASS
  thinking mode: PASS
  speaking mode: PASS
  speaking person-directed frames: 87.4%
  speaking gaze breaks: 2
  average break duration: 0.40 s

Performance
  playful max L/R eye delta: 0.33 / 0.46 px
  emphatic max L/R eye-height delta: 0.84 / 0.70 px

Artifacts
  preview.gif
  trace.json
```

All values in the summary must be calculated from the generated trace. Never hard-code example metrics.

If any hard invariant fails, the summary must name the scenario, frame/time range, invariant, and relevant measured values.

## Visual preview

Generate `preview.gif` from the exact `np.ndarray` frames returned by `FaceAnimator.tick` during the primary conversational-cycle scenario.

Do not recreate eye geometry in Pillow or another renderer.

The GIF may scale each 64×64 frame using nearest-neighbor for phone readability. Scaling is presentation-only; trace and invariant checks use native 64×64 frames.

The preview may include a narrow label area outside the original frame containing:

```text
phase
interaction mode
performance expression
```

Do not draw labels over the actual 64×64 Vess panel content.

Keep GIF generation bounded. The first scenario is about 12 seconds at 30 FPS; the GIF may export fewer display frames, such as every second or third simulation frame, as long as the underlying trace/invariant run still executes all 30 FPS frames.

## Test strategy

### Unit tests for the verification harness

`tests/test_behavior_verification.py` should cover:

- scenario durations convert to the expected deterministic frame counts;
- applying a phase mutates only declared `State` fields;
- generated traces use monotonically increasing frame/time values;
- trace values are JSON serializable and finite;
- the same scenario and seed produce identical trace/frame hashes;
- a deliberately invalid synthetic trace triggers the correct invariant failure;
- summary metrics are calculated from trace data rather than hard-coded;
- preview generation receives frames produced by the real animator path.

### Existing production tests

The GitHub workflow also runs:

```powershell
python -m unittest discover -s tests -v
```

The behavior runner then executes only if the unit suite passes, so a broken production test does not generate misleading green preview artifacts.

## CLI contract

The runner should support a simple command from the repository root:

```powershell
python tools/render_behavior_preview.py
```

Defaults:

```text
scenario = conversational_cycle
seed = 1
fps = 30
output = artifacts/behavior-verification
```

The initial implementation does not need a large CLI framework. `argparse` is enough.

Useful optional arguments:

```text
--scenario <name>
--seed <integer>
--output <path>
--no-gif
```

Do not expose tuning controls for production animator physics through this CLI. Those belong in production config, not in a test harness.

Exit status:

```text
0 = all hard invariants passed
1 = one or more hard invariants failed
2 = harness/configuration error
```

## GitHub Actions

Add one workflow, `.github/workflows/verify.yml`.

Trigger on:

```yaml
push:
pull_request:
```

The initial workflow should target a mainstream supported Python version compatible with Vess. If the repository declares an exact Python version elsewhere, use that value; otherwise prefer Python 3.11 for CI portability rather than pretending CI reproduces the owner's full Windows runtime.

Jobs can remain one job initially:

```text
checkout
setup Python
install test/preview dependencies
run unittest suite
run behavior verification
write summary to GitHub step summary
upload behavior-verification artifacts
```

### Dependency scope

Do not install heavyweight runtime dependencies that the verification path does not need.

The face/animator path needs NumPy and image/GIF support. If importing `main.py` or hardware/model packages would drag in Whisper, Kokoro, OpenCV camera backends, Ollama, or sounddevice, the verification harness must avoid those imports.

Use production `State`, `FaceAnimator`, `face.py`, JSON config, and lightweight dependencies only.

If Pillow is not already a project dependency, it may be added as a test/verification-only dependency for GIF encoding. Do not add a large graphics framework.

## GitHub artifact publication

Always upload the verification output directory as an Actions artifact named:

```text
vess-behavior-verification
```

The artifact contains at minimum:

```text
preview.gif
trace.json
summary.txt
```

The GitHub Actions job summary should contain the full readable `summary.txt` text and clearly identify whether artifacts were produced.

Do not require GitHub Pages, a deployed website, or any external hosting in the first version.

## Mobile review workflow

The intended remote workflow is:

```text
code change pushed
    -> GitHub Actions runs automatically
    -> unit tests + behavior verification execute
    -> job summary gives PASS/FAIL + measured metrics
    -> preview/trace are downloadable artifacts
    -> implementation can be reviewed from mobile
```

This provides enough evidence for ordinary iteration while the owner is away from the target PC.

Final target-machine acceptance still remains necessary when a change touches:

- microphone input;
- real camera/detector behavior;
- physical display hardware;
- speakers/audio device behavior;
- Whisper or local model performance;
- Kokoro latency or audible quality;
- CPU/GPU contention;
- subjective animation quality before merge of a significant visual change.

## Relationship to future whole-eye movement

The trace schema deliberately distinguishes:

```text
whole-face offset
left-eye offset
right-eye offset
pupil/gaze direction
```

At first, left/right independent eye offsets may remain zero or map directly from composed eye centers because production Vess does not yet have an explicit independent-eye translation layer.

When independent whole-eye movement is implemented later, it plugs into the existing trace without changing the verification architecture.

New invariants can then check:

- per-eye translation bounds;
- left/right asymmetry limits;
- reaction motion duration;
- return-to-baseline behavior;
- no collision/overlap or edge clipping;
- intentional difference between pupil attention and whole-eye expression motion.

This is preferable to designing a preview around today's pupil-only behavior and rebuilding it immediately afterward.

## Failure reporting

A failed CI run must make the problem actionable.

Bad:

```text
behavior verification failed
```

Required style:

```text
FAIL conversational_cycle frame 143 (4.767 s)
Invariant: gaze_y must remain within [-1, 1]
Observed: -1.083
Mode: thinking
Performance: thoughtful
```

For range or aggregate failures, report the relevant interval and metric:

```text
FAIL conversational_cycle speaking phase
Invariant: observed gaze break exceeded 0.60 s maximum
Observed: 0.73 s from 7.133 s to 7.867 s
```

## Avoiding brittle tests

Do not assert exact frame pixels or exact gaze coordinates for ordinary behavior.

Exact-output snapshots are allowed only for determinism checks comparing the runner to itself at the same code revision/seed. They should not become committed golden images that must be manually updated whenever an intentional animation change occurs.

Behavior tests should primarily assert relationships and bounds:

```text
up rather than exact y=-0.72
moves toward person rather than exact x=0.63
within geometry bounds rather than exact rectangle coordinates
performance does not modify mood color
```

This keeps the verification system useful during animation tuning instead of turning every intentional visual adjustment into test-maintenance debris.

## Performance and CI budget

The runner should remain cheap enough to run on every push.

At roughly 30 FPS for a few short synthetic scenarios, the face renderer processes only hundreds to low thousands of 64×64 NumPy frames. This should remain tiny compared with LLM/TTS workloads.

No Ollama model, Whisper model, camera detector, Kokoro pipeline, microphone, or audio playback is started in CI.

The behavior runner must fail if it accidentally requires any of those heavyweight runtime services.

## Security and privacy

CI scenarios contain synthetic state only.

Do not upload:

- real microphone recordings;
- camera frames;
- user conversations;
- SQLite memory databases;
- local machine identifiers;
- API keys or secrets.

The generated GIF and trace represent synthetic scripted Vess behavior only.

## Acceptance criteria

The verification subsystem is complete when:

1. a fresh GitHub Actions run automatically executes the full unittest suite;
2. the real `State` + `FaceAnimator` + `face.py` path runs headlessly without camera/audio/model services;
3. the same scenario/seed is deterministic;
4. the conversational-cycle scenario generates a native-frame trace and viewable GIF;
5. the trace contains interaction, gaze, face, eye, performance, and future independent-eye-offset fields;
6. listening/thinking/speaking/geometry/performance hard invariants are evaluated automatically;
7. invariant failures exit nonzero and identify scenario + frame/time + measured values;
8. `summary.txt` reports measured values derived from the trace;
9. the GitHub Actions job summary shows the readable result;
10. `preview.gif`, `trace.json`, and `summary.txt` are uploaded as the `vess-behavior-verification` artifact;
11. no LLM, Whisper, Kokoro, camera, microphone, or physical audio/display service is needed in CI;
12. the system remains explicitly supplemental to final target-PC and subjective visual acceptance.

## Explicitly out of scope

- remote desktop or remote control of the owner's PC;
- running Ollama/Qwen in GitHub Actions;
- running Whisper in GitHub Actions;
- generating or evaluating real speech;
- real camera/person detection in CI;
- automatic aesthetic scoring of the face;
- GitHub Pages or a permanent dashboard;
- storing historical trace analytics across every run;
- automatic merge based on behavior scores;
- replacing ordinary unit tests;
- changing animator behavior merely to make the verification harness easier;
- implementing independent whole-eye motion itself.
