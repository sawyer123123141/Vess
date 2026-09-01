# Remote Behavior Verification Design

## Goal

Make ordinary Vess development verifiable without requiring the owner to be physically at the target PC for every iteration.

The system must automatically produce three forms of evidence from the **same deterministic behavior run**:

1. machine-checkable behavioral results;
2. a frame-by-frame numeric trace;
3. a human-viewable animation.

It reduces dependence on manual PC testing but does not replace final target-machine acceptance for microphone, camera, speakers, model latency, GPU/CPU contention, or subjective visual quality.

## Architecture

The verification harness exercises production code directly:

```text
BehaviorScenario
    -> real State
    -> real FaceAnimator.tick(...)
    -> real face.render(...)
    -> exact rendered frame
    -> animator diagnostic snapshot from that same tick
    -> trace record
    -> invariants + metrics
    -> preview.gif + trace.json + summary.txt
```

Do not build a second animator, manually redraw Vess in Pillow, or reimplement gaze/eye math inside the harness. The preview must change automatically when production animation changes.

## Scope

This slice adds:

```text
tools/
  behavior_scenarios.py
  render_behavior_preview.py

.github/workflows/
  verify.yml

requirements-ci.txt

tests/
  test_behavior_verification.py
```

Generated output is not committed:

```text
artifacts/behavior-verification/
  preview.gif
  trace.json
  summary.txt
```

Out of scope:

- remote desktop/control of the target PC;
- real camera/microphone/audio testing in CI;
- Ollama, Whisper, Kokoro, or YOLO inference in CI;
- automatic aesthetic scoring;
- GitHub Pages or a permanent dashboard;
- historical analytics storage;
- automatic merge based on verification scores;
- implementing independent whole-eye motion itself.

## Responsibilities

### `tools/behavior_scenarios.py`

Defines deterministic synthetic timelines only. It contains no renderer logic and no invariant logic.

### `tools/render_behavior_preview.py`

Owns:

- scenario execution;
- deterministic frame stepping;
- trace collection;
- invariant checks;
- metric calculations;
- GIF creation from real animator frames;
- `trace.json` and `summary.txt` output;
- CLI exit status.

### `FaceAnimator`

Remains the owner of animation physics. If the harness needs internal numeric values that are not safely exposed, add one non-mutating animator-local diagnostic method:

```python
def debug_snapshot(self) -> dict[str, object]:
    ...
```

The snapshot reports values the animator already computed. It must not reconstruct animation behavior, mutate shared `State`, or become a control API.

## Deterministic simulation

Use the production target rate:

```python
FPS = 30
dt = 1.0 / FPS
```

No `sleep()` is used. Scenario duration is converted to deterministic integer frame counts.

Construct `FaceAnimator` with RNG seed `1` by default. A scenario may explicitly declare a different seed only for a named regression case.

Set scripted `State.mood_until = 0.0` unless testing expiry elsewhere so production `time.time()` mood-expiry behavior cannot make the run nondeterministic.

The runner never monkeypatches animation physics merely to make a test stable.

## Scenario model

A phase contains:

```python
ScenarioPhase(
    name="thinking",
    duration_seconds=1.5,
    state={
        "listening": False,
        "thinking": True,
        "speaking": False,
        "person_present": True,
        "person_pos": (0.80, 0.48),
        "performance": PerformanceCue("thoughtful", 0.55),
    },
)
```

Applying a phase changes only fields declared by that phase. The runner starts each scenario from a fresh `State` and fresh `FaceAnimator`.

### Scenario 1: `conversational_cycle`

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

Use person position `(0.80, 0.48)` throughout the conversational portion. This produces the primary mobile GIF.

### Scenario 2: `priority_conflicts`

Exercise overlapping flags and verify:

```text
listening > thinking > speaking > tracking > idle
```

This may be trace-only.

### Scenario 3: `geometry_stress`

Cycle every configured mood and performance expression while moving the synthetic person through several normalized positions near the useful tracking range.

Purpose: numeric validity and geometry safety, not storytelling.

## Trace schema

`trace.json` is authoritative machine-readable evidence.

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
  "person_x": 0.80,
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

  "left_eye_width": 16.0,
  "left_eye_height": 22.0,
  "right_eye_width": 12.0,
  "right_eye_height": 16.0,
  "left_eye_slant": 0.0,
  "right_eye_slant": 0.0
}
```

Eye coordinates and sizes come from the animator's final composed render parameters for that frame. They are not reconstructed from `moods.json` afterward.

The independent left/right eye offset fields exist from schema version 1. They remain zero until production whole-eye translation exists, which lets that future feature plug into verification without another trace redesign.

## Hard invariants vs metrics

Hard invariant failures fail CI. Informational metrics never fail CI in the first version.

### Global hard invariants

Every frame must satisfy:

- every traced numeric value is finite;
- no NaN or infinity;
- rendered frame shape is exactly `(64, 64, 3)`;
- rendered frame dtype is `uint8`;
- gaze X/Y stay in `[-1.0, 1.0]`;
- performance intensity stays in `[0.0, 1.0]`;
- eye dimensions remain usable;
- composed eye geometry plus whole-face offset remains inside an explicit panel safety boundary;
- the same scenario + seed produces the same trace within strict floating-point tolerance and identical native rendered frames.

Geometry validation must use the **composed** eye centers, dimensions, slants, and offsets. Merely checking static JSON ranges is insufficient.

### Listening invariants

After a short transition allowance:

- mode is `listening`;
- idle fixation does not mutate;
- with a person present, gaze is directionally toward the person's target rather than an unrelated idle target;
- whole-face horizontal offset leans toward the person's horizontal direction.

Do not assert one exact gaze coordinate.

### Thinking invariants

After transition:

- mode is `thinking`;
- gaze Y is negative, meaning upward in Vess coordinates;
- direct person tracking is not used even though the person may remain present;
- idle fixation does not mutate.

### Speaking invariants

After transition:

- mode is `speaking`;
- with a person present, most non-break frames are generally person-directed;
- every observed speaking gaze break stays within the production duration bounds;
- speaking behavior requires no waveform or audio input.

Do **not** require at least one gaze break in every short run. A fixed RNG seed can legitimately produce none in a small interval.

### Performance invariants

Across configured expressions:

- final values remain finite and inside geometry bounds;
- neutral produces zero shape deltas and unit movement scales;
- performance does not alter mood color;
- an unknown performance expression degrades to neutral.

## Informational metrics

Calculate and report useful measurements such as:

- gaze 90%-settle time after an interaction-mode change;
- peak whole-face offset;
- percentage of speaking frames generally directed toward the person;
- gaze-break count;
- average/max gaze-break duration;
- maximum frame-to-frame gaze delta;
- maximum frame-to-frame whole-face delta;
- maximum left/right eye shape delta by performance;
- maximum playful/uncertain asymmetry.

These are review evidence, not pass/fail thresholds initially. A metric becomes contractual only after a real regression establishes a justified bound.

## Summary output

`summary.txt` must be generated entirely from run results.

Example format only:

```text
Vess behavior verification

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
```

The numbers above are examples, never constants.

On failure, identify scenario, phase/frame/time, invariant, and observed values. Example:

```text
FAIL conversational_cycle frame 143 (4.767 s)
Invariant: gaze_y must remain within [-1, 1]
Observed: -1.083
Mode: thinking
Performance: thoughtful
```

## GIF preview

`preview.gif` is built from the exact NumPy frames returned by production `FaceAnimator.tick`.

Do not redraw the eyes with Pillow. Pillow is only the encoder/presentation layer.

Native simulation remains 30 FPS and 64×64. For mobile readability, GIF output may:

- nearest-neighbor upscale the native frame;
- add a label strip outside the 64×64 content;
- include phase, mode, and performance name;
- export every second or third simulation frame to reduce artifact size.

All trace/invariant processing still uses every simulated 30 FPS frame.

## Verification harness tests

`tests/test_behavior_verification.py` covers:

- phase durations convert to expected frame counts;
- phase application changes only declared `State` fields;
- frame/time values are monotonic;
- all trace records are JSON serializable;
- repeated scenario + seed produces equal trace/frame hashes;
- a deliberately invalid synthetic trace produces the correct invariant failure;
- summary numbers are calculated from supplied trace records;
- preview encoding receives frames produced by the real animator path;
- no scenario imports or starts Ollama, Whisper, Kokoro, YOLO, microphone, or audio playback.

Avoid committed golden GIFs/pixel snapshots. Intentional animation tuning should not require updating binary expected files.

## CLI

From repo root:

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

Optional arguments:

```text
--scenario <name>
--seed <integer>
--output <path>
--no-gif
```

Use standard-library `argparse`; no CLI framework.

Exit codes:

```text
0  all hard invariants passed
1  one or more hard invariants failed
2  harness/configuration error
```

Do not expose animator tuning parameters through this CLI.

## CI dependency strategy

The repository's normal `requirements.txt` includes heavy runtime packages such as `ultralytics`, `faster-whisper`, `sounddevice`, and `kokoro`. Those are unnecessary for synthetic face verification and should not be installed just to create a 64×64 preview.

Add `requirements-ci.txt` with the lightweight dependencies needed by the repository tests and preview harness:

```text
numpy
opencv-python-headless
fastapi
uvicorn
httpx
pillow
```

This works with the current import structure because heavyweight libraries are imported lazily in their runtime paths. If implementation reveals a current unit test genuinely requires another lightweight package to import, add that exact package to `requirements-ci.txt`; do not fall back to installing the entire runtime requirements file unless a test actually exercises that runtime package.

The behavior-preview code itself must import only:

- production `State`;
- production `FaceAnimator` / `face.py`;
- `moods.json` / `performance.json`;
- NumPy;
- Pillow for GIF encoding;
- Python standard library.

## GitHub Actions

Add `.github/workflows/verify.yml` triggered by:

```yaml
push:
pull_request:
```

Use Python 3.11 unless the repository later declares a different exact supported version.

Use **two jobs** so the lightweight visual harness remains conceptually separate from broad repository tests.

### Job 1: `unit-tests`

```text
checkout
setup Python 3.11
install requirements-ci.txt
run python -m unittest discover -s tests -v
```

This job proves existing code still passes its ordinary test suite without booting external models/services.

### Job 2: `behavior-preview`

Depends on `unit-tests` succeeding.

```text
checkout
setup Python 3.11
install numpy + pillow (requirements-ci.txt is acceptable if simpler)
run python tools/render_behavior_preview.py
append summary.txt to $GITHUB_STEP_SUMMARY
upload artifacts/behavior-verification
```

Artifact name:

```text
vess-behavior-verification
```

Artifact contents:

```text
preview.gif
trace.json
summary.txt
```

Do not add GitHub Pages or external hosting in version 1.

## Mobile workflow

```text
code pushed
   -> unit-tests job
   -> behavior-preview job
   -> Actions summary shows pass/fail + measured values
   -> GIF + JSON + summary downloadable from phone
```

This is sufficient for normal development iteration while away from the PC.

## Future whole-eye movement

Verification is deliberately designed before independent whole-eye translation is implemented.

The trace distinguishes:

```text
whole-face offset
left-eye offset
right-eye offset
pupil/gaze direction
```

When production whole-eye movement arrives, the existing schema begins reporting nonzero per-eye offsets.

Then add invariants for:

- independent eye translation bounds;
- left/right asymmetry limits;
- reaction duration;
- return to baseline;
- no edge clipping or destructive overlap;
- meaningful distinction between pupil attention motion and whole-eye expressive motion.

No verification architecture rewrite should be needed.

## Security/privacy

CI uses synthetic state only.

Never upload:

- real camera frames;
- microphone recordings;
- user conversations;
- `vess.db` or other memory stores;
- local machine identifiers;
- credentials/secrets.

## Acceptance criteria

Complete when all are true:

1. pushes/PRs trigger GitHub Actions automatically;
2. ordinary unit tests run in CI using `requirements-ci.txt`;
3. `State` + real `FaceAnimator` + real `face.py` run headlessly without external model/hardware services;
4. a fixed scenario + seed is deterministic;
5. `conversational_cycle` generates a native-frame trace and GIF;
6. trace includes interaction, gaze, whole-face, eye shape, performance, and future per-eye-offset fields;
7. listening/thinking/speaking/performance/geometry hard invariants run automatically;
8. failures identify scenario + frame/time + measured values;
9. `summary.txt` contains metrics calculated from the trace;
10. GitHub Actions displays the summary;
11. `preview.gif`, `trace.json`, and `summary.txt` upload as `vess-behavior-verification`;
12. CI starts no Ollama, Whisper, Kokoro, YOLO, camera, microphone, or physical audio/display path;
13. final target-PC/subjective acceptance remains required where relevant.
