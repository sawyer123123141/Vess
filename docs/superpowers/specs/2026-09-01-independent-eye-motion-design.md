# Independent Eye Motion Design

**Date:** 2026-09-01  
**Branch:** `design/independent-eye-motion`  
**Base:** `design/remote-behavior-verification`

## Goal

Give Vess more personality by allowing the left and right eye bodies to move independently, while preserving the existing separation between pupil gaze, eye shape, and whole-face motion.

The system must make eye movement feel intentional and conversational rather than constantly animated. Independent eye translation is a transient expression layer, not a replacement for gaze tracking or authored eye geometry.

## Core Model

Vess will have three separate movement channels:

1. **Pupil gaze** — where Vess is looking.
2. **Independent eye motion** — how Vess is reacting right now.
3. **Whole-face motion** — Vess's body-language equivalent.

These channels compose but do not overwrite one another.

For each eye:

```text
final eye center
= authored eye center
+ independent eye offset
+ whole-face offset
```

Pupil position remains relative to the final eye body and continues to be driven by gaze.

## Why This Architecture

The existing eye centers (`l_cx`, `l_cy`, `r_cx`, `r_cy`) define authored geometry and the intentionally asymmetric identity of Vess's eyes. Temporary reactions must not mutate those baseline values directly.

Independent offsets therefore become a distinct animation layer:

```text
base eye geometry
  -> mood interpolation
  -> performance shape overlay
  -> independent eye translation
  -> whole-face translation
  -> pupil gaze
  -> renderer
```

This keeps configuration, rendering, diagnostics, and future tuning understandable.

## Non-Goals

This version will not add:

- per-eye rotation matrices;
- arbitrary per-eye scale transforms;
- audio-amplitude-driven bouncing;
- movement on every spoken word;
- continuous random independent jitter;
- eye-body tracking of the user's position;
- new LLM performance labels;
- a new inference/model call;
- expressive TTS changes.

The existing shape system already handles height, slant, arc, and authored geometry. V1 only adds translation and reaction timing.

## Rendering Interface

`output/face.py` currently accepts one whole-face `offset` shared by both eyes. Extend rendering to also accept independent offsets:

```python
render(
    shape,
    color,
    brightness,
    openness,
    gaze,
    offset=(0.0, 0.0),
    eye_offsets=((0.0, 0.0), (0.0, 0.0)),
)
```

A clearer named structure may be used in implementation if it keeps call sites simple, but the semantics are fixed:

```text
left eye = base left center + left eye offset + face offset
right eye = base right center + right eye offset + face offset
```

`_eye(...)` receives its own local eye offset in addition to the whole-face offset.

Default zero eye offsets must produce byte-for-byte equivalent frames to the current renderer for the same inputs.

## Motion State

`FaceAnimator` owns all eye-motion timing. `State` remains unaware of individual pixel offsets.

The animator maintains:

```text
left eye current offset (x, y)
right eye current offset (x, y)
left/right target offsets
reaction phase/progress
previous performance cue identity
```

No other component writes these values.

`FaceAnimator.debug_snapshot()` exposes the exact independent offsets used for the most recent rendered frame.

## Motion Limits

V1 uses conservative hard limits because the face is only 64x64:

```text
left/right X offset: [-1.5, +1.5] px
left/right Y offset: [-1.5, +1.5] px
```

These are safety bounds, not target values. Most authored expressions should use substantially less than one pixel.

Subpixel values are intentional. The renderer's distance-field edge coverage allows subpixel translation to appear as smooth brightness changes rather than whole-pixel jumps.

## Performance Configuration

Extend each performance definition with an optional `eye_motion` block.

Example shape only:

```json
{
  "playful": {
    "intensity": 0.65,
    "shape": {},
    "eye_motion": {
      "l_x": -0.15,
      "l_y": 0.25,
      "r_x": 0.30,
      "r_y": -0.65,
      "reaction": 0.8
    },
    "movement": {}
  }
}
```

The example numbers are illustrative and are not acceptance values.

### Eye-motion fields

```text
l_x, l_y
r_x, r_y
reaction
```

`l_x/l_y/r_x/r_y` are target translation deltas in pixels before cue-intensity scaling.

`reaction` is a normalized reaction-strength value in `[0, 1]`. It controls how much transient overshoot occurs when entering the performance cue. It does not change the final settled target.

Missing `eye_motion` means all-zero offsets and zero reaction strength.

## Validation

`performance.py` validates the new block exactly like the existing shape and movement blocks.

Hard config limits:

```text
l_x, l_y, r_x, r_y: [-1.5, +1.5]
reaction: [0.0, 1.0]
```

Malformed, NaN, or infinite values fall back to neutral defaults rather than breaking startup.

The required `neutral` performance must always resolve to zero independent eye translation and zero reaction strength.

## Intensity Composition

Performance eye targets are scaled by the active cue intensity:

```text
settled_offset = configured_offset * cue.intensity
```

This mirrors the existing performance shape-overlay behavior.

The transient reaction/overshoot is also bounded by cue intensity so a low-intensity performance cannot generate a stronger physical reaction than a high-intensity one.

## Reaction Curve

Eye translation must not snap when a performance starts or ends.

V1 uses a deterministic two-stage response:

### Entry

1. Start from the current independent offsets.
2. Ease quickly toward a small overshoot target.
3. Ease back into the settled performance target.

Conceptually:

```text
baseline/current
 -> brief overshoot
 -> settled performance target
```

The overshoot magnitude is derived from `reaction` and is capped. The implementation must never exceed the hard ±1.5 px eye-offset limits.

### Hold

After settling, the eye bodies remain near the performance target. There is no continuous eye-body jitter.

### Release

When the cue ends or changes to neutral, both independent offsets ease smoothly back toward zero.

### Cue-to-cue transition

When performance changes directly from one non-neutral cue to another, the animator transitions from the current rendered offsets toward the new cue. It must not force an intermediate neutral frame.

## Timing

V1 uses fixed animator-owned timing rather than per-performance arbitrary durations.

Initial design targets:

```text
entry reaction: roughly 100-200 ms
settle: roughly 150-300 ms total from cue start
release: roughly 150-300 ms
```

These are design ranges, not exact frame assertions. Exact constants are implementation details to be tuned using the generated preview and numeric traces.

The verification suite should measure actual settle/release times but initially treat them as review metrics, not hard CI thresholds unless a safety/lifecycle requirement is violated.

## Interaction-Mode Contribution

Independent eye translation is driven primarily by performance cues. Interaction modes may add small deterministic offsets, but they remain subordinate to the distinction between gaze and expression.

### Idle

- Eye bodies normally remain near baseline.
- Rare existing face/pupil activity provides life without independent-eye noise.
- V1 does not add random per-eye idle jitter.

### Tracking

- Eye bodies remain near baseline.
- Pupil gaze and whole-face lean continue to perform tracking.

### Listening

- Eye bodies settle and remain comparatively stable.
- No independent wandering while Vess is listening.
- A tiny shared engagement bias may be used only if visual testing proves useful; V1 does not require one.

### Thinking

- Both eye bodies may shift slightly upward.
- The two offsets should not be mathematically identical, preserving Vess's asymmetry.
- Pupil thinking gaze remains independently controlled.

### Speaking

- Eye bodies are primarily driven by the active performance cue.
- Existing speaking gaze breaks remain pupil/gaze behavior and do not automatically move the eye bodies.

## Initial Expression Intent

The exact pixel constants are tuning data, not architecture, but the intended motion relationships are fixed:

### Neutral

```text
L = (0, 0)
R = (0, 0)
```

### Curious

- both eyes may lift slightly;
- right eye lifts more than left;
- motion reinforces the already-asymmetric curious shape.

### Amused

- restrained inward/downward settling;
- asymmetry should remain subtle.

### Playful

- strongest asymmetry among normal conversational cues;
- one eye may lift/outward while the other shifts slightly opposite;
- brief reaction overshoot is appropriate.

### Emphatic

- both eyes move in a more coordinated direction;
- less asymmetry than playful;
- reaction can be quick but must remain small.

### Thoughtful

- upward bias with slower-feeling settle;
- eyes may differ slightly in vertical offset.

### Sympathetic

- slight down/inward settling;
- low reaction strength.

### Uncertain

- asymmetric vertical change;
- one eye moves more than the other.

These are visual intentions only. Configuration values will be tuned against CI-generated previews rather than guessed once and declared correct.

## Priority and Composition

Independent eye translation does not introduce a new interaction priority stack.

Existing mode priority remains:

```text
listening > thinking > speaking > tracking > idle
```

Composition order for independent offsets:

```text
mode contribution
+ performance contribution
+ transient reaction component
= independent eye offset
```

The final result is hard-clamped before rendering.

Performance remains the main expressive contributor during speech. Mode contribution must be small enough that it cannot erase a performance's intended asymmetry.

## Blink Interaction

Blink openness and independent translation remain separate.

During a blink:

- eye bodies continue following their current offsets;
- translation does not reset;
- a blink must not trigger a new reaction;
- reopening resumes at the same continuous motion state.

This prevents eyes from visibly teleporting when a blink closes.

## Gaze Interaction

Pupil gaze remains relative to the independently translated eye body.

Independent eye movement must not alter the normalized gaze values.

Therefore:

```text
eye offset changes eye-body position
pupil gaze changes pupil position inside that body
```

A test must prove that applying identical gaze with different eye offsets changes the eye body's panel position without changing the gaze input or relative pupil direction.

## Whole-Face Interaction

Whole-face movement is added after independent eye movement.

Both eyes receive the same whole-face offset, while each keeps its own local eye offset.

This preserves the existing face lean/bob behavior while allowing reactions within that moving frame of reference.

## Debug and Trace Integration

The remote verification schema already contains:

```text
left_eye_offset_x
left_eye_offset_y
right_eye_offset_x
right_eye_offset_y
```

They are currently zero placeholders. This feature populates them with the exact render-time values.

`FaceAnimator.debug_snapshot()` will expose:

```text
left_eye_offset
right_eye_offset
left_eye_target
right_eye_target
reaction_active
reaction_progress or equivalent deterministic lifecycle data
```

The trace must record the values used for the actual rendered frame, not pre-clamp or pre-easing targets.

## Verification Scenarios

Extend the deterministic behavior harness rather than building a separate preview path.

### Conversational cycle

Existing performance phases begin reporting nonzero independent offsets where configured.

### Eye-reaction cycle

Add a focused scenario:

```text
neutral
curious
neutral
playful
neutral
emphatic
thoughtful
sympathetic
uncertain
neutral
```

Each phase must be long enough to show entry, settle, and release behavior in the GIF and trace.

### Geometry stress

Existing mood/performance combinations now include independent offsets in final composed geometry checks.

## Hard Invariants

CI fails if any frame violates:

1. left/right eye offsets are finite;
2. each independent eye offset remains in `[-1.5, +1.5]` on both axes;
3. final composed eye geometry remains within panel safety bounds;
4. neutral performance target has zero eye-motion target;
5. zero eye offsets preserve renderer compatibility;
6. deterministic scenario + seed produces identical traces and frame hashes;
7. performance release eventually moves toward neutral rather than becoming stuck;
8. direct cue-to-cue transitions do not require a forced neutral frame;
9. pupil gaze remains numerically independent from eye translation;
10. blink state does not reset independent eye offsets.

## Informational Metrics

The behavior summary should report, per expression when available:

```text
left peak X/Y offset
right peak X/Y offset
left/right settled X/Y offset
peak overshoot percentage or distance
entry settle time
release settle time
maximum frame-to-frame eye translation
left/right asymmetry magnitude
```

These are measured values for review and tuning, not pass/fail thresholds in V1 unless they violate a hard invariant.

## GIF Labels

The mobile preview label should include enough information to interpret motion without becoming a dashboard:

```text
phase | mode | performance
L eye: x,y   R eye: x,y
```

The GIF remains presentation only. Invariants and metrics use every native 30 FPS frame.

## Error Handling

- Missing `eye_motion`: neutral zeros.
- Unknown performance cue: existing neutral fallback behavior.
- Malformed numeric config: safe neutral value for that field.
- Non-finite computed offset: hard invariant failure in verification; runtime clamps/falls back safely rather than emitting non-finite geometry.
- Renderer caller omits independent offsets: zero offsets and legacy-equivalent behavior.

## Files Expected to Change During Implementation

Likely production files:

```text
performance.py
performance.json
output/animator.py
output/face.py
```

Likely verification/test files:

```text
tests/test_performance.py
tests/test_animator.py
tests/test_face.py or equivalent renderer test location
tests/test_behavior_verification.py
tools/behavior_scenarios.py
tools/render_behavior_preview.py
```

No audio, LLM, memory, microphone, camera, or TTS production code should need modification.

## Compatibility Requirements

- Existing callers of `face.render` remain valid through default zero eye offsets.
- Existing gaze behavior remains unchanged when independent offsets are zero.
- Existing performance shape/movement config remains valid when `eye_motion` is omitted.
- Existing 97-test baseline must remain green after updating any tests whose contracts legitimately expand.
- Remote behavior verification remains lightweight and must not import heavyweight audio/model packages.

## Acceptance Criteria

The implementation is acceptable when:

1. independent left/right eye translation is visibly present in at least several configured performance cues;
2. pupil gaze, eye-body translation, and whole-face motion remain distinct layers;
3. neutral/default rendering remains backward-compatible;
4. configured/malformed values are safely validated and clamped;
5. transitions ease and may overshoot without snapping;
6. direct cue-to-cue transitions are continuous;
7. blinks do not reset or teleport eye offsets;
8. CI trace contains real left/right eye-offset values from the rendered frames;
9. geometry/determinism/behavior invariants pass;
10. the generated GIF is the real renderer output and is useful for mobile visual review;
11. subjective tuning remains explicitly separate from hard correctness tests.

## Deferred Follow-Ups

After this feature is verified visually, possible later work includes:

- richer eyelid/partial-squint behavior;
- per-eye rotation if translation + shape prove insufficient;
- event-specific micro-reactions beyond performance cues;
- expressive TTS driven by the same performance state.

Those are not part of this implementation.