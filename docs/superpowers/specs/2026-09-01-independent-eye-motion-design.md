# Independent Eye Motion Design

**Date:** 2026-09-01  
**Branch:** `design/independent-eye-motion`  
**Base:** `design/remote-behavior-verification`

## Goal

Give Vess more personality by allowing the left and right **eye bodies** to move independently, while keeping pupil gaze, eye shape, and whole-face motion as separate systems.

Independent eye motion should feel like a reaction, not constant animation.

## Core Model

Vess has three distinct movement channels:

1. **Pupil gaze** — where Vess is looking.
2. **Independent eye motion** — how Vess is reacting.
3. **Whole-face motion** — Vess's body-language equivalent.

For each eye:

```text
final eye center
= authored eye center
+ independent eye offset
+ whole-face offset
```

Pupil motion remains relative to that final eye body.

## Why a Separate Eye-Offset Layer

The current `l_cx/l_cy/r_cx/r_cy` values are authored geometry. They define the deliberately asymmetric identity of Vess's eyes and are already interpolated by mood/performance shape changes.

Temporary reactions must not rewrite those baseline centers.

Composition therefore becomes:

```text
base eye geometry
  -> mood interpolation
  -> performance shape overlay
  -> independent eye translation
  -> whole-face translation
  -> pupil gaze inside translated eye
  -> renderer
```

## Non-Goals

V1 does **not** add:

- per-eye rotation matrices;
- arbitrary per-eye scale transforms;
- audio-amplitude bouncing;
- movement on every word;
- random independent eye jitter;
- eye-body user tracking;
- new LLM performance labels;
- another model/inference call;
- expressive TTS changes.

Existing shape fields continue handling height, slant, arc, and authored geometry. V1 adds only independent translation plus reaction timing.

## Rendering Interface

Extend `output/face.py` so the renderer can receive independent eye offsets in addition to the existing shared face offset.

Conceptual interface:

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

A named structure may be used if implementation clarity improves, but semantics are fixed:

```text
left center  = l_cx/l_cy + left eye offset  + face offset
right center = r_cx/r_cy + right eye offset + face offset
```

Default zero eye offsets must produce **byte-for-byte identical frames** to the pre-feature renderer for identical inputs.

## Ownership

`FaceAnimator` owns:

```text
left/right current eye offset
left/right settled target
left/right reaction target
reaction phase
reaction elapsed time
active performance expression
```

`State` does not store pixel offsets.

No component other than `FaceAnimator` mutates eye-motion state.

## Hard Motion Limits

Because the display is 64x64:

```text
left/right X offset: [-1.5, +1.5] px
left/right Y offset: [-1.5, +1.5] px
```

These are safety limits, not normal authored values. Most configured offsets should remain below one pixel.

Subpixel translation is intentional because the distance-field renderer already converts fractional movement into smooth edge-brightness changes.

## Performance Configuration

Each performance may optionally define:

```json
"eye_motion": {
  "l_x": -0.15,
  "l_y": 0.25,
  "r_x": 0.30,
  "r_y": -0.65,
  "reaction": 0.8
}
```

The numbers above are illustrative, not final tuning values.

Fields:

```text
l_x, l_y, r_x, r_y: settled translation before cue-intensity scaling
reaction: overshoot strength in [0, 1]
```

Missing `eye_motion` means:

```text
l_x = l_y = r_x = r_y = 0
reaction = 0
```

### Validation

`performance.py` validates:

```text
l_x/l_y/r_x/r_y: clamp to [-1.5, +1.5]
reaction: clamp to [0, 1]
```

Malformed, NaN, or infinite values use neutral defaults rather than preventing startup.

`neutral` must always resolve to zero performance eye offsets and zero reaction strength.

## Cue Intensity

Performance settled target:

```text
configured eye offset * cue.intensity
```

The reaction component is also scaled by cue intensity.

## Interaction-Mode Contribution

V1 deliberately keeps mode-driven eye-body movement minimal so gaze, face lean, and eye translation do not all chase the same target.

The fixed mode contribution is:

```text
idle:      L (0.00,  0.00)   R (0.00,  0.00)
tracking:  L (0.00,  0.00)   R (0.00,  0.00)
listening: L (0.00,  0.00)   R (0.00,  0.00)
thinking:  L (0.00, -0.12)   R (0.00, -0.22)
speaking:  L (0.00,  0.00)   R (0.00,  0.00)
```

Positive Y is down, so thinking raises both eye bodies slightly, with the right eye moving more.

There is **no mode overshoot** in V1.

Why:

- person tracking stays pupil/whole-face driven;
- listening remains visually settled;
- thinking gets a small real eye-body gesture instead of only a pupil glance;
- speech expression comes primarily from performance cues.

Existing mode priority remains:

```text
listening > thinking > speaking > tracking > idle
```

## Settled Target Composition

Per eye:

```text
settled target
= fixed interaction-mode contribution
+ performance eye target
```

Then clamp to ±1.5 px per axis.

Performance does not replace the thinking contribution; the two compose.

## Reaction Trigger Rules

A new reaction begins when the **performance expression name changes** to a non-neutral expression.

Examples:

```text
neutral -> playful      reaction
playful -> emphatic     reaction from current rendered position
playful -> neutral      release only, no overshoot
```

Changing only the intensity while the expression name remains the same retargets smoothly but does **not** start another overshoot.

Blink events, gaze breaks, and interaction-mode changes do not trigger performance overshoot.

## Deterministic Reaction Curve

The reaction is piecewise and deterministic rather than random.

Constants for V1:

```text
reaction leg: 0.12 s
settle leg:   0.16 s
release:      0.22 s
max overshoot factor: 0.15
```

### Entry/cue-change

When a non-neutral performance expression starts:

```text
start = current rendered eye offset
settled = newly composed settled target
delta = settled - start
overshoot = settled + delta * (0.15 * reaction * cue.intensity)
```

Clamp `overshoot` to hard eye-offset limits.

Then:

```text
0.00 -> 0.12 s: smoothstep(start, overshoot)
0.12 -> 0.28 s: smoothstep(overshoot, settled)
```

This means overshoot follows the direction of the actual transition rather than simply scaling coordinates away from zero.

### Hold

After 0.28 s, target the settled offset directly. No random eye-body jitter is added.

### Release to neutral

When performance becomes neutral:

```text
start = current rendered eye offset
settled = current mode contribution
0.00 -> 0.22 s: smoothstep(start, settled)
```

No overshoot is used on release.

### Mode changes during a cue

If the interaction mode changes while a performance cue remains active, recompute the settled target and ease toward it without restarting the performance reaction.

### Direct cue-to-cue transitions

A cue change such as `playful -> emphatic` starts from the **current rendered eye position** and heads directly toward the new cue's overshoot/settled target. There is no forced neutral frame.

## Initial Performance Intent

Exact config values are tuning data, but directional relationships are fixed.

### Neutral

```text
performance contribution: L (0,0), R (0,0)
```

### Curious

- both eyes lift;
- right lifts more than left;
- low-to-medium reaction.

### Amused

- restrained asymmetric settling;
- low reaction.

### Playful

- strongest normal conversational asymmetry;
- eyes move in slightly opposing directions;
- medium/high reaction.

### Emphatic

- more coordinated movement than playful;
- quick but small reaction.

### Thoughtful

- slight upward bias;
- low reaction, letting the fixed thinking-mode lift do most of the work when applicable.

### Sympathetic

- slight down/inward feeling;
- low reaction.

### Uncertain

- asymmetric vertical movement;
- medium reaction.

Final pixel values are tuned from CI-generated GIFs and traces, not treated as architectural constants.

## Blink Interaction

Blink and independent translation are separate.

During a blink:

- eye offsets continue advancing normally;
- offsets are not reset;
- blink start/end does not trigger a new reaction;
- reopening uses the continuously updated offset.

A blink therefore cannot cause the eyes to teleport.

## Gaze Interaction

Independent eye translation must not modify normalized gaze values.

```text
eye offset = body position
gaze = pupil position inside that body
```

With identical gaze inputs, changing eye offsets should move the whole eye body while preserving the pupil's relative gaze direction.

## Whole-Face Interaction

Both eyes still receive the same whole-face offset after their local eye offsets are calculated.

```text
local eye reaction
+ shared face lean/bob
```

This preserves existing face motion.

## Debug Snapshot

`FaceAnimator.debug_snapshot()` adds exact render-time values:

```text
left_eye_offset
right_eye_offset
left_eye_settled_target
right_eye_settled_target
reaction_phase
reaction_elapsed
```

The values called `left_eye_offset/right_eye_offset` must be the clamped values actually sent to the renderer.

## Verification Trace

The existing fields:

```text
left_eye_offset_x
left_eye_offset_y
right_eye_offset_x
right_eye_offset_y
```

currently contain zero placeholders. This feature replaces them with real render-time offsets.

Trace/invariants/GIF all continue deriving from the same `FaceAnimator.tick()` frames.

## Verification Scenarios

### Existing conversational cycle

Configured speaking performances should produce nonzero independent offsets.

### New `eye_reaction_cycle`

Use phases long enough to see reaction and hold:

```text
neutral       0.6 s
curious       0.8 s
neutral       0.6 s
playful       0.8 s
neutral       0.6 s
emphatic      0.8 s
thoughtful    0.8 s
sympathetic   0.8 s
uncertain     0.8 s
neutral       0.6 s
```

Use speaking mode for performance phases so the scenario isolates performance behavior from the thinking-mode contribution.

### Thinking mode case

Add a focused deterministic case proving:

```text
thinking + neutral performance
=> L target (0,-0.12), R target (0,-0.22)
```

### Geometry stress

Existing mood/performance combinations include local eye offsets in final composed panel-bound checks.

## Hard Invariants

CI fails if:

1. eye offsets contain NaN/infinity;
2. any local eye offset leaves `[-1.5,+1.5]` on either axis;
3. final composed eye geometry leaves panel safety bounds;
4. neutral performance target contains nonzero performance eye motion;
5. zero eye offsets change legacy renderer output;
6. same scenario + seed changes trace or frame hashes;
7. release does not converge toward the current mode contribution;
8. direct cue-to-cue transition inserts a forced neutral state;
9. gaze values change merely because eye translation changes;
10. blink start/end resets eye offsets;
11. thinking-mode neutral settled target differs from the specified asymmetric lift;
12. a reaction exceeds its hard local-offset limits.

## Informational Metrics

Per expression, report when available:

```text
left peak X/Y
right peak X/Y
left settled X/Y
right settled X/Y
overshoot distance
entry settle time
release settle time
max frame-to-frame eye translation
left/right asymmetry magnitude
```

These remain review metrics in V1 rather than arbitrary quality thresholds.

## Mobile GIF

The preview label should include:

```text
phase | mode | performance
L x,y | R x,y
```

The GIF is presentation only. CI checks every native 30 FPS frame.

## Error Handling

- missing `eye_motion` -> zero performance eye offsets;
- malformed config -> safe neutral value per field;
- unknown performance -> existing neutral fallback;
- non-finite computed runtime value -> replace with safe zero/mode target before rendering and expose a verification failure in deterministic tests;
- omitted renderer eye offsets -> legacy zero-offset behavior.

## Expected Files

Production:

```text
performance.py
performance.json
output/animator.py
output/face.py
```

Verification/tests:

```text
tests/test_performance.py
tests/test_animator.py
renderer tests (existing location or new focused file)
tests/test_behavior_verification.py
tools/behavior_scenarios.py
tools/render_behavior_preview.py
```

No audio, LLM, memory, microphone, camera, or TTS production code should need modification.

## Compatibility Requirements

- existing `face.render` callers work unchanged via zero-offset defaults;
- existing gaze behavior is unchanged at zero eye offsets;
- old performance config remains valid when `eye_motion` is omitted;
- the verified 97-test baseline remains green after legitimate contract updates;
- behavior CI remains lightweight and imports no heavyweight model/audio stack.

## Acceptance Criteria

Implementation is accepted when:

1. several performance cues visibly move the actual eye bodies;
2. thinking mode produces a small real asymmetric eye-body lift even with neutral performance;
3. pupil gaze, local eye translation, and whole-face motion remain separate layers;
4. zero-offset rendering is byte-compatible with current rendering;
5. config is bounded and malformed values degrade safely;
6. entry reactions follow the defined overshoot curve without snapping;
7. releases return smoothly without overshoot;
8. direct cue-to-cue transitions remain continuous;
9. blinks do not reset/teleport eye offsets;
10. CI trace records the actual per-eye offsets used for rendering;
11. geometry, lifecycle, gaze-independence, and determinism invariants pass;
12. the real CI-generated GIF can be reviewed from mobile;
13. subjective motion tuning remains separate from hard correctness checks.

## Deferred

Not part of this feature:

- richer eyelid/partial-squint system;
- per-eye rotation;
- event-specific micro-reactions unrelated to performance/mode;
- expressive TTS mapping.
