# Expressive Performance + Eyes Design

## Goal

Make Vess feel more responsive and alive during conversation by adding a transient performance layer that can drive the eyes now and a better expressive voice later.

This slice deliberately covers **performance plumbing + expressive/interacting eyes**. It does not attempt to make Kokoro emotionally expressive, choose a new TTS model, add barge-in, or redesign long-term mood. Voice-engine research is a later slice that will consume the same performance interface.

## Existing architecture this must preserve

- `PLAN.md` remains authoritative.
- Shared runtime state stays in the single locked `State` object.
- The face render loop stays non-blocking at 30fps.
- Persona remains a stable user-selected style; mood remains slower reactive state that decays.
- The LLM remains `qwen2.5:7b` with the existing 4096 context limit.
- No extra LLM call is added before speech. Performance selection must come from the same streamed response generation.
- The model may only select from fixed performance labels; unknown values fall back safely.
- Existing latest-intent generation invalidation, one-clause-ahead TTS, silence trimming, and natural clause splitting remain intact.
- Spoken/stored assistant text must never contain performance markup.

## Why a separate performance layer

Mood is too coarse for moment-to-moment delivery. Vess can be generally curious while one sentence is thoughtful and the next is playful. Reusing mood for each sentence would produce dramatic, distracting color/shape swaps and would conflate a minutes-long emotional state with a seconds-long conversational beat.

Add a transient layer:

```text
persona      stable behavioral style
mood         slower emotional background
performance  current conversational expression
```

Mood continues to define the base face and movement character. Performance is a temporary overlay applied while a clause is actually being spoken.

## Performance model

Add a frozen value object:

```python
@dataclass(frozen=True)
class PerformanceCue:
    expression: str = "neutral"
    intensity: float = 0.0
```

`State` gains:

```python
performance: PerformanceCue = PerformanceCue()
```

The first fixed expression set is:

- `neutral`
- `curious`
- `amused`
- `playful`
- `emphatic`
- `thoughtful`
- `sympathetic`
- `uncertain`

The model chooses only the expression label. It does **not** invent numeric intensity, speed, pitch, or animation parameters. Each allowed expression has a human-authored `performance.json` entry containing its fixed default intensity and visual modifiers. This preserves the closed-list design principle and gives us one place to tune behavior.

Unknown or missing labels resolve to a neutral performance overlay. The underlying mood still remains visible.

## Response markup and parsing

The same Ollama generation that writes Vess's response also chooses performance. Each sentence is instructed to begin with exactly one reserved marker:

```text
[[vess:thoughtful]] The atmosphere scatters shorter wavelengths more strongly.
[[vess:playful]] It's basically nature's giant blue canvas.
```

The marker is control metadata, never user-visible speech.

### Parser rules

The streamed parser must produce a structured clause:

```python
@dataclass(frozen=True)
class SpeechClause:
    text: str
    performance: PerformanceCue
```

Rules:

1. A recognized marker at the start of a sentence selects that sentence's cue and is stripped before TTS, memory, logs containing assistant prose, and user-visible output.
2. An unknown `[[vess:...]]` marker is stripped and falls back to neutral instead of being spoken.
3. If the model omits a marker at the start of a sentence, that sentence uses neutral performance.
4. If one logical sentence is split by the existing long-clause comma/whitespace fallback, continuation chunks inherit the same cue until a strong sentence boundary (`.`, `!`, `?`, newline) ends that sentence.
5. After a strong sentence boundary, cue inheritance resets. A following untagged sentence therefore becomes neutral rather than accidentally inheriting the previous sentence's expression.
6. Markers may arrive fragmented across Ollama stream chunks. Parsing happens from the accumulated pending text, so chunk boundaries must not affect correctness.

This keeps the current latency-oriented clause splitter while preventing the performance tag from being lost when a long sentence is divided for TTS.

## Prompt behavior

The stable prompt gains a short machine-readable instruction near the response-format rules:

```text
Prefix each sentence with exactly one performance tag from:
neutral, curious, amused, playful, emphatic, thoughtful, sympathetic, uncertain.
Use the tag that best matches how that sentence should be delivered.
Do not explain the tag.
```

The prompt should not contain long descriptions of every label. The model already has ordinary language understanding; adding a large rubric would waste the fixed context window and increase prompt noise.

No second classifier call is added. Performance selection is part of the same token stream as the answer, so the added latency is only the few control tokens emitted before each sentence.

## Playback synchronization

Performance becomes active when the corresponding audio **actually starts playback**, not when the LLM generates the sentence or when Kokoro finishes synthesis.

Flow:

```text
LLM emits SpeechClause
    -> VoiceOutput queues/synthesizes it
    -> prepared waveform waits if necessary
    -> playback begins
    -> State.performance = clause.performance
    -> animator reads it on the next frame
    -> playback completes
    -> State.performance returns to neutral overlay
```

This matters because a clause may spend time in synthesis or the ready queue. Moving the face early would make Vess react to a sentence before it speaks it.

`VoiceOutput.enqueue` therefore gains an optional `performance` argument and carries it with the existing generation metadata through queued, synthesizing, prepared, and playback stages.

A stale clause that is skipped before playback must never modify `State.performance`.

The currently playing clause may continue after a newer generation is requested because barge-in is still out of scope. Its performance cue remains active until that physical playback ends, matching what the user is actually hearing.

Performance is cleared in a `finally` path after playback so audio/device errors cannot leave the state stuck in an expression.

## Eye behavior architecture

`FaceAnimator` remains the owner of time and animation. `State` says what Vess is doing; the animator decides what that state looks like on the current frame.

The renderer (`face.py`) remains stateless. It should not learn about listening, speaking, performance labels, timers, or conversation logic.

### Runtime interaction priority

The animator should resolve conversational behavior in this order:

1. listening
2. thinking
3. speaking
4. person tracking
5. idle behavior

Performance does not replace that priority. It modifies the selected behavior.

Examples:

- listening + playful: still focuses on the person, but can carry a lighter asymmetric eye shape;
- thinking + thoughtful: still looks up/away, but holds longer and settles more;
- speaking + emphatic: keeps stronger direct fixation for the emphasized clause;
- idle + neutral: existing wandering/blinking behavior.

### Listening

While `state.listening` is true:

- gaze strongly prefers `person_pos` when available;
- whole-face movement leans in and settles slightly;
- idle fixation changes are suppressed;
- micro-drift and natural blinking remain so the face does not freeze.

### Thinking

While `state.thinking` is true:

- break direct eye contact;
- bias gaze upward and slightly sideways;
- use a longer hold rather than repeated wandering;
- keep the existing up/away whole-face offset.

Thinking behavior should read as one deliberate fixation, not nervous random motion.

### Speaking

The current animator does not meaningfully use `state.speaking`; this slice adds it.

While speaking:

- mostly track the person when present;
- periodically make brief, deliberate gaze breaks rather than maintaining unbroken eye contact;
- suppress ordinary idle wandering;
- performance overlays may change eye shape, hold duration, gaze bias, and break tendency per clause.

Speaking gaze breaks are timer-driven with bounded randomized intervals using the animator's existing seeded RNG. They are not tied to audio amplitude or individual syllables.

### No waveform-driven eye animation

Do not map audio amplitude directly to pupil position, eye size, or face bob. That produces a visualizer, not a conversational character.

Meaningful state transitions and performance cues drive the eyes. Audio is used only to determine when a clause begins/ends so the cue is synchronized with speech.

## Performance visual overlays

Performance should modify the already-interpolated mood face rather than swap to a completely different mood shape.

Each expression maps to a small set of bounded overlay values. Exact numeric tuning belongs in implementation/tests, but the intended character is fixed here:

- `neutral`: no overlay.
- `curious`: slightly wider eyes, attentive/direct gaze, shorter fixation holds.
- `amused`: mild squint/softening, brief side glance allowed, relaxed return.
- `playful`: small asymmetric eye-height/slant change, livelier gaze break, quick return to the person.
- `emphatic`: slightly wider eyes, stronger direct fixation, reduced gaze breaking for the clause.
- `thoughtful`: slightly softer/narrower eyes, upward gaze bias, longer holds, slower movement.
- `sympathetic`: softened eye height/slant, slight downward bias, calmer movement.
- `uncertain`: subtle asymmetry and sideward bias, but never jittering.

These are modifiers, not full alternate faces. Mood color remains unchanged by performance in this slice.

### Interpolation

Mood continues using the existing mood easing path. Performance overlays use a faster independent easing time constant so a clause-level expression appears quickly without snapping.

`State.performance` returns to neutral when physical playback of a clause ends, but the animator's interpolated overlay is never hard-reset. It simply retargets from whatever values are currently on screen. With the normal near-zero inter-clause playback gap, the next clause's cue arrives before a neutral target can have a meaningful visible effect, so the face transitions directly from one expression toward the next. If there is a real audible gap, the face naturally begins relaxing toward neutral during that gap.

All performance values are clamped before use so a bad config value cannot make the eyes vanish, leave the panel, or vibrate.

## Configuration

Add a small `performance.json` containing the fixed expression names, fixed default intensity for each expression, and visual modifier values.

The file is human-authored configuration. The model never writes it.

The set of valid model labels comes from these configured keys, with `neutral` required as the fallback. Startup validation should reject or fall back from malformed numeric values rather than allowing renderer-breaking values.

Keep this separate from `moods.json` because mood and performance have different lifetimes and semantics. Combining them would make every mood entry carry two unrelated responsibilities.

## Diagnostics

Add enough diagnostics to tune behavior without guessing:

- current `performance_expression`
- current `performance_intensity`
- `performance_started` event with text, expression, and generation id when playback begins
- `performance_ended` event with expression and generation id when playback ends
- parsed performance label on `llm_first_clause`

Do not log raw hidden control markup as assistant text.

The diagnostics event-history UI limitation is a separate operator-quality issue and is not expanded in this slice unless it blocks testing.

## Memory and persistence

Performance is transient runtime state and is not durable memory.

Short-term conversation memory stores only the cleaned assistant text. Performance tags are never stored in `ConversationTurn` or the durable `conversation_turn` event.

Mood classification remains unchanged and still runs after a completed response when no newer request is pending.

## Failure handling

- Missing tag: neutral cue.
- Unknown tag: strip it, neutral cue.
- Malformed marker that does not match the reserved syntax: treat it as ordinary text unless it begins with the reserved `[[vess:` prefix; reserved malformed metadata is stripped rather than spoken.
- Stale generation before playback: skip audio and never activate the cue.
- Playback exception: clear active cue in `finally`.
- Missing/malformed performance config: neutral behavior remains usable; the face must still render.
- Unknown performance in `State`: animator treats it as neutral.

## Testing

Tests should verify behavior rather than exact thread timing wherever possible.

### Parser and prompt tests

- prompt includes the fixed performance-label instruction;
- recognized markers are stripped and mapped to the expected cue;
- unknown reserved markers are stripped and fall back to neutral;
- untagged sentences use neutral;
- a marker fragmented across model chunks parses correctly;
- comma/hard-limit continuation chunks inherit the sentence cue;
- inheritance resets after `.`, `!`, `?`, or newline;
- remembered assistant text contains no markup.

### Voice synchronization tests

- queued/prepared cues do not change `State.performance` before playback;
- playback start activates the correct cue;
- playback completion clears the cue;
- stale prepared audio never activates its cue;
- playback errors clear the cue;
- consecutive clauses can change cues without changing existing generation freshness behavior.

### Animator tests

Use deterministic seeds and inspect rendered/animator parameters rather than brittle full-frame snapshots where possible.

- listening suppresses idle fixation changes and prioritizes person tracking;
- thinking biases gaze up/away;
- speaking uses conversational tracking plus bounded gaze breaks;
- neutral performance leaves the existing mood target unchanged;
- each performance overlay stays inside declared clamps;
- performance changes ease rather than snap;
- mood + performance compose rather than one replacing the other;
- renderer remains stateless and receives only numeric shape/color/gaze/offset inputs.

### Regression tests

Existing tests for:

- mood interpolation;
- person tracking;
- blinking;
- one-clause-ahead synthesis;
- stale TTS invalidation;
- silence trimming;
- natural clause splitting;
- short-term memory;

must continue to pass.

## Acceptance criteria

A live conversation is acceptable for this slice when:

1. Vess visibly focuses while the user is speaking.
2. Thinking has a deliberate look distinct from listening and idle.
3. While speaking, gaze is mostly engaged but not unnervingly fixed.
4. Different sentence-level performance labels produce visible but restrained expression changes.
5. Mood remains the base character and does not visibly flip for every sentence.
6. Eye changes begin with the matching physical audio clause, not early during LLM/TTS preparation.
7. No performance tag is ever spoken or stored in conversation memory.
8. No extra LLM request is introduced and first-response latency does not gain a new model-call stage.
9. The face continues rendering smoothly if performance metadata is missing or invalid.

## Implementation decomposition

This design is one implementation cycle because the performance plumbing and expressive eyes depend directly on each other, but work should land in small testable steps:

1. `PerformanceCue` + config validation + parser/prompt plumbing.
2. Carry cues through `VoiceOutput` and synchronize them to real playback.
3. Add runtime speaking behavior to `FaceAnimator`.
4. Add bounded performance overlays and easing.
5. Add diagnostics and live-tuning checks.

## Explicitly out of scope

- choosing or installing a replacement TTS model;
- trying to fake broad emotion using aggressive Kokoro speed/pitch changes;
- per-word or per-phoneme performance control;
- waveform/amplitude-driven eye animation;
- lip sync or a mouth;
- barge-in / interrupting currently playing audio;
- speaker identification among multiple people;
- new camera/detector architecture;
- durable performance memory;
- long-term owner facts / Step 5B memory work;
- changes to the base LLM or context size.

## Later expressive-voice slice

A later design should benchmark genuinely expressive local TTS options on the target hardware and define a voice adapter that accepts the same `PerformanceCue`.

That future work may map cues to pace, energy, pitch contour, emphasis, hesitation, or model-specific style controls, but those capabilities must be verified against the chosen engine rather than invented in this architecture.

The point of this slice is that the brain and face will already speak a stable performance language. Replacing the speech engine later should not require redesigning conversational expression.