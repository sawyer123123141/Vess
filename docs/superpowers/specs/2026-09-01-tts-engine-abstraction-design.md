# TTS Engine Abstraction Design

## Goal

Introduce a small, explicit text-to-speech engine boundary so Vess can keep Kokoro as a stable baseline while adding Chatterbox Turbo as an optional alternative, without rewriting the existing speech queue, stale-generation handling, one-clause-ahead pipeline, playback timing, or face-performance synchronization.

The architecture must be testable in CI without CUDA, model downloads, audio hardware, or a target PC. Real Chatterbox inference, VRAM use, latency, and listening quality remain target-PC acceptance work.

## Current State

`VoiceOutput` currently owns the important behavior that already works:

- synthesis and playback run on separate worker threads
- exactly one prepared waveform can stay ahead of playback
- stale generations are rejected before synthesis, after synthesis, and before playback
- acknowledgement audio can be prepared in advance
- synthesized waveforms are trimmed conservatively at their edges
- synthesis latency, playback gaps, and edge silence are recorded
- `PerformanceCue` activates exactly when a clause begins physical playback and clears afterward

The model-specific implementation is concentrated in `_make_synthesizer`, which currently constructs a CPU Kokoro `KPipeline` and returns `Callable[[str], np.ndarray]`.

`config.json` already has `voice.engine = "kokoro"`, so engine selection does not require a new top-level configuration concept.

## Non-Goals

This change does not:

- replace Kokoro by default
- claim Chatterbox is faster or better before target-PC benchmarking
- change ConversationWorker clause splitting
- change stale-generation semantics
- change playback scheduling or the one-clause-ahead ready queue
- add streaming partial-waveform playback inside one clause
- add a separate TTS server/process in V1
- automatically convert every performance cue into a laugh, sigh, gasp, or other paralinguistic event
- modify the LLM performance vocabulary
- require CUDA or Chatterbox dependencies in CI

## Architecture

The existing pipeline remains authoritative:

```text
SpeechClause(text, performance)
        |
        v
VoiceOutput
  - queueing
  - stale generation checks
  - one-clause-ahead scheduling
  - trimming
  - playback
  - diagnostics
  - physical playback performance timing
        |
        v
TTSEngine.synthesize(text, performance)
        |
        v
SynthesisResult(audio, sample_rate)
```

`VoiceOutput` depends on an engine contract, not on Kokoro or Chatterbox directly.

Initial engines:

```text
TTSEngine
  |- KokoroEngine
  `- ChatterboxTurboEngine
```

The engine boundary is intentionally small. An engine knows how to load its model and turn text plus a performance cue into audio. It does not own queueing, cancellation, playback, face state, or conversation state.

## Core Types

### `SynthesisResult`

Use an immutable result object:

```python
@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
```

Requirements:

- `audio` is a one-dimensional `np.float32` waveform when returned to `VoiceOutput`.
- `sample_rate` is a positive integer supplied by the engine.
- `VoiceOutput` must not assume every engine uses the configured Kokoro sample rate.

### `TTSEngine`

Use a structural protocol or similarly lightweight interface:

```python
class TTSEngine(Protocol):
    def synthesize(
        self,
        text: str,
        performance: PerformanceCue,
    ) -> SynthesisResult:
        ...
```

V1 deliberately omits lifecycle methods unless implementation proves they are needed. Model construction occurs through the engine factory and an engine may lazily load internal resources on first synthesis.

## File Boundaries

Recommended structure:

```text
output/
  voice.py
  tts/
    __init__.py
    base.py
    factory.py
    kokoro.py
    chatterbox_turbo.py
```

Responsibilities:

### `output/voice.py`

Keeps all existing scheduling and playback behavior. It receives either an injected engine for tests or asks the factory for the configured engine.

It must not import Kokoro or Chatterbox packages.

### `output/tts/base.py`

Contains only lightweight shared types such as `SynthesisResult` and `TTSEngine`. Importing this module must not load model libraries.

### `output/tts/factory.py`

Reads `voice.engine` and creates the requested engine.

Initial accepted names:

- `kokoro`
- `chatterbox_turbo`

Unknown values raise a clear configuration error. There is no silent fallback.

### `output/tts/kokoro.py`

Contains the existing Kokoro-specific behavior moved out of `voice.py` with no intended audio or configuration change.

It continues to use:

- configured `voice.name`, defaulting to `af_heart`
- CPU Kokoro pipeline for the current baseline

It returns Kokoro's actual sample rate in `SynthesisResult`.

### `output/tts/chatterbox_turbo.py`

Contains all Chatterbox-specific imports and behavior.

The module may be imported safely, but heavyweight package/model loading occurs only when the engine is selected and constructed or first used.

The adapter must be optional. A Kokoro installation and CI environment must not require Chatterbox dependencies.

## VoiceOutput Integration

`VoiceOutput` should accept an optional injected engine while preserving convenient test injection.

Preferred shape:

```python
VoiceOutput(
    config,
    state,
    event_log,
    engine: TTSEngine | None = None,
    play: Callable[[np.ndarray, int], None] | None = None,
)
```

If preserving the existing `synthesize=` callback temporarily materially reduces migration risk, a compatibility shim is acceptable during implementation, but the completed design should make `TTSEngine` the normal production seam.

During synthesis:

```text
result = engine.synthesize(text, performance)
audio = result.audio
sample_rate = result.sample_rate
```

The resulting sample rate travels with the prepared waveform into playback. This prevents engine-specific rate assumptions from leaking into `VoiceOutput`.

The ready-queue item therefore needs to carry `sample_rate` alongside audio.

All existing stale checks remain in the same relative positions.

## Performance Cue Semantics

The full `PerformanceCue` is passed into `TTSEngine.synthesize`.

This creates one semantic cue that can influence both visual and vocal delivery while keeping actual rendering engine-specific.

### Kokoro V1

Kokoro may ignore `performance` initially. The goal of the extraction is behavioral equivalence, not invented prosody controls.

### Chatterbox Turbo V1

Chatterbox may map approved performance labels to conservative style hints when supported and validated.

However, transient expression labels are not equivalent to explicit vocal events.

For example:

```text
performance=playful
```

must not automatically mean:

```text
[chuckle]
```

on every playful clause.

Paralinguistic events such as laughs, sighs, gasps, or coughs should eventually be represented explicitly if Vess gains that capability. They are out of scope for this abstraction.

The initial adapter may therefore use plain text for unsupported or unvalidated performance mappings. Unknown performance labels must degrade to neutral behavior.

## Chatterbox Configuration

Keep engine-specific settings under `voice` rather than creating unrelated global configuration.

Conceptual configuration:

```json
{
  "voice": {
    "engine": "chatterbox_turbo",
    "sample_rate": 24000,
    "chatterbox": {
      "device": "cuda",
      "reference_audio": "..."
    }
  }
}
```

Exact Chatterbox fields should only be added when required by the official adapter implementation.

Do not duplicate settings that the model can determine itself.

`voice.sample_rate` remains available for legacy/config compatibility, but engine-returned `SynthesisResult.sample_rate` is authoritative for generated audio.

## Voice Conditioning

Chatterbox voice conditioning should be computed or loaded once and reused when the library supports it.

Do not reprocess the reference voice for every clause.

The engine owns cached conditioning state because it is model-specific. `VoiceOutput` must not know what a reference embedding or conditioning object is.

## Loading and Failure Behavior

### Lazy/optional dependency behavior

Selecting Kokoro must not import or initialize Chatterbox.

Selecting Chatterbox when its dependency is unavailable must raise a clear error naming the selected engine and missing requirement.

### No silent fallback

If `voice.engine` is `chatterbox_turbo` and Chatterbox fails to load, Vess must not silently switch to Kokoro.

Reasons:

- benchmarks would become invalid
- users could believe expressive controls are active when they are not
- configuration problems would be hidden

Fallback may be added later only as an explicit user-configurable policy.

### Runtime synthesis errors

Model inference exceptions continue through the existing `voice_error`/`tts_error` path in `VoiceOutput`. The engine should not swallow failures and return fake silence merely to keep the pipeline moving.

## Benchmark Harness

Add a standalone harness that uses the same engine factory and engine contract as production.

Conceptual commands:

```powershell
python tools/benchmark_tts.py --engine kokoro
python tools/benchmark_tts.py --engine chatterbox_turbo
```

The harness must not require microphones, Whisper, Ollama, face rendering, or physical playback.

### Standard text set

At minimum:

```text
Sure.
That's actually pretty interesting.
I don't think that's going to work, though.
Why would that happen?
That's ridiculous.
```

A separate optional expressive set may exercise supported Chatterbox style/paralinguistic syntax, but those results must not be conflated with plain-text engine comparison.

### Measurements

Per utterance/run:

- engine name
- text identifier
- cold/warm run distinction
- synthesis milliseconds
- output audio duration
- realtime factor (`synthesis_seconds / audio_seconds`)
- sample rate
- output sample count
- success/failure
- error text when failed

When CUDA/PyTorch metrics are available without coupling the generic harness to one engine, also report:

- allocated VRAM
- reserved VRAM
- peak allocated VRAM

VRAM metrics are optional because native/non-PyTorch backends may expose different telemetry.

### Output

The harness should produce:

- human-readable console summary
- machine-readable JSON results
- WAV files for listening comparison

Generated benchmark outputs must be ignored by Git.

## Target-PC Acceptance

Remote CI can prove architecture and behavior, not real model suitability.

Chatterbox remains experimental until the target PC verifies:

1. model loads reliably
2. no CUDA OOM with the intended Ollama/Qwen configuration
3. warm short-clause latency is acceptable
4. Qwen generation speed does not degrade unacceptably while TTS is resident
5. clause-to-clause playback remains smooth
6. short replies remain stable over repeated generations
7. voice quality and emotional range clearly justify any latency/resource cost
8. performance/style mappings sound natural rather than theatrical or repetitive

Recommended product thresholds for the first test pass:

```text
warm short-clause TTS < 1.2 s    excellent
1.2-1.5 s                        strong
1.5-2.0 s                        acceptable if quality gain is clear
2.0-2.5 s                        questionable
> 2.5 s                          reject this backend/configuration
```

These are Vess design targets, not vendor benchmark claims.

Also target:

- no OOM
- no new audible clause gaps
- ideally less than ~25% LLM slowdown while both models are resident

## Testing Strategy

### Contract tests

Use fake engines to verify:

- performance cue reaches the engine unchanged
- engine-returned sample rate reaches playback
- waveform remains float32/one-dimensional at the VoiceOutput boundary
- synthesis errors enter the existing error path

### Kokoro regression tests

Prove extraction preserves current behavior:

- configured voice name is passed to Kokoro
- multiple Kokoro result chunks concatenate in order
- tensor-like outputs are converted to NumPy float32
- empty model output yields an empty float32 waveform

Tests should mock the heavyweight Kokoro dependency rather than loading the real model in CI.

### Factory tests

Verify:

- `kokoro` selects `KokoroEngine`
- `chatterbox_turbo` selects `ChatterboxTurboEngine`
- unknown engine names fail clearly
- selecting Kokoro does not require/import Chatterbox runtime dependencies

### Voice pipeline regressions

Existing voice tests remain authoritative for:

- playback order
- speaking state
- stale-generation behavior
- ready-slot behavior
- acknowledgement preparation
- synthesis/playback overlap
- gap diagnostics
- performance activation timing

### Benchmark tests

Unit-test result aggregation and file/JSON formatting with fake engines. CI does not benchmark real models.

## Migration Order

Implementation should proceed in this order:

1. Add engine contract and result type with tests.
2. Extract current Kokoro implementation behind `KokoroEngine` with regression tests.
3. Adapt `VoiceOutput` to consume `SynthesisResult` and engine-returned sample rates while keeping existing scheduling unchanged.
4. Add engine factory and `voice.engine` selection tests.
5. Add fake-engine coverage proving performance propagation.
6. Add the benchmark harness using fake engines in CI.
7. Add optional `ChatterboxTurboEngine` adapter and structural tests that do not download/load the model.
8. Run the full existing unit and behavior-verification suite.
9. Perform real Chatterbox latency/VRAM/audio acceptance later on the target PC.

## Compatibility and Rollback

Default configuration remains `voice.engine = "kokoro"`.

A completed abstraction with Kokoro selected should behave the same as the current system except for internal type boundaries and use of engine-reported sample rate.

If Chatterbox proves unsuitable, the abstraction still has value: Kokoro remains the default and future engines can be evaluated without another `VoiceOutput` refactor.

Rollback of Chatterbox should therefore mean selecting/removing that adapter, not undoing the engine boundary.

## Success Criteria

The design is successful when:

- `VoiceOutput` contains no direct Kokoro/Chatterbox model construction
- Kokoro remains the default and passes all existing voice behavior tests
- engine sample rate travels with each synthesized waveform
- `PerformanceCue` reaches the selected engine
- Chatterbox is optional and never imported/loaded for Kokoro CI/runtime paths
- invalid engine selection fails explicitly
- the benchmark harness can compare engines through the production engine contract
- CI remains model-free and CUDA-free
- no claim is made about Chatterbox production suitability until target-PC benchmarking is complete
