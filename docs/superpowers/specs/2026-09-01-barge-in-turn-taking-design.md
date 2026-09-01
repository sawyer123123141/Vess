# Vess Barge-In and Natural Turn-Taking Design

**Status:** Approved architecture, pending written-spec review.

## Goal

Let a person interrupt Vess naturally while Vess is speaking, without allowing Vess's own speaker output, incidental noise, slow transcription, or cancellation races to corrupt the conversation.

The intended interaction is:

1. Vess is speaking.
2. The microphone remains active.
3. Credible near-end speech appears while speaker playback is active.
4. Vess pauses quickly, before waiting for Whisper.
5. The overlapping human utterance continues to be captured.
6. If a real transcript arrives, the interruption is committed: the old response is cancelled and the transcript becomes the newest intent.
7. If no real utterance materializes, the interruption is rolled back: the same waveform resumes from its saved position.
8. Short-term memory records delivery state, not merely whatever text the LLM happened to generate.

The feature must preserve the independent 30 FPS face loop, newest-intent semantics, one-clause-ahead TTS bound, and local-only architecture.

## Non-Goals

This design does not add:

- a new LLM
- long-term memory
- a mandatory learned interruption classifier
- a hard-coded choice of acoustic echo canceller before hardware testing
- multi-speaker diarization
- semantic backchannel classification
- full-duplex behavior where Vess intentionally continues talking over a person
- word-level TTS/audio alignment
- planner or proactive-behavior work

## Existing Problems to Fix

The current implementation has four relevant limitations:

1. `AudioLoop` discards microphone blocks whenever `state.speaking` is true.
2. `VoiceOutput` can skip stale queued/prepared speech but cannot stop a waveform already playing.
3. `ConversationWorker.submit()` both creates a new request and invalidates older generations; there is no separate cancel-without-submit operation.
4. Conversation memory is finalized from generated clauses before physical playback is known to have completed.

Barge-in must correct those four things without turning the runtime into one giant synchronous voice loop.

---

## 1. Core Architecture

The runtime remains asynchronous and event-driven.

```text
AudioPlayer -----------------------> RenderReference
                                      |
                                      v
microphone -> CapturedAudioBlock -> CapturePreprocessor
                                      |
                                      v
                               UtteranceAssembler/VAD
                                      |
                      +---------------+---------------+
                      |                               |
                 Vess silent                    Vess speaking
                      |                               |
                      v                               v
             normal wake/follow-up            InterruptionDetector
                                                      |
                                                      v
                                                TurnCoordinator
                                                 /          \
                                                /            \
                                            rollback        commit
                                               |              |
                                               v              v
                                          VoiceOutput   ConversationWorker
```

Responsibilities stay narrow:

- **CapturePreprocessor** handles capture cleanup/echo-reduction plumbing.
- **InterruptionDetector** answers whether credible near-end speech has persisted long enough to justify a reversible pause.
- **TurnCoordinator** owns pause/commit/rollback policy.
- **AudioPlayer** owns physical output, cursor state, pausing, resuming, and render reference.
- **VoiceOutput** owns TTS ordering, clause lifecycle, generation freshness, and performance cues.
- **ConversationWorker** owns user requests, LLM generations, and conversation-turn lifecycle.

No one component gets to quietly become the whole assistant because software apparently enjoys recreating governments.

---

## 2. Always-Live Capture When Barge-In Is Enabled

When `barge_in.enabled` is true, `AudioLoop` must not discard microphone blocks merely because Vess is speaking.

The sounddevice input callback remains minimal and non-blocking. It copies samples into the existing bounded capture queue and preserves timing metadata when available.

Use a value such as:

```python
@dataclass(frozen=True)
class CapturedAudioBlock:
    samples: np.ndarray
    adc_time: float | None
    received_at: float
```

`adc_time` is retained for future echo-path/delay work and diagnostics. Tests may use `None`.

When barge-in is disabled, the current conservative behavior may remain: microphone data during Vess speech may be ignored. This preserves current behavior until hardware acceptance is complete.

---

## 3. Capture Preprocessing and Render Reference

Speaker output can return through the microphone and look like human speech to an ordinary VAD. Raw VAD alone is therefore not a production barge-in solution for speaker use.

Introduce an explicit render-aware preprocessing contract:

```python
@dataclass(frozen=True)
class RenderedAudioBlock:
    samples: np.ndarray
    sample_rate: int
    dac_time: float | None

class CapturePreprocessor(Protocol):
    def push_render_reference(self, block: RenderedAudioBlock) -> None:
        ...

    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray:
        ...
```

A passthrough implementation is sufficient for deterministic CI and headphones/non-speaker testing.

A future real backend may use WebRTC AEC3 or another local echo-cancellation implementation selected after testing the actual microphone, speakers, host API, and room.

Requirements:

- no network service is required
- no mandatory CUDA dependency
- no heavy DSP, file I/O, model inference, database writes, or ordinary blocking work occurs in the real-time audio callback
- preprocessing failure is visible in diagnostics
- if configured to fail closed, preprocessing failure disables live barge-in for the session rather than treating raw self-echo as trustworthy speech
- speaker-mode barge-in is not production-ready until real self-echo testing passes

The first implementation may provide the contract and passthrough backend only. `barge_in.enabled` remains false by default.

---

## 4. Interruption Detection Uses a Fast Reversible Pause Threshold

`InterruptionDetector` answers one question:

> Has credible preprocessed near-end speech persisted long enough to justify temporarily pausing Vess?

It does not know about prompts, LLM generations, TTS queues, or memory.

Because the first action is **reversible pause**, not permanent cancellation, the pause threshold can be faster than a traditional commit threshold.

Initial configurable value:

```text
pause_after_speech_seconds = 0.25
```

This is intentionally aligned with the existing minimum utterance scale rather than using 0.5 seconds as the audible pause delay. Hardware acceptance may move it higher or lower.

A single short block or instantaneous spike must not trigger pause. A candidate is emitted only after the configured continuous/sustained speech duration is met.

A candidate never becomes text and never directly creates a conversation generation.

---

## 5. Two-Phase Interruption Protocol

An acoustic candidate pauses speech quickly but does not immediately destroy the current response.

### Phase A: Reversible Pause

On the first candidate for the currently audible generation:

1. `TurnCoordinator` marks an interruption pending.
2. `VoiceOutput.pause_for_interruption()` requests immediate physical output abort and saves resumable playback state.
3. The current LLM generation remains current.
4. Playback is gated so the worker cannot advance to the next clause while pending.
5. The existing one-waveform-ahead synthesis bound remains in force, so pause cannot create an unbounded TTS backlog.
6. Capture continues until the overlapping human utterance ends.
7. `state.speaking` becomes false only after physical audio is actually paused.
8. `state.listening` becomes true for the accepted candidate.
9. Transient performance clears to neutral while no audio is physically playing.

Duplicate candidate notifications while already pending are idempotent.

### Phase B1: Commit a Real Interruption

A real interruption is committed when the captured utterance returns a non-empty accepted transcript.

Commit order:

1. `ConversationWorker.cancel_active_response(reason="barge_in")` advances the freshness generation without inventing a user request.
2. `VoiceOutput.commit_interruption(old_generation)` permanently discards the saved remainder and lets freshness discard queued/prepared work from that old generation.
3. The transcript is submitted normally, allocating a newer generation.
4. Existing newest-intent rules continue to handle any request that races with the barge-in transcript.

The cancel token is monotonic. Generation identifiers are never reused or decremented.

True immediate cancellation of Ollama's underlying inference process is not required for V1. The old stream must stop contributing clauses as soon as the worker observes stale generation state, and the HTTP response must close promptly afterward.

### Phase B2: Roll Back a False Interruption

If the candidate does not produce a real transcript:

1. no cancellation generation is created
2. no synthetic user request is submitted
3. the original generation remains current
4. `VoiceOutput.resume_after_false_interruption()` resumes the same waveform from its saved playback position
5. the original performance cue becomes active again only when resumed audio physically starts
6. normal subsequent clauses from that same generation may continue
7. the false interruption is recorded diagnostically

If any newer generation appeared while paused, rollback is forbidden and the old paused waveform is discarded.

V1 does not semantically classify utterances like "mhm", "right", or "okay" as backchannels. A non-empty accepted transcript is a real user turn. Backchannel policy comes later only if actual data shows it matters.

---

## 6. False-Interruption Timing Must Not Race Whisper

The false-interruption timer must never resume Vess merely because local transcription is taking longer than expected.

The pending interruption has internal decision phases:

```text
CAPTURING -> TRANSCRIBING -> DECIDED
```

Rules:

- While overlapping speech is still being assembled, rollback is not allowed.
- Once an utterance has been queued for transcription, the ordinary false-silence timer is suspended.
- A non-empty transcript commits.
- An empty transcript or explicit transcription failure rolls back if the original generation is still current.
- A bounded decision watchdog prevents a broken transcriber from leaving Vess paused forever.

Initial configurable values:

```text
false_interruption_timeout_seconds = 2.0
max_interruption_decision_seconds = 5.0
```

`false_interruption_timeout_seconds` applies to a candidate that ends without entering a valid transcription decision. `max_interruption_decision_seconds` bounds the period after speech end while a transcription decision is pending.

If the decision watchdog expires, record an error and roll back only if the original generation is still current. Otherwise discard the old paused response.

These are starting values, not hardware-tuned constants.

---

## 7. Streaming Cancellable `AudioPlayer`

The existing high-level `sounddevice.play()` / `wait()` helper hides the state required for fast abort, resume, and render-reference publication.

Introduce an `AudioPlayer` abstraction owned by `VoiceOutput`.

Conceptual values:

```python
@dataclass(frozen=True)
class PlaybackReceipt:
    status: str  # completed | paused | interrupted | error
    frames_started: int
    frames_completed: int
    total_frames: int
    sample_rate: int
```

Conceptual interface:

```python
class AudioPlayer(Protocol):
    def play(self, audio: np.ndarray, sample_rate: int) -> PlaybackReceipt:
        ...

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        ...

    def resume(self) -> PlaybackReceipt:
        ...

    def discard_paused(self) -> None:
        ...
```

`play()` may block the dedicated playback worker while the device plays, but `pause_for_interruption()` must be safe to invoke from another worker.

The production implementation uses stream-level sounddevice/PortAudio output. Interruption uses immediate **abort** semantics rather than ordinary graceful stop semantics that may wait for pending device buffers.

The player owns:

- current waveform
- sample rate
- current playback identity
- playback cursor/completed-frame estimate
- paused remainder
- render-reference publication

A small replay overlap at resume is preferable to skipping speech if device buffering makes the exact audible cursor uncertain. Target-PC tests determine whether such a safety overlap is necessary.

The real-time callback remains extremely small. It must not call the LLM, TTS, SQLite, event logging, file I/O, or unpredictable blocking code.

---

## 8. `VoiceOutput` Interruption Behavior

Add explicit operations:

```python
pause_for_interruption()
commit_interruption(generation_id)
resume_after_false_interruption()
finish_generation(generation_id)
```

### `pause_for_interruption()`

- idempotent
- affects only currently audible playback
- asks `AudioPlayer` to abort/pause
- records pause request and physical pause timing
- stops playback progression to later clauses while pending
- clears active performance only after the waveform is physically paused

### `commit_interruption(generation_id)`

- permanently discards paused remainder for that generation
- never resumes stale audio
- lets normal generation freshness reject queued/prepared old work

### `resume_after_false_interruption()`

- legal only when an interruption is pending
- verifies the paused generation is still current
- resumes the same synthesized waveform rather than synthesizing it again
- reactivates its original `PerformanceCue` at physical resumed playback start
- reopens playback progression for subsequent clauses

### `finish_generation(generation_id)`

`ConversationWorker` calls this only after the LLM stream has ended and every clause for that generation has been enqueued.

`VoiceOutput` must preserve the marker's ordering through synthesis/playback so it can emit `generation_playback_drained` only after every earlier playable clause has either completed or been explicitly abandoned.

This end marker is required for truthful memory finalization. LLM completion alone is not playback completion.

The one-ready-waveform-ahead bound remains unchanged.

---

## 9. Conversation Cancellation Without Submission

Add:

```python
ConversationWorker.cancel_active_response(reason: str)
```

It must:

- atomically advance the latest-generation token
- make the active old generation stale
- not enqueue an empty or fake request
- preserve any already pending newer request correctly
- record cancellation reason and old/new generation identifiers
- be safe and idempotent when there is nothing active to cancel

A later `submit(transcript)` allocates another newer generation normally.

Cancellation and submission remain separate because commit should first make the old generation impossible to speak, then submit the replacement user intent. Under the two-phase protocol, neither happens on the initial reversible pause.

---

## 10. Delivery-Aware Conversation Memory

Generated text is not equivalent to delivered text.

Replace the current "remember generated clauses when LLM streaming finishes" rule with a generation-scoped delivery ledger.

Each generation tracks:

- user request
- generated clauses
- clauses whose physical playback completed
- currently paused/interrupted clause, if any
- whether LLM generation ended
- whether playback drained
- final delivery status: completed or interrupted

Extend the short-term turn value:

```python
@dataclass(frozen=True)
class ConversationTurn:
    timestamp: float
    user: str
    assistant: str
    status: str = "completed"       # completed | interrupted
    interrupted_clause: str | None = None
```

### Normal completion

The turn is finalized only after:

1. LLM generation ended
2. `finish_generation(generation_id)` entered the voice pipeline
3. the corresponding playback-drained marker is reached

`assistant` contains clauses whose playback fully completed.

### Real interruption

Finalize the old turn as `interrupted` when the interruption commits.

- `assistant` contains only fully completed clauses
- `interrupted_clause` may contain the full text of the clause that had started but did not complete
- prompt construction must explicitly treat `interrupted_clause` as partially delivered and must not assume the user heard the whole thing

If no assistant clause completed, retain the prior user request anyway with `assistant=""` and `status="interrupted"`. A user question must not disappear merely because Vess was interrupted before finishing its first clause.

Prompt rendering may use:

```text
User: <prior request>
Vess (interrupted): <fully completed speech, if any>
Vess had started another clause but was interrupted; do not assume the user heard all of it.
```

No word-level claim is made about the partially heard clause.

### Required playback lifecycle receipts

Generation-scoped internal receipts distinguish:

- clause playback started
- clause playback completed
- clause paused for candidate interruption
- clause resumed
- clause abandoned on committed interruption
- generation playback drained

Stale receipts cannot finalize memory for a newer generation.

---

## 11. `TurnCoordinator`

Introduce one small coordinator for interruption policy.

Public conceptual states:

```text
IDLE
AGENT_SPEAKING
INTERRUPTION_PENDING
```

The pending state internally knows whether it is capturing, transcribing, or awaiting decision.

Transitions:

```text
IDLE -> AGENT_SPEAKING
    physical Vess playback starts

AGENT_SPEAKING -> INTERRUPTION_PENDING
    detector emits first credible sustained-speech candidate

INTERRUPTION_PENDING -> IDLE/user-turn flow
    non-empty transcript commits interruption

INTERRUPTION_PENDING -> AGENT_SPEAKING
    false decision rolls back and same generation remains current

INTERRUPTION_PENDING -> IDLE
    false decision occurs but a newer generation already superseded the paused one
```

The coordinator:

- makes duplicate candidate/commit/rollback events harmless
- never holds `State.lock` while calling potentially blocking audio, transcription, TTS, or conversation methods
- owns timer/watchdog state
- does not perform acoustic processing itself
- does not construct prompts or mutate memory directly

---

## 12. Runtime State Semantics

`State` remains the authoritative public runtime state.

Important distinction:

- `state.listening` means Vess is meaningfully attending to an accepted human speech candidate/turn
- raw microphone energy or raw VAD during speaker playback is diagnostic only

During pending interruption after physical pause:

```text
speaking = False
listening = True
performance = neutral
```

On false rollback:

- `listening` becomes false before resumed speech takes over
- `speaking` becomes true only when resumed sound physically starts
- original performance returns at that same physical start

This preserves the face's existing listening/speaking priority without letting self-echo make the eyes behave as though a person spoke.

---

## 13. Wake and Follow-Up Semantics

A committed barge-in transcript is part of the already active conversation and does not require the wake phrase.

Idle behavior remains unchanged:

- wake matching still applies when conversation is inactive
- normal follow-ups use the current conversation timeout

The acoustic candidate never becomes text and never submits a request by itself.

---

## 14. Error Handling

### Audio player cannot pause/abort

- record `barge_in_pause_error`
- do not claim physical pause succeeded
- keep capturing the user's utterance
- if a real transcript arrives, commit cancellation so no additional stale clauses begin after current blocking playback eventually returns

### Capture preprocessor fails

- record `barge_in_preprocessor_error`
- when `disable_on_preprocessor_error=true`, disable speaker-time barge-in for the session
- retain ordinary idle microphone handling if raw capture itself still works

### Transcription returns empty

- roll back if original generation is still current
- otherwise discard old paused waveform

### Transcription raises or decision watchdog expires

- record the error/timeout
- roll back if the original generation is still current
- otherwise discard
- never fabricate transcript text

### A newer request arrives while interruption is pending

Newest intent wins. The old waveform may never resume after a newer generation exists.

### Shutdown while paused

- abort/close player
- discard paused remainder
- clear speaking/listening/performance state
- cancel timers/watchdogs
- do not wait for false-interruption timeout

---

## 15. Configuration

Add conservative, explicitly non-final defaults:

```json
{
  "barge_in": {
    "enabled": false,
    "pause_after_speech_seconds": 0.25,
    "false_interruption_timeout_seconds": 2.0,
    "max_interruption_decision_seconds": 5.0,
    "preprocessor": "passthrough",
    "disable_on_preprocessor_error": true
  }
}
```

`enabled` remains false until target-PC acceptance proves the chosen preprocessing path prevents self-echo interruption with the actual speakers and microphone.

---

## 16. Diagnostics and Metrics

Record generation-scoped timestamps/events for:

- credible speech candidate start
- pause requested
- physical audio paused
- speech-start-to-pause latency
- playback frame/cursor at pause
- interruption pending phase changes
- transcript start/finish
- interruption committed
- false interruption
- resumed playback start
- stale clauses discarded after commit
- interrupted clause metadata
- new response first-clause latency after barge-in
- preprocessor/AEC failure
- player abort failure

Target-PC metrics:

- p50/p95 human-speech-start -> audible Vess pause
- real interruption detection rate
- missed interruption rate
- false interruptions per hour
- self-echo interruptions per hour
- false-interruption resume success rate
- stale clauses audibly played after commit
- completed-but-unheard clauses falsely stored as delivered

Two hard production targets remain exact:

```text
stale clauses played after committed interruption = 0
never-completed clauses recorded as fully delivered = 0
```

---

## 17. Hard Invariants

Deterministic tests must prove:

1. A committed interruption makes the old generation stale before the replacement request is submitted.
2. No stale prepared clause starts physical playback after commit.
3. Reversible pause alone does not cancel the LLM generation.
4. False rollback can resume only the same still-current generation.
5. Any newer generation permanently forbids old paused-waveform resume.
6. Active performance is neutral while physical playback is paused.
7. Original performance returns only when resumed physical playback starts.
8. Raw speaker echo/VAD alone cannot set meaningful listening state unless preprocessing/detection accepts a candidate.
9. LLM-generated but never completed clauses are never marked fully delivered.
10. Interrupted turns remain present in short-term context.
11. Duplicate candidate/commit/rollback/cancel operations are idempotent.
12. Playback, capture, synthesis, transcription, conversation, and rendering remain independent workers.
13. Barge-in queues remain bounded; paused playback cannot create unbounded synthesized-audio backlog.
14. False-interruption rollback cannot fire while a valid transcription decision is actively in flight, except through the explicit bounded watchdog path.
15. The generation-drained marker cannot overtake preceding speech.
16. Barge-in disabled preserves current idle/wake/follow-up behavior.
17. The 30 FPS face loop never waits on barge-in work.

---

## 18. Remote Test Strategy

CI tests use no microphone hardware, speakers, Ollama, Whisper model, real TTS model, or CUDA.

Use deterministic fakes for:

- `AudioPlayer`
- capture preprocessor
- transcriber
- TTS engine
- LLM stream
- clocks/timers where timing matters

Required scenarios:

### Real interruption

```text
playback starts
candidate threshold reached
player pauses
human utterance finishes
transcription starts
non-empty transcript arrives
old generation is cancelled
paused remainder is committed/discarded
new transcript is submitted
old prepared/queued clauses never play
old memory turn finalizes interrupted
```

### False interruption before transcription

```text
playback starts
candidate threshold reached
player pauses
candidate collapses without valid utterance
false timeout expires
same waveform resumes
same generation remains current
```

### Empty transcript after valid captured utterance

```text
player pauses
utterance is transcribed
transcriber returns empty
same current generation resumes
```

### Slow but valid transcription

```text
player pauses
transcription remains in flight longer than ordinary false timeout
Vess does not resume early
non-empty transcript eventually commits
```

### Decision watchdog

```text
player pauses
transcription decision hangs
watchdog expires
error is logged
same generation resumes only if still current
```

### Superseded false interruption

```text
player pauses
newer request arrives
candidate becomes false
old waveform does not resume
```

### Interrupt during first clause

No complete assistant clause exists; prior user request is still retained as interrupted memory.

### Interrupt between clauses

Completed clauses stay delivered; next prepared clause is abandoned after commit.

### Normal generation drain

`finish_generation` marker cannot finalize memory until all preceding physical playback completes.

### Duplicate candidate

Two candidate events cause one pause transition.

### Commit racing synthesis completion

A waveform produced at the race boundary becomes stale and never plays.

### Shutdown while paused

Shutdown clears paused state without deadlock or delayed resume.

### Feature disabled

Existing behavior remains unchanged.

---

## 19. Target-PC Acceptance

Remote tests prove policy and race correctness, not acoustics.

Before barge-in becomes enabled by default:

1. choose/integrate the actual local echo-preprocessing backend
2. run Vess speech at realistic speaker volume with no human speech and verify self-echo does not pause it
3. test interruptions at multiple distances, directions, and normal voice volumes
4. measure p50/p95 pause latency
5. test very short real interruptions such as "wait"
6. test coughs, keyboard noise, room noise, and accidental speech-like sounds
7. test speech beginning immediately after Vess starts and near the end of a clause
8. verify false-interruption resume sounds natural
9. verify Qwen, Whisper, TTS, preprocessing, and rendering coexist within CPU/RAM/VRAM limits
10. inspect traces for stale playback, timer races, and delivery-memory errors
11. tune pause/timeout values only from these measurements

Only then should `barge_in.enabled` become true by default.

---

## 20. Implementation Boundaries

Likely production files/modules:

- `perception/audio.py`
- new focused audio preprocessing module
- new `output/audio_player.py`
- `output/voice.py`
- `brain/llm.py`
- `brain/memory.py`
- new small turn-coordination module
- `state.py`
- `main.py`
- `config.json`

Tests extend audio, TTS pipeline, voice freshness, conversation freshness, short-term memory, and new coordinator/player behavior.

Do not combine this feature with long-term memory, proactive behavior, planner work, or TTS-engine benchmarking.

---

## References Informing the Design

- Current LiveKit voice-agent turn handling separates interruption detection from turn policy, supports false-interruption recovery, and truncates history around delivered speech.
- WebRTC AEC3 is built around capture plus render-reference processing rather than raw VAD alone.
- python-sounddevice/PortAudio stream APIs distinguish graceful stop from immediate abort; immediate abort is the appropriate primitive for fast barge-in cancellation.

These references inform architecture only. They are not automatically runtime dependencies.