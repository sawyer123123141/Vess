# Vess Barge-In and Natural Turn-Taking Design

**Status:** Approved architecture, pending written-spec review.

## Goal

Let a person interrupt Vess naturally while Vess is speaking, without allowing Vess's own speaker output, incidental noise, slow transcription, or cancellation races to corrupt the conversation.

The intended interaction is:

1. Vess is speaking.
2. The microphone remains active.
3. Credible near-end human speech appears despite speaker playback.
4. Vess pauses quickly, before waiting for Whisper.
5. The overlapping utterance continues to be captured.
6. A real transcript commits the interruption and becomes the newest intent.
7. A false candidate rolls back and resumes the same waveform.
8. Conversation memory records delivery state, not merely generated text.

The feature must preserve the independent 30 FPS face loop, newest-intent semantics, one-clause-ahead TTS bound, and local-only architecture.

## Non-Goals

This design does not add a new LLM, long-term memory, a learned interruption classifier, a mandatory echo-canceller chosen without hardware testing, multi-speaker diarization, semantic backchannel classification, intentional full-duplex talking-over, word-level TTS alignment, planner work, or proactive-behavior work.

## Existing Problems

The current implementation has four relevant limitations:

1. `AudioLoop` discards microphone blocks whenever `state.speaking` is true.
2. `VoiceOutput` can skip stale queued/prepared speech but cannot stop a waveform already playing.
3. `ConversationWorker.submit()` both creates a new request and invalidates older generations; there is no generation-specific cancel-without-submit operation.
4. Conversation memory is finalized from generated clauses before physical playback completion is known.

Barge-in must correct those four things without turning the runtime into one synchronous voice loop.

---

## 1. Core Architecture

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

- **CapturePreprocessor** owns capture cleanup/echo-reference plumbing.
- **InterruptionDetector** decides when credible near-end speech justifies a reversible pause.
- **TurnCoordinator** owns pause/commit/rollback policy and timers.
- **AudioPlayer** owns physical output, cursor state, pause/resume, and render reference.
- **VoiceOutput** owns TTS ordering, clause lifecycle, freshness, and performance cues.
- **ConversationWorker** owns user requests, LLM generations, and conversation-turn lifecycle.

---

## 2. Always-Live Capture When Enabled

When `barge_in.enabled` is true, `AudioLoop` must not discard microphone blocks merely because Vess is speaking.

The sounddevice input callback remains minimal and non-blocking. It copies samples into the existing bounded capture queue and preserves timing metadata when available.

```python
@dataclass(frozen=True)
class CapturedAudioBlock:
    samples: np.ndarray
    adc_time: float | None
    received_at: float
```

`adc_time` is retained for future echo-path/delay work and diagnostics. Tests may use `None`.

When barge-in is disabled, current speaker-time microphone behavior may remain unchanged. This preserves existing behavior until hardware acceptance.

---

## 3. Capture Preprocessing and Render Reference

Speaker output can return through the microphone and appear as speech to ordinary VAD. Raw VAD alone is therefore not a production barge-in solution for speaker use.

Use an explicit render-aware contract:

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

- no network service
- no mandatory CUDA dependency
- no heavy DSP, file I/O, model inference, database writes, or unpredictable blocking work in real-time audio callbacks
- preprocessing failure is visible
- with fail-closed policy, preprocessing failure disables speaker-time barge-in for the session rather than trusting raw self-echo
- speaker-mode barge-in is not production-ready until real self-echo tests pass

The first implementation may provide only the contract plus passthrough backend. `barge_in.enabled` remains false by default.

---

## 4. Fast Reversible Interruption Detection

`InterruptionDetector` answers:

> Has credible preprocessed near-end speech persisted long enough to justify temporarily pausing Vess?

It knows nothing about prompts, LLM generations, TTS queues, or memory.

Because the first action is reversible pause rather than permanent cancellation, the pause threshold can be shorter than a traditional interruption-commit threshold.

Initial configurable value:

```text
pause_after_speech_seconds = 0.25
```

This is a starting value, not a hardware-tuned constant.

A single block or instantaneous spike cannot trigger pause. A candidate is emitted only after sustained accepted speech reaches the configured duration.

A candidate never becomes text and never creates a generation.

---

## 5. Two-Phase Interruption Protocol

### Phase A: Reversible Pause

On the first candidate for currently audible speech:

1. `TurnCoordinator` enters pending state.
2. `VoiceOutput.pause_for_interruption()` requests immediate output abort.
3. The returned pause receipt identifies the exact audible generation and saves resumable playback position.
4. `TurnCoordinator` stores that `paused_generation` as the only generation this interruption is allowed to commit against.
5. The LLM generation is **not cancelled yet**.
6. Playback cannot advance to later clauses while pending.
7. Existing one-waveform-ahead synthesis remains bounded.
8. Capture continues until the overlapping human utterance ends.
9. `state.speaking` becomes false only after physical audio is paused.
10. `state.listening` becomes true for the accepted candidate.
11. Transient performance clears while no sound is playing.

Duplicate candidate notifications are idempotent.

### Phase B1: Commit Real Interruption

A non-empty accepted transcript commits the interruption.

Commit order:

1. `ConversationWorker.cancel_generation(paused_generation, reason="barge_in")` attempts to invalidate **only** the generation that was paused.
2. `VoiceOutput.commit_interruption(paused_generation)` permanently discards that generation's saved remainder and lets freshness discard its queued/prepared work.
3. The transcript is submitted normally, allocating a new generation.
4. Existing newest-intent rules handle any later race.

`cancel_generation()` is compare-and-cancel behavior. If `paused_generation` is already stale because another request arrived first, it must not cancel the newer generation. It simply reports that the expected generation was no longer current.

Generation identifiers remain monotonic and are never reused.

Immediate cancellation of Ollama's underlying inference process is not required in V1. An old stream stops contributing clauses as soon as it observes stale generation state and closes its response promptly afterward.

### Phase B2: Roll Back False Interruption

If no real transcript materializes:

1. no generation cancellation occurs
2. no synthetic request is created
3. the original generation remains current unless independently superseded
4. `VoiceOutput.resume_after_false_interruption(paused_generation)` resumes the saved waveform only if that same generation is still current
5. original performance returns only when resumed audio physically starts
6. normal subsequent clauses from that generation may continue
7. the false interruption is recorded

If any newer generation appeared while paused, rollback is forbidden and the old waveform is discarded.

V1 does not semantically classify "mhm", "right", "okay", etc. as backchannels. A non-empty accepted transcript is treated as a real user turn. Backchannel policy comes later only if actual data justifies it.

---

## 6. False-Interruption Timing Must Not Race Whisper

A timer must never resume Vess merely because local transcription is legitimately taking longer than expected.

Pending interruption has internal decision phases:

```text
CAPTURING -> TRANSCRIBING -> DECIDED
```

Rules:

- no rollback while overlapping speech is still being assembled
- once a valid utterance is queued for transcription, ordinary false-silence timeout is suspended
- non-empty transcript commits
- empty transcript or explicit transcription failure rolls back if the paused generation is still current
- a bounded watchdog prevents a broken transcriber from leaving Vess paused forever

Initial values:

```text
false_interruption_timeout_seconds = 2.0
max_interruption_decision_seconds = 5.0
```

The false timeout covers a candidate that ends without entering a valid transcription decision. The decision watchdog bounds the period after speech end while transcription is in flight.

On watchdog expiry: record error and roll back only if the paused generation remains current; otherwise discard the old paused response.

---

## 7. Streaming Cancellable `AudioPlayer`

The existing high-level `sounddevice.play()` / `wait()` helper hides state needed for fast abort, resume, and render reference.

Introduce an `AudioPlayer` owned by `VoiceOutput`.

```python
@dataclass(frozen=True)
class PlaybackReceipt:
    status: str  # completed | paused | interrupted | error
    generation_id: int | None
    frames_started: int
    frames_completed: int
    total_frames: int
    sample_rate: int
```

Conceptual interface:

```python
class AudioPlayer(Protocol):
    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generation_id: int | None,
    ) -> PlaybackReceipt:
        ...

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        ...

    def resume(self) -> PlaybackReceipt:
        ...

    def discard_paused(self) -> None:
        ...
```

`play()` may block the dedicated playback worker. `pause_for_interruption()` must be safe from another worker.

The production implementation uses stream-level sounddevice/PortAudio output. Barge-in uses immediate **abort** semantics rather than graceful stop semantics that may wait for queued device buffers.

The player owns current waveform, sample rate, generation identity, cursor/completed-frame estimate, paused remainder, and render-reference publication.

A tiny replay overlap is preferable to skipping speech if device buffering makes the exact audible cursor uncertain. Target-PC testing decides whether overlap is necessary.

The real-time callback cannot call LLM/TTS/SQLite/event logging/file I/O or unpredictable blocking code.

---

## 8. `VoiceOutput` Interruption and Drain Behavior

Add:

```python
pause_for_interruption()
commit_interruption(generation_id)
resume_after_false_interruption(generation_id)
finish_generation(generation_id)
```

### Pause

`pause_for_interruption()`:

- is idempotent
- pauses only current audible playback
- returns the paused generation identity
- records pause timing and cursor
- prevents playback progression while pending
- clears active performance only after physical pause

### Commit

`commit_interruption(generation_id)`:

- permanently discards paused remainder only for the supplied generation
- never resumes stale audio
- relies on normal freshness for queued/prepared stale work

### Resume

`resume_after_false_interruption(generation_id)`:

- verifies pending generation identity and freshness
- resumes the same waveform rather than resynthesizing
- restores original performance only at physical resumed playback start
- then reopens progression to later clauses

### Generation drain marker

After the LLM stream ends and all clauses are enqueued, `ConversationWorker` calls:

```python
voice.finish_generation(generation_id)
```

The marker must preserve ordering through synthesis/playback. `generation_playback_drained` occurs only after every preceding playable clause has completed or been explicitly abandoned.

LLM completion alone is not playback completion.

The existing one-ready-waveform-ahead bound remains unchanged.

---

## 9. Generation-Specific Conversation Cancellation

Add:

```python
ConversationWorker.cancel_generation(
    expected_generation: int,
    reason: str,
) -> bool
```

It must atomically:

- compare `expected_generation` with the currently valid generation
- advance freshness only when they match
- make that expected generation stale
- never cancel a generation that appeared later
- not enqueue an empty/fake request
- preserve already pending newer requests
- record cancellation reason and old/new IDs
- return whether the expected generation was actually cancelled
- be idempotent if the expected generation is already stale

A later `submit(transcript)` allocates another generation normally.

This compare-and-cancel rule is essential: a delayed barge-in decision may never destroy a newer request that arrived during the pause window.

---

## 10. Delivery-Aware Conversation Memory

Generated text is not delivered text.

Replace "remember generated clauses at LLM completion" with a generation-scoped delivery ledger containing:

- user request
- generated clauses
- fully completed playback clauses
- paused/interrupted clause, if any
- LLM-finished flag
- playback-drained flag
- final status: completed or interrupted

Extend short-term turns:

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

Finalize only after both:

- LLM generation has ended and sent `finish_generation`
- corresponding playback-drained marker has been reached

`assistant` contains fully completed speech.

### Real interruption

Finalize the paused old generation as `interrupted` when commit occurs.

- `assistant` contains only fully completed clauses
- `interrupted_clause` may contain the full text of the clause that started but did not complete
- prompt construction must not assume the whole interrupted clause was heard

If no clause completed, retain the user request with `assistant=""` and interrupted status. The previous user question must not disappear simply because Vess was cut off early.

Prompt rendering may use:

```text
User: <prior request>
Vess (interrupted): <fully completed speech, if any>
Vess had started another clause but was interrupted; do not assume the user heard all of it.
```

No word-level claim is made about the partial clause.

### Required generation-scoped receipts

Track:

- clause playback started
- clause completed
- clause paused
- clause resumed
- clause abandoned on commit
- generation playback drained

Late/stale receipts from a finalized generation are ignored and cannot mutate newer memory.

---

## 11. `TurnCoordinator`

Introduce one small policy coordinator.

Public conceptual states:

```text
IDLE
AGENT_SPEAKING
INTERRUPTION_PENDING
```

Pending state also stores:

- `paused_generation`
- candidate timestamps
- current decision phase: capturing/transcribing
- watchdog/false-timeout state

Transitions:

```text
IDLE -> AGENT_SPEAKING
    physical playback starts

AGENT_SPEAKING -> INTERRUPTION_PENDING
    detector emits first credible candidate

INTERRUPTION_PENDING -> IDLE/user-turn flow
    non-empty transcript commits

INTERRUPTION_PENDING -> AGENT_SPEAKING
    false decision and paused generation still current

INTERRUPTION_PENDING -> IDLE
    false decision but paused generation already stale
```

The coordinator:

- makes duplicate candidate/commit/rollback harmless
- never holds `State.lock` while calling blocking audio/transcription/TTS/conversation methods
- does not perform DSP
- does not build prompts or mutate memory directly

---

## 12. Runtime State Semantics

`State` remains authoritative public runtime state.

`state.listening` means meaningful accepted human speech activity, not raw microphone energy.

Raw VAD/self-echo during speaker playback stays diagnostic until preprocessing/detection accepts it.

During pending interruption after physical pause:

```text
speaking = False
listening = True
performance = neutral
```

On false rollback:

- listening clears
- speaking becomes true only when resumed sound physically starts
- original performance returns at the same start point

This preserves current face priority behavior.

---

## 13. Wake and Follow-Up Semantics

A committed barge-in transcript is automatically part of the active conversation and does not require a wake phrase.

When idle, existing wake/follow-up behavior is unchanged.

The acoustic candidate never becomes text and never submits a request.

---

## 14. Error Handling

### Player cannot pause/abort

- record `barge_in_pause_error`
- do not claim physical pause succeeded
- keep capturing the utterance
- if real transcript arrives, compare-and-cancel the expected old generation so no later stale clauses start after current playback returns

### Preprocessor fails

- record `barge_in_preprocessor_error`
- if `disable_on_preprocessor_error=true`, disable speaker-time barge-in for the session
- ordinary idle capture may continue if raw microphone still works

### Empty transcript

- roll back only if paused generation remains current
- otherwise discard old paused state

### Transcription error/watchdog expiry

- record error/timeout
- roll back only if paused generation remains current
- otherwise discard
- never fabricate text

### New request during pending interruption

Newest intent wins. Compare-and-cancel may not cancel that newer generation, and the old paused waveform may never resume.

### Shutdown while paused

Abort/close player, discard paused state, clear speaking/listening/performance, cancel timers, and do not wait for false timeout.

---

## 15. Configuration

Initial non-final defaults:

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

Barge-in stays disabled by default until target-PC tests prove self-echo rejection with the actual microphone and speakers.

---

## 16. Diagnostics and Metrics

Record generation-scoped events/timestamps for:

- credible candidate start
- pause requested
- physical pause
- speech-start-to-pause latency
- paused generation and cursor
- pending phase changes
- transcription start/finish
- commit/rollback result
- compare-and-cancel success/failure
- resumed playback start
- stale clauses discarded
- interrupted clause metadata
- new-response first-clause latency
- preprocessor error
- player abort error

Target-PC metrics:

- p50/p95 human-speech-start -> audible pause
- real interruption detection rate
- missed interruption rate
- false interruptions/hour
- self-echo interruptions/hour
- false-resume success rate
- stale clauses played after commit
- never-completed clauses incorrectly stored as fully delivered

Hard production targets:

```text
stale clauses played after committed interruption = 0
never-completed clauses recorded as fully delivered = 0
newer generation cancelled by delayed old interruption = 0
```

---

## 17. Hard Invariants

Deterministic tests must prove:

1. Commit compare-and-cancels only the exact paused generation.
2. A newer generation can never be cancelled by a delayed old interruption.
3. No stale prepared clause starts playback after committed cancellation.
4. Reversible pause alone does not cancel the LLM generation.
5. False rollback resumes only the same still-current generation.
6. Any newer generation permanently forbids old paused-waveform resume.
7. Performance is neutral while physical playback is paused.
8. Original performance returns only at resumed physical playback start.
9. Raw speaker echo/VAD alone cannot become meaningful listening state without candidate acceptance.
10. Generated but never completed clauses are never marked fully delivered.
11. Interrupted turns remain in short-term context.
12. Duplicate candidate/commit/rollback/cancel operations are idempotent.
13. Barge-in queues remain bounded during pause.
14. False rollback cannot fire while a valid transcription decision is in flight except via explicit watchdog expiry.
15. Generation-drained marker cannot overtake preceding speech.
16. Late receipts cannot mutate finalized or newer turns.
17. Barge-in disabled preserves existing idle/wake/follow-up behavior.
18. The 30 FPS face loop never waits on barge-in work.

---

## 18. Remote Test Strategy

CI uses no real microphone, speakers, Ollama, Whisper model, TTS model, or CUDA.

Use deterministic fakes for AudioPlayer, preprocessor, transcriber, TTS engine, LLM stream, and clocks/timers.

Required scenarios:

### Real interruption

```text
playback starts
candidate threshold reached
player pauses and returns generation G
utterance transcribes non-empty
cancel_generation(G) succeeds
paused G audio commits/discards
new transcript is submitted
old G queued/prepared speech never plays
old turn finalizes interrupted
```

### False interruption before transcription

```text
player pauses G
candidate collapses without valid utterance
false timeout expires
G still current
same waveform resumes
```

### Empty transcript

Valid captured utterance enters transcription, returns empty, and G resumes only if still current.

### Slow valid transcription

Transcription remains in flight longer than ordinary false timeout; Vess does not resume early; later non-empty transcript commits.

### Decision watchdog

Transcription hangs; watchdog records error and resumes only if G remains current.

### New request before commit

```text
G is paused
new request creates H
old barge-in transcript arrives late
cancel_generation(G) must not cancel H
G never resumes
```

### Interrupt during first clause

No completed assistant clause exists; prior user request remains as interrupted memory.

### Interrupt between clauses

Completed clauses remain delivered; prepared next clause is abandoned after commit.

### Normal generation drain

`finish_generation(G)` cannot finalize memory until all preceding physical playback completes.

### Duplicate candidate

Two candidate events cause one pause.

### Commit racing synthesis

Waveform created at race boundary becomes stale and never plays.

### Shutdown while paused

No deadlock, delayed resume, or stale state.

### Feature disabled

Current behavior remains unchanged.

---

## 19. Target-PC Acceptance

Before enabling barge-in by default:

1. choose/integrate actual local echo preprocessing
2. run Vess speaking at realistic speaker volume with no human speech and verify self-echo does not pause it
3. test interruptions at multiple distances/directions/voice volumes
4. measure p50/p95 pause latency
5. test very short real interruptions such as "wait"
6. test coughs, keyboard/room noise, and accidental speech-like sounds
7. test speech beginning immediately after Vess starts and near clause end
8. verify false-interruption resume sounds natural
9. verify Qwen, Whisper, TTS, preprocessing, and rendering coexist within CPU/RAM/VRAM limits
10. inspect traces for stale playback, timer races, compare-and-cancel mistakes, and delivery-memory errors
11. tune timing only from measured results

Only then should `barge_in.enabled` become true by default.

---

## 20. Implementation Boundaries

Likely production files/modules:

- `perception/audio.py`
- new focused audio-preprocessing module
- new `output/audio_player.py`
- `output/voice.py`
- `brain/llm.py`
- `brain/memory.py`
- new small turn-coordination module
- `state.py`
- `main.py`
- `config.json`

Tests extend audio, TTS pipeline, voice freshness, conversation freshness, short-term memory, coordinator, and player behavior.

Do not combine this feature with long-term memory, proactive behavior, planner work, or TTS-engine benchmarking.

## References Informing the Design

- Current LiveKit turn handling separates interruption detection from turn policy, supports false-interruption recovery, and truncates history around delivered speech.
- WebRTC AEC3 is built around capture plus render-reference processing rather than raw VAD alone.
- python-sounddevice/PortAudio stream APIs distinguish graceful stop from immediate abort; immediate abort is the appropriate primitive for fast barge-in pause.

These references inform architecture only. They are not automatically runtime dependencies.