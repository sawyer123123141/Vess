# Vess Barge-In and Natural Turn-Taking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a person interrupt Vess while it is speaking, pause quickly, commit only real interruptions, resume false interruptions, and remember only speech that was actually delivered.

**Architecture:** Keep the runtime asynchronous. Add small preprocessing, interruption-detection, playback, delivery-ledger, and coordination boundaries while preserving generation freshness and one-clause-ahead TTS. The remotely testable implementation uses passthrough preprocessing and keeps `barge_in.enabled=false` until target-PC acoustic acceptance.

**Tech Stack:** Python 3.11, `unittest`, NumPy, sounddevice/PortAudio at runtime only, existing Ollama/faster-whisper/TTS stack.

**Spec:** `docs/superpowers/specs/2026-09-01-barge-in-turn-taking-design.md`

## Global Constraints

- Preserve the independent 30 FPS face loop.
- Preserve the one-ready-waveform-ahead TTS bound.
- Barge-in stays disabled by default until real microphone/speaker acceptance.
- CI must not require a microphone, speakers, Ollama, Whisper model, TTS model, CUDA, or a real AEC library.
- No learned interruption classifier in V1.
- Raw speaker-time VAD can create only a reversible candidate, never a committed interruption.
- `stale clauses played after committed interruption = 0`.
- `never-completed clauses recorded as fully delivered = 0`.
- `newer generation cancelled by delayed old interruption = 0`.
- No timer/callback may hold `State.lock` while calling blocking audio, transcription, TTS, or conversation work.

---

## File Structure

Create:
- `perception/audio_preprocess.py` — capture/render value types and passthrough preprocessing.
- `perception/interruption.py` — deterministic sustained-speech candidate detector.
- `output/audio_player.py` — playback receipt contract and cancellable/resumable player.
- `brain/delivery.py` — generation-scoped delivery ledger.
- `brain/turn_coordinator.py` — two-phase interruption policy and timers.
- `tests/test_audio_preprocess.py`
- `tests/test_interruption.py`
- `tests/test_audio_player.py`
- `tests/test_delivery.py`
- `tests/test_turn_coordinator.py`
- `tests/test_barge_in_flow.py`

Modify:
- `perception/audio.py`
- `output/voice.py`
- `brain/llm.py`
- `brain/memory.py`
- `state.py`
- `main.py`
- `config.json`
- existing focused tests for those modules.

---

### Task 1: Capture preprocessing and interruption detector

**Files:**
- Create: `perception/audio_preprocess.py`
- Create: `perception/interruption.py`
- Create: `tests/test_audio_preprocess.py`
- Create: `tests/test_interruption.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CapturedAudioBlock:
    samples: np.ndarray
    adc_time: float | None
    received_at: float

@dataclass(frozen=True)
class RenderedAudioBlock:
    samples: np.ndarray
    sample_rate: int
    dac_time: float | None

class CapturePreprocessor(Protocol):
    def push_render_reference(self, block: RenderedAudioBlock) -> None:
        raise NotImplementedError

    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray:
        raise NotImplementedError

class PassthroughCapturePreprocessor:
    def push_render_reference(self, block: RenderedAudioBlock) -> None:
        return None

    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray:
        return np.asarray(block.samples, dtype=np.float32).reshape(-1).copy()

class InterruptionDetector:
    def __init__(self, sample_rate: int, threshold: float, pause_after_speech_seconds: float) -> None:
        ...

    def push(self, samples: np.ndarray) -> bool:
        ...

    def reset(self) -> None:
        ...
```

For the implementation of `InterruptionDetector`, use `ceil(sample_rate * pause_after_speech_seconds)` accepted samples as the required duration; quiet samples reset the count; after one candidate is emitted, do not emit again until quiet audio or `reset()`.

- [ ] **Step 1: Write failing tests**

`tests/test_audio_preprocess.py`:

```python
import unittest
import numpy as np
from perception.audio_preprocess import CapturedAudioBlock, PassthroughCapturePreprocessor, RenderedAudioBlock

class AudioPreprocessTests(unittest.TestCase):
    def test_passthrough_returns_float32_copy(self) -> None:
        source = np.array([0.1, -0.2], dtype=np.float64)
        block = CapturedAudioBlock(source, 1.0, 2.0)
        result = PassthroughCapturePreprocessor().process_capture(block)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [0.1, -0.2])
        self.assertIsNot(result, source)

    def test_render_reference_does_not_change_passthrough_capture(self) -> None:
        preprocessor = PassthroughCapturePreprocessor()
        preprocessor.push_render_reference(RenderedAudioBlock(np.array([0.8], dtype=np.float32), 24_000, None))
        result = preprocessor.process_capture(CapturedAudioBlock(np.array([0.2], dtype=np.float32), None, 1.0))
        np.testing.assert_allclose(result, [0.2])
```

`tests/test_interruption.py`:

```python
import unittest
import numpy as np
from perception.interruption import InterruptionDetector

class InterruptionDetectorTests(unittest.TestCase):
    def test_requires_sustained_speech(self) -> None:
        detector = InterruptionDetector(10, 0.1, 0.3)
        self.assertFalse(detector.push(np.array([0.2, 0.2])))
        self.assertTrue(detector.push(np.array([0.2])))
        self.assertFalse(detector.push(np.array([0.2])))

    def test_quiet_resets_progress(self) -> None:
        detector = InterruptionDetector(10, 0.1, 0.3)
        self.assertFalse(detector.push(np.array([0.2, 0.2])))
        self.assertFalse(detector.push(np.array([0.0])))
        self.assertFalse(detector.push(np.array([0.2, 0.2])))

    def test_reset_allows_future_candidate(self) -> None:
        detector = InterruptionDetector(10, 0.1, 0.2)
        self.assertTrue(detector.push(np.array([0.2, 0.2])))
        detector.reset()
        self.assertTrue(detector.push(np.array([0.2, 0.2])))
```

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_audio_preprocess tests.test_interruption -v
```

Expected: import failures for the new modules.

- [ ] **Step 3: Implement the interfaces and detector behavior exactly above**

- [ ] **Step 4: Run GREEN and full suite**

```bash
python -m unittest tests.test_audio_preprocess tests.test_interruption -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```bash
git add perception/audio_preprocess.py perception/interruption.py tests/test_audio_preprocess.py tests/test_interruption.py
git commit -m "feat: add barge-in audio detection contracts"
```

---

### Task 2: Cancellable/resumable `AudioPlayer`

**Files:**
- Create: `output/audio_player.py`
- Create: `tests/test_audio_player.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PlaybackReceipt:
    status: str
    generation_id: int | None
    frames_started: int
    frames_completed: int
    total_frames: int
    sample_rate: int

class AudioPlayer(Protocol):
    def play(self, audio: np.ndarray, sample_rate: int, generation_id: int | None) -> PlaybackReceipt:
        raise NotImplementedError

    def pause_for_interruption(self) -> PlaybackReceipt | None:
        raise NotImplementedError

    def resume(self) -> PlaybackReceipt:
        raise NotImplementedError

    def discard_paused(self) -> None:
        raise NotImplementedError
```

`SoundDeviceAudioPlayer` owns a dedicated lock protecting current/paused metadata, but does not hold it while the output backend blocks. Accept an injectable backend object in tests. Runtime backend imports `sounddevice` lazily and uses stream abort semantics.

- [ ] **Step 1: Write a deterministic fake backend and failing tests**

The fake backend exposes `started = threading.Event()`, `release = threading.Event()`, `abort_requested = threading.Event()`, and `frames_written`. Its `play` loop signals `started`, renders one frame at a time, exits when `abort_requested` is set, and otherwise blocks briefly on `release` between frames. This avoids busy-wait races.

Tests must prove:
- normal playback returns `status="completed"` with all frames completed;
- pausing from another thread returns `status="paused"`, exact generation ID, and a resumable remainder;
- `resume()` completes the same generation without resynthesis;
- `discard_paused()` makes `resume()` raise `RuntimeError`;
- render callback receives only frames that the backend actually rendered.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_audio_player -v
```

- [ ] **Step 3: Implement the player**

Use an internal `_PausedPlayback(audio, sample_rate, generation_id, completed_frames)` dataclass. `pause_for_interruption()` requests backend abort and returns a receipt only after/if current playback is known. `resume()` slices from `completed_frames` and calls the same `play` path. `discard_paused()` clears saved state.

- [ ] **Step 4: Run GREEN + full suite**

```bash
python -m unittest tests.test_audio_player -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```bash
git add output/audio_player.py tests/test_audio_player.py
git commit -m "feat: add cancellable audio player"
```

---

### Task 3: Make `VoiceOutput` interruptible and delivery-aware

**Files:**
- Modify: `output/voice.py`
- Modify: `tests/test_voice.py`
- Modify: `tests/test_performance_flow.py`

**Interfaces:**

```python
VoiceOutput(
    config,
    state,
    event_log,
    synthesize=None,
    play=None,
    engine=None,
    player: AudioPlayer | None = None,
    on_delivery: Callable[[str, dict[str, object]], None] | None = None,
)

pause_for_interruption() -> PlaybackReceipt | None
commit_interruption(generation_id: int) -> bool
resume_after_false_interruption(generation_id: int) -> bool
finish_generation(generation_id: int) -> None
```

Delivery event names are fixed:
- `clause_started`
- `clause_completed`
- `clause_paused`
- `clause_resumed`
- `clause_abandoned`
- `generation_playback_drained`

- [ ] **Step 1: Add RED tests**

Prove:
1. pausing generation 11 clears `state.speaking` and `state.performance` only after the fake player confirms pause;
2. false resume of generation 11 restores the original performance cue only while resumed physical playback is active;
3. committing generation 11 discards its paused remainder and a second commit is harmless;
4. `finish_generation(13)` cannot emit `generation_playback_drained` before two preceding clauses report `clause_completed`;
5. an old/stale prepared clause still never starts.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_voice tests.test_performance_flow -v
```

- [ ] **Step 3: Integrate `AudioPlayer`**

If tests pass legacy `play=callback`, wrap it in `CallbackAudioPlayer` so existing test ergonomics remain. Production default is `SoundDeviceAudioPlayer`. Do not change TTS synthesis queue or ready-queue bounds.

Add `finish` as an ordered queue item. It must pass through synthesis ordering and playback ordering before producing `generation_playback_drained`.

- [ ] **Step 4: Implement pause/commit/resume methods**

Track one pending paused generation and clause metadata. No playback worker may advance to the next clause while interruption is pending. Commit only matching generation. Resume only matching generation that is still current. Emit generation-scoped delivery callbacks for every physical lifecycle transition.

- [ ] **Step 5: Run GREEN + full suite**

```bash
python -m unittest tests.test_voice tests.test_performance_flow -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```bash
git add output/voice.py tests/test_voice.py tests/test_performance_flow.py
git commit -m "feat: make voice playback interruptible"
```

---

### Task 4: Generation-safe cancellation

**Files:**
- Modify: `brain/llm.py`
- Modify: `tests/test_llm.py`

**Interface:**

```python
def cancel_generation(self, expected_generation: int, reason: str) -> bool:
    ...
```

Implementation rule: inside `_request_lock`, cancel only when `expected_generation == _latest_generation`. Allocate a new monotonic invalidation generation, update `_latest_generation`, and call `voice.begin_generation(new_generation)` after leaving the lock. Do not remove or overwrite a separate newer pending request.

- [ ] **Step 1: RED tests**

Prove exact generation cancels successfully, duplicate cancel is harmless, and delayed cancel of G after H exists returns false while `_latest_generation` remains H.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_llm -v
```

- [ ] **Step 3: Implement compare-and-cancel and diagnostics**

Record `generation_cancelled` with `expected_generation`, `replacement_generation`, and `reason`. Return `True` only when cancellation actually occurred.

- [ ] **Step 4: Run GREEN + full suite**

```bash
python -m unittest tests.test_llm -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```bash
git add brain/llm.py tests/test_llm.py
git commit -m "feat: add generation-safe conversation cancellation"
```

---

### Task 5: Delivery ledger and interrupted memory

**Files:**
- Create: `brain/delivery.py`
- Create: `tests/test_delivery.py`
- Modify: `state.py`
- Modify: `brain/memory.py`
- Modify: `brain/llm.py`
- Modify: `tests/test_short_term_memory.py`
- Modify: `tests/test_llm.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ConversationTurn:
    timestamp: float
    user: str
    assistant: str
    status: str = "completed"
    interrupted_clause: str | None = None
```

`append_conversation_turn` gains keyword arguments `status="completed"` and `interrupted_clause=None`.

`DeliveryLedger` methods:

```python
begin(generation_id: int, user_request: str) -> None
generated(generation_id: int, text: str) -> None
handle(event_type: str, payload: dict[str, object]) -> None
llm_finished(generation_id: int) -> None
interrupt(generation_id: int) -> None
```

The ledger constructor receives a finalize callback `Callable[[int, str, str, str, str | None], None]` with arguments `(generation_id, user, assistant, status, interrupted_clause)`.

- [ ] **Step 1: RED compatibility and interrupted-turn tests**

Existing memory tests must still pass unchanged. Add an interrupted append test and prompt rendering test. Interrupted prompts may include fully completed assistant text, but must contain the sentence `Vess had started another clause but was interrupted; do not assume the user heard all of it.` and must not present the partial clause as certainly heard.

- [ ] **Step 2: RED ledger tests**

Normal flow finalizes only after `llm_finished` plus `generation_playback_drained`. Interrupted flow with completed A and active B finalizes `assistant="A"`, `status="interrupted"`, `interrupted_clause="B"`. Late receipts after finalization are ignored.

- [ ] **Step 3: Implement pure ledger state**

Use one lock and a private per-generation dataclass. The module performs no audio, LLM, SQLite, timer, or State mutation itself; it only invokes the supplied finalize callback once.

- [ ] **Step 4: Wire `ConversationWorker`**

At response start call `ledger.begin`. On each generated clause call `ledger.generated`. At stream end call `ledger.llm_finished` then `voice.finish_generation`; remove the existing immediate `_remember_completed_turn` completion path. Voice delivery callbacks feed `ledger.handle`.

On successful barge-in cancellation call `ledger.interrupt(expected_generation)` so the old user request remains context even if no assistant clause completed.

- [ ] **Step 5: Run GREEN + full suite**

```bash
python -m unittest tests.test_delivery tests.test_short_term_memory tests.test_llm -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```bash
git add brain/delivery.py state.py brain/memory.py brain/llm.py tests/test_delivery.py tests/test_short_term_memory.py tests/test_llm.py
git commit -m "feat: track delivered speech in conversation memory"
```

---

### Task 6: Two-phase `TurnCoordinator`

**Files:**
- Create: `brain/turn_coordinator.py`
- Create: `tests/test_turn_coordinator.py`

**Interface:**

```python
class TurnCoordinator:
    def on_candidate(self) -> bool:
        ...

    def on_utterance_queued_for_transcription(self) -> None:
        ...

    def on_transcript(self, text: str) -> None:
        ...

    def on_transcription_error(self, error: Exception) -> None:
        ...

    def close(self) -> None:
        ...
```

Constructor dependencies: `state`, `event_log`, `voice`, `conversation`, `submit_transcript`, `false_timeout_seconds`, `decision_watchdog_seconds`, injectable timer factory.

- [ ] **Step 1: RED tests**

Prove:
- real transcript call order is `cancel_generation(G)`, `commit_interruption(G)`, `submit_transcript(text)`;
- false timeout resumes G exactly once when still current;
- entering transcription cancels/suspends ordinary false timeout;
- decision watchdog rolls back only if G is still current;
- if H appears during pending G, delayed commit cannot cancel H and G never resumes;
- duplicate candidate/commit/rollback notifications are idempotent;
- `close()` cancels timers and never resumes later.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_turn_coordinator -v
```

- [ ] **Step 3: Implement explicit phases**

Use `IDLE`, `PENDING_CAPTURE`, `PENDING_TRANSCRIBE`, `CLOSED`. Under the coordinator lock, only mutate phase/generation/timer handles and decide which effects must happen. Release the lock before calling voice/conversation/state-facing operations.

Candidate pause stores exact `paused_generation` returned by the voice receipt. Non-empty transcript commits. Empty transcript or transcription error rolls back. If rollback cannot resume because generation became stale, call `voice.commit_interruption(G)` to discard paused audio.

- [ ] **Step 4: Run GREEN + full suite**

```bash
python -m unittest tests.test_turn_coordinator -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```bash
git add brain/turn_coordinator.py tests/test_turn_coordinator.py
git commit -m "feat: add reversible barge-in coordinator"
```

---

### Task 7: Always-live capture during Vess speech

**Files:**
- Modify: `perception/audio.py`
- Modify: `tests/test_audio.py`

**Interface changes:**

`AudioLoop` gains optional `preprocessor`, `interruption_detector`, and `turn_coordinator` dependencies. When barge-in is disabled, current speaking-time discard behavior remains unchanged. When enabled, microphone blocks are preprocessed and interruption candidates/overlapping utterances are routed to the coordinator.

- [ ] **Step 1: RED disabled-mode regression test**

With `barge_in.enabled=false` and `state.speaking=True`, current block ignore/debug behavior must remain.

- [ ] **Step 2: RED enabled candidate test**

With enabled mode, speaking state, passthrough preprocessing, and a fake detector that emits a candidate, raw energy alone must not set listening; after coordinator accepts candidate, `state.listening` becomes true.

- [ ] **Step 3: RED transcription-phase test**

When overlapping utterance completes, call `coordinator.on_utterance_queued_for_transcription()` before transcription begins. Its transcript result goes to `coordinator.on_transcript(text)` and does not pass through wake matching. Empty transcript is passed as empty for rollback.

- [ ] **Step 4: Preserve capture timing metadata**

Input callback creates `CapturedAudioBlock`; use `time_info.inputBufferAdcTime` when available and `time.perf_counter()` for `received_at`. Keep callback non-blocking.

- [ ] **Step 5: Fail closed on preprocessing error**

With configured fail-closed behavior, record `barge_in_preprocessor_error` and disable only speaking-time barge-in for the session. Idle microphone/wake behavior remains functional.

- [ ] **Step 6: Run GREEN + full suite**

```bash
python -m unittest tests.test_audio -v
python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit**

```bash
git add perception/audio.py tests/test_audio.py
git commit -m "feat: capture speech during Vess playback"
```

---

### Task 8: Runtime wiring and dormant config

**Files:**
- Modify: `main.py`
- Modify: `config.json`
- Modify: `tests/test_main.py`

Add exactly:

```json
"barge_in": {
  "enabled": false,
  "pause_after_speech_seconds": 0.25,
  "false_interruption_timeout_seconds": 2.0,
  "max_interruption_decision_seconds": 5.0,
  "preprocessor": "passthrough",
  "disable_on_preprocessor_error": true
}
```

- [ ] **Step 1: RED repository-config test**

Assert repository config contains the block and `enabled` is false.

- [ ] **Step 2: RED builder/wiring test**

Create a focused `_build_voice_stack` or `_build_barge_in_components` helper that accepts/injects fakes and does not open hardware. Test that player render callback feeds the same preprocessor instance later passed to `AudioLoop`, and coordinator uses the same conversation/voice instances.

- [ ] **Step 3: Implement wiring**

Construct player, voice, conversation, preprocessor, detector, coordinator, then audio. Avoid circular construction by allowing delivery callback/coordinator hooks to be attached explicitly after objects exist if needed.

- [ ] **Step 4: Shutdown order**

Close coordinator timers before conversation/voice teardown; paused audio must be discarded and no timer callback may resume after shutdown.

- [ ] **Step 5: Run GREEN + full suite**

```bash
python -m unittest tests.test_main -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```bash
git add main.py config.json tests/test_main.py
git commit -m "feat: wire dormant barge-in runtime"
```

---

### Task 9: Cross-component race verification

**Files:**
- Create: `tests/test_barge_in_flow.py`
- Modify production only when a failing integration case demonstrates a concrete bug.

- [ ] **Step 1: Real interruption scenario**

Exercise G playback -> candidate -> pause -> overlapping utterance -> non-empty transcription -> exact G cancel -> paused G discard -> new request submission. Assert no stale G clause starts afterward and G finalizes interrupted using only completed speech.

- [ ] **Step 2: False interruption scenario**

Pause G, produce no valid transcript, trigger false decision, resume same G waveform, drain normally, and finalize G as completed.

- [ ] **Step 3: Slow transcription scenario**

Pause G, mark transcription in flight, advance fake timer beyond 2 seconds, assert no resume, then resolve transcript before 5-second watchdog and commit.

- [ ] **Step 4: Newer request race**

Pause G, independently submit H, then resolve old G barge-in. Assert delayed `cancel_generation(G)` cannot cancel H and G never resumes.

- [ ] **Step 5: Shutdown while paused**

Close coordinator/voice while pending; assert no delayed resume, deadlock, speaking/listening flag, or non-neutral performance remains.

- [ ] **Step 6: Full verification**

```bash
python -m unittest tests.test_barge_in_flow -v
python -m unittest discover -s tests -v
python tools/render_behavior_preview.py
python tools/render_eye_validation.py
```

Expected: every command exits 0. CI still installs only `requirements-ci.txt` and requires no audio hardware/model stack.

- [ ] **Step 7: Scope audit**

Compare `design/barge-in-turn-taking` against `design/independent-eye-motion`. Allowed scope: approved spec/plan, focused barge-in modules, audio/player/voice/conversation/memory/state/main/config wiring, and tests. Confirm `barge_in.enabled=false`.

---

## Target-PC Acceptance After Remote Implementation

Remote implementation success does not enable the feature. On the actual PC:

1. select/integrate a real local echo-preprocessing backend if passthrough is unsafe with speakers;
2. run Vess speech with no human speech and measure self-echo interruption rate;
3. test real interruptions at multiple distances, directions, and voice levels;
4. measure p50/p95 human-speech-start -> audible pause;
5. test short interruptions such as `wait`;
6. test coughs, keyboard/room noise, and other speech-like noise;
7. test interruptions immediately after playback start and near clause boundaries;
8. judge false-interruption resume quality;
9. verify Qwen, Whisper, TTS, detector, preprocessing, and rendering resource coexistence;
10. change `barge_in.enabled` only after measured acceptance.
