# Vess Barge-In and Natural Turn-Taking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a person interrupt Vess while it is speaking, pause quickly, commit only real interruptions, resume false interruptions, and remember only speech that was actually delivered.

**Architecture:** Keep the runtime asynchronous. Add focused preprocessing, interruption-detection, playback, delivery-ledger, and coordination boundaries while preserving generation freshness and the one-clause-ahead TTS bound. The remotely testable version uses passthrough preprocessing and keeps `barge_in.enabled=false` until target-PC acoustic acceptance.

**Tech Stack:** Python 3.11, `unittest`, NumPy, sounddevice/PortAudio at runtime only, existing Ollama/faster-whisper/TTS stack.

**Spec:** `docs/superpowers/specs/2026-09-01-barge-in-turn-taking-design.md`

## Global Constraints

- Preserve the independent 30 FPS face loop.
- Preserve the one-ready-waveform-ahead TTS bound.
- Barge-in stays disabled by default until real microphone/speaker acceptance.
- CI must not require audio hardware, Ollama, Whisper model, TTS model, CUDA, or a real AEC library.
- No learned interruption classifier in V1.
- Raw speaker-time VAD can create only a reversible candidate, never a committed interruption.
- `stale clauses played after committed interruption = 0`.
- `never-completed clauses recorded as fully delivered = 0`.
- `newer generation cancelled by delayed old interruption = 0`.
- No timer/callback may hold `State.lock` while calling blocking audio, transcription, TTS, or conversation work.

## File Structure

Create `perception/audio_preprocess.py`, `perception/interruption.py`, `output/audio_player.py`, `brain/delivery.py`, `brain/turn_coordinator.py`, and focused tests for each. Modify `perception/audio.py`, `output/voice.py`, `brain/llm.py`, `brain/memory.py`, `state.py`, `main.py`, `config.json`, and their existing tests.

---

### Task 1: Capture preprocessing and interruption detector

**Files:**
- Create: `perception/audio_preprocess.py`
- Create: `perception/interruption.py`
- Create: `tests/test_audio_preprocess.py`
- Create: `tests/test_interruption.py`

**Produces:** `CapturedAudioBlock`, `RenderedAudioBlock`, `CapturePreprocessor`, `PassthroughCapturePreprocessor`, `InterruptionDetector`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio_preprocess.py
import unittest
import numpy as np
from perception.audio_preprocess import CapturedAudioBlock, PassthroughCapturePreprocessor, RenderedAudioBlock

class AudioPreprocessTests(unittest.TestCase):
    def test_passthrough_returns_float32_copy(self) -> None:
        source = np.array([0.1, -0.2], dtype=np.float64)
        result = PassthroughCapturePreprocessor().process_capture(
            CapturedAudioBlock(source, 1.0, 2.0)
        )
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [0.1, -0.2])
        self.assertIsNot(result, source)

    def test_render_reference_does_not_modify_capture(self) -> None:
        preprocessor = PassthroughCapturePreprocessor()
        preprocessor.push_render_reference(
            RenderedAudioBlock(np.array([0.8], dtype=np.float32), 24_000, None)
        )
        result = preprocessor.process_capture(
            CapturedAudioBlock(np.array([0.2], dtype=np.float32), None, 1.0)
        )
        np.testing.assert_allclose(result, [0.2])
```

```python
# tests/test_interruption.py
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

Expected: imports fail because the modules do not exist.

- [ ] **Step 3: Implement preprocessing contracts**

```python
# perception/audio_preprocess.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

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
```

- [ ] **Step 4: Implement detector**

```python
# perception/interruption.py
from __future__ import annotations
from math import ceil
import numpy as np

class InterruptionDetector:
    def __init__(self, sample_rate: int, threshold: float, pause_after_speech_seconds: float) -> None:
        self._threshold = float(threshold)
        self._required = max(1, ceil(sample_rate * pause_after_speech_seconds))
        self._audible_samples = 0
        self._emitted = False

    def push(self, samples: np.ndarray) -> bool:
        for sample in np.asarray(samples).reshape(-1):
            if abs(float(sample)) < self._threshold:
                self._audible_samples = 0
                self._emitted = False
                continue
            self._audible_samples += 1
            if self._audible_samples >= self._required and not self._emitted:
                self._emitted = True
                return True
        return False

    def reset(self) -> None:
        self._audible_samples = 0
        self._emitted = False
```

- [ ] **Step 5: Run GREEN/full suite and commit**

```bash
python -m unittest tests.test_audio_preprocess tests.test_interruption -v
python -m unittest discover -s tests -v
git add perception/audio_preprocess.py perception/interruption.py tests/test_audio_preprocess.py tests/test_interruption.py
git commit -m "feat: add barge-in audio detection contracts"
```

---

### Task 2: Cancellable/resumable `AudioPlayer`

**Files:**
- Create: `output/audio_player.py`
- Create: `tests/test_audio_player.py`

**Produces:** `PlaybackReceipt`, `AudioPlayer`, `CallbackAudioPlayer`, `SoundDeviceAudioPlayer`.

```python
@dataclass(frozen=True)
class PlaybackReceipt:
    status: str
    generation_id: int | None
    frames_started: int
    frames_completed: int
    total_frames: int
    sample_rate: int
```

- [ ] **Step 1: Write a deterministic fake-backend test suite**

The fake backend uses `threading.Event` for `started`, `release`, and `abort_requested`; it exposes `frames_written`; its play loop calls the supplied render callback only for frames actually written. Tests prove completed playback, pause from another thread, resume of the same generation/remainder, discard preventing resume, and render-reference accuracy.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_audio_player -v
```

- [ ] **Step 3: Implement player state**

Use a private `_PausedPlayback` dataclass containing the original waveform, sample rate, generation ID, and completed-frame cursor. A dedicated player lock protects metadata but is released before backend playback blocks. `pause_for_interruption()` requests backend abort and returns a `paused` receipt; `resume()` starts from saved cursor; `discard_paused()` clears resumable state. Runtime sounddevice import is lazy.

- [ ] **Step 4: Run GREEN/full suite and commit**

```bash
python -m unittest tests.test_audio_player -v
python -m unittest discover -s tests -v
git add output/audio_player.py tests/test_audio_player.py
git commit -m "feat: add cancellable audio player"
```

---

### Task 3: Make `VoiceOutput` interruptible and delivery-aware

**Files:**
- Modify: `output/voice.py`
- Modify: `tests/test_voice.py`
- Modify: `tests/test_performance_flow.py`

**Produces:**

```python
pause_for_interruption() -> PlaybackReceipt | None
commit_interruption(generation_id: int) -> bool
resume_after_false_interruption(generation_id: int) -> bool
finish_generation(generation_id: int) -> None
```

`VoiceOutput.__init__` gains optional `player` and `on_delivery`. Delivery event names are exactly `clause_started`, `clause_completed`, `clause_paused`, `clause_resumed`, `clause_abandoned`, `generation_playback_drained`.

- [ ] **Step 1: RED tests**

Add deterministic player tests proving: pause clears speaking/performance only after physical pause; false resume restores original cue only while resumed sound is playing; commit discards only matching paused generation and is idempotent; a finish marker cannot overtake preceding clauses; stale prepared clauses still never start.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_voice tests.test_performance_flow -v
```

- [ ] **Step 3: Integrate player without changing queue bounds**

Wrap existing `play=callback` tests with `CallbackAudioPlayer`; production default is `SoundDeviceAudioPlayer`. Add an ordered `finish` queue item that reaches playback only after all prior clauses for that generation. Hold playback progression while an interruption is pending.

- [ ] **Step 4: Implement delivery callbacks and interruption methods**

Every physical lifecycle event includes `generation_id` and clause text where applicable. Commit may discard only the exact paused generation; false resume requires that same generation to remain current. Performance becomes neutral during pause and restores at physical resume start.

- [ ] **Step 5: Run GREEN/full suite and commit**

```bash
python -m unittest tests.test_voice tests.test_performance_flow -v
python -m unittest discover -s tests -v
git add output/voice.py tests/test_voice.py tests/test_performance_flow.py
git commit -m "feat: make voice playback interruptible"
```

---

### Task 4: Generation-safe cancellation

**Files:**
- Modify: `brain/llm.py`
- Modify: `tests/test_llm.py`

**Produces:**

```python
def cancel_generation(self, expected_generation: int, reason: str) -> bool:
    with self._request_lock:
        if expected_generation != self._latest_generation:
            return False
        self._next_generation += 1
        replacement = self._next_generation
        self._latest_generation = replacement
    self._voice.begin_generation(replacement)
    self._state.record_debug(
        "generation_cancelled",
        expected_generation=expected_generation,
        replacement_generation=replacement,
        reason=reason,
    )
    return True
```

The final implementation also preserves any separate pending newer request and records the durable event if appropriate.

- [ ] **Step 1: RED tests**

Prove exact G cancels, duplicate G cancellation is harmless, and delayed cancellation of G after H exists returns false without changing H.

- [ ] **Step 2: Run RED, implement, run GREEN/full suite, commit**

```bash
python -m unittest tests.test_llm -v
python -m unittest discover -s tests -v
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

**Produces:**

```python
@dataclass(frozen=True)
class ConversationTurn:
    timestamp: float
    user: str
    assistant: str
    status: str = "completed"
    interrupted_clause: str | None = None
```

`append_conversation_turn` gains `status="completed"` and `interrupted_clause=None`. `DeliveryLedger` exposes `begin`, `generated`, `handle`, `llm_finished`, and `interrupt`; its finalize callback receives `(generation_id, user, assistant, status, interrupted_clause)`.

- [ ] **Step 1: RED tests**

Existing completed-turn tests remain unchanged. Add an interrupted append test. Ledger tests prove normal finalization requires both LLM-finished and playback-drained; interrupted A/B flow stores completed A and marks active B as partial; late receipts after finalization do nothing.

- [ ] **Step 2: Implement pure ledger state**

Use a private per-generation dataclass and one lock. The ledger performs no audio, LLM, SQLite, timer, or State operations. It calls its finalize callback once.

- [ ] **Step 3: Wire ConversationWorker**

At response start call `ledger.begin`; for each clause call `ledger.generated`; at stream end call `ledger.llm_finished` then `voice.finish_generation`. Remove immediate memory finalization at LLM-end. Voice delivery callbacks feed `ledger.handle`. Successful barge-in cancellation calls `ledger.interrupt(G)`.

- [ ] **Step 4: Render interrupted prompt history safely**

Use fully completed assistant text only. Add: `Vess had started another clause but was interrupted; do not assume the user heard all of it.` Never present the partial clause as certainly heard.

- [ ] **Step 5: Run GREEN/full suite and commit**

```bash
python -m unittest tests.test_delivery tests.test_short_term_memory tests.test_llm -v
python -m unittest discover -s tests -v
git add brain/delivery.py state.py brain/memory.py brain/llm.py tests/test_delivery.py tests/test_short_term_memory.py tests/test_llm.py
git commit -m "feat: track delivered speech in conversation memory"
```

---

### Task 6: Two-phase `TurnCoordinator`

**Files:**
- Create: `brain/turn_coordinator.py`
- Create: `tests/test_turn_coordinator.py`

**Public contract:**

```python
class TurnCoordinator:
    def on_candidate(self) -> bool:
        raise NotImplementedError

    def on_utterance_queued_for_transcription(self) -> None:
        raise NotImplementedError

    def on_transcript(self, text: str) -> None:
        raise NotImplementedError

    def on_transcription_error(self, error: Exception) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

Constructor dependencies are state, event log, voice, conversation, transcript submit callback, false-timeout seconds, decision-watchdog seconds, and injectable timer factory.

- [ ] **Step 1: RED tests**

Prove real transcript effect order is cancel G -> commit G -> submit text; false timeout resumes G once; entering transcription suspends ordinary false timeout; watchdog rolls back only still-current G; H appearing during pending G cannot be cancelled by delayed G; duplicate transitions are harmless; close cancels timers and cannot later resume.

- [ ] **Step 2: Implement explicit phases**

Use `IDLE`, `PENDING_CAPTURE`, `PENDING_TRANSCRIBE`, `CLOSED`. Under the coordinator lock only mutate phase, paused generation, timestamps, and timer handles. Release the lock before voice/conversation/state-facing effects. Empty transcript or transcription error rolls back. If same-generation resume fails because G is stale, discard G with `commit_interruption(G)`.

- [ ] **Step 3: Run GREEN/full suite and commit**

```bash
python -m unittest tests.test_turn_coordinator -v
python -m unittest discover -s tests -v
git add brain/turn_coordinator.py tests/test_turn_coordinator.py
git commit -m "feat: add reversible barge-in coordinator"
```

---

### Task 7: Always-live capture during Vess speech

**Files:**
- Modify: `perception/audio.py`
- Modify: `tests/test_audio.py`

`AudioLoop` gains optional preprocessor, interruption detector, and turn coordinator. Disabled mode preserves current speaking-time discard behavior. Enabled mode preprocesses speaking-time blocks, feeds the detector, and routes completed overlapping utterances through coordinator transcription rather than wake matching.

- [ ] **Step 1: RED tests**

Prove disabled mode still ignores speaking-time blocks; enabled raw energy alone does not set listening; accepted candidate does; valid overlapping utterance marks transcription in-flight before transcriber work; transcript result goes directly to coordinator; empty transcript drives rollback; preprocessing failure with fail-closed disables only speaker-time barge-in while idle wake handling survives.

- [ ] **Step 2: Preserve capture timing**

Input callback stores `CapturedAudioBlock(samples, adc_time, received_at)` using `time_info.inputBufferAdcTime` when present and `time.perf_counter()` locally. Callback remains copy/enqueue only.

- [ ] **Step 3: Run GREEN/full suite and commit**

```bash
python -m unittest tests.test_audio -v
python -m unittest discover -s tests -v
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

- [ ] **Step 1: RED config/wiring tests**

Repository config contains the block and remains disabled. Add one focused builder that can accept fakes without opening hardware; verify player render reference and AudioLoop capture use the same preprocessor and coordinator uses the same voice/conversation objects.

- [ ] **Step 2: Implement wiring and shutdown order**

Construct player/voice/conversation/preprocessor/detector/coordinator/audio without opening devices during construction. Close coordinator timers before conversation/voice teardown; paused audio cannot resume after shutdown.

- [ ] **Step 3: Run GREEN/full suite and commit**

```bash
python -m unittest tests.test_main -v
python -m unittest discover -s tests -v
git add main.py config.json tests/test_main.py
git commit -m "feat: wire dormant barge-in runtime"
```

---

### Task 9: Cross-component race verification

**Files:**
- Create: `tests/test_barge_in_flow.py`
- Modify production only when an integration test demonstrates a concrete defect.

- [ ] **Step 1: Real interruption test**

Exercise G playback -> candidate -> pause -> overlapping utterance -> non-empty transcription -> exact G cancellation -> paused G discard -> new request. Assert no stale G clause starts and G memory contains only completed delivered speech.

- [ ] **Step 2: False interruption test**

Pause G, produce no valid transcript, trigger false decision, resume the same G waveform, drain normally, finalize completed.

- [ ] **Step 3: Slow transcription test**

Pause G, mark transcription in flight, advance fake timer beyond 2 seconds without resume, resolve before 5-second watchdog, commit.

- [ ] **Step 4: Newer-request race test**

Pause G, independently submit H, resolve delayed G interruption, prove H is never cancelled and G never resumes.

- [ ] **Step 5: Shutdown-while-paused test**

Close coordinator/voice while pending; assert no delayed resume, deadlock, speaking/listening flag, or non-neutral performance remains.

- [ ] **Step 6: Full verification and scope audit**

```bash
python -m unittest tests.test_barge_in_flow -v
python -m unittest discover -s tests -v
python tools/render_behavior_preview.py
python tools/render_eye_validation.py
```

Every command must exit 0. Compare branch against `design/independent-eye-motion`; scope is limited to approved spec/plan, focused barge-in modules, audio/player/voice/conversation/memory/state/main/config wiring, and tests. Confirm `barge_in.enabled=false`.

## Target-PC Acceptance

Remote success does not enable barge-in. On the actual PC: select/integrate real local echo preprocessing if needed; measure self-echo interruptions; test real interruptions at multiple distances/directions/volumes; measure p50/p95 speech-start-to-pause; test short `wait`-style interruptions, coughs, keyboard/room noise, clause-boundary timing, false-resume quality, and full resource coexistence. Change `barge_in.enabled` only after measured acceptance.
