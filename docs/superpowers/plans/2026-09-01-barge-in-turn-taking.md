# Vess Barge-In and Natural Turn-Taking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a person interrupt Vess while it is speaking, pause quickly, commit only real interruptions, resume false interruptions, and remember only speech that was actually delivered.

**Architecture:** Keep the runtime asynchronous. Introduce small audio/preprocessing/player/coordinator boundaries, preserve generation freshness, and move conversation completion from LLM-end to delivery receipts. The first implementation is deterministic and hardware-agnostic: passthrough preprocessing, no mandatory AEC backend, and `barge_in.enabled=false` until target-PC acceptance.

**Tech Stack:** Python 3.11, `unittest`, NumPy, sounddevice/PortAudio at runtime only, existing Ollama/faster-whisper/TTS stack.

**Spec:** `docs/superpowers/specs/2026-09-01-barge-in-turn-taking-design.md`

## Global Constraints

- Preserve the independent 30 FPS face loop.
- Preserve one-ready-waveform-ahead TTS bounding.
- Barge-in stays disabled by default until real microphone/speaker acceptance.
- CI must not require a microphone, speakers, Ollama, Whisper model, TTS model, CUDA, or a real AEC library.
- No learned interruption classifier in V1.
- No raw speaker-time VAD may directly become a committed interruption.
- `stale clauses played after committed interruption = 0`.
- `never-completed clauses recorded as fully delivered = 0`.
- `newer generation cancelled by delayed old interruption = 0`.
- Timers and callbacks may not hold `State.lock` while calling blocking audio, transcription, TTS, or conversation work.

---

## File Structure

New focused modules:

- `perception/audio_preprocess.py` — captured/render block value types plus passthrough preprocessor.
- `perception/interruption.py` — deterministic sustained-speech detector only.
- `output/audio_player.py` — playback receipt contract, deterministic fake-friendly player boundary, runtime sounddevice player.
- `brain/delivery.py` — generation-scoped delivery ledger; no audio or LLM calls.
- `brain/turn_coordinator.py` — reversible pause/commit/rollback policy and timeout phases.

Existing modules modified:

- `perception/audio.py` — preserve capture timing, keep capture alive when enabled, route speaking-time utterances to coordinator.
- `output/voice.py` — use `AudioPlayer`, expose pause/commit/resume, emit delivery callbacks, process finish markers in order.
- `brain/llm.py` — generation-specific cancel, delivery-ledger lifecycle, finish-generation marker, interrupted prompt rendering.
- `brain/memory.py` / `state.py` — interrupted turn metadata.
- `main.py` — construct and wire player/preprocessor/coordinator.
- `config.json` — dormant `barge_in` defaults.

Tests:

- `tests/test_audio_preprocess.py`
- `tests/test_interruption.py`
- `tests/test_audio_player.py`
- `tests/test_delivery.py`
- `tests/test_turn_coordinator.py`
- extend `tests/test_audio.py`, `tests/test_voice.py`, `tests/test_llm.py`, `tests/test_short_term_memory.py`, `tests/test_main.py`, `tests/test_performance_flow.py`.

---

### Task 1: Capture/preprocessing contracts and sustained interruption detector

**Files:**
- Create: `perception/audio_preprocess.py`
- Create: `perception/interruption.py`
- Create: `tests/test_audio_preprocess.py`
- Create: `tests/test_interruption.py`

**Interfaces:**
- Produces `CapturedAudioBlock(samples: np.ndarray, adc_time: float | None, received_at: float)`.
- Produces `RenderedAudioBlock(samples: np.ndarray, sample_rate: int, dac_time: float | None)`.
- Produces `PassthroughCapturePreprocessor.push_render_reference(block)` and `.process_capture(block) -> np.ndarray`.
- Produces `InterruptionDetector(sample_rate: int, threshold: float, pause_after_speech_seconds: float)` with `push(samples) -> bool` and `reset()`.

- [ ] **Step 1: Write failing preprocessing tests**

```python
import unittest
import numpy as np

from perception.audio_preprocess import (
    CapturedAudioBlock,
    PassthroughCapturePreprocessor,
    RenderedAudioBlock,
)


class AudioPreprocessTests(unittest.TestCase):
    def test_passthrough_returns_float32_copy(self) -> None:
        source = np.array([0.1, -0.2], dtype=np.float64)
        block = CapturedAudioBlock(source, adc_time=1.0, received_at=2.0)
        preprocessor = PassthroughCapturePreprocessor()

        result = preprocessor.process_capture(block)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [0.1, -0.2])
        self.assertIsNot(result, source)

    def test_render_reference_is_accepted_without_mutating_capture(self) -> None:
        preprocessor = PassthroughCapturePreprocessor()
        preprocessor.push_render_reference(
            RenderedAudioBlock(np.array([0.4], dtype=np.float32), 24_000, None)
        )
        result = preprocessor.process_capture(
            CapturedAudioBlock(np.array([0.2], dtype=np.float32), None, 1.0)
        )
        np.testing.assert_allclose(result, [0.2])
```

- [ ] **Step 2: Write failing detector tests**

```python
import unittest
import numpy as np

from perception.interruption import InterruptionDetector


class InterruptionDetectorTests(unittest.TestCase):
    def test_requires_sustained_speech_before_candidate(self) -> None:
        detector = InterruptionDetector(10, threshold=0.1, pause_after_speech_seconds=0.3)
        self.assertFalse(detector.push(np.array([0.2, 0.2])))
        self.assertTrue(detector.push(np.array([0.2])))
        self.assertFalse(detector.push(np.array([0.2])))

    def test_quiet_audio_resets_candidate_progress(self) -> None:
        detector = InterruptionDetector(10, threshold=0.1, pause_after_speech_seconds=0.3)
        self.assertFalse(detector.push(np.array([0.2, 0.2])))
        self.assertFalse(detector.push(np.array([0.0])))
        self.assertFalse(detector.push(np.array([0.2, 0.2])))

    def test_reset_allows_a_future_candidate(self) -> None:
        detector = InterruptionDetector(10, threshold=0.1, pause_after_speech_seconds=0.2)
        self.assertTrue(detector.push(np.array([0.2, 0.2])))
        detector.reset()
        self.assertTrue(detector.push(np.array([0.2, 0.2])))
```

- [ ] **Step 3: Run RED**

Run:

```bash
python -m unittest tests.test_audio_preprocess tests.test_interruption -v
```

Expected: imports fail because both modules do not exist.

- [ ] **Step 4: Implement minimal contracts**

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
    def push_render_reference(self, block: RenderedAudioBlock) -> None: ...
    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray: ...


class PassthroughCapturePreprocessor:
    def push_render_reference(self, block: RenderedAudioBlock) -> None:
        return None

    def process_capture(self, block: CapturedAudioBlock) -> np.ndarray:
        return np.asarray(block.samples, dtype=np.float32).reshape(-1).copy()
```

```python
# perception/interruption.py
from __future__ import annotations
from math import ceil
import numpy as np


class InterruptionDetector:
    def __init__(self, sample_rate: int, threshold: float, pause_after_speech_seconds: float) -> None:
        self._threshold = float(threshold)
        self._required = max(1, ceil(sample_rate * pause_after_speech_seconds))
        self._audible = 0
        self._emitted = False

    def push(self, samples: np.ndarray) -> bool:
        for sample in np.asarray(samples).reshape(-1):
            if abs(float(sample)) < self._threshold:
                self._audible = 0
                self._emitted = False
                continue
            self._audible += 1
            if self._audible >= self._required and not self._emitted:
                self._emitted = True
                return True
        return False

    def reset(self) -> None:
        self._audible = 0
        self._emitted = False
```

- [ ] **Step 5: Run GREEN and full suite**

```bash
python -m unittest tests.test_audio_preprocess tests.test_interruption -v
python -m unittest discover -s tests -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add perception/audio_preprocess.py perception/interruption.py tests/test_audio_preprocess.py tests/test_interruption.py
git commit -m "feat: add barge-in audio detection contracts"
```

---

### Task 2: Cancellable player boundary with deterministic pause/resume semantics

**Files:**
- Create: `output/audio_player.py`
- Create: `tests/test_audio_player.py`

**Interfaces:**
- Produces `PlaybackReceipt(status, generation_id, frames_started, frames_completed, total_frames, sample_rate)`.
- Produces `AudioPlayer` protocol.
- Produces `SoundDeviceAudioPlayer(render_callback=None)` runtime implementation.
- `pause_for_interruption()` returns the paused receipt or `None`; `resume()` continues the saved remainder; `discard_paused()` makes resume impossible.

- [ ] **Step 1: Write RED tests around a fake backend seam**

```python
import threading
import unittest
import numpy as np

from output.audio_player import SoundDeviceAudioPlayer


class FakeStreamBackend:
    def __init__(self) -> None:
        self.abort_requested = threading.Event()
        self.frames_written = 0

    def play(self, audio: np.ndarray, sample_rate: int, on_render) -> int:
        for value in audio:
            if self.abort_requested.is_set():
                break
            frame = np.array([value], dtype=np.float32)
            on_render(frame, sample_rate)
            self.frames_written += 1
        return self.frames_written

    def abort(self) -> None:
        self.abort_requested.set()

    def clear_abort(self) -> None:
        self.abort_requested.clear()
        self.frames_written = 0


class AudioPlayerTests(unittest.TestCase):
    def test_completed_playback_reports_completed_frames(self) -> None:
        backend = FakeStreamBackend()
        player = SoundDeviceAudioPlayer(backend=backend)
        receipt = player.play(np.arange(4, dtype=np.float32), 4, 7)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.generation_id, 7)
        self.assertEqual(receipt.frames_completed, 4)

    def test_pause_saves_remainder_and_resume_finishes_same_generation(self) -> None:
        backend = FakeStreamBackend()
        player = SoundDeviceAudioPlayer(backend=backend)
        result: list[object] = []
        thread = threading.Thread(
            target=lambda: result.append(player.play(np.arange(1000, dtype=np.float32), 1000, 3))
        )
        thread.start()
        while backend.frames_written < 10:
            pass
        paused = player.pause_for_interruption()
        thread.join(timeout=1.0)
        self.assertIsNotNone(paused)
        assert paused is not None
        self.assertEqual(paused.status, "paused")
        self.assertEqual(paused.generation_id, 3)
        backend.clear_abort()
        resumed = player.resume()
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.generation_id, 3)

    def test_discard_paused_prevents_resume(self) -> None:
        backend = FakeStreamBackend()
        player = SoundDeviceAudioPlayer(backend=backend)
        player._store_paused_for_test(np.ones(5, dtype=np.float32), 5, 9, 2)
        player.discard_paused()
        with self.assertRaises(RuntimeError):
            player.resume()
```

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_audio_player -v
```

Expected: import fails.

- [ ] **Step 3: Implement player state and backend injection**

Implement `PlaybackReceipt` and `AudioPlayer` exactly as in the spec. The runtime backend wrapper imports `sounddevice` lazily. The player must guard current/paused state with a dedicated lock and must never hold that lock while invoking the backend's potentially blocking `play()` loop.

The production backend uses `sounddevice.OutputStream` and its callback/abort path; CI uses the injected fake backend. `render_callback(RenderedAudioBlock)` is invoked with frames actually submitted to output, not the entire synthesized waveform up front.

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

### Task 3: VoiceOutput uses AudioPlayer and emits ordered playback lifecycle receipts

**Files:**
- Modify: `output/voice.py`
- Modify: `tests/test_voice.py`
- Modify: `tests/test_performance_flow.py`

**Interfaces:**
- `VoiceOutput(..., player: AudioPlayer | None = None, on_delivery: Callable[[str, dict[str, object]], None] | None = None)`.
- Add `pause_for_interruption() -> PlaybackReceipt | None`.
- Add `commit_interruption(generation_id: int) -> bool`.
- Add `resume_after_false_interruption(generation_id: int) -> bool`.
- Add `finish_generation(generation_id: int) -> None`.
- Delivery event names: `clause_started`, `clause_completed`, `clause_paused`, `clause_resumed`, `clause_abandoned`, `generation_playback_drained`.

- [ ] **Step 1: Write RED test proving pause clears speaking/performance and returns exact generation**

Use a deterministic fake player whose `play()` blocks until pause, and assert:

```python
paused = voice.pause_for_interruption()
self.assertEqual(paused.generation_id, 11)
self.assertFalse(state.speaking)
self.assertEqual(state.performance, PerformanceCue())
```

- [ ] **Step 2: Write RED test proving false resume restores the original cue only during physical playback**

Queue one playful clause for generation 12, pause it, call `resume_after_false_interruption(12)`, and have the fake player assert `state.performance.expression == "playful"` inside resumed playback. After completion assert neutral again.

- [ ] **Step 3: Write RED ordering test for finish marker**

Queue two clauses for generation 13, then `finish_generation(13)`. Capture `on_delivery` events. Expected final suffix:

```python
[
    ("clause_completed", {"generation_id": 13, "text": "one"}),
    ("clause_completed", {"generation_id": 13, "text": "two"}),
    ("generation_playback_drained", {"generation_id": 13}),
]
```

The drain marker may not overtake either clause.

- [ ] **Step 4: Run RED**

```bash
python -m unittest tests.test_voice tests.test_performance_flow -v
```

- [ ] **Step 5: Replace direct `play` callback path with player adapter**

Preserve compatibility for existing tests by wrapping a supplied legacy `play(audio, sample_rate)` callback in a tiny `CallbackAudioPlayer`; production default becomes `SoundDeviceAudioPlayer`. Keep synthesis/ready queue sizing unchanged.

Add a distinct queue item kind `finish` so the synthesis queue preserves generation-end ordering. The playback worker forwards a drain item through the ready queue after all earlier clauses for that generation.

- [ ] **Step 6: Implement interruption methods**

Pause calls the player, validates the current generation, blocks playback progression, clears performance/speaking only when the player confirms pause, and emits `clause_paused`. Commit discards only matching paused audio and unblocks freshness handling. False resume verifies generation is still current, emits `clause_resumed`, restores performance at playback start, and reopens later-clause playback.

- [ ] **Step 7: Run GREEN and all tests**

```bash
python -m unittest tests.test_voice tests.test_performance_flow -v
python -m unittest discover -s tests -v
```

- [ ] **Step 8: Commit**

```bash
git add output/voice.py tests/test_voice.py tests/test_performance_flow.py
git commit -m "feat: make voice playback interruptible"
```

---

### Task 4: Generation-specific cancellation that cannot kill a newer request

**Files:**
- Modify: `brain/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Produces `ConversationWorker.cancel_generation(expected_generation: int, reason: str) -> bool`.
- Add test helper/read-only `current_generation()` only if needed internally; do not expose mutable generation state.

- [ ] **Step 1: RED tests**

Add tests covering:

```python
worker.submit("first")
first = worker._latest_generation
self.assertTrue(worker.cancel_generation(first, "barge_in"))
self.assertGreater(worker._latest_generation, first)
```

and the race invariant:

```python
worker.submit("first")
old = worker._latest_generation
worker.submit("newer")
new = worker._latest_generation
self.assertFalse(worker.cancel_generation(old, "barge_in"))
self.assertEqual(worker._latest_generation, new)
```

Also call cancel twice and prove the second call is harmless/false.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_llm -v
```

- [ ] **Step 3: Implement compare-and-cancel under `_request_lock`**

When expected equals latest, increment `_next_generation`, copy it into `_latest_generation`, call `voice.begin_generation(new_generation)`, record `generation_cancelled` with expected/replacement/reason, and return `True`. Do not dequeue a separate newer pending request and do not submit synthetic text.

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

### Task 5: Delivery ledger and interrupted short-term turns

**Files:**
- Create: `brain/delivery.py`
- Create: `tests/test_delivery.py`
- Modify: `state.py`
- Modify: `brain/memory.py`
- Modify: `tests/test_short_term_memory.py`
- Modify: `brain/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Extend `ConversationTurn` with `status: str = "completed"` and `interrupted_clause: str | None = None`.
- Extend `append_conversation_turn(..., status="completed", interrupted_clause=None)`.
- Produce `DeliveryLedger.begin(generation_id, user_request)`, `.generated(...)`, `.handle(event_type, payload)`, `.llm_finished(generation_id)`, `.interrupt(generation_id)`, and finalize callback.

- [ ] **Step 1: RED memory compatibility tests**

Existing completed-turn tests must keep passing with defaults. Add:

```python
turn = append_conversation_turn(
    state,
    "Question",
    "Delivered part.",
    timestamp=100.0,
    max_age_seconds=600.0,
    max_turns=8,
    status="interrupted",
    interrupted_clause="Started but cut off.",
)
self.assertEqual(turn.status, "interrupted")
self.assertEqual(turn.interrupted_clause, "Started but cut off.")
```

- [ ] **Step 2: RED ledger tests**

Normal case: generated A/B, completed A/B, LLM finished, then drain -> one completed finalized record `"A B"`.

Interrupted case: generated A/B, completed A, clause B started, interrupt -> finalized `assistant="A"`, `status="interrupted"`, `interrupted_clause="B"`.

Late `clause_completed` after finalization must not mutate the result.

- [ ] **Step 3: Implement ledger as a pure lock-protected state machine**

Use a private per-generation dataclass with `generated`, `completed`, `active_clause`, `llm_finished`, `drained`, `finalized`. No audio, LLM, SQLite, or timers inside this module.

- [ ] **Step 4: Wire ConversationWorker**

At request start call `ledger.begin(generation_id, user_request)`. For each emitted clause call `ledger.generated`. At end of stream call `ledger.llm_finished` then `voice.finish_generation(generation_id)` instead of immediately calling `_remember_completed_turn`.

`VoiceOutput` delivery callback feeds ledger events. Ledger finalization calls the existing bounded-memory append path and durable `conversation_turn` event log.

On real cancellation for barge-in, call `ledger.interrupt(expected_generation)` before/new alongside freshness invalidation so the old request remains in history.

- [ ] **Step 5: Update prompt rendering**

Completed turns keep:

```text
User: ...
Vess: ...
```

Interrupted turns render:

```text
User: ...
Vess (interrupted): <completed assistant text>
Vess had started another clause but was interrupted; do not assume the user heard all of it.
```

Do not print the partial clause as certainly heard.

- [ ] **Step 6: Run GREEN + full suite**

```bash
python -m unittest tests.test_delivery tests.test_short_term_memory tests.test_llm -v
python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit**

```bash
git add brain/delivery.py state.py brain/memory.py brain/llm.py tests/test_delivery.py tests/test_short_term_memory.py tests/test_llm.py
git commit -m "feat: track delivered speech in conversation memory"
```

---

### Task 6: TurnCoordinator two-phase pause/commit/rollback state machine

**Files:**
- Create: `brain/turn_coordinator.py`
- Create: `tests/test_turn_coordinator.py`

**Interfaces:**
- Constructor receives `state`, `event_log`, `voice`, `conversation`, `submit_transcript`, timeout values, and injectable clock/timer factory.
- `on_candidate() -> bool`
- `on_utterance_queued_for_transcription() -> None`
- `on_transcript(text: str) -> None`
- `on_transcription_error(error: Exception) -> None`
- `close() -> None`

- [ ] **Step 1: RED real-interruption test**

Fake voice returns pause receipt generation 5. Candidate pauses once. Mark transcription in-flight. `on_transcript("wait")` must produce call order:

```python
[
    ("cancel", 5, "barge_in"),
    ("commit", 5),
    ("submit", "wait"),
]
```

State must end `listening=False` and old audio cannot resume.

- [ ] **Step 2: RED false-interruption test**

Candidate pauses generation 6. Trigger false timeout before transcription begins. Expected `resume_after_false_interruption(6)` exactly once and no cancel/submit.

- [ ] **Step 3: RED slow-transcription test**

Candidate pauses generation 7; call `on_utterance_queued_for_transcription()`. Advance fake clock beyond ordinary false timeout but below decision watchdog. Assert no resume. Then `on_transcript("actually")` commits.

- [ ] **Step 4: RED newer-generation race test**

Candidate pauses G. Fake conversation reports `cancel_generation(G)` false because H is newer. Commit must discard G but must not touch H; transcript submit then becomes a still-newer normal generation.

- [ ] **Step 5: Implement coordinator with one private lock and explicit phase**

Use `IDLE`, `PENDING_CAPTURE`, `PENDING_TRANSCRIBE`, `CLOSED`. Store only paused generation/timestamps/timer handles. Never hold coordinator lock while calling voice/conversation methods: copy transition intent under lock, release, then invoke effects.

- [ ] **Step 6: Implement timers**

Candidate starts false timeout. Entering transcription cancels false timeout and starts decision watchdog. Any decision cancels timers. Timer callbacks re-check phase/generation before acting so stale callbacks are harmless.

- [ ] **Step 7: Run GREEN + full suite**

```bash
python -m unittest tests.test_turn_coordinator -v
python -m unittest discover -s tests -v
```

- [ ] **Step 8: Commit**

```bash
git add brain/turn_coordinator.py tests/test_turn_coordinator.py
git commit -m "feat: add reversible barge-in coordinator"
```

---

### Task 7: Keep microphone capture alive during speech and route overlapping utterances

**Files:**
- Modify: `perception/audio.py`
- Modify: `tests/test_audio.py`

**Interfaces:**
- `AudioLoop(..., preprocessor=None, interruption_detector=None, turn_coordinator=None)`.
- `_blocks` carries `CapturedAudioBlock | None` instead of bare arrays when barge-in is enabled; legacy behavior remains accepted internally when disabled/tests use direct blocks.
- Speaking-time valid utterances notify coordinator before/during transcription and bypass wake-word gating only after committed barge-in path.

- [ ] **Step 1: RED disabled-mode regression test**

With `barge_in.enabled=false` and `state.speaking=True`, feed a block and assert existing `audio_ignored=True` behavior remains.

- [ ] **Step 2: RED enabled-mode candidate test**

With barge-in enabled, speaking true, passthrough preprocessor and detector threshold reached, assert `turn_coordinator.on_candidate()` is called and `state.listening` becomes true only after candidate acceptance, not raw energy.

- [ ] **Step 3: RED transcription-phase test**

When overlapping utterance completes, assert coordinator receives `on_utterance_queued_for_transcription()` before transcriber work. A non-empty result calls `coordinator.on_transcript(text)` directly rather than ordinary wake matching. Empty result calls `on_transcript("")` for rollback policy.

- [ ] **Step 4: Preserve capture callback timing**

Read sounddevice callback `time_info.inputBufferAdcTime` when present and store it in `CapturedAudioBlock`; callback still only copies/enqueues data.

- [ ] **Step 5: Handle preprocessor failure fail-closed**

When enabled and configured fail-closed, log `barge_in_preprocessor_error`, disable speaker-time barge-in for the session, and keep ordinary idle microphone behavior alive.

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

### Task 8: Runtime wiring and dormant configuration

**Files:**
- Modify: `main.py`
- Modify: `config.json`
- Modify: `tests/test_main.py`

**Interfaces:**
- Runtime constructs one `PassthroughCapturePreprocessor`, one `InterruptionDetector`, one `TurnCoordinator`, and one `SoundDeviceAudioPlayer` when barge-in support is wired.
- Player render reference calls `preprocessor.push_render_reference`.
- `AudioLoop` gets the same preprocessor/detector/coordinator instances.

- [ ] **Step 1: Add config block**

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

- [ ] **Step 2: RED config/wiring tests**

`test_main.py` loads repository config and asserts barge-in is present and disabled. Add a small `_build_voice_stack(...)` or similarly focused builder so tests can inject fakes and verify the same preprocessor receives rendered blocks and capture blocks without starting hardware.

- [ ] **Step 3: Implement wiring**

Do not open audio devices in the builder. Device opening remains in `AudioLoop.start()` / player first-use. Connect coordinator transcript submission to `conversation.submit` and coordinator cancellation to the same `ConversationWorker` instance.

- [ ] **Step 4: Shutdown order**

On shutdown close coordinator timers before conversation/voice teardown, then close audio/player workers. Ensure a paused waveform cannot resume after shutdown.

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

### Task 9: Cross-component race verification and final CI acceptance

**Files:**
- Create: `tests/test_barge_in_flow.py`
- Modify only production files if this integration test exposes a concrete bug.

**Interfaces:**
- No new production interface is planned in this task.

- [ ] **Step 1: Real interruption flow test**

Build fakes for TTS engine/player/transcriber/client and exercise:

```text
G playback starts
candidate reaches threshold
G pauses
human utterance finishes
transcription returns non-empty
cancel_generation(G)
commit paused G
submit interruption as H/newer
G prepared speech never starts
G memory finalizes interrupted
```

Assert the three hard invariants from Global Constraints.

- [ ] **Step 2: False interruption flow test**

Pause G, candidate collapses/no valid utterance, fire false timeout, assert the same waveform/generation resumes and G later drains as a completed memory turn.

- [ ] **Step 3: Slow transcription flow test**

Pause G, enter transcribing, advance beyond 2 seconds, assert no rollback; return transcript before 5-second watchdog and commit.

- [ ] **Step 4: Newer request race flow test**

Pause G, submit unrelated H before barge-in transcript arrives, then resolve old interruption. Assert H is never cancelled and G never resumes.

- [ ] **Step 5: Shutdown-while-paused test**

Close coordinator/voice while pending and assert no deadlock, timer-driven resume, non-neutral performance, speaking, or listening state remains.

- [ ] **Step 6: Run focused and full verification**

```bash
python -m unittest tests.test_barge_in_flow -v
python -m unittest discover -s tests -v
python tools/render_behavior_preview.py
python tools/render_eye_validation.py
```

Expected: all unit tests pass; both visual verification tools exit 0. No hardware/model dependency is introduced into CI.

- [ ] **Step 7: Inspect branch diff**

Compare against `design/independent-eye-motion`. Confirm changes are limited to the approved spec/plan, focused barge-in modules, audio/voice/conversation/memory/runtime wiring, config, and tests. Confirm `barge_in.enabled` is still false.

- [ ] **Step 8: Commit final integration fixes if any**

If the integration test required a concrete correction, commit only that correction and its regression test with a descriptive message. If no correction was needed, no empty commit is created.

---

## Target-PC Acceptance After Remote Implementation

Remote CI completion does **not** enable barge-in by default. On the actual machine:

1. choose and integrate a real local echo-preprocessing backend if passthrough self-echo is not safe
2. test Vess speaking with no human speech at realistic speaker volume
3. test interruptions at multiple distances/directions/volumes
4. measure p50/p95 human speech start -> audible pause
5. test short interruptions such as "wait"
6. test coughs, keyboard noise, room speech, and speaker echo
7. test near clause boundaries and immediately after playback starts
8. evaluate false-interruption resume quality
9. verify CPU/RAM/VRAM coexistence with Qwen, Whisper, TTS, detector, and rendering
10. only after measured acceptance consider changing `barge_in.enabled` to true
