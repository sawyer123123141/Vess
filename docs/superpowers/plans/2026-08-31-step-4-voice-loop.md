# Step 4 Voice Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local threaded voice loop that fuzzy-matches “Hey Vess”, streams Ollama output to Kokoro, and retains append-only event history without freezing the face.

**Architecture:** The microphone callback, Whisper, Ollama, Kokoro, and SQLite writer all run outside the 30fps render loop. `State` remains the live source of truth; `main.py` expires moods and logs the resulting transition. The event log is deliberately limited to one append-only SQLite table so full memory work remains Step 5.

**Tech Stack:** Python 3.11, `numpy`, `sounddevice`, `faster-whisper` CPU int8, Ollama HTTP, Kokoro `KPipeline`, SQLite, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-step-4-voice-loop-design.md`

## Global Constraints

- Do not block the render loop with audio, model, speech, SQLite, or web work.
- Keep Whisper and Kokoro on CPU; do not add a GPU model or exceed 4096 context.
- Keep all `State` mutation inside `State.locked()`.
- Use local-only services and standard-library HTTP for Ollama.
- Retain all wake rejections as event records; do not add event-log reads or retrieval.
- Defer vision and a dedicated wake-word engine; document both limitations in `STATUS.md`.
- Use `unittest`; each behavior begins with a test that visibly fails.

---

## Files and interfaces

| Path | Role |
|---|---|
| `brain/memory.py` | `EventLog` queued SQLite writer. |
| `brain/llm.py` | Prompt building, clause splitting, Ollama client, conversation worker. |
| `perception/audio.py` | VAD utterance assembly, fuzzy wake matching, sounddevice/Whisper worker. |
| `output/voice.py` | Ordered Kokoro synthesis and sounddevice playback. |
| `state.py` | `expire_mood(now)` atomic transition. |
| `control/web.py` | Accept optional event log and record colour set/reset. |
| `main.py` | Wire lifecycles and call mood expiry each render tick. |
| `tests/test_memory.py` | Real SQLite persistence test. |
| `tests/test_audio.py` | Pure matching/VAD and worker-dispatch tests. |
| `tests/test_llm.py` | Prompt/clause/mood parsing tests. |
| `tests/test_voice.py` | Ordered voice worker tests with injected synthesis/playback. |

## Task 1: Event log and mood expiry

**Files:** Create `brain/__init__.py`, `brain/memory.py`, `tests/test_memory.py`; modify `state.py`, `tests/test_color_override.py`.

**Interfaces:**

```python
class EventLog:
    def __init__(self, path: Path) -> None: ...
    def append(self, event_type: str, payload: dict[str, object],
               timestamp: float | None = None) -> None: ...
    def close(self) -> None: ...

def State.expire_mood(self, now: float) -> tuple[str, float] | None: ...
```

- [ ] **Step 1: Write the failing persistence test.**

```python
def test_append_persists_one_event(self) -> None:
    log = EventLog(self.path)
    log.append("wake_rejected", {"transcript": "hey guess"}, timestamp=12.5)
    log.close()
    with sqlite3.connect(self.path) as db:
        self.assertEqual(db.execute("SELECT timestamp, event_type, payload_json FROM events").fetchone(),
                         (12.5, "wake_rejected", '{"transcript":"hey guess"}'))
```

- [ ] **Step 2: Verify red.** Run `python -m unittest tests.test_memory.EventLogTests.test_append_persists_one_event -v`; expect import failure for `brain.memory`.
- [ ] **Step 3: Implement `EventLog`.** Create `events(timestamp REAL NOT NULL, event_type TEXT NOT NULL, payload_json TEXT NOT NULL)`. Serialize payload with `json.dumps(..., separators=(",", ":"))`; enqueue each tuple to a `queue.SimpleQueue`; one daemon owns the SQLite connection. A `None` sentinel flushes and stops it in `close()`.
- [ ] **Step 4: Verify green.** Run `python -m unittest tests.test_memory -v`; expect PASS.
- [ ] **Step 5: Write and verify the red mood-expiry test.**

```python
def test_expire_mood_resets_state(self) -> None:
    state = State(mood="annoyed", mood_until=100.0)
    self.assertEqual(state.expire_mood(100.1), ("annoyed", 100.0))
    self.assertEqual((state.mood, state.mood_until), ("neutral", 0.0))
```

Run `python -m unittest tests.test_color_override.ColorOverrideTests.test_expire_mood_resets_state -v`; expect missing-method failure.

- [ ] **Step 6: Implement and verify expiry.** Under `locked()`, return `None` when neutral, untimed, or unexpired; otherwise retain old values, set neutral/`0.0`, and return the old pair. Run `python -m unittest tests.test_memory tests.test_color_override -v`; expect PASS.
- [ ] **Step 7: Commit.**

```bash
git add brain/__init__.py brain/memory.py state.py tests/test_memory.py tests/test_color_override.py
git commit -m "Add append-only event log"
```

## Task 2: VAD and fuzzy wake gate

**Files:** Create `perception/audio.py`, `tests/test_audio.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class WakeMatch:
    variant: str
    distance: int
    consumed_words: int

def match_wake_phrase(transcript: str, variants: list[str],
                      max_distance: int) -> WakeMatch | None: ...

class UtteranceAssembler:
    def push(self, samples: np.ndarray) -> np.ndarray | None: ...
```

- [ ] **Step 1: Write the failing matcher tests.**

```python
def test_matcher_accepts_whisper_mishear(self) -> None:
    self.assertEqual(match_wake_phrase("hey best tell me a joke", ["hey vess"], 2),
                     WakeMatch("hey vess", 1, 2))

def test_matcher_rejects_unrelated_speech(self) -> None:
    self.assertIsNone(match_wake_phrase("turn on the lights", ["hey vess"], 2))
```

- [ ] **Step 2: Verify red.** Run `python -m unittest tests.test_audio.WakeMatchTests -v`; expect import failure.
- [ ] **Step 3: Implement matching.** Lowercase, retain alphanumeric characters/whitespace, collapse whitespace, and compare first one/two/three word prefixes to every normalized variant with a local Levenshtein implementation. Return the lowest match within the limit; on equal distance choose the shorter prefix.
- [ ] **Step 4: Verify green.** Run `python -m unittest tests.test_audio.WakeMatchTests -v`; expect PASS.
- [ ] **Step 5: Write and verify the red VAD test.**

```python
def test_assembler_emits_speech_after_trailing_silence(self) -> None:
    assembler = UtteranceAssembler(10, 0.1, 0.2, 0.3, 2.0)
    self.assertIsNone(assembler.push(np.array([0.0, 0.2, 0.2, 0.0])))
    self.assertTrue(np.array_equal(assembler.push(np.zeros(3)), np.array([0.2, 0.2])))
```

Run `python -m unittest tests.test_audio.UtteranceAssemblerTests.test_assembler_emits_speech_after_trailing_silence -v`; expect missing-class failure.

- [ ] **Step 6: Implement and verify VAD.** Start retaining blocks at the first absolute-amplitude threshold crossing, count trailing quiet samples, require minimum retained speech, trim terminal silence, and force emit/reset at maximum utterance duration. Run `python -m unittest tests.test_audio -v`; expect PASS.
- [ ] **Step 7: Commit.**

```bash
git add perception/audio.py tests/test_audio.py
git commit -m "Add fuzzy wake phrase gate"
```

## Task 3: Whisper microphone worker

**Files:** Modify `perception/audio.py`, `tests/test_audio.py`, `config.json`.

**Interfaces:**

```python
class AudioLoop:
    def __init__(self, config: dict, state: State, event_log: EventLog,
                 on_request: Callable[[str], None],
                 transcribe: Callable[[np.ndarray], str]) -> None: ...
    def handle_utterance(self, samples: np.ndarray) -> None: ...
    def start(self) -> None: ...
    def close(self) -> None: ...
```

- [ ] **Step 1: Write the failing rejection test.**

```python
def test_rejection_logs_without_dispatch(self) -> None:
    dispatched: list[str] = []
    loop = AudioLoop(CONFIG, State(), self.log, dispatched.append,
                     transcribe=lambda _: "turn on the lights")
    loop.handle_utterance(np.ones(16000, dtype=np.float32))
    self.assertEqual(dispatched, [])
    self.assertEqual(read_events(self.path)[0]["event_type"], "wake_rejected")
```

- [ ] **Step 2: Verify red.** Run `python -m unittest tests.test_audio.AudioLoopTests.test_rejection_logs_without_dispatch -v`; expect missing `AudioLoop` failure.
- [ ] **Step 3: Implement `handle_utterance`.** Set `state.listening` while transcribing. Log raw transcript, tested prefix, closest variant, and distance for rejects; log the same metadata for accepts; remove `consumed_words` from accepted request; dispatch it. Use a CPU/int8 `WhisperModel` only in the default transcriber factory.
- [ ] **Step 4: Add config and live stream.** Add `audio.device: null`, `sample_rate: 16000`, `channels: 1`, `vad_threshold: 0.015`, `min_utterance_seconds: 0.25`, `silence_seconds: 0.8`, `max_utterance_seconds: 15.0`, initial variants, and edit distance `2`. The `sounddevice.InputStream` callback uses `put_nowait(indata[:, 0].copy())`; the worker ignores mic blocks while `state.speaking` is true. Raise `RuntimeError` when the configured device cannot open.
- [ ] **Step 5: Verify green and commit.** Run `python -m unittest tests.test_audio -v`, then:

```bash
git add perception/audio.py tests/test_audio.py config.json
git commit -m "Add local Whisper audio worker"
```

## Task 4: Ollama streaming and Kokoro worker

**Files:** Create `brain/llm.py`, `output/voice.py`, `tests/test_llm.py`, `tests/test_voice.py`.

**Interfaces:**

```python
def build_prompt(config: dict, state: State, request: str) -> str: ...
def split_clauses(chunks: Iterable[str]) -> Iterator[str]: ...

class OllamaClient:
    def stream(self, prompt: str, config: dict) -> Iterator[str]: ...
    def classify_mood(self, transcript: str, mood_names: set[str], config: dict) -> str | None: ...

class VoiceOutput:
    def enqueue(self, text: str) -> None: ...
    def prepare_acknowledgement(self, text: str = "Yeah?") -> None: ...
    def enqueue_acknowledgement(self) -> None: ...
```

- [ ] **Step 1: Write failing prompt/clause tests.**

```python
def test_split_clauses_emits_completed_punctuation(self) -> None:
    self.assertEqual(list(split_clauses(["First, then", " second.", " Last"])),
                     ["First,", "then second.", "Last"])
```

- [ ] **Step 2: Verify red.** Run `python -m unittest tests.test_llm -v`; expect import failure.
- [ ] **Step 3: Implement prompt/client.** Put stable “You are Vess” persona instruction before dynamic state. POST JSON to local `/api/generate` using `urllib.request`, `num_ctx: 4096`, configured `num_predict`/keep-alive, newline-delimited streamed JSON, and clause punctuation `, . ! ? \n`. Classifier calls the same endpoint non-streaming and accepts only names present in `moods.json`.
- [ ] **Step 4: Verify llm green.** Run `python -m unittest tests.test_llm -v`; tests inject JSON-line iterables and never contact Ollama.
- [ ] **Step 5: Write failing serial voice test.**

```python
def test_voice_plays_in_order_and_clears_speaking(self) -> None:
    played: list[int] = []
    voice = VoiceOutput(CONFIG, self.state, self.log,
                        synthesize=lambda text: np.array([len(text)], dtype=np.float32),
                        play=lambda audio, _: played.append(int(audio[0])))
    voice.start(); voice.enqueue("one"); voice.enqueue("four"); voice.close()
    self.assertEqual(played, [3, 4])
    self.assertFalse(self.state.speaking)
```

- [ ] **Step 6: Verify red, implement, and verify green.** Run the test first and expect import failure. Implement a serial queue worker that sets/clears `speaking` under lock and logs synth/play errors. Default synthesis uses `KPipeline(lang_code="a", device="cpu")`, consumes each non-`None` `result.audio`, and converts it to float32 NumPy; default playback calls `sounddevice.play` and `sounddevice.wait` only in this worker. Cache the startup “Yeah?” waveform. Run `python -m unittest tests/test_voice.py -v`; expect PASS.
- [ ] **Step 7: Commit.**

```bash
git add brain/llm.py output/voice.py tests/test_llm.py tests/test_voice.py
git commit -m "Add streaming response and voice worker"
```

## Task 5: Conversation orchestration, event instrumentation, and final verification

**Files:** Modify `brain/llm.py`, `control/web.py`, `main.py`, `tests/test_main.py`, `tests/test_web.py`, `STATUS.md`.

**Interfaces:**

```python
class ConversationWorker:
    def submit(self, request: str) -> None: ...
    def start(self) -> None: ...
    def close(self) -> None: ...

def _expire_mood(state: State, event_log: EventLog, now: float) -> None: ...
```

- [ ] **Step 1: Write the failing expiry/logging test.**

```python
def test_expired_mood_is_logged(self) -> None:
    state = State(mood="annoyed", mood_until=10.0)
    log = RecordingLog()
    _expire_mood(state, log, 10.1)
    self.assertEqual(log.events, [("mood_changed", {"from": "annoyed", "to": "neutral"})])
```

- [ ] **Step 2: Verify red, then implement main lifecycle.** Run `python -m unittest tests.test_main.MainTests.test_expired_mood_is_logged -v`; expect missing-helper failure. Construct `EventLog(ROOT / "vess.db")`, append `session_started`, call `_expire_mood` every render tick, start voice/conversation/audio after the display, and close them in order audio → conversation → voice → web → detector/camera → event log. Audio startup failure prints `voice off: ` followed by the exact `RuntimeError` message and leaves rendering alive.
- [ ] **Step 3: Add colour event records.** Extend `WebServer`/`create_app` with optional `EventLog`; `PUT /color` appends `color_override_set` with RGB and `DELETE /color` appends `color_override_cleared`. Add endpoint tests with a recording append object; run `python -m unittest tests.test_web -v`.
- [ ] **Step 4: Implement `ConversationWorker`.** A single queue serializes accepted requests. It sets `thinking`, streams clauses to `VoiceOutput`, clears `thinking` before the first clause, then classifies mood. A valid changed mood updates `State`, sets `mood_until = now + moods[name]["decay"]`, and logs `mood_changed`; empty accepted requests enqueue cached acknowledgement. Tests use real `State` plus fake client/voice and verify clause order, error cleanup, and mood logging.
- [ ] **Step 5: Run full automated verification.**

```bash
python -m unittest discover -s tests -v
python -m compileall -q brain control output perception main.py state.py
```

Expected: all tests pass; compilation exits 0 without warnings.

- [ ] **Step 6: Manual verification and release commit.** Run `python main.py`; test one rejection, one “Hey Vess” question, and bare “Hey Vess”. Confirm animation continues, event records are written, clauses stream to speech, and the cached acknowledgement plays. Record USB device, latency, and useful mishears; record continuous Whisper and deferred vision as known limitations. Then:

```bash
git add brain control main.py output perception state.py config.json tests STATUS.md
git commit -m "Add local voice loop"
git push origin main
```
