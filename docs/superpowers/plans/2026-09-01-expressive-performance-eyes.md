# Expressive Performance + Eyes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sentence-level performance metadata that is synchronized to physical speech playback and drives more deliberate listening, thinking, speaking, and expressive eye behavior without adding another LLM call.

**Architecture:** Introduce a small validated performance vocabulary shared by the LLM parser, `State`, `VoiceOutput`, and `FaceAnimator`. The LLM emits reserved sentence tags in the existing stream; the parser strips them into `SpeechClause` objects; `VoiceOutput` activates each cue only when its waveform starts; `FaceAnimator` composes a fast transient performance overlay on top of the existing slower mood target and adds explicit conversational gaze modes.

**Tech Stack:** Python 3, standard-library `dataclasses`/`json`/`unittest`, NumPy, Ollama `qwen2.5:7b`, existing Kokoro voice pipeline, existing 64x64 NumPy face renderer.

**Spec:** `docs/superpowers/specs/2026-09-01-expressive-performance-eyes-design.md`

## Global Constraints

- `PLAN.md` remains authoritative.
- Shared runtime state stays in the single locked `State` object.
- The face render loop stays non-blocking at 30fps.
- Persona stays stable; mood stays slower and reactive; performance is transient.
- Keep `qwen2.5:7b` and `num_ctx=4096`.
- Add no second LLM/classifier call before speech.
- The model may select only configured performance labels; unknown labels fall back to neutral.
- Keep latest-intent invalidation, one-clause-ahead TTS, conservative edge trimming, and natural clause splitting intact.
- Performance markers are never spoken, stored in conversation memory, or logged as assistant prose.
- Performance activates at physical playback start, not generation or synthesis time.
- `face.py` stays stateless and unaware of runtime/performance labels.
- No replacement TTS model, barge-in, per-word animation, waveform-driven eyes, or long-term performance memory in this plan.

---

## File Structure

- Create `performance.py`: shared `PerformanceCue`, performance-definition validation, label lookup, and overlay clamp definitions. This is the only module that interprets `performance.json` structure.
- Create `performance.json`: human-authored fixed performance vocabulary and bounded visual modifiers.
- Modify `state.py`: add `State.performance` using `PerformanceCue`.
- Modify `main.py`: load/validate `performance.json`, pass definitions to `ConversationWorker` and `FaceAnimator`.
- Modify `brain/llm.py`: emit performance-format prompt rules, parse reserved markers into `SpeechClause`, keep cue inheritance across soft TTS splits, and store only cleaned text.
- Modify `output/voice.py`: carry cues through queue/prepared items, activate/clear state at physical playback boundaries, and emit performance diagnostics.
- Modify `output/animator.py`: read speaking/performance state, add explicit runtime gaze modes, maintain a fast performance overlay, and compose that overlay over the mood result.
- Create `tests/test_performance.py`: config/type validation tests.
- Modify `tests/test_llm.py`: structured clause parsing, prompt, marker stripping, inheritance, and memory tests.
- Modify `tests/test_tts_pipeline.py`: playback synchronization, stale cue, and error cleanup tests.
- Create `tests/test_animator.py`: deterministic runtime-mode and performance-overlay tests.
- Modify `tests/test_main.py`: startup wiring/config validation coverage.
- Modify `STATUS.md`: record the completed slice only after the full suite and live check pass.

---

### Task 1: Performance Vocabulary, Validation, and State

**Files:**
- Create: `performance.py`
- Create: `performance.json`
- Modify: `state.py`
- Modify: `main.py`
- Create: `tests/test_performance.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Produces: `PerformanceCue(expression: str = "neutral", intensity: float = 0.0)`.
- Produces: `load_performance_definitions(raw: dict[str, object]) -> dict[str, dict[str, object]]`.
- Produces: `cue_for_label(label: str, definitions: dict[str, dict[str, object]]) -> PerformanceCue`.
- Produces: `State.performance: PerformanceCue`.
- Later tasks consume the validated definition keys `intensity`, `shape`, and `movement`.

- [ ] **Step 1: Write failing performance validation tests**

Create `tests/test_performance.py` with tests equivalent to:

```python
import unittest

from performance import PerformanceCue, cue_for_label, load_performance_definitions


class PerformanceTests(unittest.TestCase):
    def test_neutral_is_required_and_unknown_labels_fall_back(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {"intensity": 0.0},
            "playful": {"intensity": 0.65},
        })
        self.assertEqual(cue_for_label("playful", definitions),
                         PerformanceCue("playful", 0.65))
        self.assertEqual(cue_for_label("nonsense", definitions),
                         PerformanceCue())

    def test_numeric_values_are_clamped_and_missing_blocks_default_empty(self) -> None:
        definitions = load_performance_definitions({
            "neutral": {"intensity": 0.0},
            "emphatic": {
                "intensity": 4.0,
                "shape": {"l_h": 99.0, "r_h": -99.0},
                "movement": {"hold_scale": 99.0, "gaze_y_bias": -9.0},
            },
        })
        emphatic = definitions["emphatic"]
        self.assertEqual(emphatic["intensity"], 1.0)
        self.assertEqual(emphatic["shape"]["l_h"], 3.0)
        self.assertEqual(emphatic["shape"]["r_h"], -3.0)
        self.assertEqual(emphatic["movement"]["hold_scale"], 1.6)
        self.assertEqual(emphatic["movement"]["gaze_y_bias"], -0.35)

    def test_missing_neutral_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "neutral"):
            load_performance_definitions({"playful": {"intensity": 0.6}})
```

Also add a `tests/test_main.py` assertion that `_load_performances()` returns a mapping containing `neutral` when pointed at the repository `performance.json`.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m unittest tests.test_performance tests.test_main -v
```

Expected: import/function failures because `performance.py`, `performance.json`, and `_load_performances` do not exist yet.

- [ ] **Step 3: Implement the shared type and validator**

Create `performance.py` with these public pieces:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerformanceCue:
    expression: str = "neutral"
    intensity: float = 0.0


_SHAPE_LIMITS = {
    "l_h": (-3.0, 3.0),
    "r_h": (-3.0, 3.0),
    "l_slant": (-1.5, 1.5),
    "r_slant": (-1.5, 1.5),
    "l_cy": (-1.5, 1.5),
    "r_cy": (-1.5, 1.5),
}

_MOVEMENT_LIMITS = {
    "hold_scale": (0.6, 1.6),
    "ease_scale": (0.7, 1.5),
    "track_bias_scale": (0.7, 1.3),
    "gaze_y_bias": (-0.35, 0.35),
    "speaking_break_scale": (0.0, 2.0),
}


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def load_performance_definitions(raw: dict[str, object]) -> dict[str, dict[str, object]]:
    if "neutral" not in raw:
        raise ValueError("performance config requires neutral")
    cleaned: dict[str, dict[str, object]] = {}
    for name, value in raw.items():
        entry = value if isinstance(value, dict) else {}
        intensity = _clamp(float(entry.get("intensity", 0.0)), 0.0, 1.0)
        shape_raw = entry.get("shape", {}) if isinstance(entry.get("shape", {}), dict) else {}
        movement_raw = entry.get("movement", {}) if isinstance(entry.get("movement", {}), dict) else {}
        shape = {
            key: _clamp(float(shape_raw.get(key, 0.0)), low, high)
            for key, (low, high) in _SHAPE_LIMITS.items()
        }
        movement_defaults = {
            "hold_scale": 1.0,
            "ease_scale": 1.0,
            "track_bias_scale": 1.0,
            "gaze_y_bias": 0.0,
            "speaking_break_scale": 1.0,
        }
        movement = {
            key: _clamp(float(movement_raw.get(key, movement_defaults[key])), low, high)
            for key, (low, high) in _MOVEMENT_LIMITS.items()
        }
        cleaned[name] = {"intensity": intensity, "shape": shape, "movement": movement}
    return cleaned


def cue_for_label(label: str, definitions: dict[str, dict[str, object]]) -> PerformanceCue:
    name = label.strip().lower()
    entry = definitions.get(name)
    if entry is None:
        return PerformanceCue()
    return PerformanceCue(name, float(entry["intensity"]))
```

- [ ] **Step 4: Add the fixed `performance.json` vocabulary**

Create `performance.json` exactly with restrained starting values:

```json
{
  "neutral": {
    "intensity": 0.0,
    "shape": {},
    "movement": {}
  },
  "curious": {
    "intensity": 0.55,
    "shape": {"l_h": 1.0, "r_h": 1.4, "r_cy": -0.5},
    "movement": {"hold_scale": 0.8, "track_bias_scale": 1.1}
  },
  "amused": {
    "intensity": 0.5,
    "shape": {"l_h": -1.0, "r_h": -0.7, "l_slant": 0.35, "r_slant": 0.25},
    "movement": {"hold_scale": 1.05, "speaking_break_scale": 1.15}
  },
  "playful": {
    "intensity": 0.65,
    "shape": {"l_h": -0.5, "r_h": 0.3, "l_slant": 0.7, "r_slant": -0.35, "r_cy": -0.5},
    "movement": {"hold_scale": 0.75, "ease_scale": 0.85, "speaking_break_scale": 1.4}
  },
  "emphatic": {
    "intensity": 0.7,
    "shape": {"l_h": 1.2, "r_h": 1.0},
    "movement": {"hold_scale": 0.8, "track_bias_scale": 1.15, "speaking_break_scale": 0.4}
  },
  "thoughtful": {
    "intensity": 0.55,
    "shape": {"l_h": -0.5, "r_h": -0.4},
    "movement": {"hold_scale": 1.4, "ease_scale": 1.2, "gaze_y_bias": -0.22}
  },
  "sympathetic": {
    "intensity": 0.5,
    "shape": {"l_h": -0.8, "r_h": -0.6, "l_slant": -0.4, "r_slant": -0.3},
    "movement": {"hold_scale": 1.25, "ease_scale": 1.15, "gaze_y_bias": 0.15, "speaking_break_scale": 0.7}
  },
  "uncertain": {
    "intensity": 0.5,
    "shape": {"l_slant": 0.3, "r_slant": -0.35, "r_cy": -0.6},
    "movement": {"hold_scale": 1.15, "gaze_y_bias": -0.05, "speaking_break_scale": 1.3}
  }
}
```

- [ ] **Step 5: Add performance to `State` and startup wiring**

In `state.py`, import `PerformanceCue` and add:

```python
performance: PerformanceCue = field(default_factory=PerformanceCue)
```

In `main.py`, add:

```python
from performance import load_performance_definitions


def _load_performances() -> dict[str, dict[str, object]]:
    return load_performance_definitions(_load("performance.json"))
```

Then load it in `main()` immediately after moods. For this task, only instantiate/validate it; later tasks wire it into consumers.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_performance tests.test_main -v
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add performance.py performance.json state.py main.py tests/test_performance.py tests/test_main.py
git commit -m "feat: add validated performance state"
```

---

### Task 2: Structured LLM Performance Markup and Clean Memory

**Files:**
- Modify: `brain/llm.py`
- Modify: `main.py`
- Modify: `tests/test_llm.py`
- Modify: `tests/test_short_term_memory.py`

**Interfaces:**
- Consumes: `PerformanceCue`, `cue_for_label`, validated performance definitions from Task 1.
- Produces: `SpeechClause(text: str, performance: PerformanceCue)`.
- Changes: `split_clauses(chunks, performances)` returns `Iterator[SpeechClause]`.
- Changes: `ConversationWorker(..., performances, state, event_log, client, voice)` receives validated definitions.
- Later task consumes: `voice.enqueue(text, generation_id=..., performance=cue)`.

- [ ] **Step 1: Write failing parser/prompt tests**

Update `tests/test_llm.py` to cover these exact cases:

```python
from brain.llm import SpeechClause, build_prompt, split_clauses
from performance import PerformanceCue

PERFORMANCES = {
    "neutral": {"intensity": 0.0, "shape": {}, "movement": {}},
    "playful": {"intensity": 0.65, "shape": {}, "movement": {}},
    "thoughtful": {"intensity": 0.55, "shape": {}, "movement": {}},
}


def test_tagged_sentences_become_structured_clauses(self) -> None:
    clauses = list(split_clauses([
        "[[vess:thought", "ful]] Think first. ",
        "[[vess:playful]] Then joke!"
    ], PERFORMANCES))
    self.assertEqual(clauses, [
        SpeechClause("Think first.", PerformanceCue("thoughtful", 0.55)),
        SpeechClause("Then joke!", PerformanceCue("playful", 0.65)),
    ])


def test_unknown_reserved_tag_is_stripped_and_neutral(self) -> None:
    clauses = list(split_clauses(["[[vess:chaos]] Hello."], PERFORMANCES))
    self.assertEqual(clauses, [SpeechClause("Hello.", PerformanceCue())])


def test_untagged_sentence_is_neutral(self) -> None:
    self.assertEqual(
        list(split_clauses(["Hello."], PERFORMANCES)),
        [SpeechClause("Hello.", PerformanceCue())],
    )


def test_soft_split_inherits_cue_until_strong_boundary(self) -> None:
    long_prefix = "A" * 70
    long_middle = "B" * 70
    clauses = list(split_clauses([
        f"[[vess:playful]] {long_prefix}, {long_middle}, and finish. Next sentence."
    ], PERFORMANCES))
    self.assertEqual(clauses[0].performance.expression, "playful")
    self.assertEqual(clauses[1].performance.expression, "playful")
    self.assertEqual(clauses[-1].performance, PerformanceCue())
```

Add a prompt assertion that `build_prompt(...)` contains all configured labels once in the response-format instruction and does not include long per-label prose.

- [ ] **Step 2: Add failing memory/voice assertions**

Update the existing completed-conversation test so `RecordingVoice` records `(text, performance.expression)`, and assert the event/memory assistant string contains no `[[vess:` markup.

Update `FakeClient.stream` to emit tagged text such as:

```python
return ["[[vess:thoughtful]] First, then second."]
```

Expected voice text remains `First, then second.` and remembered assistant text remains `First, then second.`.

- [ ] **Step 3: Run focused tests and verify RED**

```powershell
python -m unittest tests.test_llm tests.test_short_term_memory -v
```

Expected: failures because `SpeechClause`, performance-aware parsing, and constructor wiring do not exist.

- [ ] **Step 4: Implement `SpeechClause` and marker parsing**

In `brain/llm.py`, import `PerformanceCue`/`cue_for_label` and add:

```python
@dataclass(frozen=True)
class SpeechClause:
    text: str
    performance: PerformanceCue
```

Refactor `split_clauses` to maintain:

```python
pending = ""
current_cue = PerformanceCue()
needs_cue = True
```

At the start of each logical sentence:

1. `lstrip()` only for parsing control whitespace.
2. If the buffered text is a prefix of `[[vess:` (for example `"[[ves"`), wait for more stream data instead of deciding it is normal prose.
3. If it begins with `[[vess:` and has a closing `]]`, extract the lowercase label, call `cue_for_label`, strip the entire marker, and set `needs_cue=False`.
4. If it does not begin with the reserved prefix, set neutral and `needs_cue=False`.
5. Use the existing `_clause_end` logic to find TTS boundaries.
6. Yield `SpeechClause(clean_text, current_cue)`.
7. If the emitted boundary character is one of `. ! ? \n`, reset `current_cue=PerformanceCue()` and `needs_cue=True`; comma/whitespace emergency splits retain the cue.
8. At end-of-stream, if a reserved `[[vess:` marker never closed, strip the reserved marker token through the first whitespace and emit the remaining prose as neutral rather than speaking metadata.

Do not change the current 120/180-character TTS boundary constants in this task.

- [ ] **Step 5: Add prompt format rules using the configured keys**

Extend `build_prompt` to accept `performances` and add one short stable section:

```text
Response format: Prefix every sentence with exactly one tag from:
[[vess:neutral]], [[vess:curious]], [[vess:amused]], [[vess:playful]],
[[vess:emphatic]], [[vess:thoughtful]], [[vess:sympathetic]], [[vess:uncertain]].
Choose how that sentence should be delivered. Do not explain or mention the tag.
```

Generate the list from `performances.keys()` so `performance.json` stays the single vocabulary source. Keep this instruction before recent conversation/current request so it remains stable/cache-friendly.

- [ ] **Step 6: Wire structured clauses through `ConversationWorker`**

Change `_respond` to:

```python
spoken_clauses: list[str] = []
for clause in split_clauses(self._client.stream(prompt, self._config), self._performances):
    ...
    spoken_clauses.append(clause.text)
    self._voice.enqueue(
        clause.text,
        generation_id=generation_id,
        performance=clause.performance,
    )
```

Add `performance=clause.performance.expression` to the `llm_first_clause` debug event. Store/join only `clause.text` so memory/event-log prose cannot contain tags.

Update `ConversationWorker.__init__` and `main.py` to pass the validated performance definitions loaded in Task 1.

- [ ] **Step 7: Run focused tests and verify GREEN**

```powershell
python -m unittest tests.test_llm tests.test_short_term_memory -v
```

Expected: all focused tests pass and no recorded voice/memory string contains `[[vess:`.

- [ ] **Step 8: Commit Task 2**

```powershell
git add brain/llm.py main.py tests/test_llm.py tests/test_short_term_memory.py
git commit -m "feat: parse sentence performance cues"
```

---

### Task 3: Synchronize Performance to Physical TTS Playback

**Files:**
- Modify: `output/voice.py`
- Modify: `tests/test_tts_pipeline.py`
- Modify: `tests/test_voice_freshness.py`

**Interfaces:**
- Consumes: `PerformanceCue` from Task 1 and `VoiceOutput.enqueue(..., performance=...)` calls from Task 2.
- Produces: active `State.performance` during physical playback only.
- Produces diagnostics: `performance_expression`, `performance_intensity`, `performance_started`, `performance_ended`.

- [ ] **Step 1: Write failing playback synchronization tests**

Add tests to `tests/test_tts_pipeline.py` using a blocking fake `play` callback:

```python
def test_performance_activates_only_during_physical_playback(self) -> None:
    state = State()
    play_started = threading.Event()
    release_play = threading.Event()

    def play(audio, sample_rate):
        self.assertEqual(state.performance.expression, "playful")
        play_started.set()
        release_play.wait(timeout=1.0)

    voice = VoiceOutput(..., state, ..., synthesize=lambda text: np.ones(10, np.float32), play=play)
    voice.start()
    voice.begin_generation(1)
    voice.enqueue("hello", generation_id=1,
                  performance=PerformanceCue("playful", 0.65))
    self.assertTrue(play_started.wait(timeout=0.5))
    release_play.set()
    voice.close()
    self.assertEqual(state.performance, PerformanceCue())
```

Add separate tests that:

- a prepared-but-not-yet-playing cue leaves `State.performance` neutral;
- stale prepared audio never activates its cue;
- if `play` raises, `State.performance` is neutral afterward;
- `performance_started` and `performance_ended` include expression/generation and started includes clause text.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m unittest tests.test_tts_pipeline tests.test_voice_freshness -v
```

Expected: `enqueue` does not accept `performance` and state never changes.

- [ ] **Step 3: Extend queue/prepared item types**

In `output/voice.py`, import `PerformanceCue`. Extend the normal `enqueue` signature:

```python
def enqueue(
    self,
    text: str,
    generation_id: int | None = None,
    performance: PerformanceCue | None = None,
) -> None:
    cue = performance or PerformanceCue()
    self._queue.put(("speak", text, generation_id, cue))
```

Carry `cue` through synthesis and `_ready_queue` tuples. The acknowledgement path always carries neutral and does not activate a transient performance cue.

Update existing tests/fakes that construct queue tuples only through public methods; do not expose new queue internals to tests.

- [ ] **Step 4: Activate/clear cue inside `_play_waveform`**

Immediately before recording `tts_playback_started`/calling the physical play callback:

```python
with self._state.locked():
    self._state.performance = performance
self._state.update_debug(
    performance_expression=performance.expression,
    performance_intensity=performance.intensity,
)
self._state.record_debug(
    "performance_started",
    text=text,
    expression=performance.expression,
    intensity=performance.intensity,
    generation_id=generation_id,
)
```

In the existing `finally` path after playback:

```python
with self._state.locked():
    self._state.performance = PerformanceCue()
self._state.update_debug(
    performance_expression="neutral",
    performance_intensity=0.0,
)
self._state.record_debug(
    "performance_ended",
    expression=performance.expression,
    generation_id=generation_id,
)
```

Only do this for normal response speech. A stale waveform returns before activation. Keep `state.speaking` semantics and TTS gap/silence diagnostics unchanged.

- [ ] **Step 5: Run focused tests and verify GREEN**

```powershell
python -m unittest tests.test_tts_pipeline tests.test_voice_freshness -v
```

Expected: all focused tests pass, including stale-generation behavior.

- [ ] **Step 6: Commit Task 3**

```powershell
git add output/voice.py tests/test_tts_pipeline.py tests/test_voice_freshness.py
git commit -m "feat: sync performance cues to playback"
```

---

### Task 4: Explicit Listening, Thinking, and Speaking Gaze Modes

**Files:**
- Modify: `output/animator.py`
- Create: `tests/test_animator.py`

**Interfaces:**
- Consumes: `State.listening`, `State.thinking`, `State.speaking`, `State.person_pos`.
- Produces: deterministic runtime interaction mode selection inside `FaceAnimator`.
- Later task composes performance modifiers onto these same gaze/movement values.

- [ ] **Step 1: Write deterministic runtime-mode tests**

Create `tests/test_animator.py`. Use `FaceAnimator(..., seed=1)` and inspect public/debuggable animator values after `tick` instead of whole-frame snapshots. Add tests equivalent to:

```python
def test_listening_prioritizes_person_and_suppresses_idle_fixation(self) -> None:
    state = State(listening=True, person_pos=(0.9, 0.5), person_present=True)
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    before = animator._fixation
    animator.tick(state, 0.05)
    self.assertEqual(animator._interaction_mode, "listening")
    self.assertEqual(animator._fixation, before)
    self.assertGreater(animator._gaze[0], 0.5)


def test_thinking_breaks_eye_contact_up_and_away(self) -> None:
    state = State(thinking=True, person_pos=(0.9, 0.8), person_present=True)
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(state, 0.05)
    self.assertEqual(animator._interaction_mode, "thinking")
    self.assertLess(animator._gaze[1], 0.0)


def test_speaking_outranks_plain_person_tracking(self) -> None:
    state = State(speaking=True, person_pos=(0.8, 0.5), person_present=True)
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(state, 0.05)
    self.assertEqual(animator._interaction_mode, "speaking")
```

Also test priority when multiple flags are true: listening > thinking > speaking > tracking > idle.

- [ ] **Step 2: Run animator tests and verify RED**

```powershell
python -m unittest tests.test_animator -v
```

Expected: constructor/signature/mode failures because performance config and explicit speaking mode are absent.

- [ ] **Step 3: Add runtime-mode selection**

In `FaceAnimator.__init__`, accept `performances` as the second config argument:

```python
def __init__(self, moods: dict[str, dict], performances: dict[str, dict[str, object]], seed: int | None = None) -> None:
```

Add `_interaction_mode = "idle"` and a helper:

```python
def _mode_for(self, listening: bool, thinking: bool, speaking: bool,
              person_pos: tuple[float, float] | None) -> str:
    if listening:
        return "listening"
    if thinking:
        return "thinking"
    if speaking:
        return "speaking"
    if person_pos is not None:
        return "tracking"
    return "idle"
```

Read `speaking` in `tick` and store the selected mode for the frame.

- [ ] **Step 4: Refactor gaze targeting around the selected mode**

Keep the current saccade/easing machinery but choose its target by mode:

```python
_THINK_GAZE = (-0.45, -0.72)
_SPEAK_BREAK_GAZE = (-0.55, -0.12)
_SPEAK_BREAK_CHECK = (1.8, 3.8)
_SPEAK_BREAK_LENGTH = (0.28, 0.60)
```

Behavior:

- `listening`: target person using existing `_track_target` with `track_break` forced to `0`; if no person, use `(0.0, 0.0)`; do not decrement/pick idle fixation.
- `thinking`: target `_THINK_GAZE` plus mood/performance gaze-y bias; do not use person tracking or idle fixation.
- `speaking`: normally target person (or center if absent). Maintain a separate speaking-break timer; when it fires, use `_SPEAK_BREAK_GAZE` for `_SPEAK_BREAK_LENGTH`, then return to person. Do not mutate idle fixation during speaking.
- `tracking`: preserve existing person-tracking behavior including mood `track_break`.
- `idle`: preserve existing fixation picker/drift.

Reuse the existing seeded RNG for speaking-break intervals so tests remain deterministic.

- [ ] **Step 5: Update whole-face movement to recognize speaking**

Change `_advance_face` to receive `mode` instead of separate booleans. Preserve current thinking/listening behavior. For speaking, use the existing `_TRACK_LEAN` target with `_FACE_TAU_ACTIVE`; do not create a new dramatic face offset.

This keeps speaking visually engaged while leaving expression changes to Task 5.

- [ ] **Step 6: Wire `main.py` animator constructor**

Pass the validated definitions:

```python
animator = FaceAnimator(moods, performances)
```

Update any existing constructor calls in tests.

- [ ] **Step 7: Run focused animator/regression tests and verify GREEN**

```powershell
python -m unittest tests.test_animator tests.test_color_override tests.test_main -v
```

Expected: all focused tests pass; existing mood/color rendering behavior remains intact.

- [ ] **Step 8: Commit Task 4**

```powershell
git add output/animator.py tests/test_animator.py main.py tests/test_main.py tests/test_color_override.py
git commit -m "feat: add conversational eye modes"
```

---

### Task 5: Fast Performance Overlays Composed over Mood

**Files:**
- Modify: `output/animator.py`
- Modify: `tests/test_animator.py`

**Interfaces:**
- Consumes: validated performance definitions and `State.performance`.
- Produces: `FaceAnimator.performance_current` numeric overlay state and composed mood+performance frame parameters.
- Keeps: mood color unchanged by performance.

- [ ] **Step 1: Write failing overlay composition/easing tests**

Add tests asserting:

```python
def test_neutral_performance_leaves_mood_shape_target_unchanged(self) -> None:
    state = State(mood="curious", performance=PerformanceCue())
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(state, 0.1)
    self.assertEqual(animator._performance_target["l_h"], 0.0)
    self.assertEqual(animator._performance_target["r_h"], 0.0)


def test_performance_composes_with_mood_instead_of_replacing_it(self) -> None:
    state = State(mood="curious", performance=PerformanceCue("playful", 0.65))
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    base = animator._target_for("curious")
    animator.tick(state, 0.2)
    self.assertNotEqual(animator._last_shape["l_h"], base["l_h"])
    self.assertEqual(animator._last_color, tuple(animator.current[f"color_{c}"] for c in "rgb"))


def test_performance_overlay_eases_instead_of_snapping(self) -> None:
    state = State(performance=PerformanceCue("emphatic", 0.7))
    animator = FaceAnimator(MOODS, PERFORMANCES, seed=1)
    animator.tick(state, 0.01)
    first = animator.performance_current["l_h"]
    target = animator._performance_target["l_h"]
    self.assertGreater(first, 0.0)
    self.assertLess(first, target)
```

Also test that every configured overlay stays inside the limits from `performance.py` after loading.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m unittest tests.test_animator -v
```

Expected: performance overlay fields/composition are absent.

- [ ] **Step 3: Add independent performance easing**

In `output/animator.py` add:

```python
_PERFORMANCE_TAU = 0.08
```

Initialize a neutral overlay containing all shape keys from `performance.py` plus movement keys. On each tick:

```python
cue = state.performance
entry = self._performances.get(cue.expression, self._performances["neutral"])
scale = cue.intensity
```

Build target deltas/scales:

- shape delta target = configured shape value * `scale`;
- movement scale target = `1.0 + (configured_scale - 1.0) * scale`;
- additive `gaze_y_bias` target = configured value * `scale`.

Ease every overlay field with `1 - exp(-dt / _PERFORMANCE_TAU)`.

If the cue changes directly from one expression to another, keep `performance_current` and retarget it; never zero it first. If state goes neutral between real pauses, it naturally eases toward zero/1.0.

- [ ] **Step 4: Compose shape overlay after mood interpolation**

After the existing mood `current` dict is eased, copy its non-color/non-movement shape fields and add only these deltas:

```python
for key in ("l_h", "r_h", "l_slant", "r_slant", "l_cy", "r_cy"):
    shape[key] += self.performance_current[key]
```

Clamp final eye heights to at least the renderer's practical nonzero range (`>= 2.0`) and centers/slants to the same bounded ranges already implied by the base shapes. Do not change color from performance.

Expose `_last_shape` and `_last_color` as diagnostic/test-only animator attributes updated immediately before `face.render`; they are not shared state.

- [ ] **Step 5: Compose movement overlay into interaction behavior**

Before calling gaze/face movement helpers, derive an effective movement dict:

```python
move["hold"] *= performance_current["hold_scale"]
move["ease"] *= performance_current["ease_scale"]
move["track_bias"] *= performance_current["track_bias_scale"]
move["gaze_y_bias"] += performance_current["gaze_y_bias"]
```

Multiply only the **speaking-mode** gaze-break probability/timer tendency by `speaking_break_scale`; do not alter idle or sad mood `track_break` semantics.

This yields restrained examples:

- emphatic: slightly wider eyes, more direct person tracking, fewer speaking gaze breaks;
- thoughtful: slightly narrower eyes, upward bias, longer holds/slower movement;
- playful: asymmetric shape, shorter holds, somewhat more speaking gaze breaks.

- [ ] **Step 6: Run animator tests and verify GREEN**

```powershell
python -m unittest tests.test_animator -v
```

Expected: runtime-mode, composition, easing, and bounds tests all pass.

- [ ] **Step 7: Commit Task 5**

```powershell
git add output/animator.py tests/test_animator.py
git commit -m "feat: add expressive eye performance overlays"
```

---

### Task 6: Diagnostics, Full Regression, and Live Acceptance

**Files:**
- Modify: `tests/test_llm.py`
- Modify: `tests/test_tts_pipeline.py`
- Modify: `tests/test_animator.py`
- Modify: `STATUS.md`

**Interfaces:**
- Verifies the complete flow from LLM marker -> clean `SpeechClause` -> queued cue -> physical playback state -> animator overlay.
- No new production interface should be introduced in this task unless a failing integration test exposes a missing one.

- [ ] **Step 1: Add one integration-style unit test across parser and voice**

Create a test in `tests/test_tts_pipeline.py` that obtains a `SpeechClause` from `split_clauses`, enqueues its text/cue into a fake `VoiceOutput`, and asserts during the fake physical `play` callback:

```python
self.assertEqual(state.performance.expression, "playful")
self.assertNotIn("[[vess:", clause.text)
```

After playback, assert neutral performance.

- [ ] **Step 2: Verify diagnostic payloads**

Assert the debug stream contains:

```text
llm_first_clause ... performance="<label>"
performance_started ... text="<clean clause>" expression="<label>" generation_id=<n>
performance_ended ... expression="<label>" generation_id=<n>
```

Also assert `debug_values` ends with `performance_expression="neutral"` and `performance_intensity=0.0` once speech is idle.

- [ ] **Step 3: Run the complete repository suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: **0 failures and 0 errors**. Do not update `STATUS.md` or claim completion if any test fails.

- [ ] **Step 4: Run the app and perform the live acceptance sequence**

Run:

```powershell
python main.py
```

Use a short conversation that naturally invites different delivery, for example:

```text
Hey Vess, explain why the sky is blue, but make the last part a little playful.
```

Accept only if all are true:

1. while the user speaks, eyes visibly focus rather than idle-wander;
2. thinking visibly breaks contact up/away;
3. speaking mostly engages the person with occasional bounded gaze breaks;
4. at least two sentence-level performance labels produce restrained visible differences;
5. mood color/base shape does not flip per sentence;
6. expression changes line up with actual clause playback, not early synthesis;
7. no `[[vess:...]]` text is heard;
8. `/debug` shows performance start/end events with cleaned clause text;
9. face preview stays smooth during LLM/TTS work.

If live behavior is too strong or too weak, tune only `performance.json` values first. Change animator physics only if the same defect appears across multiple expressions.

- [ ] **Step 5: Re-run the complete suite after any live tuning**

```powershell
python -m unittest discover -s tests -v
```

Expected: 0 failures and 0 errors.

- [ ] **Step 6: Update `STATUS.md` with measured behavior**

Add a concise entry recording:

- fixed performance vocabulary and marker protocol;
- no extra LLM request;
- cue activation synchronized to physical playback;
- listening/thinking/speaking eye modes;
- performance overlays layered over mood;
- actual final test count from Step 5;
- live acceptance notes, including any tuned `performance.json` values.

Do not claim expressive TTS is implemented; explicitly leave that as the next separate voice-engine research slice.

- [ ] **Step 7: Commit Task 6**

```powershell
git add tests/test_llm.py tests/test_tts_pipeline.py tests/test_animator.py STATUS.md performance.json
git commit -m "test: verify expressive performance flow"
```

---

## Final Verification Checklist

Before marking the implementation ready for review:

- [ ] `python -m unittest discover -s tests -v` reports 0 failures/errors.
- [ ] `performance.json` loads with required `neutral` and all values clamp safely.
- [ ] No prompt/memory/debug assistant prose contains raw `[[vess:` markers.
- [ ] Missing/unknown/malformed reserved tags degrade to neutral and are not spoken.
- [ ] Soft TTS splits inherit cues; strong sentence boundaries reset them.
- [ ] Prepared/stale audio cannot activate `State.performance` early.
- [ ] Playback exceptions clear performance in `finally`.
- [ ] Listening > thinking > speaking > tracking > idle priority is deterministic.
- [ ] Speaking behavior uses bounded gaze breaks rather than audio-amplitude motion.
- [ ] Performance overlays ease faster than mood and compose over it without changing mood color.
- [ ] `face.py` remains stateless and unchanged unless a renderer bug is proven.
- [ ] No second LLM request, TTS replacement, barge-in, or durable performance memory slipped into scope.
- [ ] Live preview remains smooth and expression timing matches physical speech.
