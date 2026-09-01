# TTS Engine Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a small text-to-speech engine boundary so Vess can keep Kokoro as the stable default, evaluate Chatterbox Turbo as an optional engine, and benchmark both through the same production interface without changing the proven speech scheduler.

**Architecture:** `VoiceOutput` remains responsible for queueing, stale-generation rejection, one-clause-ahead scheduling, trimming, playback, diagnostics, and physical-playback performance timing. A lightweight `TTSEngine` interface receives `text + PerformanceCue` and returns a `SynthesisResult(audio, sample_rate)`. Kokoro and Chatterbox Turbo live behind adapters; model initialization stays lazy on the existing synthesis worker.

**Tech Stack:** Python 3.11, `unittest`, NumPy, Kokoro, optional `chatterbox-tts`, `soundfile` for benchmark WAV output, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-01-tts-engine-abstraction-design.md`

## Global Constraints

- Default production engine remains `kokoro`.
- `VoiceOutput` must not import Kokoro or Chatterbox directly.
- Model initialization must happen lazily from the synthesis worker, never on the main/render thread.
- Existing queueing, stale-generation checks, ready-slot depth, playback order, performance activation timing, and silence trimming behavior must remain unchanged.
- `PerformanceCue` reaches the selected engine unchanged.
- Engine-returned `sample_rate` is authoritative for the generated waveform.
- Chatterbox remains optional; CI must not install `chatterbox-tts`, require CUDA, download a model, or perform real model inference.
- Selecting a missing/broken Chatterbox engine fails clearly; there is no silent Kokoro fallback.
- Real Chatterbox latency, VRAM, quality, and Qwen coexistence remain target-PC acceptance work.
- Tests continue to use `python -m unittest` and Python 3.11.
- Generated benchmark audio/results stay outside source control.

---

## File Structure

Create:

```text
output/tts/__init__.py                 public lightweight TTS types/factory exports
output/tts/base.py                     TTSEngine protocol + SynthesisResult validation
output/tts/kokoro.py                   lazy CPU Kokoro adapter
output/tts/factory.py                  engine-name -> cheap adapter construction
output/tts/chatterbox_turbo.py         optional lazy Chatterbox Turbo adapter

tests/tts_fakes.py                     reusable fake engine for voice/pipeline tests
tests/test_tts_base.py                 result/contract validation
tests/test_tts_kokoro.py               Kokoro adapter regression tests
tests/test_tts_factory.py              factory/lazy optional-dependency tests
tests/test_tts_chatterbox.py           structural Chatterbox tests with fake modules
tests/test_tts_benchmark.py            benchmark aggregation/output tests

tools/benchmark_tts.py                 standalone engine benchmark harness
requirements-chatterbox.txt            optional local Chatterbox dependency set
```

Modify:

```text
output/voice.py                        consume TTSEngine/SynthesisResult and per-item sample rate
config.json                            retain kokoro default; add only required Chatterbox fields
requirements.txt                       keep normal Vess baseline unchanged unless adapter truly needs shared deps
.gitignore                             ignore benchmark artifact directory
README.md or SETUP.md                  document optional Chatterbox install + benchmark command

tests/test_voice.py                    inject fake engine instead of text-only synth callback
tests/test_tts_pipeline.py             inject fake engine and verify sample-rate/performance flow
tests/test_performance_flow.py         inject fake engine
```

Do not modify LLM clause splitting, audio capture, detector, animator, face renderer, memory, or browser control for this slice.

---

### Task 1: Add the engine contract and validated synthesis result

**Files:**
- Create: `output/tts/__init__.py`
- Create: `output/tts/base.py`
- Create: `tests/test_tts_base.py`

**Interfaces:**
- Produces: `SynthesisResult(audio: np.ndarray, sample_rate: int)`
- Produces: `TTSEngine.synthesize(text: str, performance: PerformanceCue) -> SynthesisResult`
- Later tasks import these types only; `base.py` must not import model libraries.

- [ ] **Step 1: Write failing result-validation tests**

```python
import unittest

import numpy as np

from output.tts.base import SynthesisResult


class SynthesisResultTests(unittest.TestCase):
    def test_accepts_one_dimensional_float32_audio(self) -> None:
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        result = SynthesisResult(audio=audio, sample_rate=24_000)
        self.assertIs(result.audio, audio)
        self.assertEqual(result.sample_rate, 24_000)

    def test_rejects_non_float32_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "float32"):
            SynthesisResult(np.array([0.0], dtype=np.float64), 24_000)

    def test_rejects_non_1d_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            SynthesisResult(np.zeros((1, 3), dtype=np.float32), 24_000)

    def test_rejects_non_positive_sample_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            SynthesisResult(np.zeros(1, dtype=np.float32), 0)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
python -m unittest tests.test_tts_base -v
```

Expected: import failure because `output.tts.base` does not exist.

- [ ] **Step 3: Implement the minimal contract**

```python
# output/tts/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from performance import PerformanceCue


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.audio, np.ndarray):
            raise ValueError("audio must be a NumPy array")
        if self.audio.dtype != np.float32:
            raise ValueError("audio must use float32 dtype")
        if self.audio.ndim != 1:
            raise ValueError("audio must be one-dimensional")
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")


class TTSEngine(Protocol):
    def synthesize(self, text: str, performance: PerformanceCue) -> SynthesisResult:
        ...
```

Export only lightweight names from `output/tts/__init__.py`:

```python
from output.tts.base import SynthesisResult, TTSEngine

__all__ = ["SynthesisResult", "TTSEngine"]
```

- [ ] **Step 4: Run focused tests and confirm GREEN**

```bash
python -m unittest tests.test_tts_base -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add output/tts tests/test_tts_base.py
git commit -m "feat: add tts engine contract"
```

---

### Task 2: Extract Kokoro behind a lazy engine without changing its audio behavior

**Files:**
- Create: `output/tts/kokoro.py`
- Create: `tests/test_tts_kokoro.py`
- Modify later in Task 3: remove `_make_synthesizer` from `output/voice.py`

**Interfaces:**
- Consumes: `SynthesisResult`, `TTSEngine`, `PerformanceCue`
- Produces: `KokoroEngine(config: dict[str, Any])`
- Produces: `KokoroEngine.synthesize(text, performance) -> SynthesisResult`
- `KokoroEngine.__init__` must be cheap. It stores config only; no `kokoro` import and no `KPipeline` construction.

- [ ] **Step 1: Write failing Kokoro regression tests**

Use `unittest.mock.patch.dict(sys.modules, ...)` with a fake `kokoro` module so CI never loads the real model.

Tests must prove:

```python
class KokoroEngineTests(unittest.TestCase):
    def test_constructor_does_not_import_or_build_pipeline(self): ...
    def test_first_synthesis_builds_cpu_pipeline_once(self): ...
    def test_configured_voice_is_forwarded(self): ...
    def test_multiple_chunks_are_concatenated_in_order(self): ...
    def test_tensor_like_audio_becomes_numpy_float32(self): ...
    def test_empty_model_output_returns_empty_float32(self): ...
    def test_performance_is_currently_ignored_without_modifying_text(self): ...
```

The fake pipeline should expose iterable results where each item has an `.audio` attribute. Use a tensor-like fake with `detach()`, `cpu()`, and `numpy()` to preserve current conversion behavior.

- [ ] **Step 2: Run and confirm RED**

```bash
python -m unittest tests.test_tts_kokoro -v
```

Expected: import failure for `output.tts.kokoro`.

- [ ] **Step 3: Implement lazy KokoroEngine**

Use this shape:

```python
class KokoroEngine:
    SAMPLE_RATE = 24_000

    def __init__(self, config: dict[str, Any]) -> None:
        voice_config = config.get("voice", {})
        self._voice = str(voice_config.get("name", "af_heart"))
        self._pipeline: Any | None = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a", device="cpu")
        return self._pipeline

    def synthesize(self, text: str, performance: PerformanceCue) -> SynthesisResult:
        parts: list[np.ndarray] = []
        for result in self._get_pipeline()(text, voice=self._voice):
            audio = result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        joined = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return SynthesisResult(joined, self.SAMPLE_RATE)
```

Do not add invented Kokoro performance controls.

- [ ] **Step 4: Run focused tests**

```bash
python -m unittest tests.test_tts_kokoro -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add output/tts/kokoro.py tests/test_tts_kokoro.py
git commit -m "refactor: isolate kokoro tts engine"
```

---

### Task 3: Make VoiceOutput consume TTSEngine and engine-reported sample rates

**Files:**
- Create: `tests/tts_fakes.py`
- Modify: `output/voice.py`
- Modify: `tests/test_voice.py`
- Modify: `tests/test_tts_pipeline.py`
- Modify: `tests/test_performance_flow.py`

**Interfaces:**
- Consumes: `TTSEngine`, `SynthesisResult`
- `VoiceOutput(..., engine: TTSEngine | None = None, play=...)`
- A fake engine records `(text, PerformanceCue)` calls and returns configurable `SynthesisResult` values.
- Ready queue entries carry `sample_rate` beside each waveform.

- [ ] **Step 1: Add fake-engine helper and failing integration tests**

Create `tests/tts_fakes.py`:

```python
from collections.abc import Callable

import numpy as np

from output.tts.base import SynthesisResult
from performance import PerformanceCue


class FakeTTSEngine:
    def __init__(
        self,
        synthesize: Callable[[str, PerformanceCue], SynthesisResult] | None = None,
        *,
        sample_rate: int = 24_000,
    ) -> None:
        self.calls: list[tuple[str, PerformanceCue]] = []
        self._sample_rate = sample_rate
        self._synthesize = synthesize

    def synthesize(self, text: str, performance: PerformanceCue) -> SynthesisResult:
        self.calls.append((text, performance))
        if self._synthesize is not None:
            return self._synthesize(text, performance)
        return SynthesisResult(np.ones(1, dtype=np.float32), self._sample_rate)
```

Add tests proving:

1. `PerformanceCue("playful", 0.65)` reaches the engine unchanged.
2. A fake engine returning sample rate `16_000` causes the `play(audio, sample_rate)` callback to receive `16_000`, even if config still says `24_000`.
3. A prepared acknowledgement remembers both its waveform and its engine-returned sample rate.
4. Existing stale-generation tests still skip prepared stale audio before playback.

- [ ] **Step 2: Run the affected tests and confirm RED**

```bash
python -m unittest tests.test_voice tests.test_tts_pipeline tests.test_performance_flow -v
```

Expected: failures because `VoiceOutput` does not accept `engine=` and still assumes configured sample rate.

- [ ] **Step 3: Implement VoiceOutput migration**

Change constructor state from `_synthesize` to `_engine`:

```python
self._engine = engine
```

Add a lazy getter:

```python
def _get_engine(self) -> TTSEngine:
    if self._engine is None:
        from output.tts.factory import create_tts_engine
        self._engine = create_tts_engine(self._config)
    return self._engine
```

The factory import occurs inside the method so importing `output.voice` does not load optional model adapters unnecessarily.

Replace `_synthesize_text(text)` with:

```python
def _synthesize_text(self, text: str, performance: PerformanceCue) -> SynthesisResult:
    return self._get_engine().synthesize(text, performance)
```

Update normal speech synthesis:

```python
result = self._synthesize_text(text, cue)
audio = result.audio
sample_rate = result.sample_rate
```

Carry `sample_rate` in each ready-queue tuple and pass it into `_play_waveform`.

Update `_play_waveform` signature:

```python
def _play_waveform(
    self,
    audio: np.ndarray,
    *,
    sample_rate: int,
    ...
) -> None:
```

Use that argument for edge-silence diagnostics and playback. Remove the config lookup from `_play_waveform`.

For cached acknowledgements, store:

```python
self._acknowledgement_audio: np.ndarray | None
self._acknowledgement_sample_rate: int | None
```

and synthesize acknowledgement using neutral `PerformanceCue()`.

Remove `_make_synthesizer` from `output/voice.py` after all tests use engine injection.

- [ ] **Step 4: Migrate existing tests from `synthesize=` to `engine=`**

Example:

```python
engine = FakeTTSEngine(
    lambda text, performance: SynthesisResult(
        np.array([len(text)], dtype=np.float32),
        24_000,
    )
)
voice = VoiceOutput(CONFIG, state, RecordingLog(), engine=engine, play=play)
```

Do not weaken any existing scheduling/timing assertions.

- [ ] **Step 5: Run affected tests and full suite**

```bash
python -m unittest tests.test_voice tests.test_tts_pipeline tests.test_performance_flow -v
python -m unittest discover -s tests -v
```

Expected: all existing tests plus new engine-flow assertions PASS.

- [ ] **Step 6: Commit**

```bash
git add output/voice.py tests/tts_fakes.py tests/test_voice.py tests/test_tts_pipeline.py tests/test_performance_flow.py
git commit -m "refactor: route voice output through tts engines"
```

---

### Task 4: Add explicit engine factory and keep construction cheap

**Files:**
- Create: `output/tts/factory.py`
- Modify: `output/tts/__init__.py`
- Create: `tests/test_tts_factory.py`
- Verify: `config.json` already contains `voice.engine = "kokoro"`; do not churn it unless additional Chatterbox fields become necessary in Task 6.

**Interfaces:**
- Produces: `create_tts_engine(config: dict[str, Any]) -> TTSEngine`
- Accepted names: `kokoro`, `chatterbox_turbo`
- Construction returns cheap adapter objects only; models remain unloaded.

- [ ] **Step 1: Write failing factory tests**

Tests:

```python
def test_default_engine_is_kokoro(): ...
def test_explicit_kokoro_builds_kokoro_engine(): ...
def test_chatterbox_name_builds_adapter_without_importing_chatterbox_package(): ...
def test_unknown_engine_raises_clear_value_error(): ...
```

For the Chatterbox test, ensure `"chatterbox" not in sys.modules` before/after construction or patch Python import behavior so a premature import fails the test.

- [ ] **Step 2: Run and confirm RED**

```bash
python -m unittest tests.test_tts_factory -v
```

- [ ] **Step 3: Implement factory**

```python
def create_tts_engine(config: dict[str, Any]) -> TTSEngine:
    name = str(config.get("voice", {}).get("engine", "kokoro")).strip().lower()
    if name == "kokoro":
        from output.tts.kokoro import KokoroEngine
        return KokoroEngine(config)
    if name == "chatterbox_turbo":
        from output.tts.chatterbox_turbo import ChatterboxTurboEngine
        return ChatterboxTurboEngine(config)
    raise ValueError(f"unknown TTS engine: {name!r}")
```

The Chatterbox adapter module itself must also avoid importing the third-party package at module import time.

- [ ] **Step 4: Run focused + voice tests**

```bash
python -m unittest tests.test_tts_factory tests.test_voice tests.test_tts_pipeline -v
```

- [ ] **Step 5: Commit**

```bash
git add output/tts/factory.py output/tts/__init__.py tests/test_tts_factory.py
git commit -m "feat: select tts engines from config"
```

---

### Task 5: Add a model-independent benchmark harness

**Files:**
- Create: `tools/benchmark_tts.py`
- Create: `tests/test_tts_benchmark.py`
- Modify: `.gitignore`

**Interfaces:**
- Uses production `create_tts_engine()` by default.
- Accepts an injected engine in unit tests.
- Produces JSON rows and WAV files under `artifacts/tts-benchmark/<engine>/`.
- Does not import Whisper, Ollama, camera, face, or playback code.

- [ ] **Step 1: Write failing benchmark tests**

Test pure helpers rather than invoking real models:

```python
def test_benchmark_row_records_latency_duration_rtf_and_sample_rate(): ...
def test_zero_length_audio_uses_null_or_safe_rtf_instead_of_dividing_by_zero(): ...
def test_json_output_contains_every_run(): ...
def test_wav_output_uses_engine_sample_rate(): ...
def test_failed_synthesis_records_error_without_fabricating_audio(): ...
```

Use a fake clock injected into the measurement function so latency tests are deterministic.

- [ ] **Step 2: Run and confirm RED**

```bash
python -m unittest tests.test_tts_benchmark -v
```

- [ ] **Step 3: Implement benchmark data model/helpers**

Use a standard text set:

```python
STANDARD_TEXTS = (
    ("short", "Sure."),
    ("medium", "That's actually pretty interesting."),
    ("skeptical", "I don't think that's going to work, though."),
    ("question", "Why would that happen?"),
    ("emphatic", "That's ridiculous."),
)
```

Record per run:

```text
engine
text_id
text
run_index
warm
synthesis_ms
audio_duration_ms
realtime_factor
sample_rate
sample_count
success
error
```

Compute:

```python
audio_seconds = result.audio.size / result.sample_rate
rtf = elapsed_seconds / audio_seconds if audio_seconds > 0 else None
```

- [ ] **Step 4: Implement CLI/output**

Support:

```bash
python tools/benchmark_tts.py --engine kokoro --runs 3
python tools/benchmark_tts.py --engine chatterbox_turbo --runs 3
```

Write:

```text
artifacts/tts-benchmark/<engine>/results.json
artifacts/tts-benchmark/<engine>/<text_id>-run<N>.wav
```

Use `soundfile.write(path, audio, sample_rate)` only in the CLI/output layer.

Print a concise table showing median warm synthesis time and median warm RTF per text.

Optional CUDA metrics must be collected only when a safe telemetry function is available; absence of CUDA metrics is not an error.

- [ ] **Step 5: Ignore generated benchmark outputs**

Append:

```text
artifacts/tts-benchmark/
```

to `.gitignore`.

- [ ] **Step 6: Run tests**

```bash
python -m unittest tests.test_tts_benchmark -v
python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit**

```bash
git add tools/benchmark_tts.py tests/test_tts_benchmark.py .gitignore
git commit -m "feat: add tts benchmark harness"
```

---

### Task 6: Add optional Chatterbox Turbo adapter without requiring it in CI

**Files:**
- Create: `output/tts/chatterbox_turbo.py`
- Create: `tests/test_tts_chatterbox.py`
- Create: `requirements-chatterbox.txt`
- Modify: `config.json` only for concrete adapter fields
- Modify: `SETUP.md`

**Interfaces:**
- Produces: `ChatterboxTurboEngine(config: dict[str, Any])`
- Uses official Python API only after first synthesis:
  - `from chatterbox.tts_turbo import ChatterboxTurboTTS`
  - `ChatterboxTurboTTS.from_pretrained(device=...)`
  - `model.generate(text, audio_prompt_path=...)`
  - `model.sr`
- Returns flattened CPU NumPy `float32` audio in `SynthesisResult`.

- [ ] **Step 1: Write failing structural tests with a fake Chatterbox module**

Tests must prove:

```python
def test_constructor_does_not_import_or_load_model(): ...
def test_first_synthesis_loads_model_once(): ...
def test_device_config_is_passed_to_from_pretrained(): ...
def test_reference_audio_path_is_forwarded_to_generate(): ...
def test_output_tensor_is_detached_moved_to_cpu_flattened_float32(): ...
def test_model_sample_rate_is_returned(): ...
def test_missing_dependency_raises_engine_specific_runtime_error(): ...
def test_unknown_performance_does_not_modify_text(): ...
```

Do not download a model in any test.

- [ ] **Step 2: Run and confirm RED**

```bash
python -m unittest tests.test_tts_chatterbox -v
```

- [ ] **Step 3: Implement lazy adapter**

Recommended shape:

```python
class ChatterboxTurboEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        voice = config.get("voice", {})
        chatterbox = voice.get("chatterbox", {})
        self._device = str(chatterbox.get("device", "cuda"))
        self._reference_audio = chatterbox.get("reference_audio")
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
            except ImportError as error:
                raise RuntimeError(
                    "TTS engine 'chatterbox_turbo' requires the optional chatterbox-tts package"
                ) from error
            self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)
        return self._model

    def synthesize(self, text: str, performance: PerformanceCue) -> SynthesisResult:
        model = self._get_model()
        kwargs: dict[str, object] = {}
        if self._reference_audio:
            kwargs["audio_prompt_path"] = str(self._reference_audio)
        wav = model.generate(text, **kwargs)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return SynthesisResult(audio, int(model.sr))
```

V1 passes plain text through unchanged. Do not map performance labels to undocumented style tokens yet. This keeps the abstraction valid while real expressive behavior waits for listening tests.

- [ ] **Step 4: Add optional dependency file**

`requirements-chatterbox.txt`:

```text
-r requirements.txt
chatterbox-tts
```

Do not add `chatterbox-tts` to `requirements-ci.txt`.

- [ ] **Step 5: Add concrete config block while preserving Kokoro default**

Keep:

```json
"engine": "kokoro"
```

and add:

```json
"chatterbox": {
  "device": "cuda",
  "reference_audio": ""
}
```

An empty reference path means use the model's default/no prompt path behavior. Do not invent a reference file.

- [ ] **Step 6: Document local install and test command**

In `SETUP.md`, document:

```powershell
pip install -r requirements-chatterbox.txt
python tools/benchmark_tts.py --engine chatterbox_turbo --runs 3
```

State clearly that real Chatterbox testing requires the target PC and that Kokoro remains the default.

- [ ] **Step 7: Run structural tests and prove CI dependency isolation**

```bash
python -m unittest tests.test_tts_chatterbox tests.test_tts_factory -v
python -m unittest discover -s tests -v
```

Also inspect `requirements-ci.txt` and confirm it does not contain `chatterbox-tts`, `torch`, or CUDA-only dependencies introduced by this change.

- [ ] **Step 8: Commit**

```bash
git add output/tts/chatterbox_turbo.py tests/test_tts_chatterbox.py requirements-chatterbox.txt config.json SETUP.md
git commit -m "feat: add optional chatterbox turbo engine"
```

---

### Task 7: Full regression verification and branch review

**Files:**
- Potentially modify only files required to fix regressions found by the commands below.
- Do not broaden scope into TTS tuning or performance-tag mapping.

**Interfaces:**
- Final production path: `VoiceOutput -> TTSEngine -> SynthesisResult`
- Final default: `kokoro`
- Chatterbox remains experimental until target-PC acceptance.

- [ ] **Step 1: Run the complete unit suite**

```bash
python -m unittest discover -s tests -v
```

Expected: every test passes.

- [ ] **Step 2: Run headless behavior verification**

```bash
python tools/render_behavior_preview.py
python tools/render_eye_validation.py
```

Expected: both exit 0. This protects the already-approved expressive-eye behavior from accidental voice refactor regressions.

- [ ] **Step 3: Run a model-free Kokoro-factory smoke test**

Do not synthesize real audio. Construct the configured engine and confirm its constructor is cheap:

```bash
python -c "import json; from output.tts.factory import create_tts_engine; c=json.load(open('config.json')); e=create_tts_engine(c); print(type(e).__name__)"
```

Expected:

```text
KokoroEngine
```

- [ ] **Step 4: Review the diff against the approved base**

```bash
git diff --stat design/expressive-performance...HEAD
git diff design/expressive-performance...HEAD -- output/voice.py output/tts tests tools/benchmark_tts.py config.json requirements-chatterbox.txt SETUP.md .gitignore
```

Verify:

- no LLM/camera/audio-capture/animator behavior changed
- Kokoro remains default
- no model library imports occur from `output/voice.py`
- Chatterbox third-party import occurs only inside lazy load code
- benchmark artifacts are ignored
- no silent fallback exists

- [ ] **Step 5: Verify GitHub Actions on the exact branch head**

Push/current branch triggers `.github/workflows/verify.yml`. Require:

```text
unit-tests: success
behavior-preview: success
```

Do not call the implementation complete until CI for the exact final SHA is green.

- [ ] **Step 6: Record target-PC acceptance as intentionally pending**

The branch/PR description must explicitly retain these pending items:

```text
- install/load real Chatterbox Turbo
- benchmark warm short-clause synthesis
- measure peak VRAM with Qwen resident
- measure Qwen tokens/sec / first-clause slowdown
- listen to WAV comparisons
- decide whether/how PerformanceCue maps to vocal style
```

These are not remote failures; they are hardware acceptance steps.

- [ ] **Step 7: Commit any verification-only fixes**

```bash
git add <only files actually changed by fixes>
git commit -m "test: verify tts engine abstraction"
```

Skip this commit if verification required no code changes.

---

## Completion Checklist

Before opening/merging the stacked PR:

```text
[ ] SynthesisResult validates 1D float32 + positive sample rate
[ ] Kokoro model still loads lazily on synthesis worker
[ ] VoiceOutput no longer constructs Kokoro directly
[ ] engine sample rate reaches trimming, diagnostics, and playback
[ ] PerformanceCue reaches TTSEngine unchanged
[ ] acknowledgement cache preserves its sample rate
[ ] existing one-clause-ahead pipeline behavior remains green
[ ] engine factory rejects unknown names explicitly
[ ] Chatterbox adapter is optional and lazy
[ ] CI has no chatterbox/CUDA/model requirement
[ ] benchmark harness writes comparable JSON + WAV outputs
[ ] Kokoro remains default
[ ] all unit tests pass
[ ] behavior preview passes
[ ] comprehensive eye validation passes
[ ] final branch CI passes on exact final SHA
[ ] target-PC Chatterbox acceptance remains marked pending
```
