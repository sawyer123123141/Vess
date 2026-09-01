# TTS Engine Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a small text-to-speech engine boundary so Vess can keep Kokoro as the stable default, evaluate Chatterbox Turbo as an optional engine, and benchmark both through the same production interface without changing the proven speech scheduler.

**Architecture:** `VoiceOutput` keeps queueing, stale-generation rejection, one-clause-ahead scheduling, trimming, playback, diagnostics, and physical-playback performance timing. A lightweight `TTSEngine` receives text plus `PerformanceCue` and returns `SynthesisResult(audio, sample_rate)`. Kokoro and Chatterbox Turbo live behind adapters whose heavy models load lazily from the existing synthesis worker.

**Tech Stack:** Python 3.11, `unittest`, NumPy, Kokoro, optional `chatterbox-tts`, `soundfile`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-01-tts-engine-abstraction-design.md`

## Global Constraints

- Default production engine remains `kokoro`.
- `VoiceOutput` never imports Kokoro or Chatterbox directly.
- Heavy model initialization happens lazily from the synthesis worker, never from the main/render thread.
- Existing queueing, stale-generation checks, one-ready-waveform depth, playback order, performance activation timing, and silence trimming behavior remain unchanged.
- `PerformanceCue` reaches the selected engine unchanged.
- Engine-returned `sample_rate` is authoritative for generated audio.
- Chatterbox is optional. CI does not install `chatterbox-tts`, require CUDA, download a model, or perform real model inference.
- Missing/broken selected engines fail clearly. There is no silent fallback.
- Real Chatterbox latency, VRAM, quality, and Qwen coexistence remain target-PC acceptance work.
- Tests use `python -m unittest` on Python 3.11.
- Generated benchmark audio/results stay outside source control.

## Files

Create:

```text
output/tts/__init__.py
output/tts/base.py
output/tts/kokoro.py
output/tts/factory.py
output/tts/chatterbox_turbo.py
tests/tts_fakes.py
tests/test_tts_base.py
tests/test_tts_kokoro.py
tests/test_tts_factory.py
tests/test_tts_chatterbox.py
tests/test_tts_benchmark.py
tools/benchmark_tts.py
requirements-chatterbox.txt
```

Modify:

```text
output/voice.py
config.json
.gitignore
SETUP.md
tests/test_voice.py
tests/test_tts_pipeline.py
tests/test_performance_flow.py
```

Do not modify LLM clause splitting, audio capture, detector, animator, face renderer, memory, or browser control.

---

### Task 1: Add the TTS contract

**Files:**
- Create: `output/tts/__init__.py`
- Create: `output/tts/base.py`
- Create: `tests/test_tts_base.py`

**Interfaces:**
- Produces `SynthesisResult(audio: np.ndarray, sample_rate: int)`.
- Produces protocol method `TTSEngine.synthesize(text: str, performance: PerformanceCue) -> SynthesisResult`.

- [ ] **Step 1: Write the failing contract tests**

```python
import unittest

import numpy as np

from output.tts.base import SynthesisResult


class SynthesisResultTests(unittest.TestCase):
    def test_accepts_one_dimensional_float32_audio(self) -> None:
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        result = SynthesisResult(audio, 24_000)
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_tts_base -v
```

Expected: import failure because `output.tts.base` does not exist.

- [ ] **Step 3: Implement the minimal contract**

```python
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
        raise NotImplementedError
```

`output/tts/__init__.py`:

```python
from output.tts.base import SynthesisResult, TTSEngine

__all__ = ["SynthesisResult", "TTSEngine"]
```

- [ ] **Step 4: Run GREEN**

```bash
python -m unittest tests.test_tts_base -v
```

- [ ] **Step 5: Commit**

```bash
git add output/tts tests/test_tts_base.py
git commit -m "feat: add tts engine contract"
```

---

### Task 2: Extract Kokoro into a lazy adapter

**Files:**
- Create: `output/tts/kokoro.py`
- Create: `tests/test_tts_kokoro.py`

**Interfaces:**
- Produces `KokoroEngine(config)`.
- Constructor stores only configuration.
- First `synthesize()` call imports `kokoro`, builds `KPipeline(lang_code="a", device="cpu")`, and caches it.
- Returns 24 kHz `SynthesisResult` matching current behavior.

- [ ] **Step 1: Write failing regression tests using a fake Kokoro module**

The test module contains a fake result and pipeline:

```python
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from performance import PerformanceCue


class Result:
    def __init__(self, audio: object) -> None:
        self.audio = audio


class FakePipeline:
    builds = 0
    calls: list[tuple[str, str]] = []

    def __init__(self, lang_code: str, device: str) -> None:
        type(self).builds += 1
        self.lang_code = lang_code
        self.device = device

    def __call__(self, text: str, *, voice: str):
        type(self).calls.append((text, voice))
        return [
            Result(np.array([0.1, 0.2], dtype=np.float32)),
            Result(np.array([0.3], dtype=np.float32)),
        ]


class KokoroEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        FakePipeline.builds = 0
        FakePipeline.calls.clear()

    def test_constructor_does_not_build_pipeline(self) -> None:
        from output.tts.kokoro import KokoroEngine
        engine = KokoroEngine({"voice": {"name": "af_heart"}})
        self.assertIsNone(engine._pipeline)
        self.assertEqual(FakePipeline.builds, 0)

    def test_first_synthesis_builds_once_and_concatenates_chunks(self) -> None:
        fake_module = types.SimpleNamespace(KPipeline=FakePipeline)
        with patch.dict(sys.modules, {"kokoro": fake_module}):
            from output.tts.kokoro import KokoroEngine
            engine = KokoroEngine({"voice": {"name": "af_heart"}})
            first = engine.synthesize("first", PerformanceCue())
            second = engine.synthesize("second", PerformanceCue())
        self.assertEqual(FakePipeline.builds, 1)
        self.assertEqual(FakePipeline.calls, [("first", "af_heart"), ("second", "af_heart")])
        np.testing.assert_allclose(first.audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        self.assertEqual(first.sample_rate, 24_000)
        self.assertEqual(second.audio.dtype, np.float32)
```

Add two more concrete cases in the same file:
- fake pipeline returns no results, assert an empty 1D float32 array;
- fake audio implements `detach().cpu().numpy()`, assert conversion produces 1D float32.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_tts_kokoro -v
```

- [ ] **Step 3: Implement KokoroEngine**

```python
from __future__ import annotations

from typing import Any

import numpy as np

from output.tts.base import SynthesisResult
from performance import PerformanceCue


class KokoroEngine:
    SAMPLE_RATE = 24_000

    def __init__(self, config: dict[str, Any]) -> None:
        self._voice = str(config.get("voice", {}).get("name", "af_heart"))
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

Do not add Kokoro performance mapping.

- [ ] **Step 4: Run GREEN**

```bash
python -m unittest tests.test_tts_kokoro -v
```

- [ ] **Step 5: Commit**

```bash
git add output/tts/kokoro.py tests/test_tts_kokoro.py
git commit -m "refactor: isolate kokoro tts engine"
```

---

### Task 3: Route VoiceOutput through TTSEngine

**Files:**
- Create: `tests/tts_fakes.py`
- Modify: `output/voice.py`
- Modify: `tests/test_voice.py`
- Modify: `tests/test_tts_pipeline.py`
- Modify: `tests/test_performance_flow.py`

**Interfaces:**
- Constructor accepts `engine: TTSEngine | None = None` and `play`.
- Ready queue stores sample rate with each waveform.
- Acknowledgement cache stores waveform plus sample rate.
- Default engine is created lazily from the synthesis thread through the factory.

- [ ] **Step 1: Add reusable fake engine**

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

- [ ] **Step 2: Add RED assertions before changing production code**

Add this case to `tests/test_voice.py`:

```python
def test_engine_sample_rate_reaches_playback(self) -> None:
    played_rates: list[int] = []
    engine = FakeTTSEngine(sample_rate=16_000)
    voice = VoiceOutput(
        {"voice": {"sample_rate": 24_000}},
        State(),
        RecordingLog(),
        engine=engine,
        play=lambda audio, sample_rate: played_rates.append(sample_rate),
    )
    voice.start()
    voice.enqueue("hello", performance=PerformanceCue("playful", 0.65))
    voice.close()
    self.assertEqual(played_rates, [16_000])
    self.assertEqual(engine.calls, [("hello", PerformanceCue("playful", 0.65))])
```

Add an acknowledgement case that uses a 22,050 Hz fake result and asserts the playback callback receives 22,050.

- [ ] **Step 3: Run RED**

```bash
python -m unittest tests.test_voice -v
```

Expected: constructor does not yet accept `engine=`.

- [ ] **Step 4: Implement engine integration**

In `output/voice.py`:

```python
from output.tts.base import SynthesisResult, TTSEngine
```

Store `self._engine = engine` and add:

```python
def _get_engine(self) -> TTSEngine:
    if self._engine is None:
        from output.tts.factory import create_tts_engine
        self._engine = create_tts_engine(self._config)
    return self._engine


def _synthesize_text(self, text: str, performance: PerformanceCue) -> SynthesisResult:
    return self._get_engine().synthesize(text, performance)
```

For normal speech:

```python
result = self._synthesize_text(text, cue)
audio = result.audio
sample_rate = result.sample_rate
raw_edge_silence = _waveform_edge_silence_ms(audio, sample_rate)
audio = _trim_waveform_edges(audio, sample_rate)
```

Add `sample_rate` to the ready-queue tuple and require it in `_play_waveform`. Remove every playback-time lookup of `config["voice"]["sample_rate"]`.

For acknowledgement preparation:

```python
result = self._synthesize_text(text, PerformanceCue())
self._acknowledgement_audio = result.audio
self._acknowledgement_sample_rate = result.sample_rate
```

Remove `_make_synthesizer` when migration is complete.

- [ ] **Step 5: Migrate existing tests to FakeTTSEngine**

Replace each `synthesize=lambda text: audio` injection with:

```python
engine=FakeTTSEngine(
    lambda text, performance: SynthesisResult(audio_for_text(text), 24_000)
)
```

For simple fixed audio:

```python
engine=FakeTTSEngine(
    lambda text, performance: SynthesisResult(np.ones(10, dtype=np.float32), 1_000)
)
```

Do not change stale-generation, overlap, ready-depth, performance timing, or gap assertions.

- [ ] **Step 6: Run focused and full tests**

```bash
python -m unittest tests.test_voice tests.test_tts_pipeline tests.test_performance_flow -v
python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit**

```bash
git add output/voice.py tests/tts_fakes.py tests/test_voice.py tests/test_tts_pipeline.py tests/test_performance_flow.py
git commit -m "refactor: route voice output through tts engines"
```

---

### Task 4: Add engine factory

**Files:**
- Create: `output/tts/factory.py`
- Create: `tests/test_tts_factory.py`
- Modify: `output/tts/__init__.py`

**Interfaces:**
- Produces `create_tts_engine(config) -> TTSEngine`.
- Accepts `kokoro` and `chatterbox_turbo`.
- Adapter construction is cheap and does not load a model.

- [ ] **Step 1: Write RED factory tests**

```python
import unittest

from output.tts.factory import create_tts_engine


class TtsFactoryTests(unittest.TestCase):
    def test_default_engine_is_kokoro(self) -> None:
        engine = create_tts_engine({"voice": {}})
        self.assertEqual(type(engine).__name__, "KokoroEngine")

    def test_explicit_kokoro_engine_is_kokoro(self) -> None:
        engine = create_tts_engine({"voice": {"engine": "kokoro"}})
        self.assertEqual(type(engine).__name__, "KokoroEngine")

    def test_chatterbox_selection_constructs_adapter_only(self) -> None:
        engine = create_tts_engine({"voice": {"engine": "chatterbox_turbo"}})
        self.assertEqual(type(engine).__name__, "ChatterboxTurboEngine")
        self.assertIsNone(engine._model)

    def test_unknown_engine_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown TTS engine"):
            create_tts_engine({"voice": {"engine": "made_up"}})
```

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_tts_factory -v
```

- [ ] **Step 3: Implement factory**

```python
from __future__ import annotations

from typing import Any

from output.tts.base import TTSEngine


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

Export `create_tts_engine` from `output/tts/__init__.py` only if that does not create circular imports; otherwise import it directly from `output.tts.factory` at call sites.

- [ ] **Step 4: Run GREEN plus voice tests**

```bash
python -m unittest tests.test_tts_factory tests.test_voice tests.test_tts_pipeline -v
```

- [ ] **Step 5: Commit**

```bash
git add output/tts/factory.py output/tts/__init__.py tests/test_tts_factory.py
git commit -m "feat: select tts engines from config"
```

---

### Task 5: Add model-independent benchmark harness

**Files:**
- Create: `tools/benchmark_tts.py`
- Create: `tests/test_tts_benchmark.py`
- Modify: `.gitignore`

**Interfaces:**
- Uses the production engine contract.
- Unit tests inject fake engines and a fake clock.
- Writes JSON and WAVs to `artifacts/tts-benchmark/<engine>/`.

- [ ] **Step 1: Write RED helper tests**

The benchmark module exposes:

```python
measure_one(engine_name, engine, text_id, text, run_index, now)
write_results(output_dir, rows)
write_wave(path, result)
```

Use this deterministic test:

```python
def test_measure_one_records_latency_duration_and_rtf(self) -> None:
    times = iter([10.0, 10.25])
    now = lambda: next(times)
    engine = FakeTTSEngine(
        lambda text, performance: SynthesisResult(
            np.ones(24_000, dtype=np.float32),
            24_000,
        )
    )
    row, result = measure_one("fake", engine, "one", "hello", 0, now)
    self.assertTrue(row["success"])
    self.assertEqual(row["synthesis_ms"], 250.0)
    self.assertEqual(row["audio_duration_ms"], 1000.0)
    self.assertEqual(row["realtime_factor"], 0.25)
    self.assertEqual(result.sample_rate, 24_000)
```

Add exact cases for zero-length audio (`realtime_factor is None`) and synthesis exception (`success=False`, `error` contains exception text, returned result is `None`).

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_tts_benchmark -v
```

- [ ] **Step 3: Implement standard texts and measurement**

```python
STANDARD_TEXTS = (
    ("short", "Sure."),
    ("medium", "That's actually pretty interesting."),
    ("skeptical", "I don't think that's going to work, though."),
    ("question", "Why would that happen?"),
    ("emphatic", "That's ridiculous."),
)
```

Each row contains:

```text
engine, text_id, text, run_index, warm, synthesis_ms,
audio_duration_ms, realtime_factor, sample_rate, sample_count,
success, error
```

Run index 0 is cold; later runs are warm.

- [ ] **Step 4: Implement CLI and artifacts**

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

Use `soundfile.write` only when writing WAVs. Print median warm synthesis milliseconds and median warm realtime factor per text.

- [ ] **Step 5: Ignore artifacts**

Append exactly:

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

### Task 6: Add optional Chatterbox Turbo adapter

**Files:**
- Create: `output/tts/chatterbox_turbo.py`
- Create: `tests/test_tts_chatterbox.py`
- Create: `requirements-chatterbox.txt`
- Modify: `config.json`
- Modify: `SETUP.md`

**Interfaces:**
- Constructor stores config only.
- First synthesis imports `ChatterboxTurboTTS` and calls `from_pretrained(device=...)`.
- Generation uses `model.generate(text, audio_prompt_path=...)` only when a reference path is configured.
- Output uses `model.sr`.
- V1 passes text unchanged regardless of performance cue.

- [ ] **Step 1: Write RED structural tests with fake modules**

Use a fake model:

```python
class FakeTensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[0.1, 0.2]], dtype=np.float32)


class FakeModel:
    sr = 24_000

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def generate(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        return FakeTensor()
```

Tests assert:
- constructor leaves `_model is None`;
- first synthesis invokes fake `from_pretrained(device="cuda")` exactly once;
- second synthesis reuses the model;
- `reference_audio="voice.wav"` produces `audio_prompt_path="voice.wav"`;
- empty reference path omits `audio_prompt_path`;
- returned waveform is flat float32 and uses `model.sr`;
- missing package raises `RuntimeError` containing `chatterbox_turbo` and `chatterbox-tts`;
- passing `PerformanceCue("playful", 0.65)` does not alter the text sent to `generate`.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_tts_chatterbox -v
```

- [ ] **Step 3: Implement lazy adapter**

```python
from __future__ import annotations

from typing import Any

import numpy as np

from output.tts.base import SynthesisResult
from performance import PerformanceCue


class ChatterboxTurboEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        chatterbox = config.get("voice", {}).get("chatterbox", {})
        self._device = str(chatterbox.get("device", "cuda"))
        self._reference_audio = str(chatterbox.get("reference_audio", "")).strip()
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
            kwargs["audio_prompt_path"] = self._reference_audio
        wav = model.generate(text, **kwargs)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return SynthesisResult(audio, int(model.sr))
```

- [ ] **Step 4: Add optional dependencies**

`requirements-chatterbox.txt`:

```text
-r requirements.txt
chatterbox-tts
```

Do not add Chatterbox, torch, or CUDA packages to `requirements-ci.txt`.

- [ ] **Step 5: Add config without changing default**

Under `voice`, retain:

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

- [ ] **Step 6: Document local use**

In `SETUP.md`, include:

```powershell
pip install -r requirements-chatterbox.txt
python tools/benchmark_tts.py --engine chatterbox_turbo --runs 3
```

State that Kokoro remains default and real Chatterbox acceptance requires the target PC.

- [ ] **Step 7: Run structural and full tests**

```bash
python -m unittest tests.test_tts_chatterbox tests.test_tts_factory -v
python -m unittest discover -s tests -v
```

- [ ] **Step 8: Commit**

```bash
git add output/tts/chatterbox_turbo.py tests/test_tts_chatterbox.py requirements-chatterbox.txt config.json SETUP.md
git commit -m "feat: add optional chatterbox turbo engine"
```

---

### Task 7: Full regression verification

**Files:**
- Modify only files required to fix regressions found by verification.
- Do not add voice-style tuning or new performance tags in this task.

- [ ] **Step 1: Run all unit tests**

```bash
python -m unittest discover -s tests -v
```

- [ ] **Step 2: Run both headless behavior verifiers**

```bash
python tools/render_behavior_preview.py
python tools/render_eye_validation.py
```

Both must exit 0.

- [ ] **Step 3: Run model-free factory smoke test**

```bash
python -c "import json; from output.tts.factory import create_tts_engine; c=json.load(open('config.json')); e=create_tts_engine(c); print(type(e).__name__)"
```

Expected output:

```text
KokoroEngine
```

- [ ] **Step 4: Review final diff against `design/expressive-performance`**

```bash
git diff --stat design/expressive-performance...HEAD
git diff design/expressive-performance...HEAD -- output/voice.py output/tts tests tools/benchmark_tts.py config.json requirements-chatterbox.txt SETUP.md .gitignore
```

Confirm:
- no LLM/camera/audio-capture/animator behavior changed;
- Kokoro remains default;
- no model library imports occur from `output/voice.py`;
- Chatterbox third-party import exists only inside lazy model loading;
- benchmark artifacts are ignored;
- no silent fallback exists.

- [ ] **Step 5: Require GitHub Actions on the exact final SHA**

The branch-triggered workflow must finish with:

```text
unit-tests: success
behavior-preview: success
```

Do not claim completion before both are green for the final commit.

- [ ] **Step 6: Keep hardware acceptance explicitly pending**

The eventual PR description records these unresolved target-PC checks:

```text
install/load real Chatterbox Turbo
benchmark warm short-clause synthesis
measure peak VRAM with Qwen resident
measure Qwen tokens/sec and first-clause slowdown
listen to generated WAV comparisons
decide whether PerformanceCue should map to vocal style
```

- [ ] **Step 7: If verification found a regression, commit only the concrete fix**

For example, if sample-rate propagation broke a test:

```bash
git add output/voice.py tests/test_tts_pipeline.py
git commit -m "fix: preserve tts sample rate through playback"
```

If verification finds no regression, create no empty verification commit.

## Completion Checklist

```text
[ ] SynthesisResult validates 1D float32 + positive sample rate
[ ] Kokoro loads lazily from synthesis path
[ ] VoiceOutput constructs no model directly
[ ] engine sample rate reaches trimming, diagnostics, and playback
[ ] PerformanceCue reaches TTSEngine unchanged
[ ] acknowledgement cache preserves sample rate
[ ] one-clause-ahead behavior remains green
[ ] unknown engine names fail explicitly
[ ] Chatterbox adapter is optional and lazy
[ ] CI has no Chatterbox/CUDA/model requirement
[ ] benchmark harness writes comparable JSON + WAV outputs
[ ] Kokoro remains default
[ ] unit suite passes
[ ] behavior preview passes
[ ] eye validation passes
[ ] final branch CI passes on exact final SHA
[ ] target-PC Chatterbox acceptance remains pending
```
