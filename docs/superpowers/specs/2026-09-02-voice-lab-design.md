# Voice Lab Design

## Goal

Build a small, repeatable local benchmark harness for Vess voice work so microphone, endpointing, Whisper, TTS, and cancellation changes can be compared against identical inputs instead of relying on hand-timed live conversations.

## Scope

Voice Lab v1 is a developer tool. It does not change live Vess behavior, `State`, rendering, conversation policy, or repository config defaults.

It supports four experiment types:

1. **Endpoint replay** — feed recorded or synthetic 16 kHz mono WAV speech through the production `UtteranceAssembler` with a sweep of `silence_seconds` values and report over/under segmentation.
2. **Whisper replay** — transcribe the exact same corpus through the production faster-whisper construction path and report latency, realtime factor, transcript, and word error rate.
3. **TTS benchmark** — reuse the existing TTS measurement path for fixed/corpus text and preserve generated WAV artifacts for listening comparisons.
4. **Cancellation benchmark** — run a real cancellable TTS engine, invalidate it at deterministic offsets, and measure how long the synthesis worker remains occupied after cancellation.

## Corpus

A corpus is a JSON manifest plus local WAV files. WAV files may come from the owner microphone, synthetic generators, or externally sourced public datasets; the benchmark runner does not download network data.

Manifest entries contain:

- `id`: unique stable identifier
- `audio`: path relative to the manifest
- `transcript`: reference text
- `expected_utterances`: expected segmentation count, default 1
- `source`: descriptive provenance such as `owner`, `common_voice`, `librispeech`, or `synthetic`
- `tags`: optional labels such as `hesitation`, `quiet`, `fast`, `wake`, `short`
- `expression`: optional performance label for later expressive-TTS comparisons, default `neutral`
- `intensity`: optional 0..1 performance intensity, default 0

V1 deliberately accepts only mono 16 kHz PCM WAV input. It fails clearly on unsupported channels, rate, or sample widths rather than hiding resampling/conversion inside the benchmark. Public clips can be converted once before entering the corpus.

## Architecture

`voice_lab/` contains reusable benchmark logic. `tools/voice_lab.py` is the CLI and owns argument parsing plus artifact locations. Production classes/functions are reused directly wherever possible.

- `voice_lab/corpus.py` — manifest model/validation and deterministic WAV loading
- `voice_lab/endpointing.py` — offline replay through `UtteranceAssembler`
- `voice_lab/whisper.py` — WER and timed transcription measurements
- `voice_lab/tts.py` — cancellable-engine timing helpers
- `tools/voice_lab.py` — subcommands and JSON/console output

The existing `tools/benchmark_tts.py` remains the low-level TTS measurement implementation; Voice Lab calls/reuses it rather than creating a second synthesis measurement format.

## CLI

Examples:

```powershell
python tools/voice_lab.py endpoint --manifest voice_corpus/manifest.json --silence 0.30 0.35 0.40 0.45
python tools/voice_lab.py whisper --manifest voice_corpus/manifest.json --beam-size 1 3 5
python tools/voice_lab.py tts --engine chatterbox_turbo --runs 3
python tools/voice_lab.py cancel --engine chatterbox_turbo --text "This is a deliberately longer synthesis request." --after-ms 50 100 250 500
```

Every command writes machine-readable JSON under `artifacts/voice-lab/<experiment>/` and prints a compact summary.

## Endpoint semantics

Replay feeds fixed-size blocks through the same `UtteranceAssembler` used by live Vess and appends enough final silence to flush the final utterance. For each corpus item and silence setting it reports:

- emitted utterance count
- expected utterance count
- `premature_split` when emitted > expected
- `missed_split` when emitted < expected
- configured endpoint wait in milliseconds

The primary endpointing decision metric is segmentation correctness. The configured silence threshold itself is not presented as a magically measured acoustic latency.

## Whisper semantics

A Whisper variant loads once, then processes every corpus item. Reports include:

- transcript
- synthesis-independent transcription latency
- utterance duration
- realtime factor
- word error rate against the manifest reference

Corpus summaries keep owner speech and public speech distinguishable by `source`; they must not be collapsed into one score when making Vess-specific decisions.

## TTS and expression readiness

TTS rows preserve the performance cue (`expression`, `intensity`) alongside text. V1 does not alter Chatterbox expression behavior. This metadata exists so the next expressive-voice pass can render A/B variants from the exact same prompts without changing the corpus format.

## Cancellation semantics

Cancellation benchmarking targets engines exposing `synthesize_cancellable`. It starts one synthesis, requests cancellation after a deterministic delay, then reports `release_ms_after_cancel`: elapsed time from cancellation request until the synthesis call returns/raises and the worker is available again.

This is the hardware quantity corresponding to stale TTS head-of-line blocking. It does not claim CUDA kernels can be interrupted mid-kernel.

## Testing

Unit tests use synthetic numpy audio and fake transcribers/TTS engines. CI does not load Whisper or Chatterbox. Tests prove corpus validation, endpoint segmentation sweeps, WER/timing calculations, and cancellation timing semantics.

Real model execution remains an explicit local command on the RTX 3070 machine.

## Non-goals

- no web dashboard
- no automatic downloading of Common Voice/LibriSpeech
- no hidden resampling or format conversion
- no subjective "naturalness score"
- no automatic tuning that edits `config.json`
- no second GPU model or parallel Chatterbox instances

Human listening remains authoritative for prosody and naturalness; the lab narrows candidates and produces identical WAVs for comparison.
