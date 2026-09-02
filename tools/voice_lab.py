"""Unified repeatable voice benchmarks for Vess."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from performance import PerformanceCue
from tools.benchmark_tts import run_benchmark, write_results
from voice_lab.corpus import load_manifest, read_wav
from voice_lab.endpointing import replay_endpoint
from voice_lab.tts import measure_cancellation
from voice_lab.whisper import measure_transcription


def artifact_path(root: Path, experiment: str, *parts: str) -> Path:
    """Return one path under the Voice Lab artifact namespace."""
    return Path(root) / "voice-lab" / experiment / Path(*parts)


def with_whisper_beam(config: dict[str, Any], beam_size: int) -> dict[str, Any]:
    """Return an isolated Whisper config variant without mutating live config."""
    variant = deepcopy(config)
    variant.setdefault("whisper", {})["beam_size"] = int(beam_size)
    return variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    endpoint = subparsers.add_parser("endpoint", help="replay endpoint thresholds")
    endpoint.add_argument("--manifest", required=True)
    endpoint.add_argument("--silence", type=float, nargs="+", required=True)

    whisper = subparsers.add_parser("whisper", help="compare Whisper beam sizes")
    whisper.add_argument("--manifest", required=True)
    whisper.add_argument("--beam-size", type=int, nargs="+", required=True)

    tts = subparsers.add_parser("tts", help="run the standard TTS benchmark")
    tts.add_argument("--engine", default="chatterbox_turbo")
    tts.add_argument("--runs", type=int, default=3)
    tts.add_argument("--expression", default="neutral")
    tts.add_argument("--intensity", type=float, default=0.0)

    cancel = subparsers.add_parser("cancel", help="measure stale TTS release latency")
    cancel.add_argument("--engine", default="chatterbox_turbo")
    cancel.add_argument("--text", required=True)
    cancel.add_argument("--after-ms", type=float, nargs="+", required=True)
    cancel.add_argument("--expression", default="neutral")
    cancel.add_argument("--intensity", type=float, default=0.0)
    cancel.add_argument("--warmup-text", default="Ready.")
    return parser


def _load_config() -> dict[str, Any]:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def _run_endpoint(args: argparse.Namespace) -> int:
    config = _load_config()
    settings = dict(config.get("audio", {}))
    items = load_manifest(Path(args.manifest))
    loaded = [(item, read_wav(item)) for item in items]
    rows: list[dict[str, object]] = []
    for silence in args.silence:
        for item, samples in loaded:
            row = replay_endpoint(
                samples,
                settings,
                float(silence),
                item.expected_utterances,
            )
            row.update(
                {
                    "id": item.id,
                    "source": item.source,
                    "tags": list(item.tags),
                }
            )
            rows.append(row)

    destination = artifact_path(ROOT / "artifacts", "endpoint", "results.json")
    write_results(destination, rows)
    for silence in args.silence:
        matching = [row for row in rows if row["silence_seconds"] == round(float(silence), 4)]
        failures = sum(bool(row["premature_split"] or row["missed_split"]) for row in matching)
        print(f"{silence:.3f}s: {len(matching) - failures}/{len(matching)} correct")
    print(f"results: {destination}")
    return 0


def _run_whisper(args: argparse.Namespace) -> int:
    from perception.audio import _make_transcriber

    config = _load_config()
    items = load_manifest(Path(args.manifest))
    loaded = [(item, read_wav(item)) for item in items]
    rows: list[dict[str, object]] = []
    for beam_size in args.beam_size:
        if beam_size < 1:
            raise ValueError("beam size must be at least 1")
        variant = with_whisper_beam(config, beam_size)
        transcribe = _make_transcriber(variant)
        for item, samples in loaded:
            row = measure_transcription(samples, item.transcript, transcribe)
            row.update(
                {
                    "id": item.id,
                    "source": item.source,
                    "tags": list(item.tags),
                    "beam_size": int(beam_size),
                }
            )
            rows.append(row)

    destination = artifact_path(ROOT / "artifacts", "whisper", "results.json")
    write_results(destination, rows)
    for beam_size in args.beam_size:
        matching = [row for row in rows if row["beam_size"] == beam_size]
        mean_wer = sum(float(row["word_error_rate"]) for row in matching) / len(matching)
        mean_ms = sum(float(row["transcription_ms"]) for row in matching) / len(matching)
        print(f"beam {beam_size}: WER {mean_wer:.3f}, mean {mean_ms:.1f} ms")
    print(f"results: {destination}")
    return 0


def _run_tts(args: argparse.Namespace) -> int:
    if args.runs < 1:
        raise ValueError("runs must be at least 1")
    if not 0.0 <= args.intensity <= 1.0:
        raise ValueError("intensity must be between 0 and 1")
    destination = artifact_path(ROOT / "artifacts", "tts", args.engine)
    cue = PerformanceCue(args.expression, float(args.intensity))
    return run_benchmark(
        args.engine,
        args.runs,
        destination,
        performance=cue,
    )


def _run_cancel(args: argparse.Namespace) -> int:
    from output.tts.factory import create_tts_engine

    if not 0.0 <= args.intensity <= 1.0:
        raise ValueError("intensity must be between 0 and 1")
    config = deepcopy(_load_config())
    config.setdefault("voice", {})["engine"] = args.engine
    engine = create_tts_engine(config)
    cue = PerformanceCue(args.expression, float(args.intensity))

    # Model load/cold synthesis must not contaminate stale-worker release timing.
    engine.synthesize(args.warmup_text, PerformanceCue())
    rows = [
        {
            "engine": args.engine,
            "text": args.text,
            "expression": cue.expression,
            "intensity": cue.intensity,
            **measure_cancellation(engine, args.text, cue, float(offset)),
        }
        for offset in args.after_ms
    ]
    destination = artifact_path(ROOT / "artifacts", "cancel", args.engine, "results.json")
    write_results(destination, rows)
    for row in rows:
        release = row["release_ms_after_cancel"]
        release_label = "n/a" if release is None else f"{float(release):.1f} ms"
        print(f"cancel at {float(row['cancel_after_ms']):.1f} ms: {row['status']} -> {release_label}")
    print(f"results: {destination}")
    return 0 if all(row["status"] in {"cancelled", "completed_before_cancel"} for row in rows) else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "endpoint":
        return _run_endpoint(args)
    if args.command == "whisper":
        return _run_whisper(args)
    if args.command == "tts":
        return _run_tts(args)
    if args.command == "cancel":
        return _run_cancel(args)
    raise RuntimeError(f"unknown Voice Lab command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
