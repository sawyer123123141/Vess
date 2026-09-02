"""Benchmark Vess TTS engines without microphones, playback, or the LLM."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from output.tts.base import SynthesisResult, TTSEngine
from performance import PerformanceCue


ROOT = Path(__file__).resolve().parents[1]
STANDARD_TEXTS: tuple[tuple[str, str], ...] = (
    ("short", "Sure."),
    ("medium", "That's actually pretty interesting."),
    ("skeptical", "I don't think that's going to work, though."),
    ("question", "Why would that happen?"),
    ("emphatic", "That's ridiculous."),
)


def measure_one(
    engine_name: str,
    engine: TTSEngine,
    text_id: str,
    text: str,
    run_index: int,
    now: Callable[[], float] = time.perf_counter,
    performance: PerformanceCue | None = None,
) -> tuple[dict[str, object], SynthesisResult | None]:
    """Measure one synchronous engine synthesis call."""
    cue = performance or PerformanceCue()
    started = now()
    try:
        result = engine.synthesize(text, cue)
    except Exception as error:
        elapsed = now() - started
        return (
            {
                "engine": engine_name,
                "text_id": text_id,
                "text": text,
                "run_index": run_index,
                "warm": run_index > 0,
                "expression": cue.expression,
                "intensity": cue.intensity,
                "synthesis_ms": round(elapsed * 1000.0, 3),
                "audio_duration_ms": None,
                "realtime_factor": None,
                "sample_rate": None,
                "sample_count": None,
                "success": False,
                "error": str(error),
            },
            None,
        )

    elapsed = now() - started
    audio_seconds = result.audio.size / result.sample_rate
    realtime_factor = elapsed / audio_seconds if audio_seconds > 0 else None
    return (
        {
            "engine": engine_name,
            "text_id": text_id,
            "text": text,
            "run_index": run_index,
            "warm": run_index > 0,
            "expression": cue.expression,
            "intensity": cue.intensity,
            "synthesis_ms": round(elapsed * 1000.0, 3),
            "audio_duration_ms": round(audio_seconds * 1000.0, 3),
            "realtime_factor": (
                round(realtime_factor, 6) if realtime_factor is not None else None
            ),
            "sample_rate": result.sample_rate,
            "sample_count": int(result.audio.size),
            "success": True,
            "error": None,
        },
        result,
    )


def write_results(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write machine-readable benchmark rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")


def write_wave(path: Path, result: SynthesisResult) -> None:
    """Write one benchmark waveform without importing soundfile in CI imports."""
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, result.audio, result.sample_rate)


def _load_config() -> dict[str, Any]:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def _print_summary(rows: Sequence[dict[str, object]]) -> None:
    print("\nWarm-run summary")
    print("text        synth ms    RTF")
    print("----------  ----------  --------")
    for text_id, _ in STANDARD_TEXTS:
        matching = [
            row
            for row in rows
            if row["text_id"] == text_id and row["warm"] and row["success"]
        ]
        synth = [float(row["synthesis_ms"]) for row in matching]
        rtf = [
            float(row["realtime_factor"])
            for row in matching
            if row["realtime_factor"] is not None
        ]
        synth_label = f"{statistics.median(synth):.1f}" if synth else "n/a"
        rtf_label = f"{statistics.median(rtf):.3f}" if rtf else "n/a"
        print(f"{text_id:<10}  {synth_label:>10}  {rtf_label:>8}")


def run_benchmark(
    engine_name: str,
    runs: int,
    output_dir: Path | None = None,
    *,
    performance: PerformanceCue | None = None,
) -> int:
    """Run the standard local benchmark and persist comparable artifacts."""
    from output.tts.factory import create_tts_engine

    config = _load_config()
    config.setdefault("voice", {})["engine"] = engine_name
    engine = create_tts_engine(config)
    destination = output_dir or ROOT / "artifacts" / "tts-benchmark" / engine_name
    cue = performance or PerformanceCue()
    rows: list[dict[str, object]] = []

    for text_id, text in STANDARD_TEXTS:
        for run_index in range(runs):
            row, result = measure_one(
                engine_name,
                engine,
                text_id,
                text,
                run_index,
                performance=cue,
            )
            rows.append(row)
            status = "ok" if row["success"] else f"failed: {row['error']}"
            print(
                f"{text_id} run {run_index + 1}/{runs}: "
                f"{row['synthesis_ms']} ms ({status})"
            )
            if result is not None:
                write_wave(destination / f"{text_id}-run{run_index}.wav", result)

    write_results(destination / "results.json", rows)
    _print_summary(rows)
    return 0 if all(bool(row["success"]) for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="kokoro")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return run_benchmark(args.engine, args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
