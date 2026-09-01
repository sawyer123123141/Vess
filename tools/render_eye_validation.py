"""Render mobile-friendly visual evidence across every authored mood eye type."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.render_behavior_preview import (
    SimulationResult,
    check_invariants,
    simulate_scenario,
    verify_determinism,
)

_DEFAULT_OUTPUT = ROOT / "artifacts" / "behavior-verification"
_STATE_ORDER = (
    "listening",
    "thinking",
    "speaking_neutral",
    "speaking_playful",
    "speaking_emphatic",
    "speaking_uncertain",
)
_STATE_LABELS = {
    "listening": "LISTEN",
    "thinking": "THINK",
    "speaking_neutral": "NEUTRAL",
    "speaking_playful": "PLAYFUL",
    "speaking_emphatic": "EMPHATIC",
    "speaking_uncertain": "UNCERTAIN",
}


def run_eye_validation(
    *,
    output_dir: Path | None = None,
    fps: int = 30,
    seed: int = 1,
) -> tuple[int, str]:
    """Render the comprehensive mood/eye compatibility review bundle."""
    output = output_dir or _DEFAULT_OUTPUT
    try:
        output.mkdir(parents=True, exist_ok=True)
        result = simulate_scenario("mood_eye_validation", fps=fps, seed=seed)
        failures = check_invariants(result)
        deterministic = not verify_determinism(
            "mood_eye_validation",
            fps=fps,
            seed=seed,
        )

        _write_trace(result, output)
        _write_gif(result, output)
        _write_contact_sheet(result, output)
        summary = _build_summary(result, failures, deterministic)
        (output / "eye_validation_summary.txt").write_text(
            summary,
            encoding="utf-8",
        )
        return (1 if failures or not deterministic else 0), summary
    except Exception as error:
        summary = f"EYE VALIDATION ERROR\n\n{type(error).__name__}: {error}\n"
        try:
            output.mkdir(parents=True, exist_ok=True)
            (output / "eye_validation_summary.txt").write_text(
                summary,
                encoding="utf-8",
            )
        except OSError:
            pass
        return 2, summary


def _load_moods() -> dict[str, dict[str, object]]:
    return json.loads((ROOT / "moods.json").read_text(encoding="utf-8"))


def _write_trace(result: SimulationResult, output: Path) -> Path:
    path = output / "eye_validation_trace.json"
    payload = {
        "schema_version": 1,
        "scenario": result.scenario,
        "fps": result.fps,
        "seed": result.seed,
        "frames": result.trace,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_gif(
    result: SimulationResult,
    output: Path,
    *,
    sample_every: int = 3,
    scale: int = 6,
) -> Path:
    if not result.frames or len(result.frames) != len(result.trace):
        raise ValueError("eye validation requires matching frames and trace")

    moods = _load_moods()
    gif_frames: list[Image.Image] = []
    label_height = 46
    for index in range(0, len(result.frames), sample_every):
        native = result.frames[index]
        row = result.trace[index]
        mood = str(row["mood"])
        eye_type = str(moods.get(mood, {}).get("eye", "unknown"))
        phase = str(row["phase"]).split("__", 1)[-1]

        image = Image.fromarray(native)
        scaled = image.resize(
            (native.shape[1] * scale, native.shape[0] * scale),
            resample=Image.Resampling.NEAREST,
        )
        canvas = Image.new(
            "RGB",
            (scaled.width, scaled.height + label_height),
            (16, 17, 20),
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (6, 4),
            f"{mood} [{eye_type}] | {phase}\n"
            f"L {float(row['left_eye_offset_x']):+.2f},{float(row['left_eye_offset_y']):+.2f}  "
            f"R {float(row['right_eye_offset_x']):+.2f},{float(row['right_eye_offset_y']):+.2f}",
            fill=(235, 235, 240),
        )
        canvas.paste(scaled, (0, label_height))
        gif_frames.append(canvas)

    path = output / "eye_validation.gif"
    duration_ms = max(1, round(1000.0 * sample_every / result.fps))
    gif_frames[0].save(
        path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def _representative_indices(result: SimulationResult) -> dict[str, int]:
    by_phase: dict[str, list[dict[str, object]]] = {}
    for row in result.trace:
        by_phase.setdefault(str(row["phase"]), []).append(row)

    selected: dict[str, int] = {}
    for phase, rows in by_phase.items():
        candidates = rows[max(0, len(rows) // 2):]
        best = max(
            candidates,
            key=lambda row: (
                float(row["blink_openness"]),
                int(row["frame"]),
            ),
        )
        selected[phase] = int(best["frame"])
    return selected


def _write_contact_sheet(result: SimulationResult, output: Path) -> Path:
    moods = _load_moods()
    mood_order = [name for name in moods if any(row["mood"] == name for row in result.trace)]
    selected = _representative_indices(result)

    tile_width = 176
    tile_height = 158
    image_size = 128
    row_label_width = 104
    header_height = 30
    sheet = Image.new(
        "RGB",
        (
            row_label_width + tile_width * len(_STATE_ORDER),
            header_height + tile_height * len(mood_order),
        ),
        (12, 13, 16),
    )
    draw = ImageDraw.Draw(sheet)

    for column, state_name in enumerate(_STATE_ORDER):
        x = row_label_width + column * tile_width + 4
        draw.text((x, 9), _STATE_LABELS[state_name], fill=(230, 230, 235))

    for row_index, mood in enumerate(mood_order):
        eye_type = str(moods[mood].get("eye", "unknown"))
        y0 = header_height + row_index * tile_height
        draw.text((6, y0 + 56), f"{mood}\n[{eye_type}]", fill=(230, 230, 235))

        for column, state_name in enumerate(_STATE_ORDER):
            phase = f"{mood}__{state_name}"
            frame_index = selected[phase]
            trace_row = result.trace[frame_index]
            native = Image.fromarray(result.frames[frame_index]).resize(
                (image_size, image_size),
                resample=Image.Resampling.NEAREST,
            )
            x0 = row_label_width + column * tile_width
            sheet.paste(native, (x0 + 4, y0 + 4))
            draw.text(
                (x0 + 4, y0 + 134),
                f"L {float(trace_row['left_eye_offset_x']):+.2f},{float(trace_row['left_eye_offset_y']):+.2f}\n"
                f"R {float(trace_row['right_eye_offset_x']):+.2f},{float(trace_row['right_eye_offset_y']):+.2f}",
                fill=(205, 205, 212),
            )

    path = output / "eye_validation_contact_sheet.png"
    sheet.save(path)
    return path


def _phase_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "left_peak": max(
            math.hypot(
                float(row["left_eye_offset_x"]),
                float(row["left_eye_offset_y"]),
            )
            for row in rows
        ),
        "right_peak": max(
            math.hypot(
                float(row["right_eye_offset_x"]),
                float(row["right_eye_offset_y"]),
            )
            for row in rows
        ),
        "asymmetry": max(
            math.hypot(
                float(row["left_eye_offset_x"]) - float(row["right_eye_offset_x"]),
                float(row["left_eye_offset_y"]) - float(row["right_eye_offset_y"]),
            )
            for row in rows
        ),
    }


def _build_summary(
    result: SimulationResult,
    failures: list[object],
    deterministic: bool,
) -> str:
    moods = _load_moods()
    by_phase: dict[str, list[dict[str, object]]] = {}
    for row in result.trace:
        by_phase.setdefault(str(row["phase"]), []).append(row)

    lines = [
        "Vess eye visual validation",
        "",
        f"Hard invariants: {'PASS' if not failures else 'FAIL'}",
        f"Determinism: {'PASS' if deterministic else 'FAIL'}",
        f"Frames: {len(result.trace)}",
        "",
        "Review matrix",
    ]

    for mood, entry in moods.items():
        phases = [name for name in by_phase if name.startswith(f"{mood}__")]
        if not phases:
            continue
        eye_type = str(entry.get("eye", "unknown"))
        lines.extend(["", f"{mood} [{eye_type}]"])
        for state_name in _STATE_ORDER:
            phase = f"{mood}__{state_name}"
            rows = by_phase[phase]
            metrics = _phase_metrics(rows)
            lines.append(
                f"  {_STATE_LABELS[state_name]:9s} "
                f"L {metrics['left_peak']:.3f}px  "
                f"R {metrics['right_peak']:.3f}px  "
                f"asym {metrics['asymmetry']:.3f}px"
            )

    if failures:
        lines.extend(["", f"Hard failures: {len(failures)}"])
        for failure in failures[:10]:
            lines.append(f"  {failure}")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render comprehensive Vess eye validation")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    code, summary = run_eye_validation(
        output_dir=args.output,
        fps=30,
        seed=args.seed,
    )
    print(summary, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
