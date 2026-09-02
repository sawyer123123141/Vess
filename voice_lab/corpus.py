"""Voice Lab corpus manifest and deterministic WAV loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True)
class CorpusItem:
    id: str
    audio_path: Path
    transcript: str
    expected_utterances: int = 1
    source: str = "local"
    tags: tuple[str, ...] = ()
    expression: str = "neutral"
    intensity: float = 0.0


def load_manifest(path: Path) -> list[CorpusItem]:
    """Load one corpus manifest with audio paths resolved beside the manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("voice corpus manifest requires an items list")

    seen: set[str] = set()
    items: list[CorpusItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("voice corpus items must be objects")
        item_id = str(raw.get("id", "")).strip()
        if not item_id:
            raise ValueError("voice corpus item requires id")
        if item_id in seen:
            raise ValueError(f"duplicate corpus id: {item_id}")
        seen.add(item_id)

        audio_value = str(raw.get("audio", "")).strip()
        if not audio_value:
            raise ValueError(f"corpus item {item_id} requires audio")
        transcript = str(raw.get("transcript", ""))
        expected = int(raw.get("expected_utterances", 1))
        if expected < 1:
            raise ValueError(f"corpus item {item_id} expected_utterances must be >= 1")
        intensity = float(raw.get("intensity", 0.0))
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"corpus item {item_id} intensity must be between 0 and 1")
        tags_value = raw.get("tags", [])
        if not isinstance(tags_value, list):
            raise ValueError(f"corpus item {item_id} tags must be a list")

        items.append(
            CorpusItem(
                id=item_id,
                audio_path=(manifest_path.parent / audio_value).resolve(),
                transcript=transcript,
                expected_utterances=expected,
                source=str(raw.get("source", "local")).strip() or "local",
                tags=tuple(str(tag) for tag in tags_value),
                expression=str(raw.get("expression", "neutral")).strip() or "neutral",
                intensity=intensity,
            )
        )
    return items


def read_wav(item: CorpusItem, sample_rate: int = 16_000) -> np.ndarray:
    """Decode strict mono PCM16 WAV into the float32 format used by Vess."""
    try:
        with wave.open(str(item.audio_path), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except FileNotFoundError as error:
        raise ValueError(f"corpus audio not found: {item.audio_path}") from error
    except wave.Error as error:
        raise ValueError(f"invalid WAV file {item.audio_path}: {error}") from error

    if channels != 1:
        raise ValueError(f"voice corpus WAV must be mono, got {channels} channels")
    if rate != sample_rate:
        raise ValueError(f"voice corpus WAV must be {sample_rate} Hz, got {rate} Hz")
    if width != 2:
        raise ValueError(f"voice corpus WAV must be 16-bit PCM, got {width * 8}-bit")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    return np.ascontiguousarray(samples / 32768.0, dtype=np.float32)
