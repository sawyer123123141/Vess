"""Construct cheap TTS engine adapters from Vess configuration."""

from __future__ import annotations

from typing import Any

from output.tts.base import TTSEngine


def create_tts_engine(config: dict[str, Any]) -> TTSEngine:
    """Return the configured engine adapter without loading its heavy model."""
    name = str(config.get("voice", {}).get("engine", "kokoro")).strip().lower()
    if name == "kokoro":
        from output.tts.kokoro import KokoroEngine

        return KokoroEngine(config)
    raise ValueError(f"unknown TTS engine: {name!r}")
