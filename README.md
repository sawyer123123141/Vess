# Vess

An ambient AI that lives on a wall. A 64x64 LED face that watches the room,
listens, remembers, and occasionally speaks first. Runs entirely locally.

Named for nothing in particular. It's a presence, not an acronym.

## Status

Pre-alpha. See `STATUS.md`.

## Design

See `PLAN.md` for architecture, and `CLAUDE.md` for working conventions.

The short version: a YOLO detector and a mic run continuously and write into a
single `State` object. A 30fps loop renders a face from that state and never
blocks. When spoken to, it captures a frame, transcribes, asks a local model,
and speaks back — streaming audio at clause boundaries so it starts talking
before it's finished thinking.

The model never composes commands. It selects from a closed registry.

## Stack

Ollama (qwen2.5:7b) · faster-whisper · Kokoro TTS · YOLO · Pimoroni
Interstate 75 W driving a 64x64 HUB75 panel.

## Running

```
pip install -r requirements.txt
python main.py
```

Opens a config UI at `http://localhost:8080` with a live face preview. The LED
panel is optional — the preview is the same render path.
