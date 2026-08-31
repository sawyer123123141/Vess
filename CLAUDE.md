# Working instructions

Read this and `PLAN.md` at the start of every session. `PLAN.md` is the
architecture and is authoritative — if something here conflicts with it, ask
rather than guessing.

## The one rule

**Build one step at a time.** `PLAN.md` has a numbered build order. Do the
step that's asked for and stop. Do not scaffold ahead, do not stub out future
modules, do not generate steps 2 through 8 because they're obvious.

A 200-line commit that runs beats a 2000-line commit that half-runs. The owner
of this repo wants to understand every line of it.

## Before writing code

- Re-read `PLAN.md`. Not the summary in your head — the file.
- Check `STATUS.md` for what's actually done.
- If the request is ambiguous, **ask**. One clarifying question costs a
  message. Guessing wrong costs an afternoon.
- If you find yourself about to invent a design decision that isn't in
  `PLAN.md` — a new field on `State`, a new module, a different threading
  model — stop and ask. Those decisions belong to the owner.

## When you're confused

In order:

1. Re-read `PLAN.md` and `STATUS.md`.
2. Read the actual source. Don't reason from what you remember writing.
3. Ask a specific question. "Should mood decay pause while speaking?" is
   useful. "How should I handle mood?" is not.

Never paper over confusion by writing something plausible. If two parts of the
plan seem to contradict each other, say so.

## Hardware facts (already tested — don't re-benchmark)

Target machine: Ryzen 5800X, RTX 3070 8GB, 16GB DDR4, Windows.

- `qwen2.5:7b` via Ollama: 100% GPU, 4.7GB VRAM, 4096 context, fast when warm
- `faster-whisper` small int8 on **CPU**: ~2.3x realtime
- Kokoro TTS on **CPU**: ~0.5s for a short sentence when warm
- Vision model: near-instant

VRAM is the hard constraint. ~8GB total, ~1GB to Windows. Do not raise context
above 4096 or load a second GPU model without asking — overflow drops
throughput ~30x with no graceful degradation.

Whisper and Kokoro stay on CPU. That's deliberate, not an oversight.

## Code style

- Python 3.11. Type hints on function signatures.
- Standard library where reasonable. Every dependency is a thing that breaks.
- Small files. If a module passes ~300 lines, it's probably two modules.
- Comments explain *why*, never *what*. No comment restating the line below it.
- No clever one-liners. This gets read at 1am six months from now.

## Threading

Anything that can block runs in a thread: LLM calls, Whisper, the detector,
web requests, TTS generation. The main loop renders the face at ~30fps and
must **never** block. A frozen face during a slow response is the single most
visible failure mode in this project.

All shared mutation goes through `State` behind its lock. No side-channel
globals.

## Things that are deliberately not in scope

Do not add these. They were considered and cut:

- Pose estimation
- Adaptive motion-baseline detection (the detector's object list is the
  motion signal)
- Hosted/cloud model routing (local only for now)
- Any command that composes shell input, or lets the model write code, edit
  config files, or modify its own appearance beyond the fixed registry

The command registry is a **closed list**. The model selects from it; it never
constructs a command. If a new capability is wanted, a human adds it to the
registry.

## After each step

Update `STATUS.md`: what now works, what was learned, anything that surprised
you. That file is how the next session knows where it is.
