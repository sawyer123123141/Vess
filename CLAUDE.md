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

## Hardware facts (measured — evidence, not a placement prescription)

Target machine: Ryzen 5800X, RTX 3070 8GB, 16GB DDR4, Windows.

Historical baseline:

- `qwen2.5:7b` via Ollama: 100% GPU, about 4.7GB VRAM, 4096 context, fast when
  warm.
- `faster-whisper` small int8 on CPU: about 2.3x realtime.
- Kokoro TTS on CPU: about 0.5s for a short sentence when warm.
- Vision model: near-instant in the earlier test path.

September 2, 2026 voice-runtime measurements supersede the old assumption that
Whisper must stay on CPU and that Kokoro is necessarily the final voice:

- Whisper small on CUDA with `int8_float16` and beam size 5 is accurate on the
  tested owner phrase. Beam size 1 caused severe transcription errors. Warm
  integrated transcriptions measured from about 134ms to 1078ms on short
  utterances; one cold integrated load took 31.6s.
- Chatterbox Turbo 0.1.7 works on the RTX 3070 with
  `torch/torchaudio 2.6.0+cu124`, `torchvision 0.21.0+cu124`, and NumPy 1.26.4.
  Standalone warm neutral synthesis measured roughly 0.5-0.95s across the
  standard short-to-skeptical corpus. A playful `[chuckle]` pass measured
  roughly 0.76-1.28s warm.
- `resemble-perth` currently relies on deprecated `pkg_resources`. On this
  machine, setuptools 84 caused `PerthImplicitWatermarker` to become `None`
  and Chatterbox failed at startup. `setuptools==80.9.0` is known-good; keep
  setuptools below 81 until Perth removes that dependency.
- In the full Vess process with Ollama/Qwen, CUDA Whisper, Chatterbox Turbo,
  Perth, and the Windows desktop resident, `nvidia-smi` showed
  **7854MiB / 8192MiB VRAM used** at only 3% GPU utilization. VRAM capacity,
  not raw compute saturation, is the current hardware constraint.
- The first integrated Chatterbox interaction exposed a cold-start queue stall:
  `tts_worker_wait_ms=52515.1`, first-clause synthesis 3935.5ms, and
  speech-end-to-playback 57661.6ms. The stale acknowledgement `"Yeah?"` held
  the single synthesis worker while Chatterbox cold-loaded, then was correctly
  discarded as stale.
- Without restarting, the next interaction had essentially no TTS worker wait
  (0.1ms). It measured speech-to-transcript 1553.2ms, LLM first clause
  1360.6ms, first playful TTS synthesis 2477.9ms, and
  speech-end-to-playback 5392.6ms. The second neutral clause synthesized in
  1665.9ms with a 0.2ms playback gap.

**No final model-placement policy is chosen yet.** Do not assume that moving
Whisper to CPU, shrinking Whisper, unloading/reloading models, changing the
LLM, or keeping every model resident is automatically correct. The next voice
performance planning pass should compare alternatives against measured
end-to-end latency, recognition quality, VRAM headroom, startup behavior, and
steady-state responsiveness. Avoid an architecture that fixes VRAM by adding
large reload stalls.

The main branch config may remain conservative while hardware experiments are
performed locally. Record experiment settings and results in `STATUS.md`
before turning them into defaults.

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
