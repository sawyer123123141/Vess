# Status

Update this at the end of every session. Newest at the top.

## Not started

Nothing built yet. Next: **step 1** from `PLAN.md` — state, face renderer,
animator, preview display. Eyes that blink in a browser window.

## Verified working (environment, not code)

- Ollama + `qwen2.5:7b` — 100% GPU, 4.7GB, 4096 ctx, fast when warm
- Vision model — near-instant
- `faster-whisper` small int8 on CPU — 4.8s of audio in 2.1s
- Kokoro TTS on CPU — 0.75s cold, ~0.5s warm, `af_heart` voice

## Open questions

- Panel mounting spot not decided (right of desk vs above). Doesn't block
  anything — camera position is independent.
- Voice choice not final. Kokoro ships several; currently `af_heart`.
