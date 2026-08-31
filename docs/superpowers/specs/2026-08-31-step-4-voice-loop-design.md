# Step 4 — voice loop design

## Goal

Turn a spoken request into a streamed local response without blocking the
30fps face render loop:

```
microphone -> VAD -> Whisper -> fuzzy "Hey Vess" gate -> Ollama -> Kokoro
```

The browser preview and cv2 target continue receiving the same frames while
audio, LLM, TTS, and database work happen in their own threads.

## Components and flow

### Audio input

`perception/audio.py` opens a `sounddevice` input stream. Its callback only
copies 16 kHz mono samples into a bounded queue. A worker uses a configurable
energy threshold and trailing-silence duration to assemble an utterance, then
runs CPU `faster-whisper` on that completed utterance. While Vess is speaking,
input is discarded to avoid transcribing its own output; interruption handling
is not part of this step.

`state.listening` is true from detected speech through transcription. Failed
audio capture or transcription produces a log record and returns the worker
to listening; it never stops the render loop.

### Wake gate

The first one, two, and three normalised transcript words are compared to the
configured normalised wake variants using Levenshtein edit distance. The
initial variants are `hey vess`, `hey best`, `hey guess`, and `heaviest`.
The smallest configured distance wins. A match removes the matched prefix;
an empty remainder receives a short acknowledgement rather than an LLM call.
Kokoro prepares that acknowledgement once on startup in its worker, so it can
play immediately rather than waiting for a first synthesis.

Every rejected transcript logs its raw text, tested prefix, closest variant,
and distance. Accepted requests log the same match metadata. Continuous
Whisper work and its roughly one-second wake latency are accepted limitations
of this step; a proper local wake-word engine is a later separate step.

### Response and speech

`brain/llm.py` calls local Ollama over HTTP with the configured model, 4096
context, 80-token limit, and keep-alive setting. Its stable system prompt is
sent first to preserve the KV cache; dynamic persona, mood, perception fields,
and user request follow it. The response stream is split only at completed
clause boundaries; completed clauses enter the TTS queue immediately.

`output/voice.py` has one worker that synthesizes each clause with Kokoro and
plays it in order. It owns `state.speaking`; `state.thinking` is true while a
response is awaited and false once the first clause is queued. Failures are
logged and do not terminate later requests.

When the response ends, a small local classification request chooses a mood
only from `moods.json`. Invalid output leaves mood unchanged. A valid change
sets `mood_until` from that mood's configured decay and logs the transition.

## State and expiry

`State.expire_mood(now)` is called by the main loop every tick. Once expired,
it atomically sets `mood` to `neutral` and `mood_until` to `0.0`, then returns
the old mood so the caller can log the transition. No consumer independently
interprets an expired mood.

## Minimal event log

The early `brain/memory.py` exception supplies only append-only history for
data that cannot be reconstructed. SQLite contains one table:

```
events(timestamp REAL, event_type TEXT, payload_json TEXT)
```

`EventLog.append()` enqueues writes to one background SQLite worker. It has no
query, fact, retrieval, or summarisation API; those belong to Step 5.

Step 4 records session start, accepted/rejected wake utterances, explicit and
expiry mood changes, and web colour override/reset. Future persona changes
will use the same append API in Step 6.

## Configuration

An `audio` block adds input device selection, 16 kHz mono capture, VAD
threshold (`0.015`), minimum utterance (`0.25s`), trailing silence (`0.8s`),
maximum utterance (`15s`), wake variants, and max edit distance (`2`). These
are deliberately config values because the actual USB mic will need tuning.
Existing LLM and voice config values remain the single model/voice source of
truth. If the configured audio device cannot open, Vess logs the failure and
continues rendering without the voice loop.

## Explicit non-goals

- No vision-model call or frame handoff. The current camera design has no
  safe shared latest-frame path, and adding another GPU model conflicts with
  the stated VRAM rule. This remains a known limitation.
- No dedicated wake-word engine, command execution, memory retrieval, facts,
  summarisation, or database querying.
- No changes to the 30fps render path beyond calling mood expiry.

## Verification

Tests cover VAD utterance boundaries, fuzzy matching and rejection logging,
state transitions, append-only SQLite writes, prompt/clause parsing, and
worker error recovery. A manual run verifies the real USB mic, Whisper,
Ollama, Kokoro, and visible listening/thinking/speaking face states.
