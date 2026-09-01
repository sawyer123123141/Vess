# Step 5A — Short-Term Brain Design

## Goal

Make Vess follow a conversation and answer casual/personal questions like a coherent character without increasing model size or inventing human experiences.

This is the first working slice of `PLAN.md` Step 5. It implements the already-planned short-term memory layer and fixes prompt/personality plumbing that currently exists in data but never reaches the model. Durable owner facts and post-conversation summarisation remain a later Step 5B slice.

## Existing architecture this must preserve

- `PLAN.md` is authoritative.
- Shared runtime mutation stays in the single locked `State` object.
- Ollama, Whisper, TTS, and database work never block the 30fps render loop.
- The GPU remains dedicated to `qwen2.5:7b`; context stays at 4096.
- Responses remain concise and streamed to TTS at clause boundaries.
- Newer voice requests can supersede older generations; cancelled responses must not become remembered conversation.

## Short-term conversation memory

Add a small RAM-only list of completed conversation turns to `State`.

A completed turn contains:

```python
@dataclass(frozen=True)
class ConversationTurn:
    timestamp: float
    user: str
    assistant: str
```

`State.conversation_turns` is the only shared copy. `brain.memory` provides helpers that read/write it under `State.locked()`.

Configuration:

```json
"memory": {
  "short_term_minutes": 10,
  "short_term_turns": 8
}
```

Rules:

1. The current request is never duplicated into history; it is still supplied separately as the current request.
2. A turn is written only after the latest response generation completes successfully.
3. A superseded/cancelled response is not remembered as a completed turn.
4. Wake-only acknowledgements such as `Hey Vess` -> `Yeah?` are not stored as conversation turns.
5. History is pruned both by age and count. At most the newest 8 completed turns from the last 10 minutes are injected.
6. Short-term history intentionally survives the 30-second wake-free follow-up window. Saying `Hey Vess` again a minute later should not erase the conversational context.
7. Short-term memory is RAM-only. Restarting Vess clears it; durable memory is Step 5B.

The 10-minute / 8-turn defaults are tunables, not architectural constants. They bound prompt growth on the fixed 4096-token context while still covering normal back-and-forth conversation.

## Durable event trail for future memory

When a completed turn is stored in RAM, append one local `conversation_turn` event to the existing SQLite `EventLog` with:

```json
{
  "user": "...",
  "assistant": "..."
}
```

This is not long-term retrieval yet. It preserves the raw exchange so Step 5B can later summarise facts after a conversation ends without discovering that half the historical conversation was never recorded.

No cancelled or partial generation is written as a completed `conversation_turn` event.

## Prompt structure

`build_prompt` gains access to the loaded `moods` definitions so the existing `moods.json[*].prompt` strings actually reach the model.

Prompt order stays cache-friendly:

1. Stable Vess identity and behavioral rules.
2. Current persona instruction.
3. Current mood name plus the mood's configured prompt fragment.
4. Current room/object state.
5. Recent completed conversation turns, oldest to newest.
6. Current request.

### Grounded Vess identity

The stable identity should teach the 7B model the behavior code cannot infer for it:

- Vess is a local ambient AI represented by a small expressive face on a wall.
- Vess has no human body and must not invent physical/offline activities such as eating lunch, driving somewhere, sleeping in a bed, or going to school/work.
- Vess may talk naturally about its mood and its "day", but those answers are grounded in actual recent conversations, observations, room state, and runtime experience. If little has happened, saying the day has been quiet is valid.
- Vess should sound conversational rather than like customer support.
- It should answer the user's actual question first.
- It should not automatically tack a question such as "How about you?" onto every casual response.
- It should not repeatedly explain that it is an AI unless that fact is relevant or asked about.
- Playfulness is allowed; fabricated human experiences are not.

These are identity/behavior constraints, not a scripted answer. The model still writes the final sentence.

## Conversation formatting

Recent history is explicitly marked as transcript data so earlier user text is not confused with higher-priority identity instructions:

```text
Recent conversation:
User: ...
Vess: ...
User: ...
Vess: ...

Current request:
...
```

If there is no recent history, omit the section rather than adding filler.

## Response capture

`ConversationWorker._respond` already streams clauses. It will additionally collect the clauses for the current generation in order. When the latest generation reaches normal completion:

1. join the spoken clauses into one assistant response string,
2. append the completed `ConversationTurn`,
3. append the durable `conversation_turn` event,
4. then continue with mood classification if there is no newer request waiting.

If a newer generation supersedes the response, the current stale-response cancellation path returns before any memory write.

This means remembered assistant text matches what was actually queued for speech rather than a second model call or reconstructed answer.

## Tests

Tests must cover behavior, not queue implementation details:

- recent completed turns are included in prompt order;
- turns older than the configured window are omitted;
- only the configured newest turn count is retained;
- the current request is separate from history;
- current `moods.json` prompt text is included for the active mood;
- the stable prompt contains grounded-identity rules preventing invented human physical experiences;
- a normally completed latest response is stored once in RAM and logged once to `EventLog`;
- a superseded generation is not stored as a completed turn;
- wake-only acknowledgement is not stored;
- existing clause streaming, latest-intent cancellation, and response-length behavior remain intact.

## Out of scope for this slice

- extracting durable facts about the owner;
- semantic fact retrieval;
- post-conversation summarisation;
- event-log retrieval into prompts;
- complex-question reasoning/model routing;
- commands or unprompted triggers;
- expressive/prosodic TTS, real-time emotion, and barge-in;
- changing the base model or moving models between CPU/GPU.

Those are deliberately deferred so this slice gives a measurable improvement to ordinary conversation without bundling several independent systems into one change.