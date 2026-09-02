# Durable Memory Design

## Goal

Add the durable-facts layer described in `PLAN.md` without adding prompt-time model latency or blocking playback/rendering. Vess should retain a small set of explicit, useful facts the user has stated, update them when the user corrects them, and inject only relevant facts into later prompts.

## Scope

This pass adds durable facts only. It does not add commands, triggers, embeddings, a memory UI, cloud storage, autonomous profile scraping, or a second general-purpose memory model.

Short-term conversation memory and the append-only event log remain unchanged.

## Storage

Durable facts live in the existing `vess.db` SQLite database in a new `facts` table:

```sql
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    source_text TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
```

A fact key is a stable snake_case topic such as `favorite_color`, `dog_name`, or `current_project`. Upserting the same key replaces its value/source and refreshes `updated_at` while preserving `created_at`.

`source_text` keeps the exact user utterance that produced the current value. It is for audit/debugging and is never automatically spoken.

## Fact eligibility

Automatic extraction may store only durable facts explicitly stated by the user and likely to be useful later, for example names, preferences, recurring activities, relationships, or ongoing projects.

The extractor must not infer facts from Vess's own response. It receives the user utterance as the source of truth. It must return no fact for temporary moods, one-off tasks, guesses, passwords/tokens/credentials, or highly sensitive personal information.

The extractor returns at most three candidates per delivered user utterance as JSON objects:

```json
[{"key":"favorite_color","value":"navy blue"}]
```

Keys and values are validated again in deterministic Python before storage. Invalid keys, empty values, oversized values, or malformed model output are discarded.

## Asynchronous extraction

`brain/memory.py` gains `FactMemory`, a small background owner patterned after `EventLog`.

`ConversationWorker._finalize_delivered_turn()` only enqueues the user utterance. It never calls Ollama or writes durable facts synchronously. The FactMemory worker:

1. reads the existing fact keys,
2. asks the local Ollama client for explicit durable facts from the user utterance,
3. validates candidates,
4. upserts them into SQLite.

This keeps playback callbacks and the render loop free from memory-extraction latency.

Interrupted Vess responses do not invalidate the user's already-delivered utterance, so both completed and interrupted turns may queue fact extraction. A blank user request never does.

## Extraction model contract

`OllamaClient.extract_facts(transcript, known_keys, config)` performs one non-streaming local generation on the FactMemory worker thread. The prompt instructs the model to:

- use an existing key when the utterance updates that topic,
- create a concise stable snake_case key only when necessary,
- extract only information explicitly stated by the user,
- return JSON only,
- return `[]` when nothing durable is present,
- produce at most three facts.

Malformed JSON or a response with the wrong shape becomes an empty candidate list rather than breaking conversation handling.

## Retrieval

Prompt-time retrieval does not call an LLM and does not use embeddings.

`FactMemory.relevant_facts(query, limit=5)` reads stored facts on the conversation worker thread and ranks them using normalized lexical overlap between the current request and each fact's key/value. Common filler words are ignored; `updated_at` breaks ties.

A broad memory request containing phrases such as `remember about me` or `know about me` may fall back to the most recently updated facts. Ordinary requests with zero lexical overlap inject no durable facts rather than adding unrelated profile noise.

The prompt section is bounded to five facts and appears after current state but before recent conversation:

```text
Relevant durable memory:
- favorite_color: navy blue
- dog_name: Rex
```

The static Vess identity remains the first prompt section so existing KV-cache behavior is preserved.

## Failure behavior

Durable memory is supplementary. A database read failure, extraction failure, or malformed model response must not fail the user's conversation. Retrieval failure produces no durable-memory section; extraction failure drops only that extraction job.

`FactMemory.close()` drains queued jobs before shutdown, matching the existing EventLog shutdown philosophy.

## Wiring

`main.py` creates one `OllamaClient` and one `FactMemory` using `vess.db`. The same client is used for conversation, mood classification, and background fact extraction, but calls remain serialized by their owning worker paths rather than adding another GPU model.

`ConversationWorker` receives the optional FactMemory dependency. Tests may omit it, preserving existing construction paths.

## Verification

Tests must prove:

- facts persist across FactMemory instances,
- the same key updates rather than duplicates,
- invalid candidates are rejected,
- extraction is queued/non-blocking from delivered-turn finalization,
- interrupted delivered turns may still contribute user facts,
- malformed Ollama extraction output yields no candidates,
- known keys are included in the extraction prompt,
- prompt retrieval selects relevant facts and stays bounded,
- unrelated facts are not injected into ordinary prompts,
- durable-memory failures do not fail conversation generation,
- existing short-term memory, delivery, latency, behavior-preview, and eye-validation suites remain green.
