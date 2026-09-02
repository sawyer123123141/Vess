# Durable Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist explicit durable user facts, retrieve only relevant ones into prompts, and extract them asynchronously after delivered turns.

**Architecture:** Extend `brain/memory.py` with a focused `FactMemory` background owner backed by the existing SQLite database. `OllamaClient` performs JSON-only extraction on that worker thread; `ConversationWorker` does deterministic prompt-time retrieval and only enqueues extraction from delivered-turn finalization.

**Tech Stack:** Python 3.11, standard-library `sqlite3`, `threading`, `queue`, `json`, existing local Ollama client, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-02-durable-memory-design.md`

## Global Constraints

- `PLAN.md` remains authoritative.
- The render/playback path must never wait on SQLite writes or Ollama fact extraction.
- No embeddings, cloud services, extra GPU models, commands, triggers, or memory UI in this pass.
- Automatic facts come only from the user's utterance, never Vess's response.
- Prompt-time fact retrieval makes no model call.
- Existing construction paths remain compatible when durable memory is omitted in tests.

---

### Task 1: Durable fact store and deterministic retrieval

**Files:**
- Modify: `brain/memory.py`
- Create: `tests/test_durable_memory.py`

**Interfaces:**
- Produce `FactCandidate(key: str, value: str)` and `DurableFact(...)` frozen dataclasses.
- Produce `FactMemory(path, extractor)` with `remember(text)`, `relevant_facts(query, limit=5)`, `known_keys()`, and `close()`.

- [ ] Write tests proving facts persist across instances and an identical key updates instead of duplicating.
- [ ] Run focused tests and verify RED because these APIs do not exist.
- [ ] Add tests for key/value validation, lexical relevance, result limit, unrelated-query silence, and broad `remember/know about me` recent-fact fallback.
- [ ] Implement the minimal worker, `facts` schema, validated upsert, read path, lexical ranking, and draining shutdown.
- [ ] Run focused memory tests and commit `memory: add durable fact store`.

### Task 2: Local Ollama extraction contract

**Files:**
- Modify: `brain/llm.py`
- Create: `tests/test_fact_extraction.py`

**Interfaces:**
- Produce `OllamaClient.extract_facts(transcript, known_keys, config) -> list[FactCandidate]`.

- [ ] Write tests for valid JSON, malformed JSON, wrong top-level shape, invalid items, and exact prompt constraints.
- [ ] Verify RED because `extract_facts` does not exist.
- [ ] Implement one non-streaming local call that requests at most three explicit durable facts as JSON only.
- [ ] Keep storage validation in `FactMemory`; parsing failures return an empty list.
- [ ] Run focused LLM tests and commit `memory: extract explicit user facts`.

### Task 3: Prompt retrieval and delivered-turn queueing

**Files:**
- Modify: `brain/llm.py`
- Create: `tests/test_durable_memory_integration.py`

**Interfaces:**
- `build_prompt(..., durable_memory=None)` asks the store for at most five relevant facts.
- `ConversationWorker(..., durable_memory=None)` passes the same store into prompt construction and queues `remember(user_request)` from delivered-turn finalization.

- [ ] Write tests proving relevant facts appear before recent conversation, unrelated facts do not appear, and output stays bounded.
- [ ] Write tests proving completed and interrupted delivered turns enqueue only the user request, never assistant text, and enqueueing is non-blocking.
- [ ] Verify RED.
- [ ] Implement optional durable-memory wiring with failure isolation: retrieval/extraction-queue failures cannot fail conversation handling.
- [ ] Run focused conversation/delivery tests and commit `memory: wire durable facts into conversation`.

### Task 4: Runtime ownership, shutdown, docs, and full verification

**Files:**
- Modify: `main.py`
- Modify: `STATUS.md`
- Create/modify runtime wiring tests as needed.

**Interfaces:**
- `main.py` constructs one `OllamaClient`, one `FactMemory(ROOT / "vess.db", extractor=...)`, supplies both to `ConversationWorker`, and closes FactMemory after conversation work has stopped.

- [ ] Write a wiring test proving the same client-backed extractor is supplied and that shutdown drains memory after conversation shutdown.
- [ ] Verify RED.
- [ ] Implement runtime construction without changing `config.json` defaults.
- [ ] Run the complete unit suite, behavior verification, and comprehensive eye validation.
- [ ] Review the base-to-head diff for accidental voice/latency/config behavior changes.
- [ ] Update `STATUS.md` with design decisions, RED/GREEN evidence, limitations, and the next planned step.
- [ ] Run exact-head verification and commit docs.
