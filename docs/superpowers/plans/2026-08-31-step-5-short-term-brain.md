# Step 5A Short-Term Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Vess grounded identity behavior and recent conversational context without changing the local model or increasing the 4096-token context.

**Architecture:** Completed user/assistant turns live in the single locked `State` object and are pruned by helpers in `brain.memory`. `brain.llm` injects that recent history plus the existing mood prompt into a stronger stable identity prompt, and stores only successfully completed latest-generation responses. Every completed turn is also appended to the existing SQLite event log for future Step 5B summarisation.

**Tech Stack:** Python 3.11, dataclasses, existing threading/State lock, SQLite event log, unittest, Ollama qwen2.5:7b.

**Spec:** `docs/superpowers/specs/2026-08-31-step-5-short-term-brain-design.md`

## Global Constraints

- `PLAN.md` remains authoritative.
- Shared runtime mutation goes through `State` behind its lock.
- The 30fps render loop must never block.
- Ollama remains `qwen2.5:7b` on GPU with `num_ctx=4096`.
- Whisper and Kokoro remain on CPU.
- Short-term defaults are 10 minutes and 8 completed turns.
- Cancelled/stale response generations must not become completed memory.
- Durable facts, semantic retrieval, summarisation, reasoning routing, TTS expressiveness, and barge-in are out of scope.

---

### Task 1: Add bounded RAM conversation turns

**Files:**
- Modify: `state.py`
- Modify: `brain/memory.py`
- Create: `tests/test_short_term_memory.py`

**Interfaces:**
- Produces: `ConversationTurn(timestamp: float, user: str, assistant: str)` in `state.py`.
- Produces: `append_conversation_turn(state, user, assistant, *, timestamp=None, max_age_seconds, max_turns) -> ConversationTurn`.
- Produces: `recent_conversation_turns(state, *, now=None, max_age_seconds, max_turns) -> list[ConversationTurn]`.

- [ ] **Step 1: Write failing tests for append, age pruning, and count pruning**

```python
def test_recent_turns_prune_by_age_and_count():
    state = State()
    append_conversation_turn(state, "old", "old reply", timestamp=0.0,
                             max_age_seconds=600.0, max_turns=2)
    append_conversation_turn(state, "one", "reply one", timestamp=995.0,
                             max_age_seconds=600.0, max_turns=2)
    append_conversation_turn(state, "two", "reply two", timestamp=996.0,
                             max_age_seconds=600.0, max_turns=2)

    turns = recent_conversation_turns(
        state, now=1000.0, max_age_seconds=600.0, max_turns=2)

    assert [(turn.user, turn.assistant) for turn in turns] == [
        ("one", "reply one"),
        ("two", "reply two"),
    ]
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `python -m unittest tests.test_short_term_memory -v`

Expected: FAIL because `ConversationTurn`/memory helpers do not exist.

- [ ] **Step 3: Add `ConversationTurn` and `State.conversation_turns`**

```python
@dataclass(frozen=True)
class ConversationTurn:
    timestamp: float
    user: str
    assistant: str

# in State
conversation_turns: list[ConversationTurn] = field(default_factory=list)
```

- [ ] **Step 4: Implement memory helpers using `state.locked()`**

Prune expired entries first, retain only the newest `max_turns`, and return copies/lists rather than leaking the mutable state list.

- [ ] **Step 5: Run the memory tests and verify GREEN**

Run: `python -m unittest tests.test_short_term_memory -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add state.py brain/memory.py tests/test_short_term_memory.py
git commit -m "feat: add bounded short-term conversation memory"
```

---

### Task 2: Build a grounded, history-aware prompt

**Files:**
- Modify: `brain/llm.py`
- Modify: `config.json`
- Modify: `tests/test_llm.py`

**Interfaces:**
- `build_prompt(config, moods, state, request) -> str` replaces the old three-argument form.
- Reads recent turns through `recent_conversation_turns`.

- [ ] **Step 1: Add failing prompt tests**

Cover all of these assertions:

```python
prompt = build_prompt(config, moods, state, "Why?")
assert "Recent conversation:" in prompt
assert prompt.index("User: How was your day?") < prompt.index("Current request:\nWhy?")
assert "You're mildly irritated." in prompt
assert "do not invent" in prompt.lower()
assert "human body" in prompt.lower()
```

Also test that expired history is omitted and an empty history does not emit a `Recent conversation:` section.

- [ ] **Step 2: Run the focused LLM tests and verify RED**

Run: `python -m unittest tests.test_llm -v`

Expected: FAIL because the current prompt has no history, no mood prompt, and the old signature.

- [ ] **Step 3: Add memory config defaults**

```json
"memory": {
  "short_term_minutes": 10,
  "short_term_turns": 8
}
```

- [ ] **Step 4: Implement prompt composition**

Stable identity comes first and states:

```text
You are Vess, a local ambient AI represented by a small expressive face on a wall.
You do not have a human body or offline physical life. Do not invent human activities or experiences.
Treat your actual recent conversations, observations, room state, and current mood as your lived runtime experience.
For casual questions about your day or feelings, answer naturally from that real context; if little has happened, it is fine to say things have been quiet.
Speak conversationally, not like customer support. Answer the question first. Do not automatically end every response with a question, and do not repeatedly explain that you are an AI unless relevant.
```

Then append persona, mood name + `moods[mood].get("prompt", "")`, room/objects, recent turns oldest-to-newest, and the current request.

- [ ] **Step 5: Update all `build_prompt` call sites/tests**

`ConversationWorker` already owns `self._moods`; pass it into `build_prompt`.

- [ ] **Step 6: Run LLM tests and verify GREEN**

Run: `python -m unittest tests.test_llm -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add brain/llm.py config.json tests/test_llm.py
git commit -m "feat: ground Vess prompt in mood and recent context"
```

---

### Task 3: Remember only completed latest-generation responses

**Files:**
- Modify: `brain/llm.py`
- Modify: `tests/test_voice_freshness.py`
- Modify: `tests/test_memory.py`

**Interfaces:**
- Uses Task 1 `append_conversation_turn`.
- Appends `EventLog.append("conversation_turn", {"user": ..., "assistant": ...})` only after a latest generation finishes normally.

- [ ] **Step 1: Add failing completion-memory test**

Create a worker with a deterministic fake client producing two clauses, run one request, then assert exactly one RAM turn with the joined assistant text and exactly one `conversation_turn` event.

Expected assistant text uses the same clause order that was queued for speech.

- [ ] **Step 2: Add failing stale-generation test**

Use the existing latest-intent test pattern: supersede generation 1 with generation 2 before generation 1 finishes. Assert generation 1 creates no completed `ConversationTurn` and no `conversation_turn` event.

- [ ] **Step 3: Add failing acknowledgement test**

Submit an empty wake request and assert the cached `Yeah?` acknowledgement is not added to conversation memory.

- [ ] **Step 4: Run focused tests and verify RED**

Run: `python -m unittest tests.test_voice_freshness tests.test_memory -v`

Expected: FAIL because responses are not currently written to short-term memory.

- [ ] **Step 5: Collect current-generation clauses in `_respond`**

```python
spoken_clauses: list[str] = []
...
spoken_clauses.append(clause)
self._voice.enqueue(clause, generation_id=generation_id)
```

Do not write memory on any stale-return path.

- [ ] **Step 6: Store the completed turn before optional mood classification**

Read `short_term_minutes` and `short_term_turns` from config, join clauses with spaces, call `append_conversation_turn`, then append the durable event. If the model produces no spoken text, do not store an empty assistant turn.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_voice_freshness tests.test_memory -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add brain/llm.py tests/test_voice_freshness.py tests/test_memory.py
git commit -m "feat: remember completed Vess conversations"
```

---

### Task 4: Verify compatibility and document Step 5A

**Files:**
- Modify: `STATUS.md`

**Interfaces:** none new.

- [ ] **Step 1: Run the complete unit suite**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures/errors.

- [ ] **Step 2: Compile the production modules**

Run:

```bash
python -m compileall -q brain control output perception main.py state.py
```

Expected: exit code 0.

- [ ] **Step 3: Review prompt size and invariants**

Confirm the code still sends `num_ctx=4096`, no model/device settings changed, and the memory defaults are 10 minutes / 8 turns.

- [ ] **Step 4: Update `STATUS.md`**

Record:

- Step 5A short-term completed-turn memory exists;
- mood prompt fragments now reach Ollama;
- Vess has a grounded non-human self-model;
- completed turns are persisted as event-log records for later summarisation;
- durable facts/retrieval remain Step 5B;
- live validation should include `How's your day going?` followed by a context-dependent follow-up such as `Why?`.

- [ ] **Step 5: Commit**

```bash
git add STATUS.md
git commit -m "docs: record short-term brain milestone"
```
