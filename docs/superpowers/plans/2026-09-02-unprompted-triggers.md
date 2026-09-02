# Unprompted Triggers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative, deterministic Step 7 trigger system that can greet a person after a long observed absence or speak once after a long no-interaction stretch, without treating proactive context as user speech.

**Architecture:** `brain/triggers.py` owns transition/timer decisions and has no LLM dependency. `ConversationWorker` gains a separate proactive request kind that reuses generation cancellation, clause streaming, TTS, and physical-delivery accounting while beginning delivery with an empty user string. Runtime creates the trigger worker only for a live camera source and shuts it down before conversation/voice.

**Tech Stack:** Python 3.11 standard library, existing `State` lock, SQLite/event log, existing Ollama/voice/delivery pipeline, `unittest`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-02-unprompted-triggers-design.md`

## Global Constraints

- `PLAN.md` remains authoritative.
- The main render loop must never block.
- All shared mutation goes through `State` behind its lock.
- Proactive trigger decisions are deterministic code; the LLM chooses wording only.
- Runtime proactive speech is enabled only when `config.camera.source == "camera"`.
- `image` and `video` sources must never cause real proactive speech.
- No second GPU model, hosted service, shell execution, or new dependency.
- Quiet means no interaction with Vess, not acoustic silence.
- At most one accepted proactive line per no-interaction presence stretch until user interaction or genuine absence resets it.

---

### Task 1: Pure trigger decisions and user-activity clock

**Files:**
- Create: `brain/triggers.py`
- Modify: `state.py`
- Modify: `config.json`
- Modify: `brain/llm.py`
- Test: `tests/test_triggers.py`
- Test: `tests/test_trigger_activity.py`

**Interfaces:**
- Produces: `TriggerSnapshot`, `TriggerEvent`, `TriggerDecider.evaluate(snapshot, now, local_hour) -> TriggerEvent | None`, `TriggerDecider.accept(event, now) -> None`.
- Produces: `State.last_interaction: float`.
- Ordinary `ConversationWorker.submit` and `submit_with_timing` update `last_interaction`; future proactive submission will not.

- [ ] **Step 1: Write failing trigger-decision tests**

Cover these exact behaviors with deterministic timestamps:

```python
snapshot = TriggerSnapshot(
    person_present=True,
    present_since=100.0,
    last_interaction=100.0,
    muted_until=0.0,
    listening=False,
    thinking=False,
    speaking=False,
)
```

Assertions:

- first observation while absent never manufactures prior absence
- `True -> False -> True` below `min_absence_hours` returns `None`
- the same transition above threshold returns `TriggerEvent("returned_after_absence", ...)`
- present + 30 minutes no interaction returns one `quiet_interaction`
- after `accept(...)`, repeated evaluate calls return `None`
- advancing `last_interaction` resets the one-shot latch
- leaving and returning resets the presence stretch
- accepted return event blocks quiet event until interaction/absence reset
- absent, quiet-hour, muted, listening, thinking, speaking, and cooldown states return `None`

- [ ] **Step 2: Write failing activity-clock tests**

Instantiate `ConversationWorker` with fake dependencies and assert:

```python
worker.submit("hello")
assert state.last_interaction > 0
```

Also assert a duplicate ordinary submission still advances the interaction clock. Do not add proactive behavior yet.

- [ ] **Step 3: Run RED suite**

Run:

```bash
python -m unittest tests.test_triggers tests.test_trigger_activity -v
```

Expected: failures/import errors only for the missing trigger API and missing `last_interaction` behavior.

- [ ] **Step 4: Implement minimal trigger engine**

`brain/triggers.py` should contain focused immutable data structures and one stateful decider. It must not import Ollama, TTS, web, or detector code.

Use config-derived seconds:

```python
min_absence_seconds = float(settings.get("min_absence_hours", 4)) * 3600.0
idle_seconds = float(settings.get("idle_interaction_minutes", 30)) * 60.0
cooldown_seconds = float(settings.get("cooldown_minutes", 60)) * 60.0
```

Transition bookkeeping:

- `previous_present: bool | None`
- `absent_since: float | None`
- `last_accepted_at: float | None`
- `last_activity_seen: float`
- `proactive_since_interaction: bool`

`accept(...)` updates cooldown and the one-proactive latch only after caller acceptance.

- [ ] **Step 5: Add `State.last_interaction` and ordinary-submit updates**

Add:

```python
last_interaction: float = 0.0
```

At the beginning of ordinary `_submit(...)`, update it under `state.locked()` before duplicate collapse so every real accepted-user submission counts as interaction.

- [ ] **Step 6: Add config default**

Add under `triggers`:

```json
"idle_interaction_minutes": 30
```

- [ ] **Step 7: Run GREEN suite and full verification**

Run targeted tests locally/CI, then full Actions workflow. Existing conversation/command/voice tests must remain green.

- [ ] **Step 8: Commit Task 1**

Commit message:

```text
triggers: add deterministic proactive decision engine
```

---

### Task 2: Dedicated proactive conversation path

**Files:**
- Modify: `brain/llm.py`
- Modify: `brain/memory.py` only if needed for assistant-only history rendering (prefer no storage API change)
- Test: `tests/test_proactive_conversation.py`
- Test: existing `tests/test_llm.py`, `tests/test_durable_memory_integration.py`, freshness tests

**Interfaces:**
- Produces: `ConversationWorker.submit_proactive(trigger_name: str, context: str) -> bool`.
- Produces: `build_proactive_prompt(...) -> str`.
- Consumes existing `DeliveryLedger`, clause streaming, voice generation, and durable-memory guard.

- [ ] **Step 1: Write failing proactive-submission tests**

Cover:

- proactive submission returns `False` when an ordinary request is active or pending and does not replace it
- accepted proactive submission returns `True` and does **not** change `state.last_interaction`
- a later real user request advances generation and can stale-cancel proactive work
- proactive prompt contains `Proactive system observation` and does not contain `Current request` for the event context
- proactive prompt explicitly requires one short observation/greeting, no generic question, no inferred destination/activity
- model event context is never passed through command selection

- [ ] **Step 2: Write failing memory-provenance tests**

Drive physical delivery receipts for a proactive generation and assert:

- short-term history stores the assistant text with `user == ""`
- subsequent prompt history does not render a blank `User:` line for that turn
- `durable_memory.remember(...)` is never called for the proactive turn
- generated-but-undelivered proactive text is not remembered

- [ ] **Step 3: Run RED suite**

Run:

```bash
python -m unittest tests.test_proactive_conversation -v
```

Expected: missing proactive submission/prompt behavior.

- [ ] **Step 4: Introduce a typed queued request**

Replace raw `(generation_id, request)` tuples internally with a private frozen request record carrying:

```python
@dataclass(frozen=True)
class _QueuedRequest:
    generation_id: int
    text: str
    kind: str = "user"
    trigger_name: str | None = None
```

Keep public ordinary submission signatures unchanged.

- [ ] **Step 5: Implement atomic `submit_proactive`**

Under `_request_lock`:

- reject if `_active_request_key` or `_pending_request_key` is non-`None`
- allocate a generation
- begin the voice generation
- enqueue kind `proactive`
- publish debug state
- return `True`

Do not touch `state.last_interaction`.

- [ ] **Step 6: Add shared prompt-context helper and proactive prompt**

Refactor only the common prompt-section assembly needed by both prompt types. Ordinary `build_prompt` must remain behavior-compatible.

For history rendering:

```python
if turn.user.strip():
    history_lines.append(f"User: {turn.user}")
```

Always render delivered assistant text.

The proactive final section must be structurally distinct from user text.

- [ ] **Step 7: Route proactive requests through existing response machinery**

In `_run`, dispatch based on queued `kind`. Reuse the existing `_respond` streaming code with a proactive branch that:

- calls `DeliveryLedger.begin(generation_id, "")`
- skips command selection
- uses `build_proactive_prompt`
- skips mood classification from trigger context
- otherwise uses the same generation checks, performance tags, TTS enqueue, `llm_finished`, and `finish_generation`

- [ ] **Step 8: Run GREEN suite and full verification**

All ordinary conversation, command, interruption, durable-memory, latency, and voice tests must stay green.

- [ ] **Step 9: Commit Task 2**

Commit message:

```text
conversation: add provenance-safe proactive speech
```

---

### Task 3: Trigger worker, trusted-camera runtime gating, and lifecycle

**Files:**
- Modify: `brain/triggers.py`
- Modify: `main.py`
- Test: `tests/test_trigger_worker.py`
- Test: `tests/test_trigger_runtime.py`
- Test: existing `tests/test_main.py`

**Interfaces:**
- Produces: `TriggerWorker.start()`, `TriggerWorker.close()`.
- Consumes: `ConversationWorker.submit_proactive(name, context) -> bool`.
- Produces runtime helper `_build_trigger_worker(...) -> TriggerWorker | None` or equivalent focused builder.

- [ ] **Step 1: Write failing worker tests**

Use a fake state clock and fake proactive callback. Prove:

- worker/step calls `submit_proactive` only for an eligible event
- callback `False` does not call `decider.accept` and does not consume cooldown/latch
- callback `True` consumes both
- emitted event/debug payload includes trigger name and grounded duration
- polling does not spam suppression events every cycle

Prefer exposing a deterministic single `poll_once(now, local_hour)` method used by the thread loop so tests do not sleep.

- [ ] **Step 2: Write failing runtime-gating tests**

Assert:

```python
_build_trigger_worker({"camera": {"source": "image"}}, ...) is None
_build_trigger_worker({"camera": {"source": "video"}}, ...) is None
_build_trigger_worker({"camera": {"source": "camera"}}, ...) is fake_worker
```

Also assert shutdown closes triggers before audio/coordinator/conversation/voice.

- [ ] **Step 3: Run RED suite**

Run:

```bash
python -m unittest tests.test_trigger_worker tests.test_trigger_runtime -v
```

Expected: missing worker/runtime wiring.

- [ ] **Step 4: Implement worker**

The thread loop waits about 0.5 seconds between polls using an Event, never `time.sleep` in the render loop. `poll_once` snapshots state under one lock, evaluates, attempts proactive submit, and calls `decider.accept(...)` only on `True`.

- [ ] **Step 5: Wire runtime ownership**

Build the trigger worker only for live `camera` config. Start it after the conversation worker is started. Ensure close order stops triggers before any producer/consumer they call into.

With non-live sources, print one concise startup line explaining proactive speech is disabled until a live camera source is configured.

- [ ] **Step 6: Run GREEN suite and full verification**

Run the full GitHub Actions workflow. Behavior verification and comprehensive eye validation must remain green.

- [ ] **Step 7: Commit Task 3**

Commit message:

```text
runtime: wire conservative unprompted triggers
```

---

### Task 4: Review, status, and integration

**Files:**
- Modify: `STATUS.md`

- [ ] **Step 1: Review base-to-head diff**

Verify no desktop-perception stub, shell command, probabilistic scoring system, acoustic-silence detector, or non-camera runtime activation slipped into scope.

- [ ] **Step 2: Add review regressions for any discovered bug**

Any discovered behavior bug gets a tests-only RED commit before its production fix.

- [ ] **Step 3: Update `STATUS.md`**

Record:

- two trigger types and exact gates
- `last_interaction` rationale
- live-camera trust boundary and current hardware limitation
- proactive memory provenance
- RED/GREEN run IDs and final head
- deferred desktop/typing/call gates
- hardware follow-up after webcam arrives

- [ ] **Step 4: Exact-head verification**

Run GitHub Actions on the final documentation head. Require unit tests, behavior verification, comprehensive eye validation, artifact upload, and failure gate success.

- [ ] **Step 5: Integrate**

If `main` remains the exact ancestor and the feature is zero commits behind, fast-forward `main` to the verified feature head. Then verify the `main` pointer.
