# Closed Command Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `turn blue` execute a validated, allowlisted `set_color` command by voice without slowing ordinary conversation or exposing arbitrary execution.

**Architecture:** `brain/commands.py` owns the executable registry, validation, candidate gate, and state mutation. `OllamaClient` only selects from the registry after a cheap command-candidate gate, and `ConversationWorker` re-checks generation freshness before execution and speaks a deterministic acknowledgement through the existing delivery lifecycle. The web server consumes the same registry catalog and executor.

**Tech Stack:** Python 3.11 standard library, existing FastAPI web control surface, existing Ollama `/api/generate`, unittest, SQLite/event infrastructure already in Vess.

**Spec:** `docs/superpowers/specs/2026-09-02-closed-command-registry-design.md`

## Global Constraints

- The model may select only executable registry entries; it never composes shell, code, paths, or config.
- Normal conversational requests must not make a command-selection Ollama call.
- Unknown names, extra arguments, arbitrary values, and malformed JSON must fail closed before execution.
- A stale generation must never execute a command after a newer request supersedes it.
- Shared state mutation happens under `State.locked()`.
- Rendering remains non-blocking; command selection stays on the existing conversation worker.
- No new dependency.
- This pass implements only `set_color`; OS control, timers, media, and other planned commands remain out of scope.

---

### Task 1: Closed registry and `set_color`

**Files:**
- Create: `brain/commands.py`
- Modify: `config.json`
- Test: `tests/test_commands.py`

**Interfaces:**
- Produces: `CommandCall(name: str, arguments: dict[str, object])`
- Produces: `CommandResult(spoken_response: str, event_payload: dict[str, object])`
- Produces: `CommandRegistry(config: dict[str, object], state: State)`
- Produces: `CommandRegistry.catalog() -> dict[str, object]`
- Produces: `CommandRegistry.is_candidate(text: str) -> bool`
- Produces: `CommandRegistry.validate(payload: object) -> CommandCall | None`
- Produces: `CommandRegistry.execute(call: CommandCall) -> CommandResult`

- [ ] **Step 1: Write failing registry tests**

Cover the executable catalog, candidate gate, exact argument validation, unknown/extra-field rejection, and `set_color` state mutation using the configured palette.

- [ ] **Step 2: Run the full unit suite and verify RED**

Run: `python -m unittest discover -s tests -v`

Expected: existing tests remain green and only the new command-registry expectations fail because `brain.commands`/the command config do not exist.

- [ ] **Step 3: Implement the minimal registry**

Add a small human-authored palette under `config.commands.colors`; implement only `set_color`. `execute()` maps the validated name to the configured RGB tuple under the state lock and returns a short response such as `Blue.`.

- [ ] **Step 4: Run the full unit suite and verify GREEN**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit production code separately from the tests-only RED commit.

---

### Task 2: Strict Ollama command selection

**Files:**
- Modify: `brain/llm.py`
- Test: `tests/test_command_selection.py`

**Interfaces:**
- Consumes: `CommandRegistry.catalog()`
- Produces: `OllamaClient.select_command(transcript: str, catalog: dict[str, object], config: dict[str, object]) -> object | None`

- [ ] **Step 1: Write failing selector tests**

Assert that the request is non-streaming JSON selection, the prompt contains only the supplied executable catalog, `null` means no command, valid JSON is returned for Python validation, and malformed/wrong-shape output yields no selection.

- [ ] **Step 2: Verify RED with the full suite**

Run: `python -m unittest discover -s tests -v`

Expected: only the new selector expectations fail because `select_command` is missing.

- [ ] **Step 3: Implement the minimal selector**

Use the existing `/api/generate` client with `stream=false` and JSON output. Instruct the model to return one catalog command object only when the user is instructing Vess itself; otherwise return JSON `null`. Do not trust the returned fields beyond JSON decoding.

- [ ] **Step 4: Verify GREEN**

Run the full unit suite and confirm zero failures.

- [ ] **Step 5: Commit**

Commit selector production separately after GREEN.

---

### Task 3: Generation-safe voice command flow

**Files:**
- Modify: `brain/llm.py`
- Test: `tests/test_command_flow.py`

**Interfaces:**
- Consumes: `CommandRegistry.is_candidate`, `catalog`, `validate`, `execute`
- Consumes: `OllamaClient.select_command`
- Extends: `ConversationWorker(..., command_registry: CommandRegistry | None = None)`

- [ ] **Step 1: Write failing flow tests**

Cover:

1. normal questions never call the selector;
2. valid `turn blue` selection mutates state and queues `Blue.`;
3. malformed/invalid selection falls through to the normal streamed response;
4. selector errors fall through to normal conversation and record debug failure;
5. a generation superseded while selection is blocked never executes the stale call;
6. successful execution records `command_executed` with only validated name/arguments and uses the existing delivery lifecycle.

- [ ] **Step 2: Verify RED**

Run the full unit suite. New flow tests must fail for the absent command integration while existing behavior stays green.

- [ ] **Step 3: Implement the command branch in `_respond`**

Begin the existing delivery generation first. If the registry says the request is a candidate, call the selector, then re-check `_is_latest(generation_id)` before validation/execution. On a valid call, execute once, queue the deterministic acknowledgement, finish the generation, record events, and skip mood classification. On no/invalid/error selection, continue into the unchanged prompt/streaming path.

- [ ] **Step 4: Verify GREEN**

Run the full suite and confirm the existing freshness, delivery, latency, memory, and barge-in tests remain green.

- [ ] **Step 5: Commit**

Commit the production integration.

---

### Task 4: Runtime and web share the registry

**Files:**
- Modify: `main.py`
- Modify: `control/web.py`
- Modify: `tests/test_main.py`
- Modify: `tests/test_web.py`

**Interfaces:**
- `main()` creates one `CommandRegistry(config, state)`.
- `_build_voice_runtime(..., command_registry=None)` forwards the same object to `ConversationWorker`.
- `WebServer(..., command_registry=None)` forwards it to `create_app`.
- `GET /commands` returns the registry catalog when present.
- `POST /commands` accepts the same command-call shape, validates and executes through the registry, and rejects invalid calls with HTTP 422.

- [ ] **Step 1: Write failing runtime/web tests**

Assert object identity through runtime wiring, one shared catalog through `/commands`, and that web execution changes state only through a registry-valid command.

- [ ] **Step 2: Verify RED**

Run the full unit suite and confirm only the new wiring/API expectations fail.

- [ ] **Step 3: Implement shared runtime ownership**

Construct the registry once in `main`, pass it to both voice and web. Keep constructor arguments optional for existing isolated tests and degraded configurations.

- [ ] **Step 4: Verify GREEN plus behavior preview**

Use the repository `Verify Vess` workflow as the authoritative environment: unit tests, behavior verification, comprehensive eye validation, artifact gate.

- [ ] **Step 5: Commit**

Commit runtime/web production changes.

---

### Task 5: Review, status, and exact-head verification

**Files:**
- Modify: `STATUS.md`

- [ ] **Step 1: Review base-to-head diff**

Confirm no command can execute outside the closed registry, no normal request gets an extra selector call, no shell/subprocess capability was introduced, and no unrelated voice/config behavior changed beyond the explicit color palette and wiring.

- [ ] **Step 2: Update `STATUS.md`**

Record the command architecture, RED/GREEN run IDs, exact tested head, safety boundary, and the next Step 7 trigger work.

- [ ] **Step 3: Verify the exact docs head**

Run the full GitHub Actions workflow on the final branch head and require unit tests, behavior verification, comprehensive eye validation, artifact upload, and failure gate to succeed before integration.

- [ ] **Step 4: Integrate and branch for Step 7**

Fast-forward `main` only if it remains the ancestor of this verified branch. Then create a fresh trigger feature branch from the integrated SHA.
