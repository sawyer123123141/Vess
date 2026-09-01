# Vess Future Architecture Roadmap

**Status:** Directional roadmap, not an implementation spec.

This document records the architecture direction we currently think is worth pursuing after the present voice/face work. It intentionally distinguishes strong ideas from attractive-sounding overengineering. Each major feature still needs its own design review before implementation.

The goal is not to maximize the number of subsystems. The goal is for Vess to feel like one persistent, responsive entity that can run for long periods, remember what matters, notice useful things, act reliably, recover from failure, and stay within local hardware limits.

## Core Principles

1. **Keep the runtime asynchronous and event-driven.** Do not replace the current independent audio, cognition, output, perception, and rendering workers with one giant synchronous `Perceive -> Think -> Plan -> Act` loop.
2. **Preserve one authoritative runtime state.** Extend `State` with compact beliefs/status where useful; do not create competing sources of truth.
3. **Use code for deterministic policy whenever possible.** The local LLM should handle language, ambiguity, and reasoning, not things that can be enforced reliably with ordinary code.
4. **Silence is a valid action.** A proactive assistant is not improved by speaking more often.
5. **Planning is conditional.** Easy requests should stay cheap and direct. Planning exists for genuinely multi-step or uncertain tasks.
6. **Measure before adapting performance.** Add telemetry and backpressure first. Do not build a complicated automatic quality manager until profiling proves one is needed.
7. **Memory is not automatically truth or authority.** Stored information must retain provenance, confidence, freshness, and whether it is allowed to influence future actions.
8. **Completion is not the same as success.** Tool/action paths must verify intended postconditions and check for unintended side effects where practical.
9. **Prefer verified behavior over self-reflection loops.** More LLM calls and more scaffolding do not automatically make an agent more reliable.
10. **Keep expensive systems demand-driven.** Continuous cheap perception is fine; vision-language reasoning, planning, large retrieval passes, and other expensive work should be triggered by need.

---

## Recommended Development Order

This replaces the generic recommendation to immediately build memory -> attention -> world model -> proactivity -> planning. For Vess, conversational feel and reliability should come earlier.

### 1. Finish voice acceptance on the target PC

Before making Chatterbox Turbo the default:

- benchmark Kokoro vs Chatterbox on the actual machine
- measure warm first-clause latency and p95 latency
- measure VRAM while Qwen and TTS are resident together
- measure Qwen tokens/sec degradation
- test repeated short generations for stability
- listen for naturalness, consistency, and expressive range
- verify cached voice conditioning is actually effective in the real library/runtime

Keep Kokoro as the baseline until Chatterbox clearly wins the total interaction tradeoff.

### 2. Barge-in and natural turn-taking

This is higher priority than sophisticated long-term memory because it directly changes whether conversation feels responsive.

Build toward:

- VAD remaining active while Vess speaks
- reliable distinction between Vess's own speaker output and a human interruption
- interruption stopping or ducking playback quickly
- stale queued TTS being cancelled
- the interrupted response not being falsely stored as fully spoken/completed
- a clear policy for whether the new utterance supersedes or redirects the current response
- measured interruption-to-listening latency

Do not expose private model reasoning. The goal is natural conversational control, not narrated chain-of-thought.

### 3. Reliability, observability, and replay

Before adding several interacting cognitive layers, make runs inspectable.

Extend the existing event-log idea into a practical local flight recorder:

- stable correlation/run IDs across transcript -> LLM -> TTS -> playback/tool execution
- timestamp important state transitions and queue handoffs
- record tool/action inputs, outputs, errors, retries, and resulting state changes
- preserve enough evidence to reproduce failures without repeating irreversible side effects
- turn real failures into regression fixtures/tests
- record p50/p95 latency by pipeline stage
- add long-session soak tests and resource-leak checks

A replay system should replay recorded outcomes/state transitions for debugging. It must not blindly reissue real external side effects.

### 4. Action execution and failure handling foundation

Before giving a planner broad responsibilities, build a reliable executor.

For each action/command define:

- declared arguments and validation
- expected preconditions when practical
- expected postconditions
- timeout
- retry policy
- whether retry is safe/idempotent
- whether the action is reversible
- observable success evidence
- error classification

Failures should distinguish roughly:

- invalid request/configuration
- unavailable dependency/device
- transient failure worth retrying
- action executed but postcondition not reached
- unknown/ambiguous result
- user cancellation/interruption

A task is not successful merely because a tool call returned without throwing an exception.

### 5. Structured long-term memory

Do not begin with three separate memory services. Start with one local memory store containing typed records and strong metadata.

Useful memory types:

- **episodic**: specific events/interactions
- **semantic**: durable facts and learned relationships
- **preference**: user choices/tendencies that may change
- **procedural**: verified reusable skills/action patterns, much later

Each durable memory should be able to carry:

- source/provenance
- source authority (user statement, observed state, model inference, tool result, etc.)
- confidence
- created timestamp
- last-confirmed timestamp
- validity/expiry when known
- superseded/invalidated relationship
- importance/utility
- retrieval/use count if useful

#### Memory write path

Use an explicit write/manage/read loop:

1. candidate memory is extracted
2. write gate decides whether it deserves persistence
3. compare against existing related memories
4. merge, supersede, or reject contradictions rather than blindly appending
5. retain provenance/authority through consolidation
6. retrieve only when relevant

#### Forgetting matters

Memory quality is not monotonic with database size.

Support:

- explicit invalidation when a fact changes
- confidence decay for observations that become stale
- preference updates that supersede earlier preferences
- utility-aware pruning/archival of low-value duplicate memories
- tests that penalize retrieval of obsolete memories

Do **not** let an LLM-written summary silently transform an uncertain inference into an authoritative user fact.

### 6. Context compiler / prompt budget manager

Vess currently operates with a constrained local context window, so retrieval alone is insufficient. Build deterministic context assembly with explicit budgets.

Conceptual order:

1. immutable identity/safety/system instructions
2. current user request
3. immediately relevant runtime/situational state
4. recent conversational turns
5. retrieved durable memories
6. tool/planner context only when active

The exact order and token budgets need measurement, but the system should ensure one category cannot silently consume the whole prompt.

Track what was omitted and why so a bad answer can be traced back to context selection rather than vaguely blamed on the model.

### 7. Situational belief state, not a giant "world model"

Do not build a simulated universe.

Extend runtime state into a compact set of beliefs about the current situation, for example:

- who/what is currently present
- current interaction partner/subject
- current task or conversational topic
- active application/window if available
- recent meaningful transitions
- what Vess believes is happening
- confidence and last-updated time for uncertain beliefs
- whether a belief is observed, inferred, or remembered

Beliefs should expire when their supporting evidence goes stale.

This layer exists so downstream systems receive a coherent present-tense picture instead of raw sensor events.

### 8. Attention and cognition routing

Attention should initially be deterministic policy, not another neural model.

For each meaningful event, consider signals such as:

- urgency
- novelty
- relevance to current task/conversation
- confidence
- whether the user appears interruptible
- repeat frequency/habituation
- cooldowns
- whether non-verbal acknowledgement is enough
- expected value of deeper processing
- estimated compute cost

The router can choose actions like:

- ignore
- update state only
- remember candidate
- react non-verbally
- perform cheap reasoning
- perform expensive reasoning/vision
- consider proactive speech

This is the place to make Vess compute-aware before a separate performance manager exists.

### 9. Proactive behavior with abstention as a first-class outcome

Keep the existing PLAN.md direction: trigger on transitions, use hard rate limits, quiet hours, cooldowns, and prefer non-verbal reactions.

Add an explicit decision:

```text
SPEAK | REACT_NONVERBALLY | STAY_SILENT
```

Proactive evaluation must measure not only missed useful interventions but also **nuisance rate**: times Vess spoke when silence would have been better.

Start conservative. A proactive assistant should earn permission to interrupt more often through evidence, not start chatty and hope the user develops Stockholm syndrome.

### 10. Conditional planner

Do not route ordinary conversation through a planner.

A planner should activate only when a request has properties such as:

- multiple dependent actions
- uncertain intermediate state
- several possible tools/routes
- need to recover from failure
- explicit goal requiring verification

Preferred execution style:

```text
Goal
 -> choose next step
 -> execute one bounded action
 -> observe result
 -> verify postcondition
 -> revise/continue/stop
```

Avoid generating a giant fixed plan and blindly executing every step. Tool environments change and long-horizon errors compound.

Planner rules:

- bounded steps and retries
- explicit stop conditions
- detect impossible/blocked tasks
- do not fabricate success
- skip planning for trivial direct actions
- re-plan from observed results, not imagined results

### 11. Resource telemetry and backpressure

Collect before controlling.

Track at minimum:

- CPU utilization
- process RAM
- GPU utilization when available
- VRAM use when available
- Ollama throughput / first-token latency
- TTS synthesis latency
- transcription latency
- perception queue depth / dropped work
- render/display frame timing

Then add simple backpressure rules if measurements show contention, such as:

- skip stale vision frames rather than queueing them
- reduce optional detector/vision frequency while interaction is latency-critical
- postpone background consolidation while actively speaking/listening
- preserve face/render responsiveness
- reject or delay low-priority background work when queues are saturated

Only after this is proven insufficient should Vess gain user-facing modes like Low/Normal/High/Auto.

### 12. Procedural learning / skills, later

Do not continuously fine-tune local neural networks as the first form of learning.

A much cheaper and more controllable progression is:

```text
observe -> store -> retrieve -> adapt policy -> save verified reusable skill
```

A procedural skill should come from a successful, verified action trace and remain inspectable/editable. Do not let one lucky or unsafe trajectory become permanent procedure automatically.

Model fine-tuning or learned attention/resource policies can be revisited only when measured failures justify them.

---

## Cross-Cutting Architecture

The future architecture should look conceptually like this while remaining asynchronous:

```text
cheap perception producers
        |
        v
raw events ------------------------------+
        |                                 |
        v                                 v
state / situational belief update     event log / trace
        |
        v
attention + cognition router
        |
        +---- ignore / state only / remember candidate
        |
        +---- non-verbal reaction
        |
        +---- direct response
        |
        `---- complex-task path
                 |
                 v
              planner
                 |
                 v
              executor
                 |
                 v
             verifier
                 |
                 +---- success
                 `---- retry / revise / stop / report uncertainty
```

Memory and context assembly feed cognition when relevant. Persona/mood/performance affect expression after the system has decided what it believes and what it intends to do.

The face/render loop remains independent throughout.

---

## Confidence, Provenance, and Authority

This should become a shared convention across memory, situational beliefs, perception, and actions.

Do not treat these statements as equivalent:

- user explicitly said X
- camera detector observed something consistent with X
- Vess inferred X
- an old memory claims X
- a tool/API reported X

Where a decision matters, retain enough metadata to tell the difference.

Useful conceptual fields:

```text
value
source_type
source_id / evidence reference
confidence
observed_at
last_confirmed_at
expires_at
supersedes
```

Not every runtime field needs all metadata. Use it where uncertainty or future reuse matters.

---

## Evaluation Roadmap

Do not use a vague overall "Vess score" as the main measure. Track subsystem and end-to-end behavior.

### Conversation

- wake-word false accept / false reject
- speech-end -> transcript latency p50/p95
- transcript -> first LLM clause p50/p95
- transcript -> first audible sample p50/p95
- clause-to-clause audible gap
- barge-in detection and interruption latency
- incorrect self-interruption rate

### Memory

- relevant-memory recall
- irrelevant-memory injection rate
- stale/invalidated-memory retrieval rate
- contradiction resolution accuracy
- source/authority preservation
- temporal questions / updated-fact handling
- retrieval latency

### Attention and proactivity

- useful proactive intervention precision
- missed high-value events
- nuisance/over-interruption rate
- repeated-trigger suppression
- silent/non-verbal choice quality
- compute spent per useful intervention

### Planning/actions

- direct task success
- verified postcondition success
- unintended side-effect rate
- correct detection of impossible tasks
- recovery success after tool failure
- unnecessary planner invocation rate
- average steps/retries per task

### Reliability

- crashes per hour/day
- successful multi-hour/day soak sessions
- thread/queue stalls
- memory/resource growth over time
- restart/recovery correctness
- device/model failure degradation

### Performance

- CPU/RAM/GPU/VRAM over time
- LLM tokens/sec
- TTS real-time factor and first-clause latency
- frame timing
- dropped/stale perception work
- interaction latency under simultaneous workloads

Use distributions (especially p50/p95), not only averages.

---

## Ideas to Avoid or Delay

Unless later evidence changes the tradeoff, do **not** prioritize:

- one giant synchronous cognitive loop
- a huge simulated "world model"
- running a planner on every utterance
- a separate LLM call for attention when deterministic rules work
- several independent memory services before a typed store is proven insufficient
- knowledge graphs merely because they sound sophisticated
- continuous self-reflection loops after every action
- multi-agent architecture by default
- continuous local neural-network training
- arbitrary target FPS/quality numbers without profiling
- an automatic performance manager before telemetry/backpressure exists
- proactive speech without abstention metrics and hard limits
- storing model inferences as trusted user facts without provenance
- replay systems that reissue irreversible side effects

---

## Research Notes Behind This Roadmap

This roadmap was reviewed against recent agent/memory/reliability work rather than relying only on a generic recommendation list.

Useful findings included:

- **Mem0 (2025)**: selective persistent memory can improve long-session retrieval while avoiding full-history context cost.
- **Memory for Autonomous LLM Agents survey (2026)**: memory is better treated as a write/manage/read control loop; contradiction handling, consolidation, forgetting, latency, and privacy are first-class engineering concerns.
- **Memora / forgetting-aware memory evaluation (2026)**: long-term agents frequently reuse obsolete memories, so forgetting/invalidation must be evaluated rather than only recall.
- **Authority-collapse memory work (2026)**: memory consolidation can erase who/what authorized a claim, turning weak evidence into overly authoritative future behavior. Preserve source authority.
- **Recent long-horizon planning benchmarks (2026)**: performance collapses as tool environments become blocked/noisy and task horizons grow; planning should be iterative, bounded, and feedback-driven rather than assumed reliable.
- **Runtime agent safety benchmarks (2026)**: agents can complete a task while still creating unsafe/unwanted side effects, so completion and side-effect verification must be distinct.
- **Async voice-agent work (2025)**: decoupled voice/reasoning architectures support barge-in and user steering better than monolithic voice flows, matching Vess's existing asynchronous direction.
- **Proactive-assistant/HCI work (2026)**: whether and when to intervene is itself a prediction problem. Vess should explicitly evaluate abstention and interruption cost.

These sources guide architecture priorities; they are not instructions to copy any one framework.

---

## Current Priority Snapshot

At the time this roadmap was written:

```text
NOW
  target-PC TTS acceptance
  natural interruption / barge-in
  reliability + observability

NEXT
  reliable action executor + verification
  structured memory
  deterministic context compiler
  situational belief state
  attention/cognition routing

LATER
  proactive speech
  conditional planner
  measured resource adaptation
  verified procedural skills

MUCH LATER / ONLY IF JUSTIFIED
  learned attention/resource policies
  model fine-tuning / continual neural learning
  more elaborate relational/graph memory
```

The order can change when measurements or real use reveal a stronger bottleneck. Features should move forward because Vess has a demonstrated need, not because an architecture diagram has an empty box.