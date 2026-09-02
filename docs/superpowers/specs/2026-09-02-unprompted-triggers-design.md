# Unprompted Triggers Design

## Goal

Give Vess a conservative first ability to speak without being prompted, while keeping the decision to interrupt entirely in deterministic code and keeping proactive observations out of user-authored memory.

## Scope

The first version has exactly two trigger types:

1. `returned_after_absence` — Vess observed a live person leave, stay absent long enough, then return.
2. `quiet_interaction` — the person is visibly present but has not interacted with Vess for a configured interval.

No active-window/typing/call suppression is implemented yet because `perception/desktop.py` does not exist. No probabilistic scoring system, trigger-learning model, recurring nag loop, or generic event framework is added.

## Trust boundary for presence

Presence-based proactive speech is enabled only when `config.camera.source == "camera"`.

`image` and `video` are developer/test sources. They may exercise YOLO and `State.person_present`, but they are not evidence that a person is physically in the room now. Runtime proactive speech therefore stays disabled for those sources. This is especially important on the current machine because the repository uses `camera.source = "image"` until the real webcam arrives.

Tests may construct state snapshots directly and enable trusted presence explicitly, so the complete trigger behavior is CI-testable before hardware arrives.

## Trigger configuration

Extend `config.triggers` with:

```json
{
  "min_absence_hours": 4,
  "idle_interaction_minutes": 30,
  "cooldown_minutes": 60,
  "quiet_after_hour": 22,
  "quiet_before_hour": 8
}
```

Existing values stay authoritative. `idle_interaction_minutes` means time since meaningful Vess interaction, not acoustic silence in the room.

## User-activity clock

Add `State.last_interaction: float = 0.0` as the one shared timestamp for accepted user interaction.

Ordinary `ConversationWorker.submit(...)` / `submit_with_timing(...)` update this timestamp as soon as a real user request is accepted for handling, including an empty wake acknowledgement. Duplicate submissions still count as interaction even if they collapse to the already-active request. Proactive submissions never update it.

This field is intentionally separate from `state.last_spoke`. `last_spoke` means Vess physically produced audible output, including proactive output, so using it to reset the quiet trigger would allow Vess to restart its own idle timer and eventually talk again without the person ever responding.

## Trigger decision model

`brain/triggers.py` owns the decision. It has no LLM dependency.

A lightweight worker reads a lock-consistent state snapshot at a low fixed cadence (about 2 Hz). It maintains only local bookkeeping needed to recognize transitions and one-shot idle stretches.

Every trigger must pass all gates:

- live presence is trusted
- a person is currently present
- current local hour is outside configured quiet hours
- `state.muted_until <= now`
- Vess is not listening, thinking, or speaking
- the global unprompted cooldown has elapsed
- there is no active or pending conversational work when submission is attempted

The LLM never decides whether any of those gates pass.

### Return after absence

The worker must actually observe `person_present: true -> false` and remember when the absence began. Starting Vess while nobody is present does not create a synthetic multi-hour absence.

When it later observes `false -> true`, it fires only if the observed absence lasted at least `min_absence_hours`.

The event context includes only grounded timing information, for example:

```text
A person just returned after being absent for about 4 hours.
```

It must not claim to know where the person went or what they did.

### Quiet interaction

"Quiet" means no interaction with Vess, not no sound in the room.

The idle baseline is the newer of:

- `state.last_interaction`
- `state.present_since`

That means a newly detected person gets a fresh idle window rather than an immediate quiet trigger, and a user request resets the window immediately rather than waiting for the response to finish playing.

After `idle_interaction_minutes`, the trigger may fire once for that continuous no-interaction stretch. It does not repeat every N minutes.

More conservatively, once *any* proactive line has been accepted during the current no-interaction stretch — whether it came from the return trigger or quiet trigger — Vess stays proactively silent until either:

- the user interacts again (`state.last_interaction` advances), or
- the person genuinely leaves and a new presence stretch begins.

This prevents a return greeting followed an hour later by a quiet-time remark when the person never responded.

The event context is intentionally plain, for example:

```text
The room has been quiet between Vess and the person for about 30 minutes.
```

## Proactive conversation path

Do not call ordinary `ConversationWorker.submit()` with fake user text.

Add a dedicated proactive submission path that reuses the existing generation, cancellation, streaming, TTS, and physical-delivery machinery while preserving provenance.

A proactive request has:

- a generation ID
- kind `proactive`
- trigger name
- grounded event context

Submission is accepted only if there is no active or pending user/proactive request at the commit point. A real user request arriving afterward may supersede the proactive generation through the existing freshness rules.

### Proactive prompt

The proactive prompt reuses Vess identity, persona, mood, current room state, performance tags, durable facts when relevant, and recent delivered history. It ends with a clearly labeled system observation instead of `Current request`.

It instructs the model to:

- produce at most one short sentence
- make a natural observation or greeting
- avoid a generic question
- never mention trigger logic, timers, gates, or implementation
- not infer where the person was or what they were doing

The model chooses wording only. It cannot cause a trigger to fire.

## Delivery and memory provenance

Proactive speech must use the same physical-delivery ledger as ordinary speech so generated text is not remembered unless it was actually played.

For proactive generations the ledger begins with an empty user string. Finalization may store an assistant-only short-term turn, but prompt rendering must omit a blank `User:` line.

Because the user string is empty, durable fact extraction must not run. Event-log entries should identify proactive trigger submission/firing separately so operator diagnostics can distinguish it from user speech.

This preserves the rule that only actual user statements may become durable user facts.

## Cooldown ownership

The trigger worker tracks the most recent successfully accepted proactive trigger time. The global `cooldown_minutes` applies across both trigger types.

A trigger that fails to submit because conversation work appeared concurrently does not consume the cooldown or the one-proactive-per-idle-stretch latch. A successfully accepted proactive generation consumes both even if a newer user request later supersedes it; this avoids immediate retry loops.

## Runtime lifecycle

Create the trigger worker only when live presence is trusted.

Startup order:

1. build state, command registry, memory, voice/conversation runtime
2. start voice/conversation/audio/perception as already designed
3. start trigger worker after the conversation worker is available

Shutdown order must stop the trigger worker before closing conversation and voice, so it cannot submit into a draining runtime.

With `camera.source` set to `image` or `video`, no trigger thread is started and runtime prints a concise reason that proactive speech is disabled until a live camera source is configured.

## Observability

Record debug/event-log entries for:

- trigger eligible/fired with trigger name and grounded duration
- trigger suppressed by cooldown/quiet-hours/mute/busy when useful for diagnostics without spamming every 500 ms poll
- proactive submission accepted/rejected due to concurrent conversation
- proactive generation lifecycle continues through existing LLM/TTS debug events

Do not log a suppression event on every poll; only transitions or attempted firings should produce history.

## Testing

Tests-first coverage must prove:

- static image/video runtime never enables proactive speech
- startup while absent does not manufacture an absence duration
- observed leave/return below threshold does nothing
- observed leave/return above threshold produces exactly one return event
- ordinary user submission updates `state.last_interaction`; proactive submission does not
- quiet interaction fires once per no-interaction stretch and resets only after user interaction or a new presence stretch
- a successful return trigger prevents a later quiet trigger until the user interacts or leaves
- quiet interaction never fires while person is absent
- quiet hours, mute, busy state, and global cooldown suppress both trigger types
- a failed proactive submission does not consume cooldown or the one-shot latch
- a successful proactive submission consumes both
- proactive submission never replaces an already active/pending user request
- a later real user request can supersede proactive generation
- proactive prompt labels event context as observation, not user speech
- proactive delivered memory contains no fake user line and queues no durable fact extraction
- trigger worker shuts down before conversation/voice
- existing ordinary conversation, command, delivery, and latency tests stay green

## Deferred work

- active-window / typing / call suppression after a trustworthy desktop-perception producer exists
- `not now` / explicit mute command behavior beyond the existing `muted_until` gate
- trigger-repeat decay across days
- probabilities or scoring systems
- additional environmental triggers
- acoustic-room-silence detection
- hardware validation of actual leave/return behavior after the webcam arrives
