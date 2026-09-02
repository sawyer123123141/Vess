# Closed Command Registry Design

## Goal

Implement Step 6 from `PLAN.md` as a working closed command path where a spoken request such as `turn blue` can change Vess's color without allowing the model to compose shell commands, code, paths, or arbitrary actions.

## Scope

This pass ships one executable capability: `set_color`.

The broader command names shown in `PLAN.md` remain future capabilities until each has a concrete safe handler. They are not exposed to the model merely because they are planned. A closed registry is only useful if every advertised entry is actually bounded and executable.

## Architecture

`brain/commands.py` owns the command boundary. It contains the registry definition, command-call validation, deterministic candidate gating, execution, and a public JSON-safe catalog. The registry mutates `State` only through its lock and never invokes shell text supplied by the model.

`OllamaClient` gains a narrow non-streaming command-selection call. It receives only the currently executable registry catalog and must return JSON for one command call or `null`. Python validates the returned object after generation; model compliance is not a security boundary.

`ConversationWorker` checks the cheap Python candidate gate before building a normal response. Only command-like requests pay the selector call. If selection is invalid, malformed, stale, or returns `null`, the request falls through to the existing conversational path. Ordinary questions therefore keep the current latency path unchanged.

## Initial command

### `set_color`

Input schema:

```json
{"name":"set_color","arguments":{"name":"blue"}}
```

Allowed names come from `config.json` under `commands.colors`. Values are fixed RGB triples written by a human. The first palette is deliberately small:

- blue
- red
- green
- purple
- orange
- white

Execution sets `state.color` to the configured RGB tuple. The spoken result is deterministic and short (`Blue.` for the example) so a command does not require a second generative response.

## Candidate gating

The selector is considered only when the utterance contains both:

1. an imperative/control verb such as `turn`, `set`, `make`, or `change`; and
2. one currently registered color name.

This gate is intentionally permissive enough that the model still decides whether the user is commanding Vess. For example, `turn blue into RGB` may reach the selector, but the selector is instructed to return `null` unless the user is asking Vess itself to perform the action.

A normal query such as `why is the sky blue?` never invokes command selection.

## Validation and safety

Python rejects a selection unless all of these are true:

- top level is an object with exactly `name` and `arguments`;
- `name` exists in the executable registry;
- `arguments` is an object;
- argument names match the declared schema exactly;
- every value is from the declared allowlist;
- no extra arguments are present.

Unknown commands, extra fields, arbitrary RGB arrays, paths, shell text, or malformed JSON are never executed.

The model never receives a shell-capable interface. `set_color` only maps an allowlisted name to a human-authored RGB tuple.

## Generation safety

Command selection is a blocking local-model call on the existing conversation worker. A newer user request can supersede the generation while selection is in progress. The worker must re-check generation freshness after selection and before execution. A stale selection is discarded without changing state.

Execution happens at most once for one accepted current generation.

## Delivery and memory

A successful command uses the existing voice generation and physical-delivery accounting. The deterministic acknowledgement is registered with `DeliveryLedger`, queued through `VoiceOutput`, and the generation is finished through the existing lifecycle.

This preserves the project's rule that generated text is not remembered as heard until playback receipts arrive. The user's command can therefore appear in short-term conversation history only through the same delivered-turn path as normal responses.

Durable fact extraction still receives only user text after delivered turns. `turn blue` should naturally yield no durable fact.

## Events and diagnostics

Successful execution appends `command_executed` with the validated command name and arguments, and records the same action in local debug history. Invalid model output records a debug-only rejection and falls back to chat rather than breaking the request.

## Web control surface

The registry exposes a JSON-safe catalog so the local web control surface can consume the same definitions rather than duplicating command names or palette values. This pass adds registry access to the web app and a command execution endpoint using the same validator/executor. The existing arbitrary RGB color picker remains available because it is a direct local operator control, not model-selected capability.

## Failure behavior

- selector network/model error: record debug error and fall back to normal conversation;
- malformed selector JSON: no command, fall back to conversation;
- invalid/unknown command: no execution, fall back to conversation;
- stale generation after selector: return without execution;
- command execution error: log `command_error`, do not silently run a different action.

## Testing

Tests are written before production code and cover:

- registry catalog contains only executable commands;
- `set_color` allowlist validation and state mutation;
- arbitrary/unknown/extra arguments are rejected;
- candidate gate skips normal blue-related questions;
- Ollama selector prompt contains only registry capabilities and malformed output returns no call;
- ordinary conversation does not invoke selector;
- valid command executes and speaks a deterministic acknowledgement;
- stale command selection never mutates state;
- invalid selector output falls back to normal conversation;
- web command catalog/execution uses the same registry;
- runtime wiring shares one registry object between conversation and web server where applicable.

No shell execution, OS app control, timers, media keys, or new dependency is part of this pass.
