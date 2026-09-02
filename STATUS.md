# Status

Update this at the end of every session. Newest at the top.

## Stale TTS preemption — cancel obsolete Chatterbox generation

Real-PC rapid-follow-up telemetry exposed a concrete head-of-line delay: a new
response measured **700.6 ms** of `tts_worker_wait_ms` while an obsolete
Chatterbox generation was still synthesizing. The current generation's own
first synthesis then took **627.6 ms**. Debug events showed generation 6 being
skipped only `after_synthesis`/`queued`, proving that generation invalidation
could discard stale results but could not stop the synchronous Chatterbox
`generate(text)` already occupying the single synthesis worker.

The same microphone experiment also rejected `audio.silence_seconds = 0.30`
for normal use: a natural hesitation was finalized before the speaker could get
the intended thought out. The repository remains at **0.45**; do not treat the
~150 ms endpoint reduction from 0.30 as a free latency win.

### Change

`VoiceOutput` now passes a generation-staleness callback only to TTS engines
that expose an optional `synthesize_cancellable(...)` entry point. Ordinary
engines and the legacy synthesis callable retain their existing blocking
contract, so their worker-wait telemetry remains truthful.

`ChatterboxTurboEngine` implements that optional path without loading a second
model or running concurrent GPU synthesis. It checks cancellation before work,
at each safe transformer forward boundary during Turbo speech-token generation,
before the S3Gen flow/vocoder stages, and once more after generation. A
`SynthesisCancelled` exception unwinds obsolete work back to the existing
serial worker. `VoiceOutput` records this as `stale_tts_skipped` with stage
`during_synthesis`; cooperative cancellation is not a `voice_error`.

This is cooperative rather than magical GPU preemption. A CUDA kernel already
executing cannot be interrupted halfway through by Python. The obsolete turn
stops at the next checked Python/module boundary, so the real reduction in
`tts_worker_wait_ms` still requires hardware measurement.

### Verification

Tests were written before production code. Commit
`402408f143d8c8aa366e7d3631eb2c33ab0b1f53` added two regressions: a newer
generation must start without waiting for an obsolete cancellable synthesis,
and the Chatterbox adapter must abort inside its token loop and remove its
cancellation hook. GitHub Actions run **107** ran **204 tests** and failed only
those two new expectations.

Production commit `e0e5134380b583115c5a971475d3e9eb9712247c` added the
cooperative cancellation path. GitHub Actions run **108** then ran **204/204
unit tests** successfully, followed by successful behavior verification and
comprehensive eye validation. The older regression that intentionally uses an
uncancellable/legacy synthesis call still passes, confirming worker-wait
telemetry was not weakened to manufacture a green result.

A production diff review found only `output/tts/base.py` (+4),
`output/tts/chatterbox_turbo.py` (+61/-2), and `output/voice.py` (+15/-2).
There are no config, delivery-ledger, playback, memory, or queue-policy changes,
and the physical `on_delivery` lifecycle boundary remains untouched.

### Next hardware check

Pull the feature branch, ensure local `audio.silence_seconds` is back at
**0.45**, restart Vess, and create several rapid follow-ups while an earlier
response is still being synthesized. A successful preemption should produce
`stale_tts_skipped` with `stage = "during_synthesis"` and materially reduce the
new generation's `tts_worker_wait_ms` from the measured **700.6 ms** case. Do
not assume a particular residual delay because it depends on which Chatterbox
stage or CUDA kernel was active when the generation changed.

## First-clause TTS latency pass — early balanced comma

Real-PC telemetry with `audio.silence_seconds` temporarily set to **0.30**
locally measured **320.7 ms** endpoint wait, **235.3 ms** Whisper execution,
**422.7 ms** to the first LLM clause, **106.5 ms** TTS worker wait, and
**1590.8 ms** first-clause TTS synthesis. Total last-speech to playback was
**2678.1 ms**. A later 36-character clause synthesized in **698.6 ms**, while
the inter-clause playback gap stayed at **0.5 ms**. That isolates the dominant
latency in this sample to the size of the first Chatterbox synthesis request,
not the already-pipelined later clauses.

The slow first sentence was `Imagine floating together in a starry expanse,
where our thoughts are the only light.` Its comma lands at character 45, but
the old splitter only considered comma boundaries after the buffer reached
**120 characters**. The full 85-character sentence therefore went to
Chatterbox in one synchronous `generate(text)` call. The current Chatterbox
Turbo adapter exposes whole-chunk generation rather than incremental audio, so
there is no native token-streaming switch to enable in Vess's existing path.

### Change

Only the first spoken chunk gets a lower-latency comma option. A comma between
characters **40 and 59** can become the first TTS boundary if it occurs before
strong punctuation. After the first chunk is emitted, the existing 120/180
character clause policy is unchanged. Very short introductory phrases such as
`First, then second.` remain intact, and a comma split keeps the same
sentence-level performance cue because comma is not a strong performance
boundary.

This is deliberately narrower than globally lowering the clause-size limits.
The goal is to get useful first audio moving without turning every later
sentence into a chain of tiny TTS jobs. Physical prosody still needs a real
speaker check: CI can prove which text chunks are produced, but it cannot prove
that the early comma sounds natural.

### Verification

Tests were written before production code. Commit
`5656eecb14d0b1a393246d2d60ed8c46b0f38868` added the measured slow sentence
plus a short-comma guard; GitHub Actions run **104** failed the unit-test job on
the old splitter and skipped behavior preview as expected. Production commit
`9ccf757bca9a0ed91be62bee48146745b6468e65` then implemented only the
first-chunk rule. Run **105** passed the full unit-test job, behavior
verification, and comprehensive eye validation.

A production-commit diff review found only `brain/llm.py`, with **14 additions
and 2 deletions**. No TTS engine, delivery, memory, config, or later-clause
policy changed in that commit.

The repository still keeps `audio.silence_seconds` at **0.45**. The measured
0.30 value is a local hardware experiment and should not become the repository
default until natural mid-sentence pauses are confirmed not to split.

### Next hardware check

Pull this branch, keep the local 0.30 endpoint experiment if desired, restart
Vess, and exercise several normal prompts. Compare `tts_first_synthesis_ms`,
`tts_first_audio_ms`, and `speech_end_to_playback_ms` on responses whose first
sentence contains a natural comma near the new window. Also listen for an
unnatural pause or prosody reset at that comma. Keep the heuristic only if the
real first-audio reduction is meaningful and the spoken result remains natural.

## Latency telemetry follow-up — beam 5 and TTS worker wait

Real-PC measurement on the RTX 3070 showed that Whisper `small` with CUDA and
`int8_float16` was fast enough at beam 5 and that beam 1 was not an acceptable
accuracy trade. Changing only `beam_size` from 1 to 5 corrected the measured
transcript to `Say, plants turn sunlight into energy.` in **393.1 ms** at
**0.14 RTF**. The repository and `_make_transcriber()` fallback now therefore
use **beam 5**. This commit deliberately does **not** also change the repository
`device` or `compute_type`; those remain separate variables rather than being
silently bundled into the beam decision.

The same hardware run measured a warm LLM first clause at **184.5 ms**, first
TTS synthesis at **595.2 ms**, and an approximately **470 ms** endpoint wait.
A long second TTS clause took **1908.3 ms**, but because it synthesized during
playback the audible inter-clause gap was only **69.6 ms**. Rapid follow-ups
then exposed the different problem that matters here: a stale synthesis already
executing in the serial TTS worker can hold the newer generation behind it.

### New telemetry

- `audio_blocks_dropped` is published as **0** from `AudioLoop` construction,
  instead of appearing as `null` until the first detected capture gap.
- `tts_worker_wait_ms` is generation-scoped and measures **clause enqueue to
  actual synthesis start**. That definition is intentional: it includes time a
  new clause waits in the TTS queue behind an obsolete synthesis, plus any
  one-ahead prepared-audio backpressure before the engine can start.
- `tts_first_synthesis_ms` is the engine execution time for the **first
  successfully synthesized spoken clause** of the current generation. It is
  frozen once reported, while the older `tts_synthesis_ms` remains clause-level
  playback telemetry and can therefore show a later clause.
- Both new first-clause values reset with the existing generation-scoped latency
  bundle. Delayed timing from an obsolete generation is ignored.
- Existing `stale_tts_skipped` events remain separate from these durations.
  Worker wait tells us how long the current generation was blocked; stale-skip
  events tell us whether obsolete work was involved. Synthesis time is not
  inflated to hide either condition.

Synthesis timing now has its own callback from `VoiceOutput` to
`ConversationWorker`. It is **not** a physical-delivery receipt. The first green
attempt incorrectly sent `clause_synthesized` through `on_delivery`; an existing
exact lifecycle test rejected that because delivery accounting is deliberately
limited to playback events. The implementation was corrected rather than
weakening that test. The dedicated timing callback is generation-filtered by
`ConversationWorker`, preserving the same atomic debug-bundle ownership as the
rest of the latency telemetry.

### Verification

The initial tests-only commit ran **200 tests** and failed only the four new
expectations, proving beam 5, zero-drop initialization, first-TTS telemetry, and
head-of-line worker wait were not already present. The first implementation
made the new tests pass but failed one existing physical-delivery lifecycle
test, which exposed the callback-boundary mistake above. A second tests-first
red run then failed exactly on the missing dedicated synthesis-timing API and
the still-polluted delivery sequence.

After the boundary fix, GitHub Actions run 102 passed the full **200/200 unit
tests**, `tools/render_behavior_preview.py`, and the comprehensive eye
validation. A base-to-head review from `505951f3229a7d3cc39e7bf8f78e7f01c6bbbc2e`
confirmed that the only config behavior changed in this pass is Whisper beam
1 -> 5; `audio.silence_seconds` is still **0.45**, and the repository Whisper
`device`/`compute_type` defaults are unchanged.

### Next latency experiment

Do **not** lower `audio.silence_seconds` in the telemetry commit. The next
behavioral pass should change only **0.45 -> 0.30** on the real microphone and
exercise ordinary sentences containing natural mid-sentence pauses, short
hesitations, and sentence endings. Compare endpoint wait and transcription
boundaries against the 0.45 baseline, and keep 0.30 only if it reduces endpoint
latency without creating premature utterance splits. CI can verify segmentation
mechanics, but it cannot determine whether a human pause sounds natural enough
not to be cut off.

## Generation-scoped conversational latency telemetry

The audio path now distinguishes the last sample VAD considered voiced from
the later moment when the configured silence window finalizes the utterance.
It derives the last-voiced timestamp from the captured block's monotonic
arrival time and the sample offset within that block instead of stamping
"speech ended" after endpointing has already finished.

`GET /debug` now adds these live values for accepted spoken requests:

- `latency_generation_id` — response generation owning every displayed latency
- `latency_timing_valid` — false when a capture gap makes endpoint timing unsafe
- `endpoint_wait_ms` — last voiced sample to utterance finalization
- `transcription_queue_ms` — finalization to Whisper actually starting
- `transcription_ms` — Whisper execution
- `tts_first_audio_ms` — first LLM clause ready to pre-delivery playback marker
- `speech_end_to_playback_ms` — last voiced sample to that same marker

The complete timing bundle follows the accepted request through ordinary and
barge-in paths via an explicit timed-submit interface; the original one-argument
request callback remains compatible. All public latency fields are reset and
published together for one generation. Rejected or empty transcripts therefore
cannot overwrite Whisper values while leaving LLM/TTS values from an older
accepted turn, and delayed receipts from obsolete or cancelled responses cannot
update the current bundle.

If the bounded microphone queue overflows, `audio_blocks_dropped` increments and
the active utterance's sample-derived endpoint is invalidated instead of
reporting false precision. No VAD threshold, silence duration, Whisper setting,
clause split, or interruption policy changed.

`tts_first_audio_ms` and `speech_end_to_playback_ms` currently end at the
the marker immediately before synchronous delivery/debug callbacks. They are
not player-call or DAC timestamps; `latency_playback_marker` reports
`pre_delivery_callback` to make that boundary explicit. The existing
leading-silence telemetry remains separate,
so hardware results must not claim sample-accurate physical onset from these
values. Real-PC measurement is still required before choosing an optimization.

## Voice diagnostics console added

The browser preview now includes a compact Diagnostics panel, refreshed every
0.5 seconds from `GET /debug`. It reports mic peak, VAD activity/buffered
duration, whether audio is being ignored during speech, and the existing
listening/thinking/speaking flags. It retains the latest 20 local-only worker
events: transcript and wake decision, Ollama request/first clause/completion,
Kokoro preparation/playback, and errors. This is an in-memory operator aid;
it does not query or extend the Step 5 memory database.

Live check: the console showed microphone input and Whisper transcribed
`Can you, can you help me and give me something, brother?`; it was correctly
recorded as a rejected wake because it did not begin with a configured
variant. Current idle mic peaks were about **0.008**, below the configured
VAD threshold of **0.015**, so quiet speech may need a lower threshold after
testing a deliberate “Hey Vess” request.

Follow-up: the first console build emitted literal newlines into JavaScript
string literals, which stopped both browser polling loops. The page now emits
escaped `\\n` separators; a regression test checks the served source. After
restart/reload, the in-app browser showed a live face `blob:` URL and populated
diagnostics with no new console error.

## Step 4 done — local threaded voice loop and append-only event history

Vess now runs the local path `microphone -> energy VAD -> CPU Whisper ->
fuzzy wake gate -> Ollama -> Kokoro` without moving model, audio, database,
or web work into the 30fps face loop. The microphone callback only queues
16kHz mono blocks; a background audio worker segments utterances and sets
`state.listening` only during transcription. It discards input while Vess is
speaking, so interruption handling remains intentionally out of scope.

### Voice behavior

- Wake matching normalises punctuation/case and checks the first one, two, or
  three words against configurable variants using Levenshtein distance. The
  initial variants are `hey vess`, `hey best`, `hey guess`, and `heaviest`;
  the threshold is 2 and all of it is in `config.json` for microphone tuning.
- Every rejected transcript is retained as `wake_rejected` with its raw text,
  tested prefix, closest variant, and edit distance. Accepted wakes receive
  the same metadata and dispatch only the text after the matched prefix.
- Empty accepted requests play the cached `Yeah?` acknowledgement. Other
  requests use a single local Ollama worker, stream completed clauses to a
  serial CPU Kokoro worker, and keep the face thinking until the first clause
  is queued.
- A valid classifier result changes `state.mood`, gives it that mood's decay,
  and appends a `mood_changed` record. The main loop calls
  `State.expire_mood()` each tick, which atomically makes expired state truly
  neutral and logs the reverse transition.

### Event history

`brain/memory.py` provides only the deliberately minimal, append-only
`events(timestamp, event_type, payload_json)` SQLite table. Writes have a
dedicated background owner. A session start, wake accept/reject, both mood
directions, and browser colour override set/reset are recorded. Querying,
facts, retrieval, and summarisation remain Step 5 work.

### Verification

`python -m unittest discover -s tests -v` passes **22 tests**, covering VAD
boundaries, fuzzy acceptance/rejection, queued SQLite persistence, clause
streaming, serial speech, event instrumentation, colour history, and state
expiry. `python -m compileall -q brain control output perception main.py
state.py` exits cleanly.

A bounded live `python main.py` run started the configured image detector,
opened the browser preview (real `GET /frame.png` returned HTTP 200), loaded
Kokoro, and wrote session/rejected-wake records. The production `OllamaClient`
also streamed `Vess local model ready.` from the installed local `qwen2.5:7b`
model. The exact USB microphone, a spoken accepted request, and audible
speaker playback still need a physical in-room check; the event log will show
which wake variants Whisper actually needs.

### Known limitations, intentionally deferred

- Step 4 transcribes every segmented utterance with Whisper on CPU, giving
  continuous CPU load and roughly one second of wake latency. A dedicated
  local wake-word engine such as openWakeWord is a later separate step, not an
  addition to this loop.
- There is no vision-model call at wake time: the current camera path has no
  safe shared latest-frame handoff, and another GPU model conflicts with the
  VRAM constraint. This remains an open later design task.

## Step 3 done — browser config UI with a live native preview

`control/web.py` adds the local browser target. It joins the existing
`Display` fan-out, so every 64x64 render can go to the browser, cv2, and later
the LED panel at the same time. The browser target is enabled by default;
set `display.cv2_enabled` to `true` for the existing cv2 fallback alongside
it. The web server and browser launch run in their own threads, and
`WebPreview.show()` only copies a frame under a lock — encoding and HTTP work
never enter the 30fps render loop.

### Browser UI

- `http://127.0.0.1:8080/` shows the exact 64x64 RGB output, upscaled with
  browser nearest-neighbour.
- It polls lossless `/frame.png` at **30fps**, not a reduced preview rate.
- The colour picker sends `PUT /color`; **Use mood colour** sends `DELETE
  /color` to clear the override explicitly.

`State.color` is now `None` by default, meaning use the active mood colour.
An explicit RGB tuple overrides only colour, persists through mood changes,
and uses the animator's existing 0.4s interpolation on both set and reset.
Mood eye shape, blink rate, and movement continue unchanged.

### Measured

90 real `GET /frame.png` requests at the intended 30fps cadence completed in
3.001s (**29.99fps**). The live face frame was **1,062 bytes** as PNG; mean
localhost request latency was **0.842ms**, p95 **0.988ms**, max **1.065ms**.
PNG polling has ample headroom, so no WebSocket transport is needed.

### Verification

`python -m unittest discover -s tests -v` passes all 8 tests, covering colour
override/reset, lossless frame delivery, the 30fps UI contract, local server
lifecycle, and browser-target fan-out. `python -m compileall -q control
output perception main.py state.py` also exits cleanly.

## Step 2 done — eyes track a person from a real image

`config.camera.source` is now `"image"` with `path` set to `"test.jpg"`, so
the application can exercise perception without a webcam. `test.jpg` is
Wikimedia Commons' *Example of person portrait.jpg* by Marin Bobek, used under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

### End-to-end verification

The configured `ImageSource` loaded and mirrored the photo, then downscaled it
to **512x342**. Real CPU YOLO inference returned four `person` detections;
`pick_subject` chose the largest and `write_state` set:

| state value | result |
|-------------|--------|
| `person_present` | `true` |
| `person_pos` | `(0.3778, 0.2507)` |
| `objects` | `["person"]` |

Feeding that exact state into `FaceAnimator` produced a pupil gaze of
`(-0.244, -0.399)` and an initial whole-face lean of `(-0.032, -0.035)` px.
The face is therefore following the real selected subject through the same
state path the running detector thread uses.

### CPU benchmark

After one warm-up inference, ten calls to `Detector.detect()` at the configured
512px maximum measured **24.2–36.8 ms/frame**, **27.1 ms mean** (25.6 ms
median), or **36.92 fps**. The configured 3 fps means inference occupies about
8.1% of each 333 ms detector interval, leaving the render loop independent in
its own thread as designed.

The webcam mirror-direction check remains a later hardware follow-up, not a
blocker for this image-based completion of step 2.

## YOLO weights downloaded and load verified

`yolo11n.pt` (5,613,764 bytes) was downloaded from the Ultralytics GitHub
release and loaded successfully with `ultralytics.YOLO`. It is located at
`C:\Users\sawye\Vess\yolo11n.pt`; Ultralytics saved the relative model name in
the repository working directory, rather than in the previously expected
`C:\Users\sawye\weights` folder.

This completes only the model-download prerequisite. `Detector.detect()` has
not been run against an image or camera yet.

## Step 2 implementation — before real-model verification

`perception/camera.py` and `perception/detector.py` exist and are wired into
`main.py`. The detector runs in its own thread; the render loop never waits on
it. **Nothing here has been run against the real YOLO model or a real camera**,
so step 2 is not done.

### Historical blockers

- **No room image or webcam.** The model is ready, but `Detector.detect()`
  still needs a real frame before the detection path can be checked.
- **No webcam.** On order. `config.camera.source` is `"camera"` and there is
  nothing at index 0, so `main.py` prints `perception off: no camera at index
  0` and carries on rendering. The `t` key still injects a fake `person_pos`,
  which is why it was kept rather than replaced.

### What was built

- **`camera.py`** — `FrameSource` with three implementations: `CameraSource`
  (live, index from config), `ImageSource` (one still, returned forever) and
  `VideoSource` (looped). Which one is used is `config.camera.source`, so
  switching to the real camera when it arrives is one line of JSON.
  `Camera` wraps a source and does the two per-frame jobs — mirror, then
  downscale to `max_frame_px`.
- **`detector.py`** — `Detection`, `pick_subject`, `Detector`, `write_state`,
  `run_detection_loop`. The detector is pure: frames in, detections out, no
  clock and no state. `write_state` folds one frame into `State`.
- **`main.py`** — starts the detector thread if a source and a model can be
  had, prints why not otherwise, and joins it on the way out.
- **`config.json`** — new `camera` block (`source`, `index`, `path`, `mirror`,
  `max_frame_px`), and `detector.confidence`.

### Mirroring

`camera.mirror` defaults to `true`. It is a horizontal flip of the frame in
`Camera.read()`, not a coordinate flip further down, so exactly one thing
decides which way round the room is. Moving to your left sends the eyes to
your left, which is what a face on a wall should do.

One consequence to remember: the tier 3 vision model will be handed the
mirrored frame, so any text in the room reads backwards to it. Nothing depends
on that yet. If it matters later, the fix is exposing the unflipped frame
alongside rather than moving the flip.

### Multiple people

`pick_subject(detections)` is the only place a subject is chosen. Largest box
for now. Everything that needs a subject calls it, so swapping in speaker
identification later is one function body changing. Nothing upstream of it
collapses the candidate list — the detector reports every person it sees and
the picker decides. The rule is written into `PLAN.md` under Eye movement.

### Verified without hardware or the model

| check | result |
|-------|--------|
| `pick_subject` | chose the larger of two people, ignored a bigger chair |
| gaze point | 0.365 down the box, against a 0.575 centre — head, not navel |
| mirror | left-side block lands right when mirrored, not when off |
| downscale | 1920x1080 -> 512x288 |
| `open_camera` | refuses missing file / unknown kind / absent camera, with reasons |
| video source | loops: frame means `[0, 77, 157, 0, 77, 157, 0]` |
| absence grace | survives a dropped frame; `present_since` not restarted by a blip |
| real absence | clears `person_present`, `person_pos`, `present_since` |
| return after absence | starts a fresh `present_since` |
| detection loop | 11 frames in 0.6s at 20fps, stopped cleanly on the event |

### Two judgement calls

- **Gaze targets 22% down the person's box**, not its centre. Box centre is
  the torso and the face would appear to stare at your chest.
- **Presence has a 1.5s grace period.** One dropped detection would otherwise
  make the face look away and snap back. `present_since` survives a blip and
  only restarts after a real absence.

Both are constants in `detector.py` with the reasoning beside them.

### Detector runs on the CPU, deliberately

`torch` here is `2.13.0+cpu`, `cuda False`. That is now written into
`PLAN.md`'s hardware section as a design decision rather than an accident: the
GPU belongs to Ollama, and overflowing the 8GB drops throughput ~30x with no
graceful degradation. If the detector is ever too slow the fix is lowering
`detector.fps` or `camera.max_frame_px`, never moving it to the GPU.

## Next task

**Step 4: voice loop** — audio -> Whisper -> Ollama -> Kokoro. When the
webcam arrives, separately switch `config.camera.source` back to `"camera"`
and confirm mirror direction in its installed position.

## Mood drives movement

Runtime state still picks *where* the face goes. Mood now scales *how* it gets
there, through an optional `movement` block per entry in `moods.json` —
multipliers against the constants in `animator.py`. The constants are the
physics; the multipliers are the character. A mood with no block moves exactly
as neutral, so nothing that already existed had to change.

Keys and limits: `hold` 0.25-3.0, `spread` 0.3-1.6, `ease` 0.3-3.0, `bob`
0.0-2.0, `track_bias` 0.25-2.0, `gaze_lag` 0.0-1.0, `gaze_y_bias` -0.6-0.6,
`track_break` 0.0-0.6. Multipliers default to 1.0, lags and biases to 0.0.

### Measured, 180s idle per mood

| mood    | snaps | face x range | gaze reach | mean gaze y | settle into thinking |
|---------|-------|--------------|------------|----------------------|
| neutral | 72    | 3.38px       | 1.00       | -0.06       | 1.10s                |
| happy   | 86    | 3.40px       | 1.00       | +0.02       | 0.77s                |
| sad     | 0     | 2.71px       | 0.80       | **+0.40**   | 2.13s                |
| annoyed | 150   | 2.36px       | 0.70       | +0.03       | 0.83s                |
| curious | 124   | 4.54px       | 1.00       | +0.13       | 1.00s                |

Annoyed is tense: twice neutral's snap rate in a third less space. Curious is
the opposite — a third more snaps across a third more room. Sad records zero
snaps because it is the one mood that *eases* its pupils rather than
snapping them, which is what reluctance turned out to mean in practice.

### Tracking a person at x=0.95

| mood    | gaze x reached |
|---------|----------------|
| curious | +1.00          |
| neutral | +0.90          |
| sad     | +0.54          |

Sad lands short and breaks off to its resting point on ~7% of frames, then
looks back. It never stops knowing where the person is — `track_bias` is
clamped so it can never reach zero, because a face that ignores you reads as
broken rather than as sad.

### The two careful bits

- **Multipliers ride in the same dict as the shape and colour**, so the one
  existing easing loop interpolates them for free. Measured across a
  neutral to sad transition: `move_ease` 1.00 -> 1.48 -> 1.74 -> 1.90 over
  0.4s. Timing never snaps while the shape morphs.
- **Clamped on the way in**, at the point a mood entry is read. Because an
  eased value is a blend of two already-clamped numbers, every intermediate
  frame is in range too. Fed a deliberately hostile block (`hold` 0, `bob`
  1000, `spread` 99, `track_bias` 0, `gaze_y_bias` -12) the face still
  renders, still moves, still tracks, and stays inside its edge budget.

### Also

- The bob's phase is now accumulated rather than computed from elapsed time.
  A mood that changes the bob period would otherwise jump the sine mid-breath.
- The edge budget adapts: the eased offset is capped at `_FACE_MAX` minus
  whatever the bob is currently using, so happy breathing 1.35x harder simply
  travels less. Verified across every mood plus the hostile one.
- Fixations pushed to the full +/-1.0. Biggest single-frame snap now moves the
  left pupil 5.6px, against ~1.7px before.
- `tick()` 0.22 ms/frame, unchanged in practice.

### PLAN.md — and a correction

This clone was three commits behind `origin/main`. The remote already had a
`PLAN.md` revision carrying the asymmetric geometry **and** an `Eye movement`
section — I reported that neither existed, having checked only the local
`git log` without fetching. They did exist. Merged now, resolved toward the
remote: its Face design and Eye movement sections are the owner's own text
and say more than mine did.

My additions are folded in under `### Eye movement` as three subsections:
whole-face movement, mood changes how it moves, and a **future** note that
with several people in frame gaze should target whoever is speaking. That
note also records the `pick_subject(detections)` rule for step 2, and that
direction-of-arrival needs a mic array the hardware list does not have.

### Two rules from the merged plan the code does not yet follow

- **"Nothing moves randomly."** Idle fixation choice and every hold duration
  come from `random.Random`. The plan wants each movement traceable to
  something in `State`, and idle gaze to "drift toward the last thing that was
  interesting" — which needs the world model that does not exist yet. Blink
  irregularity is explicitly sanctioned, so that use of the RNG is fine.
- **`speaking` has no glance behaviour.** The plan asks for mostly-on-person
  with occasional looks away, because unbroken eye contact is unsettling.
  `state.speaking` is currently unread by the animator.

## Whole-face movement

The eye pair now drifts around the canvas, separately from the pupils moving
inside the eyes. Both happen at once and at different scales.

**The two motion models are deliberately different.** The face eases
(exponential, max 0.088 px/frame measured); the pupils snap (1.24 gaze-units
in a single frame). Same tick, same frames, opposite feel. That contrast is
the point — matching them would flatten it.

Written into `PLAN.md` as a new `### Movement` subsection under Rendering.
There was no section called "Eye movement" to add it to.

### What drives it

| state             | offset                          | tau  |
|-------------------|---------------------------------|------|
| thinking          | (-1.6, -2.4) — up and away      | 0.5s |
| listening         | lean 2.4px toward + 0.6 down    | 0.5s |
| tracking a person | lean 2.0px toward               | 0.5s |
| idle              | wander 6 points, hold 4-9s      | 1.4s |

Priority is top to bottom. Vertical lean is damped to 0.55 of horizontal — a
person high in frame shouldn't lift the face as far as a person at the edge
moves it sideways. With no person, `listening` settles centred rather than
leaning nowhere.

Idle holds 4-9s against the pupils' 1-3s, so the two systems never look
synchronised.

### Bob

1.0px amplitude, 5.2s period, applied *outside* the easing so it stays a
clean sine instead of something the ease is forever chasing. Runs under
every state.

### Staying off the edges

The eased offset is capped at 2.5px and the bob adds at most 1.0px, so total
travel is 3.5px in any direction — enforced by construction rather than by a
final clamp. The pair spans x 16-48 and y 23-48 at rest, so worst case still
leaves a 12px margin.

`offset` is a float and stays one. A sub-pixel shift renders as a change in
edge brightness, which is what makes a one-pixel drift look like drifting
rather than jumping.

### New keys

`k` toggles thinking, `l` toggles listening. Nothing sets either field yet, so
without them three of the four documented behaviours can't be seen. `t` still
cycles the fake `person_pos`, and tracking now leans the face *and* points the
pupils in the same frame.

### Measured

- `tick()` still 0.20 ms/frame. The movement is a handful of scalar ops.
- Idle wander over 60s: x -1.79..+1.60, y -1.40..+0.60.

## Step 1 done — eyes that blink

`python main.py` opens a window showing the 64x64 face at 8x nearest-neighbour.
It blinks irregularly, wanders its gaze, and morphs between all five eye
shapes in `moods.json`.

**The two eyes are different.** Left is 16x22 with its top-left at (16, 26);
right is 12x16 at (36, 23) — smaller in both dimensions and sitting 6px
higher. Not a mirror, not a slant flip. Every shape in `EYE_SHAPES` defines
both eyes independently and none derives one from the other, so a mood can
move them by different amounts (`curious` raises the right eye 8px above the
left, against 6px at neutral).

Built: `state.py`, `output/face.py`, `output/animator.py`, `output/display.py`,
`main.py`. 526 lines total, largest file 164.

Keys in the preview window:

- `1`-`5` — set mood (order follows `moods.json`: neutral, happy, sad,
  annoyed, curious)
- `t` — cycle a fake `person_pos` (off, left, top, right, bottom) so the
  tracking path works before `camera.py` exists
- `q` / Esc — quit

### What each piece does

- **`state.py`** — the `State` dataclass exactly as `PLAN.md` specifies, plus
  an `RLock` and a `locked()` context manager. The lock is set in
  `__post_init__` rather than declared as a field, so it stays out of the
  repr and equality.
- **`face.py`** — pure renderer. Takes a flat dict of numbers and returns a
  64x64x3 `uint8` array. Knows nothing about time or `State`.
- **`animator.py`** — owns time. Reads `State` under its lock, never writes.
- **`display.py`** — `DisplayTarget` base with `show()`; `PreviewWindow` is
  the only implementation. The websocket preview (step 3) and the LED panel
  (step 8) become more targets, not edits to this file.
- **`main.py`** — load config, build the three objects, loop at 30fps.

### Design notes

- **Eye shapes are a parameter vector, not ten drawing routines.** Every
  shape carries the same ten keys per eye (`cx`, `cy`, `w`, `h`, `radius`,
  `slant`, `arc`, `pupil`, `pupil_r`, `pupil_y`), flattened to `l_w`/`r_w` and
  so on, so a mood change is one componentwise lerp across both eyes. Colour
  rides along as `color_r`/`color_g`/`color_b` in the same dict — prefixed to
  stay clear of the right eye's `r_` keys.
- **The asymmetry is fixed.** It never drifts, is never randomised, and does
  not breathe between two states. Neutral is the same face every run; only
  mood morphs change the relationship between the eyes, and they always
  resolve back to the same baseline.
- **Easing is exponential**, tau 0.133 (~95% within the 0.4s the plan asks
  for), rather than a timed ramp. Retargeting mid-ease stays smooth instead of
  restarting.
- **Edges are antialiased.** The shapes are drawn from signed distance fields
  with a one-pixel soft edge. The panel is full RGB per pixel, so a partly lit
  edge pixel is real output, not a resolution lie — and at 16px wide it is the
  difference between a curve and a staircase. The upscale in the preview stays
  strictly nearest-neighbour.
- **The pupil is a hole**, not a bright dot: eye body in the mood colour,
  pupil punched out to black. Highest contrast on an LED panel, and it shrinks
  to nothing as the lid closes so a blink ends on a solid line.
- **Each pupil travels as far as its own eye allows** — reach is computed
  from that eye's half-width and pupil radius, so the small right pupil moves
  about 1.8px against the left's 2.8px for the same gaze value. Falls out of
  the asymmetry rather than being tuned.
- **Gaze is one code path.** Idle picks from seven fixed fixation points, snaps
  to one, holds 1-3s, snaps again. A sub-pixel drift from two sine waves with
  no common period rides on top, so the eyes are never perfectly still. When
  `person_pos` is set it simply becomes the fixation point every frame; step 2
  replaces the fake key with the detector and nothing else changes.
- **Blinks** run 0.17s, shutting faster than they open, gap 2.2-6.0s scaled by
  the mood's `blink_rate`, with an 18% chance of a double blink.

### Measured

- `animator.tick()` costs **0.20 ms** against a 33.3 ms budget. Rendering is
  not going to be what makes the face freeze; blocking calls in the loop will
  be.
- Blink irregularity over a simulated 120s: 29 blinks, gaps 0.33-6.07s.

### Learned / surprising

- `PLAN.md` does not contain the eye geometry. The only face measurements in
  it are "an eye is ~12-16px across" (line 100) and "rounded-rect eyes with
  pupils" (line 104) — no coordinates, no per-eye sizes, no mention of
  asymmetry, and `git log` shows the file untouched since the upload commit.
  The numbers above came from the owner directly. **They should go into
  `PLAN.md`**, because right now the authoritative file and the code disagree
  about whether the face is symmetric.
- `cv2.destroyWindow` raises `cv2.error: NULL window` if the window is already
  gone — which it is whenever the user closes it with the title bar rather
  than pressing `q`. Every run that ended by clicking X exited code 1.
  `PreviewWindow.close()` swallows it now.

- Shearing the whole eye box for `slant` reads as a brow angle, but the first
  values (2.5 and -3.0 px) turned a 7px-tall eye into a rotated sliver. The
  smaller right eye needs less slant than the left for the same read, which is
  why `narrow` uses 2.2 and 1.8 rather than one value.
- The happy arc is a circle subtracted from below. Its radius controls
  curvature and has to be *smaller* than the eye is wide (0.62 × w) or the arch
  comes out nearly straight.

## Open questions (new)

- **`animator.py` is 382 lines.** `CLAUDE.md` says past ~300 it is probably
  two modules, and it now holds four separate clocks: blink, mood easing,
  gaze, and face drift. The clean split is the two movement systems into
  `output/motion.py`, leaving the animator to orchestrate. That is a new
  module and therefore the owner's call, so it has not been done.
- **The detector must not assume one person.** `PLAN.md` now says the
  subject comes from `pick_subject(detections)` and that nothing upstream of
  it may collapse the candidates. Step 2 has to honour that even though the
  picker's body is one line today.

- **Pupil travel is small in real pixels.** Gaze is -1..1, but that maps to a
  reach of only 2.8px in the left eye and 1.8px in the right, so a full
  fixation swing is ~3.5px peak-to-peak and the micro-drift is ±0.2px —
  sub-pixel, visible only as edge shimmer. Correct by construction (the pupil
  must stay inside its eye) but worth deciding whether the fixation points
  should push closer to ±1.0 now that the face moves too.

- **The pair sits low and tight.** Together the eyes span x 16-48 (centred)
  but y 23-48, which is ~3px below the panel's vertical centre, and the gap
  between them is only 4px. That is what the given coordinates produce. Easy
  to nudge if it reads wrong on the panel.

- **Who owns `state.color`?** The animator currently takes colour from the
  mood entry in `moods.json`, per `PLAN.md`'s "target: from state.mood". That
  leaves `state.color` unread. Step 6's `set_color` command needs a rule for
  which wins — override until the next mood change, or mood always wins.
  Decide before writing `commands.py`.
- **`config.json` `display.width`/`height` are unread.** `face.py` has 64x64
  as constants because the plan says render natively. Either the config keys
  go, or `face.py` reads them. No urgency.
- **Mood decay is read-only.** The animator treats a mood as expired once
  `mood_until` has passed, but nothing writes `state.mood` back to `neutral`.
  Whoever sets moods in step 4 should decide whether the field gets reset.

## Verified working (environment, not code)

- Ollama + `qwen2.5:7b` — 100% GPU, 4.7GB, 4096 ctx, fast when warm
- Vision model — near-instant
- `faster-whisper` small int8 on CPU — 4.8s of audio in 2.1s
- Kokoro TTS on CPU — 0.75s cold, ~0.5s warm, `af_heart` voice

## Open questions (carried over)

- Panel mounting spot not decided (right of desk vs above). Doesn't block
  anything — camera position is independent.
- Voice choice not final. Kokoro ships several; currently `af_heart`.
