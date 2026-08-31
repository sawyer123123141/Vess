# Status

Update this at the end of every session. Newest at the top.

## Step 2 written — logic verified, model not

`perception/camera.py` and `perception/detector.py` exist and are wired into
`main.py`. The detector runs in its own thread; the render loop never waits on
it. **Nothing here has been run against the real YOLO model or a real camera**,
so step 2 is not done.

### Blocked on two things

- **`yolo11n.pt` is NOT downloaded.** No YOLO weights are cached anywhere on
  this machine — `C:\Users\sawye\weights` is where ultralytics would put it,
  and it is empty. Constructing `Detector` triggers a ~5.4 MB fetch from the
  ultralytics GitHub releases. That download was not authorised, so it has not
  happened. Until it does, `Detector.detect()` has never executed.
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

In order:

1. **Approve the `yolo11n.pt` download** (~5.4 MB, ultralytics GitHub
   releases, lands in `C:\Users\sawye\weights`). Nothing downstream can be
   checked until the model can load.
2. **Drop a photo of the room somewhere** and set `config.camera.source` to
   `"image"` with `path` pointing at it. That makes the whole pipeline
   verifiable with no camera: real detections, a subject picked, `person_pos`
   written, and the face visibly tracking toward the person in the photo.
3. **Benchmark `Detector.detect()` on the CPU** at 512px and record the
   per-frame cost, so the 3fps in `config.json` is grounded in a number rather
   than an assumption.
4. **When the webcam arrives**, flip `config.camera.source` back to
   `"camera"` and confirm the mirror direction is right in the room, with the
   camera in its actual position.

Only then is step 2 done. Step 3 is `web.py` — the config UI with a live
preview.

## Mood drives movement

Runtime state still picks *where* the face goes. Mood now scales *how* it gets
there, through an optional `movement` block per entry in `moods.json` —
multipliers against the constants in `animator.py`. The constants are the
physics, the multipliers are the character. A mood with no block moves exactly
as neutral, so nothing that already existed had to change.

Keys and limits: `hold` 0.25-3.0, `spread` 0.3-1.6, `ease` 0.3-3.0, `bob`
0.0-2.0, `track_bias` 0.25-2.0, `gaze_lag` 0.0-1.0, `gaze_y_bias` -0.6-0.6,
`track_break` 0.0-0.6. Multipliers default to 1.0, lags and biases to 0.0.

### Measured, 180s idle per mood

| mood    | snaps | face x range | gaze reach | mean gaze y | settle into thinking |
|---------|-------|--------------|------------|-------------|----------------------|
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
