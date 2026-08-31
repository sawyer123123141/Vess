# Status

Update this at the end of every session. Newest at the top.

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

- `cv2.destroyWindow` raises `cv2.error: NULL window` if the window is already
  gone — which it is whenever the user closes it with the title bar rather
  than pressing `q`. Every run that ended by clicking X exited code 1.
  `PreviewWindow.close()` swallows it.
- Shearing the whole eye box for `slant` reads as a brow angle, but the first
  values (2.5 and -3.0 px) turned a 7px-tall eye into a rotated sliver. The
  smaller right eye needs less slant than the left for the same read, which is
  why `narrow` uses 2.2 and 1.8 rather than one value.
- `PLAN.md` did not contain the eye geometry — the only face measurements in
  it were "an eye is ~12-16px across" and "rounded-rect eyes with pupils".
  The numbers came from the owner directly and are now written into the
  Rendering section, so the authoritative file and the code agree.
- The happy arc is a circle subtracted from below. Its radius controls
  curvature and has to be *smaller* than the eye is wide (0.62 × w) or the arch
  comes out nearly straight.

## Open questions (new)

- **Who owns `state.color`?** The animator currently takes colour from the
  mood entry in `moods.json`, per `PLAN.md`'s "target: from state.mood". That
  leaves `state.color` unread. Step 6's `set_color` command needs a rule for
  which wins — override until the next mood change, or mood always wins.
  Decide before writing `commands.py`.
- **`config.json` `display.width`/`height` are unread.** `face.py` has 64x64
  as constants because the plan says render natively. Either the config keys
  go, or `face.py` reads them. No urgency.
- **The pair sits low and tight.** Together the eyes span x 16-48 (centred)
  but y 23-48, which is ~3px below the panel's vertical centre, and the gap
  between them is only 4px. That is what the given coordinates produce. Easy
  to nudge if it reads wrong on the panel.
- **Tracking is not mirrored.** A person at `person_pos.x = 0.1` makes the
  pupils go to the viewer's left. Whether that is correct depends on whether
  the camera image is mirrored — settle it against a real camera in step 2.
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
