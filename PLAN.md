# Vess — architecture

An ambient presence: an LED face on the wall that sees the room, hears you,
remembers, and occasionally speaks first. Runs entirely on a local machine.

## Principles

1. **Intelligence in the plumbing, not the model.** Every decision made in code
   is a decision a 7B model can't get wrong. The model's job is to write the
   final sentence, given context the code assembled.
2. **One state object.** Every feature is "a field in `State`" plus "something
   that reads it." Not a new subsystem.
3. **The face never freezes.** Rendering runs at 30fps regardless of what else
   is happening.
4. **Nothing composed.** The model picks from fixed lists. It never writes
   shell, code, or config.

---

## State

Single source of truth. Lives in `state.py`, guarded by a lock.

```python
@dataclass
class State:
    # identity
    persona: str = "friendly"       # friendly | concise | dry
    mood: str = "neutral"
    mood_until: float = 0.0         # epoch; decays to neutral past this

    # appearance
    color: tuple[int, int, int] = (100, 180, 255)
    brightness: float = 0.7

    # perception
    person_present: bool = False
    person_pos: tuple[float, float] | None = None   # normalised 0-1, for eye tracking
    present_since: float | None = None
    last_seen: float | None = None
    active_window: str = ""
    objects: list[str] = field(default_factory=list)

    # runtime
    listening: bool = False
    thinking: bool = False
    speaking: bool = False
    muted_until: float = 0.0
    last_spoke: float = 0.0
```

Producers (threads) write to it. The main loop and prompt builder read it.

---

## Modules

```
vess/
  state.py            State dataclass + lock
  config.json         tunables (thresholds, paths, voice, model names)
  moods.json          mood definitions — hot-reloadable, edit freely

  perception/
    camera.py         capture, downscale to ~512px
    detector.py       YOLO -> object list + person bbox
    audio.py          mic, VAD, wake word, whisper
    desktop.py        active window title

  brain/
    llm.py            ollama calls, prompt assembly, clause streaming
    memory.py         sqlite: facts + event log
    triggers.py       decides IF it speaks unprompted
    commands.py       the closed command registry

  output/
    voice.py          kokoro, played per clause
    face.py           renders 64x64 numpy array
    animator.py       owns time: blinking, easing between moods
    display.py        sends array to preview socket AND/OR LED panel

  control/
    web.py            localhost config UI + live face preview

  main.py             event loop
```

### The boundary that matters

`face.py` produces a 64x64 array. `display.py` decides where it goes — browser
preview, LED panel, or both. Same code path. This is what makes the whole thing
testable without hardware.

---

## Rendering

**Native 64x64. Never render large and downscale.** The preview upscales with
nearest-neighbour so the browser shows exactly what the panel will show,
blocky edges included. An eye is ~12-16px across — shapes must be simple.

### Face design — asymmetric eyes, no body, no mouth

Two eyes floating on black. Nothing else. No outline, no body, no mouth — the
panel is 64x64 and a stroked body silhouette costs more pixels than it earns,
while shrinking the eyes and killing expression range.

**The eyes are deliberately mismatched.** This is the character:

- left eye: larger, sits lower
- right eye: smaller, sits a few pixels higher
- both rounded rects with dark pupils cut out

Approximate geometry at 64x64 (tune by eye, keep the asymmetry):

```
left  eye:  16w x 22h, rounded, top-left about (16, 26)
right eye:  12w x 16h, rounded, top-left about (36, 23)
left  pupil: r 4      right pupil: r 3
```

The mismatch is the point. It reads as a someone rather than a UI element, and
it gives a distinctive resting face to deviate from — so widening or narrowing
registers harder than it would from a symmetric baseline.

**Mood morphs the eye shape**, interpolated by the animator:

- neutral — as above
- happy — both squash to upward arcs, pupils hidden
- annoyed — both narrow vertically, pupils small
- sad — tops droop, pupils sit low in the eye
- curious — both widen, right eye rises further

Optional later: two small solid marks above the eyes acting as brows. Solid
shapes only, never strokes. Not in scope for step 1.

### Eye movement

**Nothing moves randomly.** Random motion reads as broken. Every movement
traces to something in `State`. If there is no reason to move, drift toward
the last thing that was interesting — do not jitter.

Priority order for where the pupils point:

1. **Tracking** — follow `state.person_pos` from the detector bbox. Strongest
   aliveness signal available and it costs nothing extra.
2. **State glances** —
   - `thinking` — drift up and away, break contact
   - `listening` — lock onto the person
   - `speaking` — mostly on the person, with occasional brief looks away.
     Unbroken eye contact is unsettling.
   - idle — slow drift toward the most recently changed object in the world
     model
3. **Micro-drift** — small continuous motion even when locked. Perfect
   stillness reads as frozen.

**Pupils snap, they do not glide.** Real eyes move in saccades: fast jumps
between fixation points with brief holds. Easing the pupil smoothly toward a
target is the single most common mistake in digital eyes and it makes them
look like a cursor. Move in 1-2 frames, then hold.

Eye *shape* changes (mood morphs) do ease smoothly over ~0.4s. Only pupil
position snaps.

**Blinking**: irregular interval, roughly every 2-6s, with occasional double
blinks. Blink rate scales with `mood.blink_rate`. Irregularity matters more
than frequency — a metronome blink reads as mechanical.

### Animator

Owns time, separate from state:

```python
class FaceAnimator:
    current: dict   # live interpolated params
    target: dict    # from state.mood
    blink_phase: float
    next_blink: float

    def tick(self, state, dt) -> np.ndarray
```

Mood changes ease over ~0.4s rather than snapping. Blink runs on its own
irregular clock. Idle pupil drift when nothing's happening.

---

## Personas and moods

**Separate fields, deliberately.** Persona is a stable setting you choose;
mood is reactive and decays. "concise and sad" must be expressible.

`personas` (in config.json) are prompt fragments:

```json
{
  "friendly": "Warm, casual, a bit playful. Two sentences max.",
  "concise":  "Terse. Answer and stop. Usually one sentence.",
  "dry":      "Understated, mildly sardonic. Never enthusiastic."
}
```

`moods.json` — add a mood by adding an entry, no code:

```json
{
  "neutral": { "color": [100,180,255], "eye": "normal", "blink_rate": 1.0, "prompt": "" },
  "happy":   { "color": [255,220,100], "eye": "arc",    "blink_rate": 1.3,
               "prompt": "You're in a good mood.", "decay": 300 },
  "sad":     { "color": [80,100,160],  "eye": "droop",  "blink_rate": 0.6,
               "prompt": "You're a bit deflated.", "decay": 600 },
  "annoyed": { "color": [255,120,80],  "eye": "narrow", "blink_rate": 0.8,
               "prompt": "You're mildly irritated.", "decay": 400 }
}
```

Mood is set automatically: after each exchange, one cheap LLM call classifies
tone and returns a mood name **from the list**. Invalid names are ignored.

---

## Commands — closed registry

The model never composes. It selects a name and fills declared arguments.
Anything not in the registry is rejected before execution.

```python
COMMANDS = {
    "open_app":       {"values": ["browser", "spotify", "unity", "discord", "vscode"]},
    "close_app":      {"values": [...]},
    "play_pause":     {},
    "volume":         {"args": {"level": "float 0-1"}},
    "set_timer":      {"args": {"minutes": "int"}},
    "set_color":      {"args": {"name": "str from palette"}},
    "set_persona":    {"values": ["friendly", "concise", "dry"]},
    "set_mood":       {"values": [...from moods.json...]},
    "mute":           {"args": {"minutes": "int"}},
    "sleep":          {},
    "wake":           {},
    "elaborate":      {},   # re-ask last question with the length cap lifted
}
```

App names map to paths a human wrote in `config.json`. Adding a capability is
a human editing that file — that friction is the safety model.

Same registry drives the web UI and voice commands. One definition.

---

## Perception tiers

- **Tier 1 (continuous, free):** YOLO detector at ~2-5fps -> object list,
  person bbox. Diffing the object list between frames *is* the motion signal;
  no separate motion detector.
- **Tier 2 (on event):** wake word -> whisper -> LLM -> kokoro.
- **Tier 3 (on demand):** vision model. Fires when spoken to, or when the
  detector sees something it can't classify. Downscale frames to ~512px first
  — image tokens scale with resolution and this is the biggest latency win.

**If you speak to it, it looks.** No cleverness about whether the question
"needs" vision. Capture the frame at wake-word, in parallel with
transcription, so it costs no added latency.

---

## Memory

SQLite. Three stores:

- **short-term** — last N minutes of transcript + events, in RAM, injected
  every prompt
- **facts** — durable things about the owner. Written by a summarisation pass
  *after* a conversation ends, not during
- **event log** — every state change, timestamped

**Start the event log on day one**, before anything reads it. It costs nothing
and it's the one thing that can't be reconstructed later. Every "remembers you"
feature depends on history existing.

Retrieval quality matters more than model size. Pulling the right five facts
beats a bigger model with none.

---

## Speaking unprompted

**Trigger on transitions, never states.** "Person present" is true constantly.
"Person went absent 4h then returned" happens once.

Gates, all of them:

- one unprompted line per hour, hard cap
- silent after a configured hour
- suppress while typing, on a call, or already talking
- decay repeats — same trigger two days running is less likely
- "not now" buys an hour of silence and is remembered

**Most reactions are non-verbal.** Eyes tracking, a colour shift. Speech for
maybe 10% of noticed events. Start almost mute and loosen — much easier than
the reverse.

**Observations, not questions.** "Still on the compiler?" beats "what are you
working on?" — it should know from the active window. Ladder: read the window
-> guess and confirm -> ask openly only when it genuinely has no signal.

Start with hard rules. Do not build a scoring system until the rules visibly
fail.

---

## Response length

Three layers, because prompts alone leak:

1. Persona says to be brief
2. `num_predict` ~80 tokens — it physically can't ramble
3. `elaborate` command lifts the cap for "tell me more"

---

## Latency

Perceived speed matters more than model quality for presence.

- **Stream TTS on clause boundaries** — start speaking at the first comma
- **Static system prompt first** so the KV cache stays valid across turns
- **`OLLAMA_KEEP_ALIVE=-1`** — a cold load is 5-15s and reads as broken
- **Canned reflexes** — wake acknowledgement, "one sec", greetings as
  pre-rendered audio, played instantly while the real response generates
- **Show thinking** — eyes drift, slow pulse. Makes a pause feel intentional

---

## Build order

Each step is a working thing. 1-7 need no hardware beyond a webcam.

1. `state.py` + `face.py` + `animator.py` + `display.py` (preview only) — eyes
   that blink
2. `camera.py` + `detector.py` — eyes that track you
3. `web.py` — config UI with live preview; change colour and see it
4. Voice loop: audio -> whisper -> ollama -> kokoro
5. `memory.py` + event log
6. `commands.py` — "turn blue" works by voice
7. `triggers.py` — speaks unprompted
8. `display.py` gains the LED panel target

---

## Hardware

- 64x64 HUB75 LED panel, P3 (3mm pitch), 192x192mm
- Pimoroni Interstate 75 W (RP2350) — receives frames over WiFi
- 5V 4A supply for the panel (USB-C powers the board only)
- USB webcam
- USB mic (owned)

Panel gotchas: 64-tall panels use 5-address (ABCDE) multiplexing, which many
drivers don't support. Some batches use the FM6126A driver chip and need to be
told so explicitly.
