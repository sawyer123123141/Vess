"""Owns time.

State says what mood Vess is in; the animator decides what that looks like
right now -- mid-blink, mid-ease, pupils a third of the way to a new fixation.
It reads State and never writes to it.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from output import face
from state import State

# ~95% of a mood change lands within 0.4s. Exponential rather than a timed
# ramp so retargeting mid-ease stays smooth instead of restarting.
_MOOD_TAU = 0.133

_BLINK_DURATION = 0.17
_BLINK_CLOSE_FRAC = 0.4         # lids shut faster than they open
_BLINK_GAP = (2.2, 6.0)         # seconds between blinks, before mood scaling
_DOUBLE_BLINK_CHANCE = 0.18
_DOUBLE_BLINK_GAP = 0.13

# Where the eyes rest between saccades, in gaze space (-1..1 per axis), which
# maps to each pupil's own reach: 2.8px in the left eye, 1.8px in the right.
# Pushed out to the full +/-1.0 so a snap is unmistakable -- at the old +/-0.62
# a jump moved the left pupil under 2px and read as a shimmer.
_FIXATIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, -0.25),
    (-1.0, -0.20),
    (0.55, 0.70),
    (-0.62, 0.62),
    (0.0, -0.85),
    (0.22, 0.32),
)
_HOLD = (1.0, 3.0)

# Pupils snap by default. `gaze_lag` scales this time constant, which is how
# sad tracks reluctantly without becoming a separate mode.
_GAZE_TAU_MAX = 0.45
_BREAK_CHECK = (1.5, 3.0)        # how often a reluctant tracker may look away
_BREAK_LENGTH = (0.7, 1.4)

# Sub-pixel wander so the eyes are never perfectly still. Two frequencies with
# no common period, so it never visibly loops. At a fixation of +/-1.0 the
# clamp eats half the drift's swing; that costs a fifth of a pixel and is not
# worth pulling the fixations back in for.
_DRIFT_AMP = 0.07
_DRIFT_FREQ = (0.71, 0.53)

# Whole-face drift. Deliberately a different motion model from the pupils:
# the face eases, the pupils snap. That contrast is what reads as a creature.
#
# The eased offset is capped at _FACE_MAX minus whatever the bob is currently
# using, so the total never grows past _FACE_MAX however a mood scales the bob.
# That is how "never near the edges" is enforced -- the pair spans x 16-48 and
# y 23-48 at rest, so 3.5px of travel still leaves a 12px margin.
_FACE_MAX = 3.5
_FACE_BOB_AMP = 1.0

# Idle floats; anything driven by a state snaps to attention faster.
_FACE_TAU_IDLE = 1.4
_FACE_TAU_ACTIVE = 0.5

_FACE_IDLE_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.6, -1.0),
    (-1.8, 0.6),
    (0.8, 1.4),
    (-1.2, -1.4),
    (1.4, 0.9),
)
_FACE_IDLE_HOLD = (4.0, 9.0)     # far slower than the pupils' 1-3s

_TRACK_LEAN = 2.0                # px at the edge of frame
_LISTEN_LEAN = 2.4               # leaning in is a bigger move than watching
_LISTEN_SETTLE = 0.6             # ...and it settles downward as it does
_THINK_OFFSET = (-1.6, -2.4)     # up and away

# Slow enough to read as breathing rather than motion.
_BOB_PERIOD = 5.2

# Vertical lean is damped: a person high in frame should not lift the face as
# far as a person at the edge moves it sideways.
_LEAN_Y_DAMP = 0.55

# How a mood moves, as multipliers against the constants above. The constants
# are the physics; these are the character. A mood with no "movement" block
# behaves exactly as neutral, so every mood that already exists stays valid.
#
#   hold           length of pupil and face holds -- below 1.0 is twitchier
#   spread         how far fixations and face wander reach from centre
#   ease           face easing time constants -- below 1.0 is quicker
#   bob            breath amplitude, and inversely its period
#   track_bias     how far toward a person the gaze and the lean travel
#   gaze_lag       0.0 snaps; higher eases the pupils toward their target
#   gaze_y_bias    where the gaze rests, positive is downward (additive)
#   track_break    chance of briefly looking away from a person
_MOVEMENT_DEFAULTS: dict[str, float] = {
    "hold": 1.0,
    "spread": 1.0,
    "ease": 1.0,
    "bob": 1.0,
    "track_bias": 1.0,
    "gaze_lag": 0.0,
    "gaze_y_bias": 0.0,
    "track_break": 0.0,
}

# Clamped on the way in, so a bad number in a hand-edited JSON file cannot
# produce a face that vibrates or freezes. The limits apply to the declared
# values, and an eased result is a blend of two clamped numbers, so every
# intermediate frame of a mood transition is in range as well.
_MOVEMENT_LIMITS: dict[str, tuple[float, float]] = {
    "hold": (0.25, 3.0),
    "spread": (0.3, 1.6),
    "ease": (0.3, 3.0),
    "bob": (0.0, 2.0),
    "track_bias": (0.25, 2.0),   # never 0: a face that ignores you reads broken
    "gaze_lag": (0.0, 1.0),
    "gaze_y_bias": (-0.6, 0.6),
    "track_break": (0.0, 0.6),
}

_FALLBACK_MOOD = "neutral"


def _clamp(value: float, limit: float) -> float:
    return min(max(value, -limit), limit)


def _clamp_range(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class FaceAnimator:
    def __init__(self, moods: dict[str, dict], seed: int | None = None) -> None:
        self._moods = moods
        self._rng = random.Random(seed)

        self.target: dict[str, float] = self._target_for(_FALLBACK_MOOD)
        self.current: dict[str, float] = dict(self.target)

        self.blink_phase: float = -1.0       # <0 means not blinking
        self.next_blink: float = self._rng.uniform(*_BLINK_GAP)
        self._blink_rate: float = 1.0
        self._double_pending = False

        # _fixation holds the *unscaled* point. Spread and the resting bias are
        # applied every frame, so a mood transition slides the pupils rather
        # than waiting for the next saccade to take effect.
        self._fixation: tuple[float, float] = (0.0, 0.0)
        self._gaze: tuple[float, float] = (0.0, 0.0)
        self._hold: float = self._rng.uniform(*_HOLD)
        self._drift_t: float = 0.0
        self._break_left: float = 0.0
        self._break_check: float = self._rng.uniform(*_BREAK_CHECK)

        self.face_offset: tuple[float, float] = (0.0, 0.0)
        self._face_target: tuple[float, float] = (0.0, 0.0)
        self._face_hold: float = self._rng.uniform(*_FACE_IDLE_HOLD)
        self._bob_phase: float = 0.0

    def tick(self, state: State, dt: float) -> np.ndarray:
        with state.locked():
            mood = state.mood
            mood_until = state.mood_until
            brightness = state.brightness
            person_pos = state.person_pos
            thinking = state.thinking
            listening = state.listening

        if mood_until and time.time() > mood_until:
            mood = _FALLBACK_MOOD

        entry = self._mood_entry(mood)
        self._blink_rate = float(entry.get("blink_rate", 1.0)) or 1.0
        self.target = self._target_for(mood)

        alpha = 1.0 - math.exp(-dt / _MOOD_TAU)
        for key, value in self.target.items():
            self.current[key] += (value - self.current[key]) * alpha

        # Read after the ease, not from the mood entry, so the multipliers
        # interpolate through a transition alongside the shape and the colour.
        move = {key: self.current[f"move_{key}"] for key in _MOVEMENT_DEFAULTS}

        self._advance_blink(dt)
        gaze = self._advance_gaze(dt, person_pos, move)
        offset = self._advance_face(dt, thinking, listening, person_pos, move)

        color = tuple(self.current[f"color_{c}"] for c in "rgb")
        shape = {k: v for k, v in self.current.items()
                 if not k.startswith(("color_", "move_"))}
        return face.render(shape, color, brightness, self._openness(), gaze,
                           offset)

    def _mood_entry(self, mood: str) -> dict:
        """Unknown mood names are ignored rather than raising -- the classifier
        in a later step returns free text and will get it wrong sometimes."""
        entry = self._moods.get(mood)
        if entry is None:
            entry = self._moods.get(_FALLBACK_MOOD, {})
        return entry

    def _target_for(self, mood: str) -> dict[str, float]:
        entry = self._mood_entry(mood)
        params = face.shape_params(str(entry.get("eye", "normal")))
        # Colour rides along in the same dict as the shape numbers so one
        # easing loop handles both. Prefixed to stay clear of the right eye's
        # own `r_` keys.
        color = entry.get("color", (255, 255, 255))
        for channel, value in zip("rgb", color):
            params[f"color_{channel}"] = float(value)

        movement = entry.get("movement", {})
        for key, default in _MOVEMENT_DEFAULTS.items():
            low, high = _MOVEMENT_LIMITS[key]
            params[f"move_{key}"] = _clamp_range(
                float(movement.get(key, default)), low, high)
        return params

    def _advance_blink(self, dt: float) -> None:
        if self.blink_phase >= 0.0:
            self.blink_phase += dt / _BLINK_DURATION
            if self.blink_phase >= 1.0:
                self.blink_phase = -1.0
                if self._double_pending:
                    self._double_pending = False
                    self.next_blink = _DOUBLE_BLINK_GAP
                else:
                    self.next_blink = self._rng.uniform(*_BLINK_GAP) / self._blink_rate
            return

        self.next_blink -= dt
        if self.next_blink <= 0.0:
            self.blink_phase = 0.0
            self._double_pending = self._rng.random() < _DOUBLE_BLINK_CHANCE

    def _openness(self) -> float:
        if self.blink_phase < 0.0:
            return 1.0
        if self.blink_phase < _BLINK_CLOSE_FRAC:
            t = self.blink_phase / _BLINK_CLOSE_FRAC
        else:
            t = 1.0 - (self.blink_phase - _BLINK_CLOSE_FRAC) / (1.0 - _BLINK_CLOSE_FRAC)
        return 1.0 - _smoothstep(t)

    def _advance_gaze(self, dt: float, person_pos: tuple[float, float] | None,
                      move: dict[str, float]) -> tuple[float, float]:
        rest = (0.0, move["gaze_y_bias"])

        if person_pos is not None:
            # A person outranks idle wandering. Same code path -- they are just
            # a fixation point that keeps moving.
            target = self._track_target(dt, person_pos, rest, move)
            self._hold = 0.0
        else:
            self._hold -= dt
            if self._hold <= 0.0:
                self._fixation = self._pick_fixation()
                self._hold = self._rng.uniform(*_HOLD) * move["hold"]
            spread = move["spread"]
            target = (_clamp(self._fixation[0] * spread, 1.0),
                      _clamp(self._fixation[1] * spread + rest[1], 1.0))

        # Snapping is the default, and the contrast with the easing face is the
        # whole point. gaze_lag above 0.0 is the one case that eases.
        tau = _GAZE_TAU_MAX * move["gaze_lag"]
        if tau <= 1e-4:
            self._gaze = target
        else:
            alpha = 1.0 - math.exp(-dt / tau)
            self._gaze = (self._gaze[0] + (target[0] - self._gaze[0]) * alpha,
                          self._gaze[1] + (target[1] - self._gaze[1]) * alpha)

        self._drift_t += dt
        dx = _DRIFT_AMP * math.sin(self._drift_t * _DRIFT_FREQ[0])
        dy = _DRIFT_AMP * math.sin(self._drift_t * _DRIFT_FREQ[1] + 1.3)
        return (_clamp(self._gaze[0] + dx, 1.0), _clamp(self._gaze[1] + dy, 1.0))

    def _track_target(self, dt: float, person_pos: tuple[float, float],
                      rest: tuple[float, float],
                      move: dict[str, float]) -> tuple[float, float]:
        """Where to look with someone in the room.

        `track_bias` is how far along the line from the resting gaze to the
        person the eyes actually travel, so a reluctant mood lands short of
        them and toward its own rest, rather than merely nearer the centre.
        """
        if move["track_break"] > 0.0:
            if self._break_left > 0.0:
                self._break_left -= dt
                return rest
            self._break_check -= dt
            if self._break_check <= 0.0:
                self._break_check = self._rng.uniform(*_BREAK_CHECK)
                if self._rng.random() < move["track_break"]:
                    self._break_left = self._rng.uniform(*_BREAK_LENGTH)
                    return rest

        px, py = person_pos
        full = (_clamp((px - 0.5) * 2.0, 1.0), _clamp((py - 0.5) * 1.6, 1.0))
        bias = move["track_bias"]
        return (_clamp(rest[0] + (full[0] - rest[0]) * bias, 1.0),
                _clamp(rest[1] + (full[1] - rest[1]) * bias, 1.0))

    def _pick_fixation(self) -> tuple[float, float]:
        choices = [p for p in _FIXATIONS if p != self._fixation]
        return self._rng.choice(choices)

    def _advance_face(self, dt: float, thinking: bool, listening: bool,
                      person_pos: tuple[float, float] | None,
                      move: dict[str, float]) -> tuple[float, float]:
        """Where the whole eye pair sits, in panel pixels from its rest place.

        Compounds with the pupils rather than replacing them: leaning toward a
        person and pointing the pupils at them happen together.
        """
        ease, bias, spread = move["ease"], move["track_bias"], move["spread"]

        if thinking:
            target, tau = _THINK_OFFSET, _FACE_TAU_ACTIVE * ease
        elif listening:
            lean = self._lean(person_pos, _LISTEN_LEAN * bias)
            target = (lean[0], lean[1] + _LISTEN_SETTLE)
            tau = _FACE_TAU_ACTIVE * ease
        elif person_pos is not None:
            target = self._lean(person_pos, _TRACK_LEAN * bias)
            tau = _FACE_TAU_ACTIVE * ease
        else:
            self._face_hold -= dt
            if self._face_hold <= 0.0:
                choices = [p for p in _FACE_IDLE_POINTS if p != self._face_target]
                self._face_target = self._rng.choice(choices)
                self._face_hold = self._rng.uniform(*_FACE_IDLE_HOLD) * move["hold"]
            target = (self._face_target[0] * spread, self._face_target[1] * spread)
            tau = _FACE_TAU_IDLE * ease

        # Give back to the ease whatever the bob is not using, so a mood that
        # breathes harder travels less and the total stays inside _FACE_MAX.
        amp = _FACE_BOB_AMP * move["bob"]
        reach = max(_FACE_MAX - amp, 0.5)

        alpha = 1.0 - math.exp(-dt / max(tau, 1e-3))
        self.face_offset = (
            _clamp(self.face_offset[0] + (target[0] - self.face_offset[0]) * alpha,
                   reach),
            _clamp(self.face_offset[1] + (target[1] - self.face_offset[1]) * alpha,
                   reach),
        )

        # The bob sits outside the easing so it stays a clean sine rather than
        # something the ease is forever chasing. Phase is accumulated rather
        # than derived from elapsed time, because a mood that changes the
        # period would otherwise jump the sine mid-breath.
        period = _BOB_PERIOD / max(move["bob"], 0.25)
        self._bob_phase += 2.0 * math.pi * dt / period
        return (self.face_offset[0], self.face_offset[1] + amp * math.sin(self._bob_phase))

    @staticmethod
    def _lean(person_pos: tuple[float, float] | None,
              gain: float) -> tuple[float, float]:
        """Lean toward a person. With nobody there, hold still and centred."""
        if person_pos is None:
            return (0.0, 0.0)
        px, py = person_pos
        return (
            _clamp((px - 0.5) * 2.0 * gain, _FACE_MAX),
            _clamp((py - 0.5) * 2.0 * gain * _LEAN_Y_DAMP, _FACE_MAX),
        )
