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

# Where the eyes rest between saccades, in gaze space (-1..1 per axis).
_FIXATIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.62, -0.15),
    (-0.62, -0.12),
    (0.34, 0.42),
    (-0.38, 0.38),
    (0.0, -0.50),
    (0.14, 0.20),
)
_HOLD = (1.0, 3.0)

# Sub-pixel wander so the eyes are never perfectly still. Two frequencies with
# no common period, so it never visibly loops.
_DRIFT_AMP = 0.07
_DRIFT_FREQ = (0.71, 0.53)

_FALLBACK_MOOD = "neutral"


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

        self._fixation: tuple[float, float] = (0.0, 0.0)
        self._hold: float = self._rng.uniform(*_HOLD)
        self._drift_t: float = 0.0

    def tick(self, state: State, dt: float) -> np.ndarray:
        with state.locked():
            mood = state.mood
            mood_until = state.mood_until
            brightness = state.brightness
            person_pos = state.person_pos

        if mood_until and time.time() > mood_until:
            mood = _FALLBACK_MOOD

        entry = self._mood_entry(mood)
        self._blink_rate = float(entry.get("blink_rate", 1.0)) or 1.0
        self.target = self._target_for(mood)

        alpha = 1.0 - math.exp(-dt / _MOOD_TAU)
        for key, value in self.target.items():
            self.current[key] += (value - self.current[key]) * alpha

        self._advance_blink(dt)
        gaze = self._advance_gaze(dt, person_pos)

        color = tuple(self.current[f"color_{c}"] for c in "rgb")
        shape = {k: v for k, v in self.current.items()
                 if not k.startswith("color_")}
        return face.render(shape, color, brightness, self._openness(), gaze)

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

    def _advance_gaze(self, dt: float,
                      person_pos: tuple[float, float] | None) -> tuple[float, float]:
        if person_pos is not None:
            # A person outranks idle wandering. Same code path -- they are just
            # a fixation point that keeps moving.
            px, py = person_pos
            self._fixation = (
                min(max((px - 0.5) * 2.0, -1.0), 1.0),
                min(max((py - 0.5) * 1.6, -1.0), 1.0),
            )
            self._hold = 0.0
        else:
            self._hold -= dt
            if self._hold <= 0.0:
                self._fixation = self._pick_fixation()
                self._hold = self._rng.uniform(*_HOLD)

        self._drift_t += dt
        dx = _DRIFT_AMP * math.sin(self._drift_t * _DRIFT_FREQ[0])
        dy = _DRIFT_AMP * math.sin(self._drift_t * _DRIFT_FREQ[1] + 1.3)
        return (
            min(max(self._fixation[0] + dx, -1.0), 1.0),
            min(max(self._fixation[1] + dy, -1.0), 1.0),
        )

    def _pick_fixation(self) -> tuple[float, float]:
        choices = [p for p in _FIXATIONS if p != self._fixation]
        return self._rng.choice(choices)
