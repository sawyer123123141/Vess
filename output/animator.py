"""Owns time.

State says what Vess is doing; the animator decides what that looks like right
now. It reads State and never writes to it.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from output import face
from performance import PerformanceCue
from state import State

_MOOD_TAU = 0.133
_PERFORMANCE_TAU = 0.08

_BLINK_DURATION = 0.17
_BLINK_CLOSE_FRAC = 0.4
_BLINK_GAP = (2.2, 6.0)
_DOUBLE_BLINK_CHANCE = 0.18
_DOUBLE_BLINK_GAP = 0.13

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

_GAZE_TAU_MAX = 0.45
_BREAK_CHECK = (1.5, 3.0)
_BREAK_LENGTH = (0.7, 1.4)
_THINK_GAZE = (-0.45, -0.72)
_SPEAK_BREAK_GAZE = (-0.55, -0.12)
_SPEAK_BREAK_CHECK = (1.8, 3.8)
_SPEAK_BREAK_LENGTH = (0.28, 0.60)

_DRIFT_AMP = 0.07
_DRIFT_FREQ = (0.71, 0.53)

_FACE_MAX = 3.5
_FACE_BOB_AMP = 1.0
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
_FACE_IDLE_HOLD = (4.0, 9.0)

_TRACK_LEAN = 2.0
_LISTEN_LEAN = 2.4
_LISTEN_SETTLE = 0.6
_THINK_OFFSET = (-1.6, -2.4)
_BOB_PERIOD = 5.2
_LEAN_Y_DAMP = 0.55

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

_MOVEMENT_LIMITS: dict[str, tuple[float, float]] = {
    "hold": (0.25, 3.0),
    "spread": (0.3, 1.6),
    "ease": (0.3, 3.0),
    "bob": (0.0, 2.0),
    "track_bias": (0.25, 2.0),
    "gaze_lag": (0.0, 1.0),
    "gaze_y_bias": (-0.6, 0.6),
    "track_break": (0.0, 0.6),
}

_PERFORMANCE_SHAPE_KEYS = (
    "l_h",
    "r_h",
    "l_slant",
    "r_slant",
    "l_cy",
    "r_cy",
)
_PERFORMANCE_MOVEMENT_DEFAULTS = {
    "hold_scale": 1.0,
    "ease_scale": 1.0,
    "track_bias_scale": 1.0,
    "gaze_y_bias": 0.0,
    "speaking_break_scale": 1.0,
}

_FALLBACK_MOOD = "neutral"


def _clamp(value: float, limit: float) -> float:
    return min(max(value, -limit), limit)


def _clamp_range(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _neutral_performance_values() -> dict[str, float]:
    values = {key: 0.0 for key in _PERFORMANCE_SHAPE_KEYS}
    values.update(_PERFORMANCE_MOVEMENT_DEFAULTS)
    return values


class FaceAnimator:
    def __init__(
        self,
        moods: dict[str, dict],
        performances: dict[str, dict[str, object]] | None = None,
        seed: int | None = None,
    ) -> None:
        self._moods = moods
        self._performances = performances or {
            "neutral": {
                "intensity": 0.0,
                "shape": {key: 0.0 for key in _PERFORMANCE_SHAPE_KEYS},
                "movement": dict(_PERFORMANCE_MOVEMENT_DEFAULTS),
            }
        }
        self._rng = random.Random(seed)

        self.target: dict[str, float] = self._target_for(_FALLBACK_MOOD)
        self.current: dict[str, float] = dict(self.target)

        self.performance_current = _neutral_performance_values()
        self._performance_target = _neutral_performance_values()
        self._interaction_mode = "idle"
        self._last_shape: dict[str, float] = {
            key: value
            for key, value in self.current.items()
            if not key.startswith(("color_", "move_"))
        }
        self._last_color = tuple(self.current[f"color_{channel}"] for channel in "rgb")
        self._last_render_gaze: tuple[float, float] = (0.0, 0.0)
        self._last_render_offset: tuple[float, float] = (0.0, 0.0)

        self.blink_phase: float = -1.0
        self.next_blink: float = self._rng.uniform(*_BLINK_GAP)
        self._blink_rate: float = 1.0
        self._double_pending = False

        self._fixation: tuple[float, float] = (0.0, 0.0)
        self._gaze: tuple[float, float] = (0.0, 0.0)
        self._hold: float = self._rng.uniform(*_HOLD)
        self._drift_t: float = 0.0
        self._break_left: float = 0.0
        self._break_check: float = self._rng.uniform(*_BREAK_CHECK)
        self._speak_break_left: float = 0.0
        self._speak_break_check: float = self._rng.uniform(*_SPEAK_BREAK_CHECK)

        self.face_offset: tuple[float, float] = (0.0, 0.0)
        self._face_target: tuple[float, float] = (0.0, 0.0)
        self._face_hold: float = self._rng.uniform(*_FACE_IDLE_HOLD)
        self._bob_phase: float = 0.0

    def tick(self, state: State, dt: float) -> np.ndarray:
        with state.locked():
            mood = state.mood
            mood_until = state.mood_until
            performance = state.performance
            color_override = state.color
            brightness = state.brightness
            person_pos = state.person_pos
            thinking = state.thinking
            listening = state.listening
            speaking = state.speaking

        if mood_until and time.time() > mood_until:
            mood = _FALLBACK_MOOD

        entry = self._mood_entry(mood)
        self._blink_rate = float(entry.get("blink_rate", 1.0)) or 1.0
        self.target = self._target_for(mood)
        if color_override is not None:
            for channel, value in zip("rgb", color_override):
                self.target[f"color_{channel}"] = float(value)

        mood_alpha = 1.0 - math.exp(-dt / _MOOD_TAU)
        for key, value in self.target.items():
            self.current[key] += (value - self.current[key]) * mood_alpha

        self._performance_target = self._performance_target_for(performance)
        performance_alpha = 1.0 - math.exp(-dt / _PERFORMANCE_TAU)
        for key, value in self._performance_target.items():
            self.performance_current[key] += (
                value - self.performance_current[key]
            ) * performance_alpha

        move = {key: self.current[f"move_{key}"] for key in _MOVEMENT_DEFAULTS}
        move["hold"] *= self.performance_current["hold_scale"]
        move["ease"] *= self.performance_current["ease_scale"]
        move["track_bias"] *= self.performance_current["track_bias_scale"]
        move["gaze_y_bias"] += self.performance_current["gaze_y_bias"]

        self._interaction_mode = self._mode_for(
            listening,
            thinking,
            speaking,
            person_pos,
        )

        self._advance_blink(dt)
        gaze = self._advance_gaze(
            dt,
            person_pos,
            move,
            self._interaction_mode,
            self.performance_current["speaking_break_scale"],
        )
        offset = self._advance_face(
            dt,
            self._interaction_mode,
            person_pos,
            move,
        )

        color = tuple(self.current[f"color_{channel}"] for channel in "rgb")
        shape = {
            key: value
            for key, value in self.current.items()
            if not key.startswith(("color_", "move_"))
        }
        for key in _PERFORMANCE_SHAPE_KEYS:
            shape[key] += self.performance_current[key]
        shape["l_h"] = max(shape["l_h"], 2.0)
        shape["r_h"] = max(shape["r_h"], 2.0)
        shape["l_slant"] = _clamp(shape["l_slant"], 4.0)
        shape["r_slant"] = _clamp(shape["r_slant"], 4.0)
        shape["l_cy"] = _clamp_range(shape["l_cy"], 12.0, 52.0)
        shape["r_cy"] = _clamp_range(shape["r_cy"], 12.0, 52.0)

        self._last_shape = dict(shape)
        self._last_color = color
        self._last_render_gaze = gaze
        self._last_render_offset = offset
        return face.render(
            shape,
            color,
            brightness,
            self._openness(),
            gaze,
            offset,
        )

    def debug_snapshot(self) -> dict[str, object]:
        return {
            "interaction_mode": self._interaction_mode,
            "render_gaze": tuple(self._last_render_gaze),
            "render_offset": tuple(self._last_render_offset),
            "blink_openness": self._openness(),
            "shape": dict(self._last_shape),
            "color": tuple(self._last_color),
            "fixation": tuple(self._fixation),
            "speaking_break_active": self._speak_break_left > 0.0,
            "speaking_break_remaining": max(self._speak_break_left, 0.0),
            "performance_current": dict(self.performance_current),
            "performance_target": dict(self._performance_target),
        }

    @staticmethod
    def _mode_for(
        listening: bool,
        thinking: bool,
        speaking: bool,
        person_pos: tuple[float, float] | None,
    ) -> str:
        if listening:
            return "listening"
        if thinking:
            return "thinking"
        if speaking:
            return "speaking"
        if person_pos is not None:
            return "tracking"
        return "idle"

    def _mood_entry(self, mood: str) -> dict:
        entry = self._moods.get(mood)
        if entry is None:
            entry = self._moods.get(_FALLBACK_MOOD, {})
        return entry

    def _target_for(self, mood: str) -> dict[str, float]:
        entry = self._mood_entry(mood)
        params = face.shape_params(str(entry.get("eye", "normal")))
        color = entry.get("color", (255, 255, 255))
        for channel, value in zip("rgb", color):
            params[f"color_{channel}"] = float(value)

        movement = entry.get("movement", {})
        for key, default in _MOVEMENT_DEFAULTS.items():
            low, high = _MOVEMENT_LIMITS[key]
            params[f"move_{key}"] = _clamp_range(
                float(movement.get(key, default)),
                low,
                high,
            )
        return params

    def _performance_target_for(self, cue: PerformanceCue) -> dict[str, float]:
        entry = self._performances.get(cue.expression)
        if entry is None:
            entry = self._performances.get("neutral", {})
            cue = PerformanceCue()

        intensity = _clamp_range(float(cue.intensity), 0.0, 1.0)
        shape = entry.get("shape", {}) if isinstance(entry, dict) else {}
        movement = entry.get("movement", {}) if isinstance(entry, dict) else {}

        target = _neutral_performance_values()
        for key in _PERFORMANCE_SHAPE_KEYS:
            target[key] = float(shape.get(key, 0.0)) * intensity
        for key in ("hold_scale", "ease_scale", "track_bias_scale", "speaking_break_scale"):
            configured = float(movement.get(key, 1.0))
            target[key] = 1.0 + (configured - 1.0) * intensity
        target["gaze_y_bias"] = float(movement.get("gaze_y_bias", 0.0)) * intensity
        return target

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
            t = 1.0 - (self.blink_phase - _BLINK_CLOSE_FRAC) / (
                1.0 - _BLINK_CLOSE_FRAC
            )
        return 1.0 - _smoothstep(t)

    def _advance_gaze(
        self,
        dt: float,
        person_pos: tuple[float, float] | None,
        move: dict[str, float],
        mode: str,
        speaking_break_scale: float,
    ) -> tuple[float, float]:
        rest = (0.0, move["gaze_y_bias"])

        if mode == "listening":
            target = (
                self._track_target(dt, person_pos, rest, move, allow_break=False)
                if person_pos is not None
                else rest
            )
        elif mode == "thinking":
            target = (
                _THINK_GAZE[0],
                _clamp(_THINK_GAZE[1] + rest[1], 1.0),
            )
        elif mode == "speaking":
            target = self._speaking_target(
                dt,
                person_pos,
                rest,
                move,
                speaking_break_scale,
            )
        elif mode == "tracking":
            target = self._track_target(dt, person_pos, rest, move, allow_break=True)
        else:
            self._hold -= dt
            if self._hold <= 0.0:
                self._fixation = self._pick_fixation()
                self._hold = self._rng.uniform(*_HOLD) * move["hold"]
            spread = move["spread"]
            target = (
                _clamp(self._fixation[0] * spread, 1.0),
                _clamp(self._fixation[1] * spread + rest[1], 1.0),
            )

        tau = _GAZE_TAU_MAX * move["gaze_lag"]
        if tau <= 1e-4:
            self._gaze = target
        else:
            alpha = 1.0 - math.exp(-dt / tau)
            self._gaze = (
                self._gaze[0] + (target[0] - self._gaze[0]) * alpha,
                self._gaze[1] + (target[1] - self._gaze[1]) * alpha,
            )

        self._drift_t += dt
        dx = _DRIFT_AMP * math.sin(self._drift_t * _DRIFT_FREQ[0])
        dy = _DRIFT_AMP * math.sin(self._drift_t * _DRIFT_FREQ[1] + 1.3)
        return (
            _clamp(self._gaze[0] + dx, 1.0),
            _clamp(self._gaze[1] + dy, 1.0),
        )

    def _speaking_target(
        self,
        dt: float,
        person_pos: tuple[float, float] | None,
        rest: tuple[float, float],
        move: dict[str, float],
        break_scale: float,
    ) -> tuple[float, float]:
        if self._speak_break_left > 0.0:
            self._speak_break_left -= dt
            return (
                _SPEAK_BREAK_GAZE[0],
                _clamp(_SPEAK_BREAK_GAZE[1] + rest[1], 1.0),
            )

        scale = max(float(break_scale), 0.0)
        if scale > 0.0:
            self._speak_break_check -= dt * scale
            if self._speak_break_check <= 0.0:
                self._speak_break_check = self._rng.uniform(*_SPEAK_BREAK_CHECK)
                self._speak_break_left = self._rng.uniform(*_SPEAK_BREAK_LENGTH)
                return (
                    _SPEAK_BREAK_GAZE[0],
                    _clamp(_SPEAK_BREAK_GAZE[1] + rest[1], 1.0),
                )

        if person_pos is None:
            return rest
        return self._track_target(dt, person_pos, rest, move, allow_break=False)

    def _track_target(
        self,
        dt: float,
        person_pos: tuple[float, float] | None,
        rest: tuple[float, float],
        move: dict[str, float],
        *,
        allow_break: bool,
    ) -> tuple[float, float]:
        if person_pos is None:
            return rest

        if allow_break and move["track_break"] > 0.0:
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
        full = (
            _clamp((px - 0.5) * 2.0, 1.0),
            _clamp((py - 0.5) * 1.6, 1.0),
        )
        bias = move["track_bias"]
        return (
            _clamp(rest[0] + (full[0] - rest[0]) * bias, 1.0),
            _clamp(rest[1] + (full[1] - rest[1]) * bias, 1.0),
        )

    def _pick_fixation(self) -> tuple[float, float]:
        choices = [point for point in _FIXATIONS if point != self._fixation]
        return self._rng.choice(choices)

    def _advance_face(
        self,
        dt: float,
        mode: str,
        person_pos: tuple[float, float] | None,
        move: dict[str, float],
    ) -> tuple[float, float]:
        ease = move["ease"]
        bias = move["track_bias"]
        spread = move["spread"]

        if mode == "thinking":
            target, tau = _THINK_OFFSET, _FACE_TAU_ACTIVE * ease
        elif mode == "listening":
            lean = self._lean(person_pos, _LISTEN_LEAN * bias)
            target = (lean[0], lean[1] + _LISTEN_SETTLE)
            tau = _FACE_TAU_ACTIVE * ease
        elif mode in ("speaking", "tracking"):
            target = self._lean(person_pos, _TRACK_LEAN * bias)
            tau = _FACE_TAU_ACTIVE * ease
        else:
            self._face_hold -= dt
            if self._face_hold <= 0.0:
                choices = [point for point in _FACE_IDLE_POINTS if point != self._face_target]
                self._face_target = self._rng.choice(choices)
                self._face_hold = self._rng.uniform(*_FACE_IDLE_HOLD) * move["hold"]
            target = (
                self._face_target[0] * spread,
                self._face_target[1] * spread,
            )
            tau = _FACE_TAU_IDLE * ease

        amp = _FACE_BOB_AMP * move["bob"]
        reach = max(_FACE_MAX - amp, 0.5)

        alpha = 1.0 - math.exp(-dt / max(tau, 1e-3))
        self.face_offset = (
            _clamp(
                self.face_offset[0] + (target[0] - self.face_offset[0]) * alpha,
                reach,
            ),
            _clamp(
                self.face_offset[1] + (target[1] - self.face_offset[1]) * alpha,
                reach,
            ),
        )

        period = _BOB_PERIOD / max(move["bob"], 0.25)
        self._bob_phase += 2.0 * math.pi * dt / period
        return (
            self.face_offset[0],
            self.face_offset[1] + amp * math.sin(self._bob_phase),
        )

    @staticmethod
    def _lean(
        person_pos: tuple[float, float] | None,
        gain: float,
    ) -> tuple[float, float]:
        if person_pos is None:
            return (0.0, 0.0)
        px, py = person_pos
        return (
            _clamp((px - 0.5) * 2.0 * gain, _FACE_MAX),
            _clamp((py - 0.5) * 2.0 * gain * _LEAN_Y_DAMP, _FACE_MAX),
        )