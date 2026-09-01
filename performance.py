"""Transient performance cues and validated visual overlay definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceCue:
    expression: str = "neutral"
    intensity: float = 0.0


_SHAPE_LIMITS = {
    "l_h": (-3.0, 3.0),
    "r_h": (-3.0, 3.0),
    "l_slant": (-1.5, 1.5),
    "r_slant": (-1.5, 1.5),
    "l_cy": (-1.5, 1.5),
    "r_cy": (-1.5, 1.5),
}

_EYE_MOTION_LIMITS = {
    "l_x": (-1.5, 1.5),
    "l_y": (-1.5, 1.5),
    "r_x": (-1.5, 1.5),
    "r_y": (-1.5, 1.5),
    "reaction": (0.0, 1.0),
}

_EYE_MOTION_DEFAULTS = {
    "l_x": 0.0,
    "l_y": 0.0,
    "r_x": 0.0,
    "r_y": 0.0,
    "reaction": 0.0,
}

_MOVEMENT_LIMITS = {
    "hold_scale": (0.6, 1.6),
    "ease_scale": (0.7, 1.5),
    "track_bias_scale": (0.7, 1.3),
    "gaze_y_bias": (-0.35, 0.35),
    "speaking_break_scale": (0.0, 2.0),
}

_MOVEMENT_DEFAULTS = {
    "hold_scale": 1.0,
    "ease_scale": 1.0,
    "track_bias_scale": 1.0,
    "gaze_y_bias": 0.0,
    "speaking_break_scale": 1.0,
}


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _number(value: object, default: float) -> float:
    """Return a finite numeric config value or its safe default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def load_performance_definitions(
    raw: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Validate a human-authored performance mapping into bounded numeric values."""
    if "neutral" not in raw:
        raise ValueError("performance config requires neutral")

    cleaned: dict[str, dict[str, object]] = {}
    for name, value in raw.items():
        entry = value if isinstance(value, dict) else {}
        intensity = _clamp(_number(entry.get("intensity", 0.0), 0.0), 0.0, 1.0)

        shape_value = entry.get("shape", {})
        shape_raw = shape_value if isinstance(shape_value, dict) else {}
        shape = {
            key: _clamp(_number(shape_raw.get(key, 0.0), 0.0), low, high)
            for key, (low, high) in _SHAPE_LIMITS.items()
        }

        eye_motion_value = entry.get("eye_motion", {})
        eye_motion_raw = eye_motion_value if isinstance(eye_motion_value, dict) else {}
        eye_motion = {
            key: _clamp(
                _number(
                    eye_motion_raw.get(key, _EYE_MOTION_DEFAULTS[key]),
                    _EYE_MOTION_DEFAULTS[key],
                ),
                low,
                high,
            )
            for key, (low, high) in _EYE_MOTION_LIMITS.items()
        }
        if name == "neutral":
            eye_motion = dict(_EYE_MOTION_DEFAULTS)

        movement_value = entry.get("movement", {})
        movement_raw = movement_value if isinstance(movement_value, dict) else {}
        movement = {
            key: _clamp(
                _number(movement_raw.get(key, _MOVEMENT_DEFAULTS[key]), _MOVEMENT_DEFAULTS[key]),
                low,
                high,
            )
            for key, (low, high) in _MOVEMENT_LIMITS.items()
        }

        cleaned[name] = {
            "intensity": intensity,
            "shape": shape,
            "eye_motion": eye_motion,
            "movement": movement,
        }

    return cleaned


def cue_for_label(
    label: str,
    definitions: dict[str, dict[str, object]],
) -> PerformanceCue:
    """Return one configured cue, falling back to neutral for unknown labels."""
    name = label.strip().lower()
    entry = definitions.get(name)
    if entry is None:
        return PerformanceCue()
    return PerformanceCue(name, float(entry["intensity"]))
