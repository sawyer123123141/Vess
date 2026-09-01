"""Deterministic synthetic State timelines for headless Vess verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from performance import PerformanceCue
from state import State

_PERSON = (0.80, 0.48)
_STRESS_POSITIONS = (
    (0.10, 0.15),
    (0.90, 0.15),
    (0.10, 0.85),
    (0.90, 0.85),
)


@dataclass(frozen=True)
class ScenarioPhase:
    name: str
    duration_seconds: float
    state: dict[str, object]


@dataclass(frozen=True)
class BehaviorScenario:
    name: str
    phases: tuple[ScenarioPhase, ...]
    seed: int = 1


def phase_frame_count(phase: ScenarioPhase, fps: int) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not math.isfinite(phase.duration_seconds):
        raise ValueError(f"phase {phase.name!r} duration must be finite")
    frames = int(round(phase.duration_seconds * fps))
    if frames <= 0:
        raise ValueError(f"phase {phase.name!r} must contain at least one frame")
    return frames


def apply_phase(state: State, phase: ScenarioPhase) -> None:
    """Apply only fields explicitly declared by one synthetic phase."""
    with state.locked():
        for name, value in phase.state.items():
            if not hasattr(state, name):
                raise ValueError(f"unknown State field in phase {phase.name}: {name}")
            setattr(state, name, value)


def get_scenario(
    name: str,
    *,
    moods: Iterable[str],
    performances: dict[str, PerformanceCue],
) -> BehaviorScenario:
    """Build one named deterministic scenario from the configured vocabulary."""
    if name == "conversational_cycle":
        return _conversational_cycle(performances)
    if name == "priority_conflicts":
        return _priority_conflicts(performances)
    if name == "geometry_stress":
        return _geometry_stress(tuple(moods), performances)
    if name == "eye_reaction_cycle":
        return _eye_reaction_cycle(performances)
    raise ValueError(f"unknown behavior scenario: {name}")


def _require_cue(
    performances: dict[str, PerformanceCue],
    name: str,
) -> PerformanceCue:
    cue = performances.get(name)
    if cue is None:
        raise ValueError(f"behavior scenario requires performance {name!r}")
    return cue


def _phase_state(
    *,
    performance: PerformanceCue,
    person: bool,
    listening: bool,
    thinking: bool,
    speaking: bool,
    mood: str = "neutral",
) -> dict[str, object]:
    return {
        "mood": mood,
        "mood_until": 0.0,
        "performance": performance,
        "person_present": person,
        "person_pos": _PERSON if person else None,
        "listening": listening,
        "thinking": thinking,
        "speaking": speaking,
    }


def _conversational_cycle(
    performances: dict[str, PerformanceCue],
) -> BehaviorScenario:
    neutral = _require_cue(performances, "neutral")
    thoughtful = _require_cue(performances, "thoughtful")
    playful = _require_cue(performances, "playful")
    emphatic = _require_cue(performances, "emphatic")

    phases = (
        ScenarioPhase(
            "idle",
            1.0,
            _phase_state(
                performance=neutral,
                person=False,
                listening=False,
                thinking=False,
                speaking=False,
            ),
        ),
        ScenarioPhase(
            "tracking",
            1.0,
            _phase_state(
                performance=neutral,
                person=True,
                listening=False,
                thinking=False,
                speaking=False,
            ),
        ),
        ScenarioPhase(
            "listening",
            1.5,
            _phase_state(
                performance=neutral,
                person=True,
                listening=True,
                thinking=False,
                speaking=False,
            ),
        ),
        ScenarioPhase(
            "thinking",
            1.5,
            _phase_state(
                performance=thoughtful,
                person=True,
                listening=False,
                thinking=True,
                speaking=False,
            ),
        ),
        ScenarioPhase(
            "speaking_neutral",
            2.0,
            _phase_state(
                performance=neutral,
                person=True,
                listening=False,
                thinking=False,
                speaking=True,
            ),
        ),
        ScenarioPhase(
            "speaking_playful",
            2.0,
            _phase_state(
                performance=playful,
                person=True,
                listening=False,
                thinking=False,
                speaking=True,
            ),
        ),
        ScenarioPhase(
            "speaking_emphatic",
            2.0,
            _phase_state(
                performance=emphatic,
                person=True,
                listening=False,
                thinking=False,
                speaking=True,
            ),
        ),
        ScenarioPhase(
            "return_idle",
            1.0,
            _phase_state(
                performance=neutral,
                person=False,
                listening=False,
                thinking=False,
                speaking=False,
            ),
        ),
    )
    return BehaviorScenario("conversational_cycle", phases)


def _priority_conflicts(
    performances: dict[str, PerformanceCue],
) -> BehaviorScenario:
    neutral = _require_cue(performances, "neutral")
    phases = (
        ScenarioPhase(
            "idle",
            0.5,
            _phase_state(
                performance=neutral,
                person=False,
                listening=False,
                thinking=False,
                speaking=False,
            ),
        ),
        ScenarioPhase(
            "tracking",
            0.5,
            _phase_state(
                performance=neutral,
                person=True,
                listening=False,
                thinking=False,
                speaking=False,
            ),
        ),
        ScenarioPhase(
            "speaking",
            0.5,
            _phase_state(
                performance=neutral,
                person=True,
                listening=False,
                thinking=False,
                speaking=True,
            ),
        ),
        ScenarioPhase(
            "thinking_over_speaking",
            0.5,
            _phase_state(
                performance=neutral,
                person=True,
                listening=False,
                thinking=True,
                speaking=True,
            ),
        ),
        ScenarioPhase(
            "listening_over_all",
            0.5,
            _phase_state(
                performance=neutral,
                person=True,
                listening=True,
                thinking=True,
                speaking=True,
            ),
        ),
    )
    return BehaviorScenario("priority_conflicts", phases)


def _geometry_stress(
    moods: tuple[str, ...],
    performances: dict[str, PerformanceCue],
) -> BehaviorScenario:
    if not moods:
        raise ValueError("geometry_stress requires at least one mood")
    if not performances:
        raise ValueError("geometry_stress requires at least one performance")

    phases: list[ScenarioPhase] = []
    index = 0
    for mood in moods:
        for expression, cue in performances.items():
            position = _STRESS_POSITIONS[index % len(_STRESS_POSITIONS)]
            phases.append(
                ScenarioPhase(
                    f"geometry_{mood}_{expression}_{index}",
                    0.2,
                    {
                        "mood": mood,
                        "mood_until": 0.0,
                        "performance": cue,
                        "person_present": True,
                        "person_pos": position,
                        "listening": False,
                        "thinking": False,
                        "speaking": True,
                    },
                )
            )
            index += 1
    return BehaviorScenario("geometry_stress", tuple(phases))


def _eye_reaction_cycle(
    performances: dict[str, PerformanceCue],
) -> BehaviorScenario:
    neutral = _require_cue(performances, "neutral")
    curious = _require_cue(performances, "curious")
    playful = _require_cue(performances, "playful")
    emphatic = _require_cue(performances, "emphatic")
    thoughtful = _require_cue(performances, "thoughtful")
    sympathetic = _require_cue(performances, "sympathetic")
    uncertain = _require_cue(performances, "uncertain")

    sequence = (
        ("neutral_1", 0.6, neutral),
        ("curious", 0.8, curious),
        ("neutral_2", 0.6, neutral),
        ("playful", 0.8, playful),
        ("neutral_3", 0.6, neutral),
        ("emphatic", 0.8, emphatic),
        ("thoughtful", 0.8, thoughtful),
        ("sympathetic", 0.8, sympathetic),
        ("uncertain", 0.8, uncertain),
        ("neutral_4", 0.8, neutral),
    )
    phases = tuple(
        ScenarioPhase(
            phase_name,
            duration,
            _phase_state(
                performance=cue,
                person=True,
                listening=False,
                thinking=False,
                speaking=True,
            ),
        )
        for phase_name, duration, cue in sequence
    )
    return BehaviorScenario("eye_reaction_cycle", phases)
