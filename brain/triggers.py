"""Deterministic decisions for conservative unprompted speech."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TriggerSnapshot:
    person_present: bool
    present_since: float | None
    last_interaction: float
    muted_until: float
    listening: bool
    thinking: bool
    speaking: bool


@dataclass(frozen=True)
class TriggerEvent:
    name: str
    context: str
    duration_seconds: float


class TriggerDecider:
    """Recognize meaningful transitions and apply hard proactive-speech gates."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._min_absence_seconds = max(
            float(settings.get("min_absence_hours", 4)), 0.0
        ) * 3600.0
        self._idle_seconds = max(
            float(settings.get("idle_interaction_minutes", 30)), 0.0
        ) * 60.0
        self._cooldown_seconds = max(
            float(settings.get("cooldown_minutes", 60)), 0.0
        ) * 60.0
        self._quiet_after_hour = int(settings.get("quiet_after_hour", 22)) % 24
        self._quiet_before_hour = int(settings.get("quiet_before_hour", 8)) % 24

        self._previous_present: bool | None = None
        self._absent_since: float | None = None
        self._last_accepted_at: float | None = None
        self._last_interaction_seen = 0.0
        self._proactive_since_interaction = False

    def evaluate(
        self,
        snapshot: TriggerSnapshot,
        *,
        now: float,
        local_hour: int,
    ) -> TriggerEvent | None:
        """Return one eligible event without consuming its cooldown or one-shot latch."""
        if snapshot.last_interaction > self._last_interaction_seen:
            self._last_interaction_seen = snapshot.last_interaction
            self._proactive_since_interaction = False

        returned_duration = self._observe_presence(snapshot.person_present, now)
        if not self._passes_common_gates(snapshot, now, local_hour):
            return None

        if (
            returned_duration is not None
            and returned_duration >= self._min_absence_seconds
            and not self._proactive_since_interaction
        ):
            hours = max(int(returned_duration // 3600.0), 1)
            return TriggerEvent(
                "returned_after_absence",
                f"A person just returned after being absent for about {hours} hours.",
                returned_duration,
            )

        if self._proactive_since_interaction:
            return None

        baseline = max(
            snapshot.last_interaction,
            snapshot.present_since if snapshot.present_since is not None else now,
        )
        idle_duration = max(now - baseline, 0.0)
        if idle_duration < self._idle_seconds:
            return None

        minutes = max(int(idle_duration // 60.0), 1)
        return TriggerEvent(
            "quiet_interaction",
            "The room has been quiet between Vess and the person for "
            f"about {minutes} minutes.",
            idle_duration,
        )

    def accept(self, event: TriggerEvent, now: float) -> None:
        """Consume the global cooldown and one-shot latch after submission succeeds."""
        self._last_accepted_at = now
        self._proactive_since_interaction = True

    def _observe_presence(self, present: bool, now: float) -> float | None:
        previous = self._previous_present
        self._previous_present = present

        if previous is None:
            return None
        if previous and not present:
            self._absent_since = now
            self._proactive_since_interaction = False
            return None
        if not previous and present:
            absent_since = self._absent_since
            self._absent_since = None
            if absent_since is None:
                return None
            return max(now - absent_since, 0.0)
        return None

    def _passes_common_gates(
        self,
        snapshot: TriggerSnapshot,
        now: float,
        local_hour: int,
    ) -> bool:
        if not snapshot.person_present:
            return False
        if snapshot.muted_until > now:
            return False
        if snapshot.listening or snapshot.thinking or snapshot.speaking:
            return False
        if self._in_quiet_hours(local_hour):
            return False
        if (
            self._last_accepted_at is not None
            and now - self._last_accepted_at < self._cooldown_seconds
        ):
            return False
        return True

    def _in_quiet_hours(self, local_hour: int) -> bool:
        hour = int(local_hour) % 24
        after = self._quiet_after_hour
        before = self._quiet_before_hour
        if after == before:
            return False
        if after < before:
            return after <= hour < before
        return hour >= after or hour < before
