"""Deterministic decisions for conservative unprompted speech."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


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


class TriggerWorker:
    """Poll State off the render thread and submit only eligible proactive events."""

    def __init__(
        self,
        state: Any,
        settings: dict[str, Any],
        submit: Callable[[str, str], bool],
        event_log: Any,
        *,
        poll_seconds: float = 0.5,
    ) -> None:
        self._state = state
        self._decider = TriggerDecider(settings)
        self._submit = submit
        self._event_log = event_log
        self._poll_seconds = max(float(poll_seconds), 0.001)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> TriggerSnapshot:
        with self._state.locked():
            return TriggerSnapshot(
                person_present=self._state.person_present,
                present_since=self._state.present_since,
                last_interaction=self._state.last_interaction,
                muted_until=self._state.muted_until,
                listening=self._state.listening,
                thinking=self._state.thinking,
                speaking=self._state.speaking,
            )

    def poll_once(
        self,
        *,
        now: float | None = None,
        local_hour: int | None = None,
    ) -> TriggerEvent | None:
        current_time = time.time() if now is None else float(now)
        hour = time.localtime(current_time).tm_hour if local_hour is None else int(local_hour)
        event = self._decider.evaluate(
            self.snapshot(),
            now=current_time,
            local_hour=hour,
        )
        if event is None:
            return None
        if not self._submit(event.name, event.context):
            return event

        self._decider.accept(event, current_time)
        self._event_log.append(
            "trigger_fired",
            {
                "trigger": event.name,
                "context": event.context,
                "duration_seconds": event.duration_seconds,
            },
        )
        return event

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="triggers",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join()
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self._poll_seconds)
