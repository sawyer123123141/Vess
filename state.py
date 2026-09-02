"""Everything Vess knows about itself and the room, in one place.

Producer threads write here; the render loop and the prompt builder read.
There are no other shared globals.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from performance import PerformanceCue


@dataclass(frozen=True)
class ConversationTurn:
    timestamp: float
    user: str
    assistant: str
    status: str = "completed"
    interrupted_clause: str | None = None


@dataclass
class State:
    # identity
    persona: str = "friendly"
    mood: str = "neutral"
    mood_until: float = 0.0         # epoch; decays to neutral past this
    performance: PerformanceCue = field(default_factory=PerformanceCue)

    # appearance
    color: tuple[int, int, int] | None = None
    brightness: float = 0.7

    # perception
    person_present: bool = False
    person_pos: tuple[float, float] | None = None   # normalised 0-1
    present_since: float | None = None
    last_seen: float | None = None
    active_window: str = ""
    objects: list[str] = field(default_factory=list)

    # short-term memory
    conversation_turns: list[ConversationTurn] = field(default_factory=list)

    # runtime
    listening: bool = False
    thinking: bool = False
    speaking: bool = False
    muted_until: float = 0.0
    last_spoke: float = 0.0
    last_interaction: float = 0.0

    # Local-only operator diagnostics. These are not Vess memory.
    debug_values: dict[str, object] = field(default_factory=dict, repr=False)
    debug_events: list[dict[str, object]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # Reentrant because a producer holding the lock may call a helper that
        # takes it again. Assigned here rather than declared as a field so it
        # stays out of the dataclass repr and equality.
        self.lock = threading.RLock()

    @contextmanager
    def locked(self) -> Iterator["State"]:
        with self.lock:
            yield self

    def expire_mood(self, now: float) -> tuple[str, float] | None:
        """Return an expired mood to neutral and report the transition."""
        with self.locked():
            if self.mood == "neutral" or self.mood_until <= 0.0 or now < self.mood_until:
                return None

            previous_mood = self.mood
            previous_until = self.mood_until
            self.mood = "neutral"
            self.mood_until = 0.0
            return previous_mood, previous_until

    def update_debug(self, **values: object) -> None:
        """Replace live diagnostic values without retaining a history entry."""
        with self.locked():
            self.debug_values.update(values)

    def record_debug(self, event_type: str, **payload: object) -> None:
        """Keep a small in-memory operator event history."""
        with self.locked():
            self.debug_events.append(
                {"timestamp": time.time(), "event": event_type, **payload}
            )
            del self.debug_events[:-20]

    def debug_snapshot(self) -> dict[str, object]:
        """Return a lock-consistent browser diagnostics snapshot."""
        with self.locked():
            return {
                "runtime": {
                    "listening": self.listening,
                    "thinking": self.thinking,
                    "speaking": self.speaking,
                },
                "values": dict(self.debug_values),
                "events": [dict(event) for event in self.debug_events],
            }
