"""Everything Vess knows about itself and the room, in one place.

Producer threads write here; the render loop and the prompt builder read.
There are no other shared globals.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class State:
    # identity
    persona: str = "friendly"
    mood: str = "neutral"
    mood_until: float = 0.0         # epoch; decays to neutral past this

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

    # runtime
    listening: bool = False
    thinking: bool = False
    speaking: bool = False
    muted_until: float = 0.0
    last_spoke: float = 0.0

    def __post_init__(self) -> None:
        # Reentrant because a producer holding the lock may call a helper that
        # takes it again. Assigned here rather than declared as a field so it
        # stays out of the dataclass repr and equality.
        self.lock = threading.RLock()

    @contextmanager
    def locked(self) -> Iterator["State"]:
        with self.lock:
            yield self
