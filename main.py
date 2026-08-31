"""Vess.

Step 1: build the state, render the face, show it in a window. No perception,
no voice, no threads yet -- the loop is the only thing running.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from output.animator import FaceAnimator
from output.display import Display, PreviewWindow
from state import State

ROOT = Path(__file__).resolve().parent
FRAME_TIME = 1.0 / 30.0

# Stand-ins for the detector, so the tracking path can be exercised before
# camera.py exists. Normalised 0-1, same as state.person_pos will be.
FAKE_PERSON_POSITIONS: tuple[tuple[float, float] | None, ...] = (
    None,
    (0.12, 0.50),
    (0.50, 0.25),
    (0.88, 0.50),
    (0.50, 0.78),
)


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> None:
    config = _load("config.json")
    moods = _load("moods.json")
    mood_names = list(moods)

    state = State()
    animator = FaceAnimator(moods)
    display = Display([PreviewWindow(scale=config["display"]["preview_scale"])])

    print("keys:")
    for index, name in enumerate(mood_names, start=1):
        print(f"  {index}  mood -> {name}")
    print("  t  cycle a fake person_pos")
    print("  k  toggle thinking   (face drifts up and away)")
    print("  l  toggle listening  (face settles and leans in)")
    print("  q  quit")

    fake = 0
    last = time.perf_counter()
    try:
        while display.is_open():
            now = time.perf_counter()
            # A stall shouldn't fast-forward the animation to catch up.
            dt = min(now - last, 0.1)
            last = now

            display.show(animator.tick(state, dt))

            key = display.poll_key()
            if key in (27, ord("q")):
                break
            if ord("1") <= key < ord("1") + len(mood_names):
                mood = mood_names[key - ord("1")]
                with state.locked():
                    state.mood = mood
                    state.mood_until = 0.0
                print(f"mood -> {mood}")
            elif key == ord("t"):
                fake = (fake + 1) % len(FAKE_PERSON_POSITIONS)
                position = FAKE_PERSON_POSITIONS[fake]
                with state.locked():
                    state.person_pos = position
                    state.person_present = position is not None
                print(f"person_pos -> {position}")
            elif key in (ord("k"), ord("l")):
                # The face reacts to thinking and listening, and nothing sets
                # either yet, so they need a key to be visible at all.
                field = "thinking" if key == ord("k") else "listening"
                with state.locked():
                    value = not getattr(state, field)
                    setattr(state, field, value)
                print(f"{field} -> {value}")

            time.sleep(max(0.0, FRAME_TIME - (time.perf_counter() - now)))
    finally:
        display.close()


if __name__ == "__main__":
    main()
