"""Vess.

Steps 1-2: build the state, run the detector in its own thread, render the
face from what it finds. The render loop never waits on perception.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from pathlib import Path

from control.web import WebServer
from output.animator import FaceAnimator
from output.display import Display, PreviewWindow
from perception import camera as camera_module
from perception.detector import Detector, run_detection_loop
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


def _start_perception(config: dict, state: State,
                      stop: threading.Event) -> tuple[threading.Thread | None, object]:
    """Bring up tier 1, or say why not and carry on without it.

    A missing camera must not stop the face rendering -- and without a source
    the fake-position key below is the only way to exercise tracking, so it
    stays useful rather than being replaced.
    """
    try:
        camera = camera_module.open_camera(config)
    except RuntimeError as error:
        print(f"perception off: {error}")
        return None, None

    try:
        detector = Detector(config["detector"]["model"],
                            float(config["detector"].get("confidence", 0.4)))
    except Exception as error:                      # model load, download, weights
        print(f"perception off: cannot load detector -- {error}")
        camera.close()
        return None, None

    thread = threading.Thread(
        target=run_detection_loop,
        args=(state, camera, detector, float(config["detector"]["fps"]), stop),
        name="detector",
        daemon=True,
    )
    thread.start()
    source = config.get("camera", {}).get("source", "camera")
    mirrored = "mirrored" if config.get("camera", {}).get("mirror", True) else "not mirrored"
    print(f"perception on: {source}, {mirrored}, "
          f"{config['detector']['fps']}fps on cpu")
    return thread, camera


def _build_display(config: dict, state: State) -> tuple[Display, WebServer | None]:
    """Build every enabled output target around the one rendered frame."""
    display_config = config.get("display", {})
    web_config = config.get("web", {})
    targets = []
    web_server = None

    if web_config.get("enabled", True):
        web_server = WebServer(state, int(web_config.get("port", 8080)))
        targets.append(web_server.preview)
    if display_config.get("cv2_enabled", False):
        targets.append(PreviewWindow(scale=display_config.get("preview_scale", 8)))
    if not targets:
        raise RuntimeError("no display targets enabled")
    return Display(targets), web_server


def _open_browser(port: int) -> None:
    webbrowser.open(f"http://127.0.0.1:{port}")


def main() -> None:
    config = _load("config.json")
    moods = _load("moods.json")
    mood_names = list(moods)

    state = State()
    animator = FaceAnimator(moods)
    display, web_server = _build_display(config, state)

    stop = threading.Event()
    thread, camera = _start_perception(config, state, stop)
    if web_server is not None:
        web_server.start()
        if config.get("web", {}).get("open_browser_on_start", True):
            threading.Thread(
                target=_open_browser,
                args=(int(config["web"].get("port", 8080)),),
                name="browser",
                daemon=True,
            ).start()

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
        stop.set()
        if thread is not None:
            thread.join(timeout=2.0)
        if camera is not None:
            camera.close()
        if web_server is not None:
            web_server.close()
        display.close()


if __name__ == "__main__":
    main()
