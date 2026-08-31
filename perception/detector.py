"""Tier 1 perception: what is in the room, continuously and cheaply.

YOLO on every frame gives an object list and a box per person. Diffing the
object list between frames is the motion signal -- there is no separate motion
detector.

Runs on the CPU by design. The GPU is reserved for Ollama, and PLAN.md is
explicit that overflowing 8GB drops throughput ~30x with no graceful
degradation. If this is ever too slow, lower the frame rate or the frame size
rather than moving it to the GPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from state import State

# A detector that drops one frame should not make the face look away and back.
# Presence survives this long past the last sighting.
_ABSENCE_GRACE = 1.5

# Gaze goes to the head, not the middle of the torso. A person's head sits
# roughly this far down their box.
_HEAD_FRACTION = 0.22


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2, normalised 0-1

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)

    @property
    def gaze_point(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y1 + (y2 - y1) * _HEAD_FRACTION)


def pick_subject(detections: list[Detection]) -> Detection | None:
    """Choose which person the face pays attention to.

    Largest box for now, which is a reasonable stand-in for nearest.

    **This is the only place that choice is made.** See "Future -- more than
    one person" under Eye movement in PLAN.md: the subject should eventually be
    whoever is speaking, which needs direction-of-arrival the perception layer
    does not have yet. When it arrives, this function body changes and nothing
    else does -- so never collapse the candidate list before calling this, and
    never re-derive a subject anywhere else.
    """
    people = [d for d in detections if d.label == "person"]
    if not people:
        return None
    return max(people, key=lambda d: d.area)


class Detector:
    """Frames in, detections out. No state, no threads, no clock."""

    def __init__(self, model_path: str, confidence: float = 0.4) -> None:
        from ultralytics import YOLO       # imported late; it is slow to load

        self._model = YOLO(model_path)
        self._confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self._model.predict(
            frame, conf=self._confidence, device="cpu", verbose=False)

        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(Detection(
                    label=str(names[int(box.cls[0])]),
                    confidence=float(box.conf[0]),
                    box=(x1 / width, y1 / height, x2 / width, y2 / height),
                ))
        return detections


def write_state(state: State, detections: list[Detection], now: float) -> None:
    """Fold one frame's detections into State.

    Every person the detector saw stays in the running until `pick_subject`
    chooses; nothing upstream of it collapses the candidates.
    """
    subject = pick_subject(detections)
    labels = sorted({d.label for d in detections})

    with state.locked():
        state.objects = labels
        if subject is not None:
            if not state.person_present:
                state.present_since = now
            state.person_present = True
            state.person_pos = subject.gaze_point
            state.last_seen = now
        elif state.person_present and now - (state.last_seen or now) > _ABSENCE_GRACE:
            state.person_present = False
            state.person_pos = None
            state.present_since = None


def run_detection_loop(state: State, camera, detector: Detector, fps: float,
                       stop: threading.Event) -> None:
    """Tier 1's own thread. Never touches the render loop's timing."""
    interval = 1.0 / max(fps, 0.1)
    while not stop.is_set():
        started = time.perf_counter()
        frame = camera.read()
        if frame is None:
            break
        write_state(state, detector.detect(frame), time.time())
        stop.wait(max(0.0, interval - (time.perf_counter() - started)))
