"""Where frames come from.

A source is anything with `read()`. The live camera is one implementation; a
still image and a video file are the others, so the whole perception pipeline
runs and can be verified with no hardware attached. Which one is used is a
config value, not a code change.

Frames are downscaled here, once, before anything else sees them.
"""

from __future__ import annotations

import cv2
import numpy as np


class FrameSource:
    def read(self) -> np.ndarray | None:
        """Next frame as BGR, or None when the source is finished."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraSource(FrameSource):
    def __init__(self, index: int) -> None:
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"no camera at index {index}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        self._capture.release()


class ImageSource(FrameSource):
    """One still image, returned forever.

    Drop a photo of the room in and the detector, the subject picker and
    everything downstream of them can be checked without a camera.
    """

    def __init__(self, path: str) -> None:
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"cannot read image: {path}")
        self._frame = frame

    def read(self) -> np.ndarray | None:
        # A copy, because consumers are free to draw on what they are given.
        return self._frame.copy()


class VideoSource(FrameSource):
    """A video file, looped."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"cannot read video: {path}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        self._capture.release()


class Camera:
    """A source plus the two things every frame needs doing to it.

    Mirroring happens here rather than by flipping coordinates later, so there
    is exactly one place that decides which way round the room is. It is on by
    default: this is a face on a wall looking at a room, so moving to your left
    should send the eyes to your left.
    """

    def __init__(self, source: FrameSource, mirror: bool = True,
                 max_px: int = 512) -> None:
        self._source = source
        self._mirror = mirror
        self._max_px = max_px

    def read(self) -> np.ndarray | None:
        frame = self._source.read()
        if frame is None:
            return None
        if self._mirror:
            frame = cv2.flip(frame, 1)
        return _downscale(frame, self._max_px)

    def close(self) -> None:
        self._source.close()


def _downscale(frame: np.ndarray, max_px: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_px:
        return frame
    scale = max_px / longest
    return cv2.resize(frame, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA)


def open_camera(config: dict) -> Camera:
    """Build the camera described by config's `camera` block.

    Raises RuntimeError if the source cannot be opened -- the caller decides
    whether that is fatal.
    """
    settings = config.get("camera", {})
    kind = settings.get("source", "camera")
    path = settings.get("path", "")

    if kind == "camera":
        source: FrameSource = CameraSource(int(settings.get("index", 0)))
    elif kind == "image":
        source = ImageSource(path)
    elif kind == "video":
        source = VideoSource(path)
    else:
        raise RuntimeError(f"unknown camera source: {kind!r}")

    return Camera(
        source,
        mirror=bool(settings.get("mirror", True)),
        max_px=int(settings.get("max_frame_px", 512)),
    )
