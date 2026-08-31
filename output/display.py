"""Where a rendered frame goes.

face.py decides what the face looks like; this decides who sees it. A target
is anything with show(). The websocket preview and the LED panel arrive later
as more targets, not as changes to this file.
"""

from __future__ import annotations

import cv2
import numpy as np

_NO_KEY = -1


class DisplayTarget:
    def show(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def poll_key(self) -> int:
        """Last key pressed on this target, or -1. Most targets have no input."""
        return _NO_KEY

    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        pass


class PreviewWindow(DisplayTarget):
    """A desktop window showing the panel at `scale` times its real size.

    Upscaling is nearest-neighbour on purpose: blocky edges are what the panel
    will actually look like, and smoothing them here would hide problems until
    the hardware arrives.
    """

    def __init__(self, scale: int = 8, title: str = "Vess") -> None:
        self._scale = max(int(scale), 1)
        self._title = title
        self._last_key = _NO_KEY
        cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)

    def show(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        big = cv2.resize(
            frame,
            (width * self._scale, height * self._scale),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(self._title, cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        # waitKey is also what pumps the window's event loop, so it has to be
        # called every frame whether or not anyone is pressing anything.
        key = cv2.waitKey(1)
        if key != _NO_KEY:
            self._last_key = key & 0xFF

    def poll_key(self) -> int:
        key, self._last_key = self._last_key, _NO_KEY
        return key

    def is_open(self) -> bool:
        try:
            return cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        # The window is already gone if the user closed it with the title bar,
        # and destroying it again raises rather than no-opping.
        try:
            cv2.destroyWindow(self._title)
        except cv2.error:
            pass


class Display:
    """Fans one frame out to every target."""

    def __init__(self, targets: list[DisplayTarget]) -> None:
        self._targets = list(targets)

    def show(self, frame: np.ndarray) -> None:
        for target in self._targets:
            target.show(frame)

    def poll_key(self) -> int:
        for target in self._targets:
            key = target.poll_key()
            if key != _NO_KEY:
                return key
        return _NO_KEY

    def is_open(self) -> bool:
        return all(target.is_open() for target in self._targets)

    def close(self) -> None:
        for target in self._targets:
            target.close()
