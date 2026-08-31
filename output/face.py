"""Rasterises the face into a 64x64 RGB array.

Nothing here knows about time or state. It takes a flat dict of shape numbers
and draws them. Everything that decides *what* those numbers should be lives
in animator.py.

Rendering is native 64x64 -- never larger and downscaled -- so what the
preview shows is exactly what the LED panel will show.
"""

from __future__ import annotations

import numpy as np

WIDTH = 64
HEIGHT = 64

# Two 16px-wide eyes centred at 32 +/- 12 leave a 12px margin each side, and
# sitting a little above centre reads as a face rather than two rectangles.
_EYE_DX = 12.0
_EYE_CY = 30.0

# A fully shut eye still draws a line. A gap that vanishes entirely reads as
# a crash, not a blink.
_MIN_EYE_H = 1.4

# Pixel centres, so the distance fields are correct at half-pixel offsets.
_XX, _YY = np.meshgrid(
    np.arange(WIDTH, dtype=np.float32) + 0.5,
    np.arange(HEIGHT, dtype=np.float32) + 0.5,
)

# One parameter vector per eye shape named in moods.json. They share keys on
# purpose: the animator interpolates between any two of them componentwise,
# so a mood change is a lerp rather than a swap.
#
#   w, h        eye box size in pixels
#   radius      corner rounding
#   slant       vertical px the inner corner drops (negative = outer drops)
#   arc         0 = full box, 1 = thin arch; carves the box from below
#   pupil       pupil opacity
#   pupil_r     pupil radius
#   pupil_y     resting pupil offset, positive = low in the eye
EYE_SHAPES: dict[str, dict[str, float]] = {
    "normal": {"w": 16.0, "h": 14.0, "radius": 5.0, "slant": 0.0,
               "arc": 0.0, "pupil": 1.0, "pupil_r": 3.4, "pupil_y": 0.0},
    "arc":    {"w": 17.0, "h": 14.0, "radius": 5.0, "slant": 0.0,
               "arc": 0.72, "pupil": 0.0, "pupil_r": 3.4, "pupil_y": 0.0},
    "narrow": {"w": 16.0, "h": 7.5, "radius": 3.2, "slant": 1.9,
               "arc": 0.0, "pupil": 1.0, "pupil_r": 2.8, "pupil_y": 0.0},
    "droop":  {"w": 15.0, "h": 12.0, "radius": 4.6, "slant": -2.1,
               "arc": 0.0, "pupil": 1.0, "pupil_r": 3.2, "pupil_y": 2.0},
    "wide":   {"w": 17.0, "h": 17.0, "radius": 6.0, "slant": 0.0,
               "arc": 0.0, "pupil": 1.0, "pupil_r": 3.0, "pupil_y": -0.4},
}


def shape_params(name: str) -> dict[str, float]:
    """Parameters for a named eye shape. Unknown names fall back to normal."""
    return dict(EYE_SHAPES.get(name, EYE_SHAPES["normal"]))


def _rounded_box(px: np.ndarray, py: np.ndarray,
                 half_w: float, half_h: float, radius: float) -> np.ndarray:
    """Signed distance to a rounded rectangle. Negative inside."""
    radius = min(radius, half_w, half_h)
    qx = np.abs(px) - (half_w - radius)
    qy = np.abs(py) - (half_h - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - radius


def _coverage(distance: np.ndarray) -> np.ndarray:
    """Distance field to alpha, with a one-pixel soft edge.

    The panel is full-colour per pixel, so a partly lit edge pixel is real
    output, not a resolution lie. At this size it is the difference between a
    curve and a staircase.
    """
    return np.clip(0.5 - distance, 0.0, 1.0)


def _eye(cx: float, inner_dir: float, shape: dict[str, float],
         openness: float, gaze: tuple[float, float]) -> np.ndarray:
    half_w = shape["w"] / 2.0
    half_h = max(shape["h"] * openness, _MIN_EYE_H) / 2.0

    px = _XX - cx
    # The slant ramps from zero at the outer edge to its full value at the
    # inner edge, tilting the eye the way a brow does.
    tilt = shape["slant"] * inner_dir * (px / half_w)
    py = _YY - _EYE_CY - tilt

    cover = _coverage(_rounded_box(px, py, half_w, half_h, shape["radius"]))

    arc = shape["arc"]
    if arc > 1e-3:
        # A circle rising from below. At arc=0 its top just grazes the bottom
        # edge and removes nothing; at arc=1 only a thin arch survives. The
        # radius has to be smaller than the eye is wide or the arch comes out
        # nearly straight.
        r = shape["w"] * 0.62
        cut_cy = half_h + r - arc * (2.0 * half_h + 1.0)
        cover = cover * (1.0 - _coverage(np.hypot(px, py - cut_cy) - r))

    pupil_alpha = shape["pupil"]
    pupil_r = min(shape["pupil_r"], max(half_h - 0.8, 0.0))
    if pupil_alpha > 1e-3 and pupil_r > 0.2:
        reach_x = max(half_w - pupil_r - 1.2, 0.0)
        reach_y = max(half_h - pupil_r - 1.2, 0.0)
        ox = gaze[0] * reach_x
        oy = float(np.clip(gaze[1] * reach_y + shape["pupil_y"], -reach_y, reach_y))
        pupil = _coverage(np.hypot(px - ox, py - oy) - pupil_r)
        cover = cover * (1.0 - pupil_alpha * pupil)

    return cover


def render(shape: dict[str, float], color: tuple[float, float, float],
           brightness: float, openness: float,
           gaze: tuple[float, float]) -> np.ndarray:
    """Draw one frame.

    `openness` is 1 for open and 0 for shut. `gaze` is -1..1 in each axis,
    where +x is the viewer's right and +y is down.
    """
    cover = np.maximum(
        _eye(32.0 - _EYE_DX, 1.0, shape, openness, gaze),
        _eye(32.0 + _EYE_DX, -1.0, shape, openness, gaze),
    )
    rgb = np.asarray(color, dtype=np.float32) * float(np.clip(brightness, 0.0, 1.0))
    frame = cover[:, :, None] * rgb
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)
