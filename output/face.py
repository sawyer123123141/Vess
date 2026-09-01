"""Rasterises the face into a 64x64 RGB array.

Nothing here knows about time or state. It takes a flat dict of numbers and
draws them. Everything that decides *what* those numbers should be lives in
animator.py.

Rendering is native 64x64 -- never larger and downscaled -- so what the
preview shows is exactly what the LED panel will show.

The two eyes are not a mirrored pair. They are different sizes at different
heights, and every shape below defines each one independently. The mismatch
is the character's identity, so no shape derives one eye from the other.
"""

from __future__ import annotations

import numpy as np

WIDTH = 64
HEIGHT = 64

# A fully shut eye still draws a line. A gap that vanishes entirely reads as
# a crash, not a blink.
_MIN_EYE_H = 1.4

# Which way each eye's inner corner faces. Geometry, not expression, so it is
# fixed here rather than being a shape parameter.
_INNER = {"l": 1.0, "r": -1.0}

_EYE_KEYS = ("cx", "cy", "w", "h", "radius", "slant",
             "arc", "pupil", "pupil_r", "pupil_y")

# Pixel centres, so the distance fields are correct at half-pixel offsets.
_XX, _YY = np.meshgrid(
    np.arange(WIDTH, dtype=np.float32) + 0.5,
    np.arange(HEIGHT, dtype=np.float32) + 0.5,
)

# One parameter set per eye per named shape. Both eyes carry the same keys on
# purpose: the animator interpolates any two shapes componentwise, so a mood
# change is a lerp rather than a swap, and each eye morphs on its own terms.
#
#   cx, cy      eye centre in panel pixels
#   w, h        eye box size
#   radius      corner rounding
#   slant       vertical px the inner corner drops (negative = outer drops)
#   arc         0 = full box, 1 = thin arch; carves the box from below
#   pupil       pupil opacity
#   pupil_r     pupil radius
#   pupil_y     resting pupil offset, positive = low in the eye
#
# Neutral is the baseline the owner specified: left eye 16x22 with its
# top-left at (16, 26), right eye 12x16 at (36, 23). Every other shape is
# authored from that, never derived from it by a transform.
EYE_SHAPES: dict[str, dict[str, dict[str, float]]] = {
    "normal": {
        "l": {"cx": 24.0, "cy": 37.0, "w": 16.0, "h": 22.0, "radius": 6.0,
              "slant": 0.0, "arc": 0.0, "pupil": 1.0, "pupil_r": 4.0,
              "pupil_y": 0.0},
        "r": {"cx": 42.0, "cy": 31.0, "w": 12.0, "h": 16.0, "radius": 4.5,
              "slant": 0.0, "arc": 0.0, "pupil": 1.0, "pupil_r": 3.0,
              "pupil_y": 0.0},
    },
    "arc": {
        "l": {"cx": 24.0, "cy": 37.0, "w": 16.0, "h": 22.0, "radius": 6.0,
              "slant": 0.0, "arc": 0.72, "pupil": 0.0, "pupil_r": 4.0,
              "pupil_y": 0.0},
        "r": {"cx": 42.0, "cy": 31.0, "w": 12.0, "h": 16.0, "radius": 4.5,
              "slant": 0.0, "arc": 0.72, "pupil": 0.0, "pupil_r": 3.0,
              "pupil_y": 0.0},
    },
    "narrow": {
        "l": {"cx": 24.0, "cy": 37.0, "w": 16.0, "h": 10.0, "radius": 4.0,
              "slant": 2.2, "arc": 0.0, "pupil": 1.0, "pupil_r": 3.4,
              "pupil_y": 0.0},
        "r": {"cx": 42.0, "cy": 31.0, "w": 12.0, "h": 8.0, "radius": 3.4,
              "slant": 1.8, "arc": 0.0, "pupil": 1.0, "pupil_r": 2.6,
              "pupil_y": 0.0},
    },
    "droop": {
        "l": {"cx": 24.0, "cy": 38.0, "w": 16.0, "h": 18.0, "radius": 6.0,
              "slant": -2.6, "arc": 0.0, "pupil": 1.0, "pupil_r": 4.0,
              "pupil_y": 2.5},
        "r": {"cx": 42.0, "cy": 32.5, "w": 12.0, "h": 13.0, "radius": 4.5,
              "slant": -2.0, "arc": 0.0, "pupil": 1.0, "pupil_r": 3.0,
              "pupil_y": 2.0},
    },
    # Curious raises the right eye much further than the left, so the gap
    # between them widens rather than the whole face lifting.
    "wide": {
        "l": {"cx": 24.0, "cy": 36.5, "w": 17.0, "h": 23.5, "radius": 6.5,
              "slant": 0.0, "arc": 0.0, "pupil": 1.0, "pupil_r": 3.7,
              "pupil_y": -0.4},
        "r": {"cx": 42.0, "cy": 28.5, "w": 13.5, "h": 18.0, "radius": 5.0,
              "slant": 0.0, "arc": 0.0, "pupil": 1.0, "pupil_r": 2.9,
              "pupil_y": -0.4},
    },
}


def shape_params(name: str) -> dict[str, float]:
    """Parameters for a named shape, flattened to `l_w`, `r_w` and so on.

    Flat because the animator eases the whole face with one loop over the
    keys. Unknown names fall back to normal.
    """
    shape = EYE_SHAPES.get(name, EYE_SHAPES["normal"])
    return {f"{side}_{key}": shape[side][key]
            for side in ("l", "r") for key in _EYE_KEYS}


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


def _eye(
    side: str,
    shape: dict[str, float],
    openness: float,
    gaze: tuple[float, float],
    offset: tuple[float, float],
    eye_offset: tuple[float, float],
) -> np.ndarray:
    def p(key: str) -> float:
        return shape[f"{side}_{key}"]

    half_w = p("w") / 2.0
    half_h = max(p("h") * openness, _MIN_EYE_H) / 2.0

    # `offset` shifts the pair together; `eye_offset` is local to this eye.
    center_x = p("cx") + offset[0] + eye_offset[0]
    center_y = p("cy") + offset[1] + eye_offset[1]
    px = _XX - center_x
    # The slant ramps from zero at the outer edge to its full value at the
    # inner edge, tilting the eye the way a brow does.
    tilt = p("slant") * _INNER[side] * (px / half_w)
    py = _YY - center_y - tilt

    cover = _coverage(_rounded_box(px, py, half_w, half_h, p("radius")))

    arc = p("arc")
    if arc > 1e-3:
        # A circle rising from below. At arc=0 its top just grazes the bottom
        # edge and removes nothing; at arc=1 only a thin arch survives. The
        # radius has to be smaller than the eye is wide or the arch comes out
        # nearly straight.
        r = p("w") * 0.62
        cut_cy = half_h + r - arc * (2.0 * half_h + 1.0)
        cover = cover * (1.0 - _coverage(np.hypot(px, py - cut_cy) - r))

    pupil_alpha = p("pupil")
    pupil_r = min(p("pupil_r"), max(half_h - 0.8, 0.0))
    if pupil_alpha > 1e-3 and pupil_r > 0.2:
        reach_x = max(half_w - pupil_r - 1.2, 0.0)
        reach_y = max(half_h - pupil_r - 1.2, 0.0)
        ox = gaze[0] * reach_x
        oy = float(np.clip(gaze[1] * reach_y + p("pupil_y"), -reach_y, reach_y))
        pupil = _coverage(np.hypot(px - ox, py - oy) - pupil_r)
        cover = cover * (1.0 - pupil_alpha * pupil)

    return cover


def render(
    shape: dict[str, float],
    color: tuple[float, float, float],
    brightness: float,
    openness: float,
    gaze: tuple[float, float],
    offset: tuple[float, float] = (0.0, 0.0),
    eye_offsets: tuple[
        tuple[float, float],
        tuple[float, float],
    ] = ((0.0, 0.0), (0.0, 0.0)),
) -> np.ndarray:
    """Draw one frame.

    `openness` is 1 for open and 0 for shut. `gaze` is -1..1 in each axis,
    where +x is the viewer's right and +y is down. Each pupil travels as far
    as its own eye allows, so the smaller eye moves less.

    `offset` moves the whole pair in panel pixels. `eye_offsets` moves the
    left and right eye bodies independently before pupil gaze is applied.
    Fractional values are intentional: sub-pixel movement changes edge
    brightness instead of snapping by whole pixels.
    """
    cover = np.maximum(
        _eye("l", shape, openness, gaze, offset, eye_offsets[0]),
        _eye("r", shape, openness, gaze, offset, eye_offsets[1]),
    )
    rgb = np.asarray(color, dtype=np.float32) * float(np.clip(brightness, 0.0, 1.0))
    frame = cover[:, :, None] * rgb
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)
