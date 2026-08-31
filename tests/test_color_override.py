"""Colour-override behavior for the face animator."""

import unittest
from pathlib import Path

from output.animator import FaceAnimator

from state import State


class ColorOverrideTests(unittest.TestCase):
    def test_new_state_uses_mood_color_until_overridden(self) -> None:
        state = State()

        self.assertIsNone(state.color)

    def test_explicit_color_override_becomes_the_eased_color_target(self) -> None:
        moods = _load_moods()
        state = State(color=(12, 34, 56))
        animator = FaceAnimator(moods, seed=1)

        animator.tick(state, 0.05)

        self.assertEqual(
            (animator.target["color_r"], animator.target["color_g"],
             animator.target["color_b"]),
            (12.0, 34.0, 56.0),
        )
        self.assertNotEqual(
            (animator.current["color_r"], animator.current["color_g"],
             animator.current["color_b"]),
            (12.0, 34.0, 56.0),
        )


def _load_moods() -> dict:
    root = Path(__file__).resolve().parents[1]
    return __import__("json").loads((root / "moods.json").read_text())


if __name__ == "__main__":
    unittest.main()
