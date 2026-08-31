"""Runtime wiring for Vess targets."""

import unittest

import numpy as np

import main
from state import State


class MainTests(unittest.TestCase):
    def test_default_display_fans_frames_to_the_browser_target(self) -> None:
        build_display = getattr(main, "_build_display", None)
        self.assertIsNotNone(build_display, "display target builder is missing")
        config = {
            "display": {"cv2_enabled": False, "preview_scale": 8},
            "web": {"enabled": True, "port": 8080},
        }
        display, server = build_display(config, State())
        self.assertIsNotNone(server)

        display.show(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertIsNotNone(server.preview.png())


if __name__ == "__main__":
    unittest.main()
