"""Independent eye-offset renderer regressions."""

import unittest

import numpy as np

from output import face


class FaceRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = face.shape_params("normal")
        self.color = (100.0, 180.0, 255.0)

    def test_zero_eye_offsets_are_byte_identical_to_legacy_call(self) -> None:
        legacy = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.25, -0.10),
            (0.4, -0.3),
        )
        explicit = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.25, -0.10),
            (0.4, -0.3),
            eye_offsets=((0.0, 0.0), (0.0, 0.0)),
        )
        np.testing.assert_array_equal(legacy, explicit)

    def test_left_eye_offset_moves_left_eye_without_moving_right_eye(self) -> None:
        baseline = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.0, 0.0),
            eye_offsets=((0.0, 0.0), (0.0, 0.0)),
        )
        shifted = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.0, 0.0),
            eye_offsets=((1.0, 0.0), (0.0, 0.0)),
        )

        left_slice = np.s_[:, :34, :]
        right_slice = np.s_[:, 34:, :]
        self.assertFalse(np.array_equal(baseline[left_slice], shifted[left_slice]))
        np.testing.assert_array_equal(baseline[right_slice], shifted[right_slice])

    def test_gaze_stays_relative_to_translated_eye_body(self) -> None:
        centered = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.8, 0.0),
            eye_offsets=((0.0, 0.0), (0.0, 0.0)),
        )
        translated = face.render(
            self.shape,
            self.color,
            1.0,
            1.0,
            (0.8, 0.0),
            eye_offsets=((0.75, -0.25), (0.0, 0.0)),
        )
        self.assertFalse(np.array_equal(centered, translated))


if __name__ == "__main__":
    unittest.main()
