from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from detecteur_image_ia.core import RATIO_LOG_MAX, image_to_gray16, vector_field_from_gray16


def _rec709_gray16(red: int, green: int, blue: int) -> int:
    gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return int(gray * 257 + 0.5)


class CoreTests(unittest.TestCase):
    def test_rgb_to_gray16_does_not_overflow(self) -> None:
        pixels = np.array(
            [
                [[0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checker.png"
            Image.fromarray(pixels, mode="RGB").save(path)

            gray = image_to_gray16(path)

        self.assertEqual(int(gray[0, 0]), 0)
        self.assertEqual(int(gray[0, 1]), 65535)
        self.assertEqual(int(gray[1, 0]), 65535)

        vx, vy = vector_field_from_gray16(gray)
        self.assertEqual(int(vx[0, 0]), RATIO_LOG_MAX)
        self.assertEqual(int(vy[0, 0]), RATIO_LOG_MAX)

    def test_ratio_vector_is_centered_on_one(self) -> None:
        gray = np.array([[10, 10], [20, 20]], dtype=np.uint16)

        vx, vy = vector_field_from_gray16(gray)

        self.assertEqual(int(vx[0, 0]), 0)
        self.assertGreater(int(vy[0, 0]), 0)

    def test_rgb_to_gray16_uses_rec709_luma(self) -> None:
        pixels = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [10, 20, 30]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "colors.png"
            Image.fromarray(pixels, mode="RGB").save(path)

            gray = image_to_gray16(path)

        expected = np.array(
            [
                [_rec709_gray16(255, 0, 0), _rec709_gray16(0, 255, 0)],
                [_rec709_gray16(0, 0, 255), _rec709_gray16(10, 20, 30)],
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(gray, expected)


if __name__ == "__main__":
    unittest.main()
