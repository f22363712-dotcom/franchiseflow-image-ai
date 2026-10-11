import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR.parent))

from occlusion import estimate_occlusion_mask  # noqa: E402


def _textured_frame(width=520, height=360):
    rng = np.random.default_rng(4)
    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    for y in range(80, 280, 20):
        for x in range(80, 440, 20):
            color = tuple(int(value) for value in rng.integers(0, 255, 3))
            cv2.rectangle(frame, (x, y), (x + 15, y + 15), color, -1)
    return frame


class OcclusionHintTests(unittest.TestCase):
    def setUp(self):
        self.anchor = _textured_frame()
        self.quad = np.float32([[80, 80], [440, 80], [440, 280], [80, 280]])
        self.motion = np.float32([[1.05, 0.03, 8], [0.01, 1.02, 4], [0, 0, 1]])

    def test_strong_planar_motion_does_not_create_hint(self):
        current = cv2.warpPerspective(self.anchor, self.motion, (520, 360))
        result = estimate_occlusion_mask(self.anchor, current, self.motion, self.quad)
        self.assertEqual(result["status"], "none")
        self.assertEqual(int(np.count_nonzero(result["mask"])), 0)

    def test_large_foreground_block_creates_current_frame_mask(self):
        current = cv2.warpPerspective(self.anchor, self.motion, (520, 360))
        cv2.rectangle(current, (180, 120), (280, 260), (0, 0, 0), -1)
        result = estimate_occlusion_mask(self.anchor, current, self.motion, self.quad)
        self.assertEqual(result["status"], "occlusion_hint")
        self.assertGreater(result["area_fraction"], 0.10)
        self.assertGreater(int(np.count_nonzero(result["mask"])), 1000)


if __name__ == "__main__":
    unittest.main()
