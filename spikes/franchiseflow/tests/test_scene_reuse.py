import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_reuse import transfer_candidate


def _textured_frame(width=360, height=260):
    frame = np.full((height, width, 3), 72, dtype=np.uint8)
    for y in range(48, 150, 12):
        for x in range(55, 275, 12):
            cv2.rectangle(frame, (x, y), (x + 8, y + 8), ((x * 3) % 255, (y * 5) % 255, 220), -1)
            cv2.putText(frame, f"{x}{y}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 255), 1)
    return frame


class SceneReuseTests(unittest.TestCase):
    def test_transfers_verified_planar_candidate_to_new_anchor(self):
        anchor = _textured_frame()
        matrix = np.float32([[1.0, 0.015, 28.0], [-0.01, 1.0, 12.0], [0.00002, 0.0001, 1.0]])
        target = cv2.warpPerspective(anchor, matrix, (anchor.shape[1], anchor.shape[0]))
        result = transfer_candidate(
            anchor,
            target,
            [[50, 40], [280, 40], [280, 155], [50, 155]],
            min_inliers=8,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "validated_cross_scene_transfer")
        self.assertGreaterEqual(result["transfer"]["inliers"], 8)
        self.assertLess(result["transfer"]["median_residual_px"], 2.0)

    def test_weak_scene_does_not_fabricate_a_candidate(self):
        anchor = _textured_frame()
        target = np.full_like(anchor, 18)
        result = transfer_candidate(anchor, target, [[50, 40], [280, 40], [280, 155], [50, 155]])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
