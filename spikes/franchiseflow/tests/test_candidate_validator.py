import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_validator import validate_candidate


def _textured_frame(width=320, height=240):
    frame = np.full((height, width, 3), 80, dtype=np.uint8)
    for y in range(45, 145, 12):
        for x in range(60, 260, 12):
            cv2.rectangle(frame, (x, y), (x + 8, y + 8), ((x * 3) % 255, (y * 5) % 255, 220), -1)
            cv2.putText(frame, f"{x}{y}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 255), 1)
    return frame


class CandidateValidatorTests(unittest.TestCase):
    def test_accepts_planar_candidate_consistent_across_three_samples(self):
        first = _textured_frame()
        matrix = np.float32([[1, 0, 5], [0, 1, 3], [0, 0, 1]])
        middle = cv2.warpPerspective(first, matrix, (first.shape[1], first.shape[0]))
        last = cv2.warpPerspective(first, np.float32([[1, 0, 12], [0, 1, 7], [0, 0, 1]]), (first.shape[1], first.shape[0]))
        result = validate_candidate(
            [(0, first), (5, middle), (9, last)],
            {"quad": [[55, 40], [265, 40], [265, 150], [55, 150]], "confidence": 0.9, "reason": "board"},
        )
        self.assertTrue(result["accepted"])
        self.assertFalse(result["requires_human_review"])
        self.assertEqual(len(result["sample_metrics"]), 3)

    def test_requires_review_when_later_sample_leaves_image(self):
        first = _textured_frame()
        later = np.roll(first, 180, axis=1)
        result = validate_candidate(
            [(0, first), (1, later)],
            {"quad": [[55, 40], [265, 40], [265, 150], [55, 150]], "confidence": 0.9, "reason": "board"},
        )
        self.assertFalse(result["accepted"])
        self.assertTrue(result["requires_human_review"])
        self.assertTrue(any(reason in result["reasons"] for reason in ("quad_out_of_bounds", "inconsistent_samples")))

    def test_rejects_candidate_without_fabricating_quad(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = validate_candidate(
            [(0, frame)],
            {"quad": [[10, 10], [10, 10], [10, 10], [10, 10]], "confidence": 0.1, "reason": "unclear"},
        )
        self.assertFalse(result["accepted"])
        self.assertIn("invalid_quad", result["reasons"])
        self.assertNotIn("quad", result)


if __name__ == "__main__":
    unittest.main()
