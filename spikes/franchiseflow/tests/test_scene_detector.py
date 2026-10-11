import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "spikes" / "franchiseflow"))

from scene_detector import (  # noqa: E402
    SceneSegment,
    detect_scenes,
    frame_difference_score,
    histogram_correlation,
)


def _textured_frame(seed: int, colour: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.empty((64, 96, 3), dtype=np.uint8)
    frame[:] = colour
    noise = rng.integers(0, 20, size=frame.shape, dtype=np.uint8)
    return cv2.add(frame, noise)


class SceneDetectorTests(unittest.TestCase):
    def test_similarity_scores_are_bounded_and_distinguish_small_motion(self):
        first = _textured_frame(1, (40, 80, 140))
        nearby = np.roll(first, 1, axis=1)
        far = _textured_frame(2, (180, 30, 20))

        self.assertGreaterEqual(frame_difference_score(first, nearby), 0.0)
        self.assertLessEqual(frame_difference_score(first, nearby), 1.0)
        self.assertLess(frame_difference_score(first, nearby), frame_difference_score(first, far))
        self.assertGreater(histogram_correlation(first, nearby), histogram_correlation(first, far))

    def test_detects_high_confidence_hard_cut_and_keeps_frame_boundaries(self):
        frames = [
            (10, _textured_frame(1, (40, 80, 140))),
            (11, _textured_frame(2, (40, 80, 140))),
            (12, _textured_frame(3, (40, 80, 140))),
            (13, _textured_frame(4, (180, 30, 20))),
            (14, _textured_frame(5, (180, 30, 20))),
        ]

        scenes = detect_scenes(frames)

        self.assertEqual(scenes, [
            SceneSegment(scene_id="scene_001", start_frame=10, end_frame=12, cut_before=None, cut_score=0.0),
            SceneSegment(scene_id="scene_002", start_frame=13, end_frame=14, cut_before=13, cut_score=scenes[1].cut_score),
        ])
        self.assertGreater(scenes[1].cut_score, 0.5)

    def test_continuous_camera_motion_does_not_create_scene(self):
        base = _textured_frame(7, (70, 90, 120))
        frames = [(i, np.roll(base, i, axis=1)) for i in range(6)]

        scenes = detect_scenes(frames)

        self.assertEqual(len(scenes), 1)
        self.assertEqual((scenes[0].start_frame, scenes[0].end_frame), (0, 5))

    def test_empty_and_single_frame_inputs_have_defined_results(self):
        self.assertEqual(detect_scenes([]), [])
        self.assertEqual(
            detect_scenes([(42, _textured_frame(1, (10, 20, 30)))]),
            [SceneSegment(scene_id="scene_001", start_frame=42, end_frame=42, cut_before=None, cut_score=0.0)],
        )

    def test_rejects_non_monotonic_frame_indices(self):
        frame = _textured_frame(1, (10, 20, 30))
        with self.assertRaises(ValueError):
            detect_scenes([(4, frame), (4, frame)])


if __name__ == "__main__":
    unittest.main()
